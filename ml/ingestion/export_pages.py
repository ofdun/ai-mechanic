"""Export physical PDF pages as JSONL or one ParsedDocument JSON per manual."""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pypdfium2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PAGE = re.compile(r"(?:(\d{4}-\d{2}-\d{2})\s+)?Страница\s+([1-9]\d*)", re.IGNORECASE)


def read_manifest(manifest_path, data_dir=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if not documents:
        raise ValueError("В каталоге нет PDF.")

    filenames = defaultdict(list)
    for path in (data_dir or manifest_path.parent).iterdir():
        filenames[unicodedata.normalize("NFC", path.name)].append(path)

    seen = set()
    for document in documents:
        document_id = document["document_id"]
        if document_id in seen:
            raise ValueError(f"Повторяющийся document_id: {document_id}")
        seen.add(document_id)

        filename = document["file"]
        matches = filenames[unicodedata.normalize("NFC", filename)]
        if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
            raise ValueError(f"Не найден единственный файл для {document_id}: {filename}")
        yield document, matches[0]


def page_content(raw_text, pdf_page, offset):
    lines = [line.rstrip() for line in raw_text.strip().splitlines()]
    if not lines:
        return None, ""

    printed_page = None
    if offset is None:
        # Module numbering can restart, so read the explicit folio on this page.
        markers = [MODULE_PAGE.fullmatch(line.strip()) for line in lines]
        numbers = {marker[2] for marker in markers if marker}
        if len(numbers) == 1:
            printed_page = numbers.pop()
            lines = [(marker[1] or "") if marker else line for line, marker in zip(lines, markers)]
    else:
        # A matching boundary line confirms the expected offset for this edition.
        expected = str(pdf_page + offset)
        if lines[0].strip() == expected:
            printed_page = lines.pop(0).strip()
        elif lines[-1].strip() == expected:
            printed_page = lines.pop().strip()

    return printed_page, "\n".join(lines).replace("\x02", "").strip()


def format_table_row(text_page, raw_text, row):
    # PDFium follows the cell order here. Preserve the pairing in a two-column layout.
    cause = text_page.get_text_bounded(*row["cause_rect"]).strip().splitlines()
    operation = text_page.get_text_bounded(*row["operation_rect"]).strip().splitlines()
    original = "\n".join(cause + operation)
    if not cause or not operation or raw_text.count(original) != 1:
        raise ValueError(f"Не удалось сопоставить строку таблицы на PDF {row['pdf_page']}.")
    width = max(map(len, cause)) + 2
    aligned = "\n".join(
        f"{left:<{width}}{right}" for left, right in zip_longest(cause, operation, fillvalue="")
    )
    return raw_text.replace(original, aligned, 1)


def manual_name(document):
    return (document.get("designation") or "").strip() or document["title"]


def visual_table_text(text_page):
    """Read the reviewed table whose PDF content stream stores digits out of display order."""
    glyphs = []
    for index in range(text_page.count_chars()):
        char = text_page.get_text_range(index, 1)
        if char.isprintable() and char.strip():
            left, bottom, right, top = text_page.get_charbox(index, loose=True)
            glyphs.append(((bottom + top) / 2, left, right, char))

    lines = []
    for height, left, right, char in sorted(glyphs, reverse=True):
        if not lines or lines[-1][0] - height > 2.8:
            lines.append((height, []))
        lines[-1][1].append((left, right, char))
    result = []
    for _, line in lines:
        pieces = []
        previous = None
        for left, right, char in sorted(line):
            if previous is not None and left - previous > 1.4:
                pieces.append("    " if left - previous > 8 else " ")
            pieces.append(char)
            previous = right
        result.append("".join(pieces))
    return "\n".join(result)


def extract_text(text_page, document, pdf_page):
    if pdf_page in document.get("visual_order_pages", []):
        return visual_table_text(text_page)
    text = text_page.get_text_bounded().replace("\r\n", "\n")
    for row in document.get("table_rows", []):
        if row["pdf_page"] == pdf_page:
            text = format_table_row(text_page, text, row)
    return text


def verify_source(document, source):
    digest = hashlib.sha256()
    with source.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != document["sha256"]:
        raise ValueError(f"SHA-256 отличается от каталога: {source.name}")


def read_document_pages(document, source, *, text_engine="pdfium", ocr=None, mixed_ocr=False):
    if mixed_ocr and ocr is None:
        raise ValueError("Распознавание смешанных страниц требует --ocr.")
    verify_source(document, source)
    sections = document.get("sections", [])
    offset = document.get("printed_page_offset")
    reviewed_pages = set(document.get("visual_order_pages", []))
    reviewed_pages.update(row["pdf_page"] for row in document.get("table_rows", []))
    with ExitStack() as stack:
        pdf = stack.enter_context(pypdfium2.PdfDocument(source))
        mu_pdf = None
        if text_engine == "pymupdf" or ocr is not None:
            try:
                import pymupdf
            except ImportError as error:
                raise ValueError("Для этого режима нужен PyMuPDF.") from error
            mu_pdf = stack.enter_context(pymupdf.open(source))
        page_count = len(pdf)
        if page_count != document["expected_pages"]:
            raise ValueError(
                f"{source.name}: {page_count} страниц вместо {document['expected_pages']}"
            )

        for page_index in range(page_count):
            pdf_page = page_index + 1
            if text_engine == "pymupdf" and pdf_page not in reviewed_pages:
                raw_text = mu_pdf[page_index].get_text(sort=True)
            else:
                page = pdf[page_index]
                try:
                    raw_text = extract_text(page.get_textpage(), document, pdf_page)
                finally:
                    page.close()
            printed_page, text = page_content(raw_text, pdf_page, offset)
            structures = {}
            if ocr is not None:
                try:
                    if not text:
                        regions = [None]
                    elif mixed_ocr and pdf_page not in reviewed_pages:
                        if __package__:
                            from .page_ocr import raster_regions
                        else:
                            from page_ocr import raster_regions
                        regions = raster_regions(mu_pdf[page_index])
                    else:
                        regions = []
                    extra_text, blocks = [], []
                    for region in regions:
                        recognized = ocr.read(mu_pdf[page_index], region)
                        ocr_folio, recognized_text = page_content(recognized.text, pdf_page, offset)
                        printed_page = printed_page or ocr_folio
                        if recognized_text:
                            extra_text.append(recognized_text)
                        blocks.extend(recognized.blocks)
                    if extra_text:
                        text = "\n\n".join(part for part in (text, *extra_text) if part)
                    if blocks:
                        structures = {
                            "blocks": blocks,
                            "extraction": {
                                "ocr_engine": ocr.mode, "dpi": ocr.dpi,
                                "table_html": getattr(ocr, "table_html", "model"),
                                "padding_pixels": getattr(ocr, "padding", 0),
                                "coordinate_system": "displayed_pdf_points",
                                "page_size": list(mu_pdf[page_index].rect)[2:],
                                "regions": len(regions),
                            },
                        }
                except (RuntimeError, ValueError) as error:
                    raise ValueError(f"{document['document_id']}, физическая стр. {pdf_page}: {error}") from error
            section = next(
                (item["title"] for item in sections if item["first_page"] <= pdf_page <= item["last_page"]),
                None,
            )
            yield {
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "section": section,
                "text": text,
                **structures,
            }


def export_pages(manifest_path, output, *, data_dir=None, text_engine="pdfium", ocr=None, mixed_ocr=False):
    documents = list(read_manifest(manifest_path, data_dir))
    input_paths = {manifest_path.resolve(), *(source.resolve() for _, source in documents)}
    if output.resolve() in input_paths:
        raise ValueError(f"Путь выгрузки совпадает с исходным файлом: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            page_count = 0
            for document, source in documents:
                manual = manual_name(document)
                for page in read_document_pages(document, source, text_engine=text_engine, ocr=ocr,
                                                mixed_ocr=mixed_ocr):
                    record = {
                        "document_id": document["document_id"],
                        "manual": manual,
                        "models": document["models"],
                        **page,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    page_count += 1
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return len(documents), page_count


def export_parsed_documents(manifest_path, output, *, data_dir=None, text_engine="pdfium", ocr=None, mixed_ocr=False):
    documents = list(read_manifest(manifest_path, data_dir))
    input_paths = {manifest_path.resolve(), *(source.resolve() for _, source in documents)}
    filenames = []
    for document, _ in documents:
        document_id = document["document_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", document_id):
            raise ValueError(f"document_id непригоден для имени файла: {document_id}")
        filename = f"{document_id}.json"
        target = output / filename
        if target.resolve() in input_paths or target.is_dir():
            raise ValueError(f"Нельзя записать выгрузку по пути: {target}")
        filenames.append(filename)
    if len({name.casefold() for name in filenames}) != len(filenames):
        raise ValueError("Имена файлов документов различаются только регистром.")

    output.mkdir(parents=True, exist_ok=True)
    page_count = 0
    # Prepare the whole corpus before replacing any previously exported JSON.
    with TemporaryDirectory(dir=output, prefix=".parsed-") as temp:
        staging = Path(temp)
        for (document, source), filename in zip(documents, filenames):
            pages = [
                {key: value for key, value in page.items() if value is not None}
                for page in read_document_pages(document, source, text_engine=text_engine, ocr=ocr,
                                                mixed_ocr=mixed_ocr)
            ]
            parsed_document = {
                "document_id": document["document_id"],
                "manual": manual_name(document),
                "engine_models": document["models"],
                "pages": pages,
            }
            with (staging / filename).open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(parsed_document, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            page_count += len(pages)
        for filename in filenames:
            (staging / filename).replace(output / filename)

    return len(documents), page_count


def main():
    parser = argparse.ArgumentParser(description="Выгрузка текстового слоя PDF по страницам или руководствам.")
    default_data = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
    parser.add_argument("--manifest", type=Path, default=default_data / "manifest.json")
    parser.add_argument("--data-dir", type=Path, help="Каталог PDF, если они лежат отдельно от манифеста.")
    parser.add_argument("--format", choices=("pages", "parsed"), default="pages")
    parser.add_argument("--text-engine", choices=("pdfium", "pymupdf"), default="pdfium")
    parser.add_argument("--ocr", choices=("paddleocr", "pp-structure"),
                        help="Локальное OCR для страниц без текста после очистки.")
    parser.add_argument("--mixed-ocr", action="store_true",
                        help="Дополнительно распознавать растровые области страниц с текстом.")
    parser.add_argument("--table-html", choices=("model", "cells"), default="model",
                        help="Сборка HTML по структуре модели или по ячейкам линованной таблицы.")
    parser.add_argument("--ocr-padding", type=int, default=0,
                        help="Ширина белых полей вокруг изображения для PaddleOCR, в пикселях.")
    parser.add_argument(
        "--output", type=Path,
        help="Файл pages.jsonl для pages или каталог artifacts/parsed для parsed.",
    )
    args = parser.parse_args()

    try:
        ocr = None
        if args.ocr:
            if __package__:
                from .page_ocr import PageOCR
            else:
                from page_ocr import PageOCR
            ocr = PageOCR(args.ocr, table_html=args.table_html, padding=args.ocr_padding)
        elif args.table_html != "model":
            raise ValueError("--table-html cells требует --ocr pp-structure.")
        elif args.ocr_padding:
            raise ValueError("--ocr-padding требует --ocr paddleocr.")
        options = {"data_dir": args.data_dir, "text_engine": args.text_engine, "ocr": ocr,
                   "mixed_ocr": args.mixed_ocr}
        if args.format == "parsed":
            artifacts = Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
            output = args.output or artifacts / "parsed"
            documents, pages = export_parsed_documents(args.manifest, output, **options)
        else:
            output = args.output or args.manifest.with_name("pages.jsonl")
            documents, pages = export_pages(args.manifest, output, **options)
    except (OSError, ValueError, KeyError, TypeError, pypdfium2.PdfiumError) as error:
        print(f"Ошибка выгрузки: {error}", file=sys.stderr)
        return 1

    print(f"{output}: {documents} PDF, {pages} страниц")
    return 0


if __name__ == "__main__":
    sys.exit(main())
