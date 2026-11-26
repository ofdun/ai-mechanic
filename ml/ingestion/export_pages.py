"""Export physical PDF pages as JSONL or one ParsedDocument JSON per manual."""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pypdfium2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PAGE = re.compile(r"(?:(\d{4}-\d{2}-\d{2})\s+)?Страница\s+([1-9]\d*)", re.IGNORECASE)


def read_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if not documents:
        raise ValueError("В каталоге нет PDF.")

    filenames = defaultdict(list)
    for path in manifest_path.parent.iterdir():
        filenames[unicodedata.normalize("NFC", path.name)].append(path)

    seen = set()
    for document in documents:
        document_id = document["document_id"]
        if document_id in seen:
            raise ValueError(f"Повторяющийся document_id: {document_id}")
        seen.add(document_id)

        filename = document["file"]
        matches = filenames[unicodedata.normalize("NFC", filename)]
        if len(matches) != 1 or not matches[0].is_file():
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


def read_document_pages(document, source):
    digest = hashlib.sha256()
    with source.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != document["sha256"]:
        raise ValueError(f"SHA-256 отличается от каталога: {source.name}")

    sections = document.get("sections", [])
    offset = document.get("printed_page_offset")
    table_rows = {row["pdf_page"]: row for row in document.get("table_rows", [])}
    with pypdfium2.PdfDocument(source) as pdf:
        page_count = len(pdf)
        if page_count != document["expected_pages"]:
            raise ValueError(
                f"{source.name}: {page_count} страниц вместо {document['expected_pages']}"
            )

        for page_index in range(page_count):
            pdf_page = page_index + 1
            page = pdf[page_index]
            try:
                text_page = page.get_textpage()
                raw_text = text_page.get_text_bounded()
                if pdf_page in table_rows:
                    raw_text = format_table_row(
                        text_page, raw_text.replace("\r\n", "\n"), table_rows[pdf_page]
                    )
            finally:
                page.close()
            printed_page, text = page_content(raw_text, pdf_page, offset)
            section = next(
                (item["title"] for item in sections if item["first_page"] <= pdf_page <= item["last_page"]),
                None,
            )
            yield {
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "section": section,
                "text": text,
            }


def export_pages(manifest_path, output):
    documents = list(read_manifest(manifest_path))
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
                for page in read_document_pages(document, source):
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


def export_parsed_documents(manifest_path, output):
    documents = list(read_manifest(manifest_path))
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
                for page in read_document_pages(document, source)
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
    parser.add_argument("--format", choices=("pages", "parsed"), default="pages")
    parser.add_argument(
        "--output", type=Path,
        help="Файл pages.jsonl для pages или каталог artifacts/parsed для parsed.",
    )
    args = parser.parse_args()

    try:
        if args.format == "parsed":
            artifacts = Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
            output = args.output or artifacts / "parsed"
            documents, pages = export_parsed_documents(args.manifest, output)
        else:
            output = args.output or args.manifest.with_name("pages.jsonl")
            documents, pages = export_pages(args.manifest, output)
    except (OSError, ValueError, KeyError, TypeError, pypdfium2.PdfiumError) as error:
        print(f"Ошибка выгрузки: {error}", file=sys.stderr)
        return 1

    print(f"{output}: {documents} PDF, {pages} страниц")
    return 0


if __name__ == "__main__":
    sys.exit(main())
