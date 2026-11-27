"""Compare parser outputs on manually transcribed regions of verified PDF pages."""

import argparse
import json
import re
import runpy
import sys
import unicodedata
from contextlib import closing, redirect_stdout
from html import escape
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import pymupdf
import pypdfium2
from rapidfuzz.distance import Levenshtein

from ml.ingestion.export_pages import extract_text, read_manifest, verify_source
from ml.ingestion.page_ocr import OCRPage, PageOCR


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def reading_order(text, anchors):
    expected = [normalize(anchor) for anchor in anchors]
    if any(not anchor for anchor in expected) or len(set(expected)) != len(expected):
        raise ValueError("Метки порядка чтения должны быть непустыми и уникальными.")
    text = normalize(text)
    found = [(text.find(anchor), index) for index, anchor in enumerate(expected) if anchor in text]
    observed = [index for _, index in sorted(found)]
    return Levenshtein.normalized_distance(list(range(len(expected))), observed)


def matrix_html(rows):
    return "<table><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(value or '')}</td>" for value in row) + "</tr>"
        for row in rows
    ) + "</tbody></table>"


def score(case, result, teds):
    text = normalize(result.text)
    metrics = {"has_text": bool(text)}
    if "text" in case:
        metrics["text_edit_distance"] = Levenshtein.normalized_distance(normalize(case["text"]), text)
    if "anchors" in case:
        metrics["reading_order_edit_distance"] = reading_order(text, case["anchors"])
    if "rows" in case:
        lines = [normalize(line) for line in result.text.splitlines()]
        expected = [
            [re.compile(rf"(?<!\w){re.escape(normalize(field))}(?!\w)") for field in row]
            for row in case["rows"]
        ]
        matches = [
            any(all(field.search(line) for field in row) for line in lines)
            for row in expected
        ]
        metrics["exact_rows"] = mean(matches)
    if "table_html" in case:
        from lxml.html import document_fromstring, tostring

        def wrap_table(value):
            tables = document_fromstring(value).xpath("//table")
            return "<html><body>" + tostring(tables[0], encoding="unicode") + "</body></html>" if tables else ""

        expected = wrap_table(case["table_html"])
        observed = wrap_table(result.tables[0]) if result.tables else ""
        metrics["detected_tables"] = len(result.tables)
        metrics["teds"] = teds[0].evaluate(observed, expected)
        metrics["teds_structure"] = teds[1].evaluate(observed, expected)
    return metrics


def extract(path, document, case, engine, ocr):
    number = case["pdf_page"]
    if not isinstance(number, int) or not 1 <= number <= document["expected_pages"]:
        raise ValueError(f"Недопустимая физическая страница: {number}")
    box = case.get("bbox")
    with pymupdf.open(path) as pdf:
        page = pdf[number - 1]
        clip = None
        if box is not None:
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"Некорректный bbox: {box}")
            clip = pymupdf.Rect(x0 * page.rect.width, y0 * page.rect.height,
                                x1 * page.rect.width, y1 * page.rect.height)
        if ocr is not None:
            return ocr.read(page, clip)
        if engine == "pymupdf":
            raw = page.get_text(sort=True, clip=clip)
            tables = ()
            if "table_html" in case:
                with redirect_stdout(sys.stderr):
                    tables = tuple(matrix_html(table.extract()) for table in page.find_tables(clip=clip).tables)
            return OCRPage(raw, tables)
    with pypdfium2.PdfDocument(path) as pdf, closing(pdf[number - 1]) as page, closing(page.get_textpage()) as text:
        if box is None:
            return OCRPage(extract_text(text, document, number))
        left, bottom, right, top = page.get_bbox()
        width, height = right - left, top - bottom
        return OCRPage(text.get_text_bounded(left + x0 * width, top - y1 * height,
                                            left + x1 * width, top - y0 * height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=("pdfium", "pymupdf", "paddleocr", "pp-structure"), required=True)
    parser.add_argument("--omnidocbench", type=Path, help="Локальный checkout OmniDocBench для официального TEDS.")
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    teds = None
    if any("table_html" in case for case in cases):
        if args.omnidocbench is None:
            parser.error("Для табличных эталонов нужен --omnidocbench.")
        scorer = runpy.run_path(str(args.omnidocbench / "src/metrics/table_metric.py"))["TEDS"]
        teds = (scorer(), scorer(structure_only=True))
    documents = {document["document_id"]: (document, path) for document, path in
                 read_manifest(args.manifest, args.data_dir)}
    for document_id in {case["document_id"] for case in cases}:
        verify_source(*documents[document_id])
    ocr = PageOCR(args.engine) if args.engine in {"paddleocr", "pp-structure"} else None
    startup = perf_counter()
    if ocr is not None:
        _ = ocr.pipeline
    startup = perf_counter() - startup
    results = []
    for case in cases:
        document, path = documents[case["document_id"]]
        start = perf_counter()
        result = extract(path, document, case, args.engine, ocr)
        elapsed = perf_counter() - start
        results.append({"id": case["id"], "seconds": elapsed, **score(case, result, teds)})
    measures = set().union(*(row.keys() for row in results)) - {"id", "seconds"}
    summary = {key: mean(row[key] for row in results if key in row) for key in sorted(measures)}
    if results:
        summary["median_seconds"] = median(row["seconds"] for row in results)
    print(json.dumps({"engine": args.engine, "model_startup_seconds": startup,
                      "summary": summary, "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
