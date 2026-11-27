"""Compare OCR settings on a fixed local set of hand-checked PDF regions."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from statistics import mean
from time import perf_counter

from evaluation.ingestion import extract, score
from ml.ingestion.export_pages import read_manifest, verify_source
from ml.ingestion.page_ocr import PageOCR


class PredictionOptions:
    def __init__(self, pipeline, options, grayscale=False):
        self.pipeline = pipeline
        self.options = options
        self.grayscale = grayscale

    def predict(self, input, **options):
        if self.grayscale:
            import cv2
            input = cv2.cvtColor(cv2.cvtColor(input, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return self.pipeline.predict(input=input, **{**options, **self.options})


class TableRouter:
    def __init__(self, structure, text):
        self.structure = structure
        self.text = text

    def read(self, page, clip=None):
        result = self.structure.read(page, clip)
        return result if result.tables else self.text.read(page, clip)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--omnidocbench", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    configs = json.loads(args.configs.read_text())
    if isinstance(configs, dict):
        configs = [row["config"] for row in configs["profiles"]]
    documents = {entry["document_id"]: (entry, path) for entry, path in read_manifest(args.manifest, args.data_dir)}
    for document_id in {case["document_id"] for case in cases}:
        verify_source(*documents[document_id])
    metric = runpy.run_path(str(args.omnidocbench / "src/metrics/table_metric.py"))["TEDS"]
    teds = (metric(), metric(structure_only=True))
    report = {"gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(), "profiles": []}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report["gold_sha256"] != hashlib.sha256(args.gold.read_bytes()).hexdigest():
            raise ValueError("The output belongs to a different gold dataset.")
    pipelines = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        if any(row["config"] == config for row in report["profiles"]):
            continue
        mode = config["mode"]
        detector = config.get("detector", "PP-OCRv5_mobile_det")
        orientation = config.get("orientation", False)
        key = (mode, detector, orientation)
        if key not in pipelines:
            if mode == "paddleocr" and (orientation or detector != "PP-OCRv5_mobile_det"):
                from paddleocr import PaddleOCR
                pipelines[key] = PaddleOCR(
                    device="cpu", enable_mkldnn=False, cpu_threads=4,
                    text_detection_model_name=detector,
                    text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=orientation,
                    text_det_limit_side_len=1600, text_det_limit_type="max",
                )
            else:
                pipelines[key] = PageOCR(mode).pipeline
        reader = PageOCR(mode, dpi=config.get("dpi", 250))
        reader.pipeline = PredictionOptions(pipelines[key], config.get("predict", {}), config.get("grayscale", False))
        if config.get("route_text", False):
            text = PageOCR("paddleocr")
            text_key = ("paddleocr", "PP-OCRv5_mobile_det", False)
            if text_key not in pipelines:
                pipelines[text_key] = text.pipeline
            text.pipeline = pipelines[text_key]
            reader = TableRouter(reader, text)
        rows = []
        for case in cases:
            if "table_html" in case and mode != "pp-structure":
                continue
            document, source = documents[case["document_id"]]
            started = perf_counter()
            result = extract(source, document, case, mode, reader)
            row = {"id": case["id"], "seconds": perf_counter() - started, **score(case, result, teds)}
            rows.append(row)
        keys = set().union(*(row.keys() for row in rows)) - {"id"} if rows else set()
        summary = {key: mean(row[key] for row in rows if key in row) for key in sorted(keys)}
        report["profiles"].append({"config": config, "summary": summary, "cases": rows})
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
        print(config["id"], json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
