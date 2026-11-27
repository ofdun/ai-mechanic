import hashlib
import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from ml.ingestion.export_pages import export_pages, export_parsed_documents, read_document_pages
from ml.ingestion.page_ocr import OCRPage, raster_regions
from ml.ingestion.page_ocr import PageOCR
from ml.retrieval import load_documents


@pytest.fixture
def mixed(tmp_path):
    image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 300, 120), False)
    image.clear_with(220)
    source = tmp_path / "mixed.pdf"
    with pymupdf.open() as pdf:
        for rotation in (0, 90):
            page = pdf.new_page(width=400, height=500)
            page.insert_text((30, 30), "Native heading for this mixed page")
            page.insert_image(pymupdf.Rect(30, 100, 330, 220), pixmap=image)
            page.set_rotation(rotation)
        page = pdf.new_page(width=400, height=500)
        page.insert_text((30, 30), "Native text page without images")
        pdf.save(source)
    document = {"document_id": "mixed", "models": ["Synthetic"], "title": "Fixture",
                "file": source.name, "expected_pages": 3,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [document]}))
    return document, source, manifest


class OCR:
    mode = "pp-structure"
    dpi = 250

    def __init__(self):
        self.regions = []

    def read(self, page, clip=None):
        self.regions.append(clip)
        html = '<table><tr><td rowspan="2">Fastener</td><td>100-120</td></tr></table>'
        return OCRPage("Fastener | 100-120", (html,), [
            {"type": "table", "text": "Fastener | 100-120", "html": html, "bbox": list(clip)},
        ])


def test_mixed_page_recovers_raster_and_keeps_native_text(mixed):
    document, source, _ = mixed
    reader = OCR()
    legacy = list(read_document_pages(document, source, ocr=reader))
    assert reader.regions == []
    improved = list(read_document_pages(document, source, ocr=reader, mixed_ocr=True))
    assert len(reader.regions) == 2
    assert improved[2] == legacy[2]
    for old, new in zip(legacy[:2], improved[:2]):
        assert new["text"].startswith(old["text"])
        assert new["text"].count("Native heading") == 1
        assert "Fastener | 100-120" in new["text"]
        assert 'rowspan="2"' in new["blocks"][0]["html"]
        assert new["extraction"]["coordinate_system"] == "displayed_pdf_points"
    with pymupdf.open(source) as pdf:
        assert reader.regions[0] == pymupdf.Rect(30, 100, 330, 220)
        assert reader.regions[1] == pymupdf.Rect(30, 100, 330, 220) * pdf[1].rotation_matrix


def test_exports_keep_html_and_existing_index_loader_accepts_both(mixed, tmp_path):
    _, _, manifest = mixed
    output = tmp_path / "pages.jsonl"
    parsed = tmp_path / "parsed"
    assert export_pages(manifest, output, ocr=OCR(), mixed_ocr=True) == (1, 3)
    assert export_parsed_documents(manifest, parsed, ocr=OCR(), mixed_ocr=True) == (1, 3)
    record = json.loads(output.read_text().splitlines()[0])
    manual = json.loads((parsed / "mixed.json").read_text())
    assert record["blocks"] == manual["pages"][0]["blocks"]
    assert load_documents(output, manifest)[0].pages[0].text == load_documents(parsed)[0].pages[0].text


def test_existing_native_text_inside_image_is_not_read_twice(mixed):
    _, source, _ = mixed
    with pymupdf.open(source) as pdf:
        page = pdf[0]
        page.insert_text((40, 140), "An existing searchable OCR layer over the image")
        assert raster_regions(page) == []


def test_mixed_flag_needs_an_ocr_engine(mixed):
    document, source, _ = mixed
    with pytest.raises(ValueError, match="--ocr"):
        list(read_document_pages(document, source, mixed_ocr=True))


def test_table_html_mode_reaches_the_recognition_pipeline(mixed):
    _, source, _ = mixed
    calls = []

    class Pipeline:
        def predict(self, input, **options):
            calls.append(options)
            return [{"parsing_res_list": []}]

    with pymupdf.open(source) as pdf:
        for mode in ("model", "cells"):
            reader = PageOCR("pp-structure", table_html=mode)
            reader.pipeline = Pipeline()
            reader.read(pdf[0])
    assert calls == [{"use_wired_table_cells_trans_to_html": False},
                     {"use_wired_table_cells_trans_to_html": True}]
    with pytest.raises(ValueError, match="pp-structure"):
        PageOCR("paddleocr", table_html="cells")


def test_padding_is_white_and_coordinates_stay_on_the_original_page(mixed):
    import numpy as np

    _, source, _ = mixed
    padding = 64
    reader = PageOCR("paddleocr", padding=padding)

    class Pipeline:
        def predict(self, input, **options):
            assert (input[:padding] == 255).all()
            assert (input[-padding:] == 255).all()
            assert (input[:, :padding] == 255).all()
            assert (input[:, -padding:] == 255).all()
            return [{"rec_texts": ["Detected"], "rec_scores": [0.9],
                     "rec_boxes": np.array([[padding, padding, padding + 50, padding + 20]])}]

    reader.pipeline = Pipeline()
    with pymupdf.open(source) as pdf:
        clip = pymupdf.Rect(30, 100, 330, 220)
        result = reader.read(pdf[0], clip)
        x0, y0, x1, y1 = result.blocks[0]["bbox"]
        assert x0 == pytest.approx(30)
        assert y0 == pytest.approx(100)
        assert x1 == pytest.approx(44.352, abs=0.3)
        assert y1 == pytest.approx(105.696, abs=0.3)
    with pytest.raises(ValueError, match="paddleocr"):
        PageOCR("pp-structure", padding=64)
    with pytest.raises(ValueError, match="неотрицательным"):
        PageOCR("paddleocr", padding=-1)
