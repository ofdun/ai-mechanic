"""Optional local OCR and structured table extraction for rendered PDF pages."""

from dataclasses import dataclass
from functools import cached_property
from html.parser import HTMLParser


@dataclass
class OCRPage:
    text: str
    tables: tuple[str, ...] = ()


class TableText(HTMLParser):
    """Keep recognized rows and cells as separate fields in the searchable text."""

    def __init__(self):
        super().__init__()
        self.rows: list[str] = []
        self.cells: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.cells = []
        elif tag in {"th", "td"} and self.cells is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.cells.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.cells is not None:
            self.rows.append(" | ".join(self.cells))
            self.cells = None


def table_text(html):
    parser = TableText()
    parser.feed(html)
    parser.close()
    return "\n".join(parser.rows)


class PageOCR:
    def __init__(self, mode, dpi=250):
        self.mode = mode
        self.dpi = dpi

    @cached_property
    def pipeline(self):
        try:
            from paddleocr import PaddleOCR, PPStructureV3
        except ImportError as error:
            raise ValueError("Установите зависимости из ml/ingestion/requirements-ocr.txt.") from error
        common = dict(
            device="cpu", enable_mkldnn=False, cpu_threads=4,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False, use_doc_unwarping=False,
        )
        if self.mode == "paddleocr":
            return PaddleOCR(**common, use_textline_orientation=False,
                             text_det_limit_side_len=1600, text_det_limit_type="max")
        return PPStructureV3(
            **common, use_textline_orientation=False, use_formula_recognition=False,
            use_table_recognition=True, use_region_detection=False,
        )

    def read(self, page, clip=None):
        import numpy as np

        image = page.get_pixmap(dpi=self.dpi, clip=clip, alpha=False)
        # PaddleOCR's ndarray input expects BGR.
        pixels = np.frombuffer(image.samples, dtype=np.uint8).reshape(image.height, image.width, 3)
        result, = self.pipeline.predict(input=pixels[:, :, ::-1].copy())
        if self.mode == "paddleocr":
            return OCRPage("\n".join(result["rec_texts"]))

        blocks, tables = [], []
        for block in result["parsing_res_list"]:
            content = block.content.strip()
            if block.label == "table":
                tables.append(content)
                content = table_text(content)
            if content:
                blocks.append(content)
        return OCRPage("\n".join(blocks), tuple(tables))
