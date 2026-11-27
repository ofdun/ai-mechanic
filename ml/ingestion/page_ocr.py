"""Optional local OCR and structured table extraction for rendered PDF pages."""

from dataclasses import dataclass, field
from functools import cached_property
from html.parser import HTMLParser


@dataclass
class OCRPage:
    text: str
    tables: tuple[str, ...] = ()
    blocks: list[dict] = field(default_factory=list)


def raster_regions(page):
    """Return displayed raster areas with little native text, in rendered page coordinates."""
    import pymupdf

    regions = []
    for image in page.get_image_info():
        rect = (pymupdf.Rect(image["bbox"]) * page.rotation_matrix) & page.rect
        if rect.is_empty or rect.width < 36 or rect.height < 36:
            continue
        # Merge overlapping tiles and repeated occurrences of the same displayed area.
        changed = True
        while changed:
            changed = False
            for existing in list(regions):
                if not (existing & rect).is_empty:
                    regions.remove(existing)
                    rect |= existing
                    changed = True
        regions.append(rect)
    found = []
    for rect in sorted(regions, key=lambda box: (box.y0, box.x0)):
        native = page.get_text(clip=rect * page.derotation_matrix).strip()
        if len(native) < 20:
            found.append(rect)
    return found


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
    def __init__(self, mode, dpi=250, table_html="model", padding=0):
        if table_html not in {"model", "cells"}:
            raise ValueError("Неизвестный способ сборки HTML таблиц.")
        if table_html == "cells" and mode != "pp-structure":
            raise ValueError("--table-html cells требует --ocr pp-structure.")
        if type(padding) is not int or padding < 0:
            raise ValueError("--ocr-padding должен быть целым неотрицательным числом.")
        if padding and mode != "paddleocr":
            raise ValueError("--ocr-padding поддерживается с --ocr paddleocr.")
        self.mode = mode
        self.dpi = dpi
        self.table_html = table_html
        self.padding = padding

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
        if self.padding:
            pixels = np.pad(pixels, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                            mode="constant", constant_values=255)
        options = {}
        if self.mode == "pp-structure":
            options["use_wired_table_cells_trans_to_html"] = self.table_html == "cells"
        result, = self.pipeline.predict(input=pixels[:, :, ::-1].copy(), **options)
        area = page.rect if clip is None else clip & page.rect

        def bbox(box):
            # Map model pixels back to points on the displayed PDF page.
            x0, y0, x1, y1 = map(float, box)
            return [
                min(area.x1, max(area.x0, (image.x + x0 - self.padding) * 72 / self.dpi)),
                min(area.y1, max(area.y0, (image.y + y0 - self.padding) * 72 / self.dpi)),
                min(area.x1, max(area.x0, (image.x + x1 - self.padding) * 72 / self.dpi)),
                min(area.y1, max(area.y0, (image.y + y1 - self.padding) * 72 / self.dpi)),
            ]

        if self.mode == "paddleocr":
            blocks = [
                {"type": "text", "text": text, "bbox": bbox(box), "confidence": float(confidence)}
                for text, box, confidence in zip(
                    result["rec_texts"], result["rec_boxes"], result["rec_scores"], strict=True
                ) if text.strip()
            ]
            return OCRPage("\n".join(block["text"] for block in blocks), blocks=blocks)

        blocks, tables = [], []
        for block in result["parsing_res_list"]:
            content = block.content.strip()
            item = {"type": block.label, "bbox": bbox(block.bbox)}
            if block.label == "table":
                tables.append(content)
                item["html"] = content
                content = table_text(content)
            if content:
                item["text"] = content
                blocks.append(item)
        return OCRPage("\n".join(block["text"] for block in blocks), tuple(tables), blocks)
