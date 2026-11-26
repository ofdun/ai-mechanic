import hashlib
from contextlib import closing
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock

import pypdfium2 as pdfium
from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response
from ml.ingestion.export_pages import extract_text


# PDFium calls, including handle disposal, must not run concurrently in one process.
_pdf_lock = Lock()


@lru_cache(maxsize=24)
def _source_digest(path: Path, version: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_digest(path: Path) -> str:
    stat = path.stat()
    return _source_digest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns, stat.st_mtime_ns))


def read_manual_page(path: Path, number: int, view: str, metadata: dict | None = None) -> Response:
    try:
        with _pdf_lock, pdfium.PdfDocument(path) as document:
            total = len(document)
            if not 1 <= number <= total:
                raise HTTPException(404, 'В руководстве нет такой страницы.')
            with closing(document[number - 1]) as page:
                if view == 'text':
                    with closing(page.get_textpage()) as text:
                        return JSONResponse({'page': number, 'total': total, 'text': extract_text(text, metadata or {}, number)},
                                            headers={'Cache-Control': 'private'})
                scale = 1800 / max(page.get_size())
                with closing(page.render(scale=scale)) as bitmap, closing(bitmap.to_pil()) as image:
                    output = BytesIO()
                    image.save(output, format='PNG')
                    return Response(output.getvalue(), media_type='image/png', headers={'Cache-Control': 'private'})
    except (OSError, pdfium.PdfiumError):
        raise HTTPException(503, 'Страницу руководства сейчас не удаётся открыть.') from None
