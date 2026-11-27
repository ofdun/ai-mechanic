import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ml import pdf_baseline
from ml.document_ids import document_id_variants
from ml.manual_corpus import document_catalog
from ml.manual_pages import checked_digest, read_manual_page


def manual_router(settings, index, documents=None):
    router = APIRouter()
    catalog = document_catalog(settings.manual_manifest_path) if documents is None else {}

    def checked_document(document_id: str) -> Path:
        document_id, legacy_id = document_id_variants(document_id)
        if documents is not None and document_id in documents:
            name, digest = documents[document_id]
        elif documents is not None and legacy_id is not None and legacy_id in documents:
            name, digest = documents[legacy_id]
        elif document_id in catalog:
            entry = catalog[document_id]
            name, digest = entry['file'], entry['sha256']
        elif documents is None and document_id == pdf_baseline.DOCUMENT_ID:
            name, digest = pdf_baseline.PDF_NAME, pdf_baseline.PDF_SHA256
        else:
            raise HTTPException(404, 'Источник не найден.')
        try:
            with closing(sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                indexed = db.execute('SELECT 1 FROM manual_pages WHERE document_id IN (?, ?) LIMIT 1',
                                     (document_id, legacy_id)).fetchone()
        except sqlite3.Error:
            raise HTTPException(503, 'Индекс руководств недоступен.') from None
        if not indexed:
            raise HTTPException(404, 'Источник отсутствует в индексе.')
        matches = [p for p in settings.data_dir.glob('*.pdf') if unicodedata.normalize('NFC', p.name) == name
                   and not p.is_symlink() and p.is_file()]
        if len(matches) != 1 or checked_digest(matches[0]) != digest:
            raise HTTPException(404, 'Проверенная версия руководства недоступна.')
        return matches[0]

    @router.get('/api/documents/{document_id}', response_class=FileResponse)
    def document(path: Path = Depends(checked_document)):
        return FileResponse(path, media_type='application/pdf', headers={'Cache-Control': 'private'})

    @router.get('/api/documents/{document_id}/pages/{page_number}')
    def manual_page(document_id: str, page_number: int, view: Literal['text', 'image'] = 'text',
                    path: Path = Depends(checked_document)):
        return read_manual_page(path, page_number, view, catalog.get(document_id_variants(document_id)[0]))

    return router
