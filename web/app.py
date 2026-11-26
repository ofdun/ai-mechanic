import sqlite3
import unicodedata
from contextlib import asynccontextmanager, closing
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml import pdf_baseline
from ml.api import create_agent_router
from ml.config import Settings
from ml.manual_corpus import document_catalog
from ml.pipeline import Pipeline
from web.manuals import checked_digest, read_manual_page
from web.sessions import SESSION_SECONDS, WebSessions


class SessionOption(BaseModel):
    session_id: str
    checked_at: datetime


class VehicleOption(BaseModel):
    vehicle_id: str
    make: str | None = None
    model: str | None = None
    engine_model: str
    sessions: list[SessionOption]


class VehicleList(BaseModel):
    vehicles: list[VehicleOption]


def create_app(
    settings: Settings | None = None, *,
    transport: httpx.BaseTransport | None = None,
    frontend_dist: Path | None = None,
    pipeline_factory: Callable[[Settings, httpx.Client, Path], Pipeline] = pdf_baseline.build_pdf_pipeline,
    documents: dict[str, tuple[str, str]] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    index = settings.artifacts_dir / 'manuals.sqlite3'
    sessions = WebSessions(settings.artifacts_dir / 'web-sessions.sqlite3')
    http = httpx.Client(timeout=settings.http_timeout_seconds, transport=transport)
    try:
        pipeline = pipeline_factory(settings, http, index)
    except BaseException:
        http.close()
        raise

    @asynccontextmanager
    async def lifespan(app):
        with http:
            yield

    app = FastAPI(title='Цифровой механик', lifespan=lifespan)
    catalog = document_catalog() if documents is None else {}

    def current_actor(request: Request, response: Response) -> str:
        actor, token = sessions.resolve(request.cookies.get('mechanic_session'))
        if token:
            response.set_cookie('mechanic_session', token, max_age=SESSION_SECONDS,
                                httponly=True, secure=request.url.scheme == 'https', samesite='strict')
        response.headers['Cache-Control'] = 'private, no-store'
        return actor

    app.include_router(create_agent_router(pipeline, current_actor, run_turn=lambda request: sessions.run(pipeline, request)))

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/api/vehicles', response_model=VehicleList)
    def vehicles(actor: str = Depends(current_actor)):
        return VehicleList(vehicles=[
            VehicleOption.model_validate(vehicle, from_attributes=True)
            for vehicle in pipeline.vehicles.vehicles.values()
        ])

    def checked_document(document_id: str) -> Path:
        if documents is not None and document_id in documents:
            name, digest = documents[document_id]
        elif document_id in catalog:
            entry = catalog[document_id]
            name, digest = entry['file'], entry['sha256']
        elif documents is None and document_id == pdf_baseline.DOCUMENT_ID:
            name, digest = pdf_baseline.PDF_NAME, pdf_baseline.PDF_SHA256
        else:
            raise HTTPException(404, 'Источник не найден.')
        try:
            with closing(sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                indexed = db.execute('SELECT 1 FROM manual_pages WHERE document_id=? LIMIT 1', (document_id,)).fetchone()
        except sqlite3.Error:
            raise HTTPException(503, 'Индекс руководств недоступен.') from None
        if not indexed:
            raise HTTPException(404, 'Источник отсутствует в индексе.')
        matches = [p for p in settings.data_dir.glob('*.pdf') if unicodedata.normalize('NFC', p.name) == name
                   and not p.is_symlink() and p.is_file()]
        if len(matches) != 1 or checked_digest(matches[0]) != digest:
            raise HTTPException(404, 'Проверенная версия руководства недоступна.')
        return matches[0]

    @app.get('/api/documents/{document_id}', response_class=FileResponse)
    def document(path: Path = Depends(checked_document)):
        return FileResponse(path, media_type='application/pdf', headers={'Cache-Control': 'private'})

    @app.get('/api/documents/{document_id}/pages/{page_number}')
    def manual_page(document_id: str, page_number: int, view: Literal['text', 'image'] = 'text',
                    path: Path = Depends(checked_document)):
        return read_manual_page(path, page_number, view, catalog.get(document_id))

    dist = frontend_dist or Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    return app
