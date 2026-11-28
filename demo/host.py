"""In-process host for the synthetic demo and Python contract tests."""
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml import pdf_baseline
from ml.api import create_agent_router
from ml.config import Settings
from ml.pipeline import Pipeline
from ml.manual_api import manual_router
from ml.tools.fleet_status import FleetStatus, fleet_statuses
from demo.sessions import SESSION_SECONDS, WebSessions


class SessionOption(BaseModel):
    session_id: str
    checked_at: datetime
    status: str
    anomaly_score: float | None = None
    anomaly_type: str | None = None
    anomaly_title: str | None = None


class VehicleOption(BaseModel):
    vehicle_id: str
    make: str | None = None
    model: str | None = None
    engine_model: str
    sessions: list[SessionOption]
    fleet_status: FleetStatus | None = None


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
        vehicles = list(pipeline.vehicles.vehicles.values())
        statuses = fleet_statuses(vehicles, pipeline)
        return VehicleList(vehicles=[
            VehicleOption.model_validate(vehicle, from_attributes=True).model_copy(
                update={'fleet_status': statuses[vehicle.vehicle_id]})
            for vehicle in vehicles
        ])

    app.include_router(manual_router(settings, index, documents))

    dist = frontend_dist or Path(__file__).resolve().parents[1] / 'app' / 'frontend' / 'dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    return app
