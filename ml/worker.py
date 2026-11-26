"""Private HTTP contract between the public backend and the engine-manual pipeline."""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from ml.api import api_errors
from ml.config import Settings
from ml.contracts import AgentRequest, ConversationContext, TicketConfirmation, TicketDecision
from ml.manual_api import manual_router
from ml.pdf_baseline import build_pdf_pipeline


class Run(BaseModel):
    turn: AgentRequest
    context: ConversationContext | None = None


class Confirm(BaseModel):
    actor_id: str
    decision: TicketDecision


def create_app():
    settings = Settings()
    index = settings.artifacts_dir / 'manuals.sqlite3'
    client = httpx.Client(timeout=settings.http_timeout_seconds)
    pipeline = build_pdf_pipeline(settings, client, index)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            client.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.include_router(manual_router(settings, index))

    @app.get('/fleet')
    def fleet():
        return {'vehicles': [vehicle.model_dump(include={
            'vehicle_id': True, 'engine_model': True, 'make': True, 'model': True,
            'sessions': {'__all__': {'session_id', 'checked_at'}},
        }) for vehicle in pipeline.vehicles.vehicles.values()]}

    @app.post('/run')
    def run(body: Run):
        with api_errors():
            return pipeline.run(body.turn, context=body.context)

    @app.post('/confirm')
    def confirm(body: Confirm):
        with api_errors():
            return pipeline.confirm_ticket(TicketConfirmation(actor_id=body.actor_id, **body.decision.model_dump()))

    return app
