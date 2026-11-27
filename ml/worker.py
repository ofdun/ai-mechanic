"""Private HTTP contract between the public backend and the engine-manual pipeline."""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ml.api import api_errors
from ml.config import Settings
from ml.contracts import AgentRequest, ConversationContext, TicketConfirmation, TicketDecision
from ml.manual_api import manual_router
from ml.pdf_baseline import PdfPageRetriever, build_manual_pipeline
from ml.plates import read_plate
from ml.retrieval import RetrievalConfig, RetrievalEngine


MAX_PLATE_PHOTO = 8 << 20


class Run(BaseModel):
    turn: AgentRequest
    context: ConversationContext | None = None


class Confirm(BaseModel):
    actor_id: str
    decision: TicketDecision


def create_app():
    settings = Settings()
    index = settings.artifacts_dir / 'manuals.sqlite3'
    if settings.web_retrieval_mode == 'team':
        config = RetrievalConfig.load(settings.retrieval_config_path)
        retriever = RetrievalEngine.load(settings.artifacts_dir / 'retrieval', config)
        logging.getLogger('uvicorn.error').info(
            'Loaded %s retrieval, corpus %s (%d manuals)', config.mode,
            retriever.manifest.corpus_version, len(retriever.manifest.documents),
        )
    else:
        retriever = PdfPageRetriever(index)
    client = httpx.Client(timeout=settings.http_timeout_seconds)
    pipeline = build_manual_pipeline(settings, client, retriever)

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
            'sessions': {'__all__': {'session_id', 'checked_at', 'status'}},
        }) for vehicle in pipeline.vehicles.vehicles.values()]}

    @app.post('/plate')
    async def plate(request: Request):
        # The photo stays in memory: it is not logged, stored or passed to the agent.
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_PLATE_PHOTO:
                raise HTTPException(413)
            data.extend(chunk)
        if not data:
            raise HTTPException(400)
        with api_errors():
            return await run_in_threadpool(read_plate, bytes(data), list(pipeline.vehicles.vehicles))

    @app.post('/run')
    def run(body: Run):
        with api_errors():
            return pipeline.run(body.turn, context=body.context)

    @app.post('/confirm')
    def confirm(body: Confirm):
        with api_errors():
            return pipeline.confirm_ticket(TicketConfirmation(actor_id=body.actor_id, **body.decision.model_dump()))

    return app
