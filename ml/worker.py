"""Private HTTP contract between the public backend and the engine-manual pipeline."""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ml.api import api_errors
from ml.config import Settings
from ml.contracts import AgentError, AgentRequest, ConversationContext, TicketConfirmation, TicketDecision
from ml.manual_api import manual_router
from ml.pdf_baseline import PdfPageRetriever, build_manual_pipeline
from ml.plates import read_plate
from ml.retrieval import RetrievalConfig, RetrievalEngine
from ml.tools.fleet_status import fleet_statuses


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
        vehicles = list(pipeline.vehicles.vehicles.values())
        # B3: the chief mechanic's status and order, by the same rules as diagnosis and ticket urgency.
        statuses = fleet_statuses(vehicles, pipeline)
        return {'vehicles': [{**vehicle.model_dump(include={
            'vehicle_id': True, 'engine_model': True, 'make': True, 'model': True,
            # anomaly_score feeds the score chart (B2); a failed check has none and is drawn as a gap.
            'sessions': {'__all__': {'session_id', 'checked_at', 'status', 'anomaly_score', 'anomaly_type',
                                     'anomaly_title'}},
        }), 'fleet_status': statuses[vehicle.vehicle_id].model_dump()} for vehicle in vehicles]}

    @app.get('/inspections')
    def inspections():
        # Export for the acoustic model team. Not routed through the web backend; the mechanic id is not included.
        with api_errors():
            return {'inspections': [record.model_dump(mode='json') for record in pipeline.inspections.export()]}

    @app.get('/audit')
    def audit(session_id: str, vehicle_id: str | None = None):
        # The chain for one check, from the first question to the ticket. Internal: the web backend does not expose it.
        with api_errors():
            if pipeline.audit is None:
                raise AgentError('AUDIT_DISABLED', 'Журнал аудита выключен (AUDIT_ENABLED=false).')
            return {'entries': pipeline.audit.chain(vehicle_id, session_id)}

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
