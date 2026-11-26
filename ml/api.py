from contextlib import contextmanager
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from ml.contracts import (
    AgentError, AgentRequest, AgentResponse, AgentTurn, ConversationContext,
    TicketConfirmation, TicketDecision, TicketReceipt,
)
from ml.pipeline import Pipeline


ERROR_HTTP_STATUS = {
    'VEHICLE_OR_SESSION_NOT_FOUND': 404,
    'DRAFT_NOT_FOUND': 404,
    'CONTEXT_MISMATCH': 409,
    'DRAFT_STALE': 409,
    'DRAFT_EXPIRED': 409,
    'DUPLICATE_TICKET': 409,
    'TICKET_IN_PROGRESS': 409,
    'TICKET_OUTCOME_UNKNOWN': 409,
    'REMEASUREMENT_REQUIRED': 409,
    'CONFIRMATION_REQUIRED': 422,
    'LLM_EVIDENCE_NOT_ALLOWED': 403,
    'LLM_NOT_CONFIGURED': 503,
    'RAG_INDEX_UNAVAILABLE': 503,
    'TICKET_STORAGE_UNAVAILABLE': 503,
    'TOIR_UNAVAILABLE': 503,
}


@contextmanager
def api_errors():
    try:
        yield
    except AgentError as error:
        raise HTTPException(ERROR_HTTP_STATUS.get(error.code, 502), detail={
            'code': error.code, 'message': str(error), 'retryable': error.retryable,
        }) from error


def create_agent_router(
    pipeline: Pipeline,
    resolve_actor: Callable[..., str],
    *,
    load_context: Callable[[str, str], ConversationContext | None] | None = None,
) -> APIRouter:
    """Mount under a host backend with an authenticated actor dependency."""
    router = APIRouter(prefix='/agent', tags=['agent'])
    Actor = Annotated[str, Depends(resolve_actor)]

    @router.post('/run', response_model=AgentResponse)
    def run_text(turn: AgentTurn, actor_id: Actor) -> AgentResponse:
        try:
            request = AgentRequest(actor_id=actor_id, **turn.model_dump())
        except ValidationError as error:
            raise HTTPException(422, detail={'code': 'INVALID_INPUT', 'message': 'Нужен текст или явная команда.'}) from error
        with api_errors():
            context = load_context(actor_id, request.conversation_id) if load_context and request.conversation_id else None
            return pipeline.run(request, context=context)

    @router.post('/tickets/confirm', response_model=TicketReceipt)
    def confirm(body: TicketDecision, actor_id: Actor) -> TicketReceipt:
        with api_errors():
            return pipeline.confirm_ticket(TicketConfirmation(actor_id=actor_id, **body.model_dump()))

    return router
