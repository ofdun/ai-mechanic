from contextlib import contextmanager
from typing import Annotated, Callable

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import ValidationError

from ml.contracts import (
    MAX_AUDIO_BYTES, AgentError, AgentRequest, AgentResponse, AgentTurn, AudioInput, ConversationContext,
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
    'ASR_ERROR': 422,
    'EMPTY_TRANSCRIPT': 422,
    'INVALID_TRANSCRIPT': 422,
    'AUDIO_TOO_LONG': 413,
    'ASR_NOT_INSTALLED': 503,
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

    def run(actor_id: str, turn: AgentTurn, audio: AudioInput | None = None) -> AgentResponse:
        try:
            request = AgentRequest(actor_id=actor_id, **turn.model_dump(), audio=audio)
        except ValidationError as error:
            raise HTTPException(422, detail={'code': 'INVALID_INPUT', 'message': 'Нужен текст, аудио или явная команда.'}) from error
        with api_errors():
            context = load_context(actor_id, request.conversation_id) if load_context and request.conversation_id else None
            return pipeline.run(request, context=context)

    @router.post('/run', response_model=AgentResponse)
    def run_text(turn: AgentTurn, actor_id: Actor) -> AgentResponse:
        return run(actor_id, turn)

    @router.post('/audio', response_model=AgentResponse)
    def run_audio(
        actor_id: Actor, audio: UploadFile, vehicle_id: Annotated[str, Form(min_length=1)],
        session_id: Annotated[str, Form(min_length=1)], conversation_id: Annotated[str | None, Form(min_length=1)] = None,
    ) -> AgentResponse:
        content = audio.file.read(MAX_AUDIO_BYTES + 1)
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(413, detail={'code': 'AUDIO_TOO_LARGE', 'message': 'Аудио превышает 20 МиБ.'})
        if not content:
            raise HTTPException(422, detail={'code': 'INVALID_INPUT', 'message': 'Пустой аудиофайл.'})
        try:
            turn = AgentTurn(vehicle_id=vehicle_id, session_id=session_id, conversation_id=conversation_id)
        except ValidationError as error:
            raise HTTPException(422, detail={'code': 'INVALID_INPUT', 'message': 'Укажите ТС и диагностическую сессию.'}) from error
        return run(actor_id, turn, AudioInput(content=content))

    @router.post('/tickets/confirm', response_model=TicketReceipt)
    def confirm(body: TicketDecision, actor_id: Actor) -> TicketReceipt:
        with api_errors():
            return pipeline.confirm_ticket(TicketConfirmation(actor_id=actor_id, **body.model_dump()))

    return router
