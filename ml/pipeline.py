from datetime import datetime, timedelta, timezone
from functools import cached_property
from uuid import uuid4

import httpx

from ml.config import Settings
from ml.contracts import (
    AgentError, AgentRequest, AgentResponse, ConversationContext, DiagnosticResult, EvidenceBundle, Generator, Retriever,
    SearchPlan, SearchWorkflow, TechnicalClaim, TicketConfirmation, TicketDraft, TicketPayload, TicketReceipt,
    TicketReference, Vehicle, Verifier,
)
from ml.generation.finalizer import MechanicFinalizer, conflict_notes, ticket_fields
from ml.planner import Decision, LLMPlanner, OfflinePlanner, Planner, diagnostic_search
from ml.providers.openrouter import OpenRouterClient
from ml.tools.mocks import MockServices, VehicleDirectory
from ml.tools.tickets import TicketStore


def scope_evidence(evidence: EvidenceBundle, engine_model: str) -> EvidenceBundle:
    items = [item for item in evidence.items if engine_model in item.engine_models]
    if len({item.evidence_id for item in items}) != len(items):
        raise AgentError('RAG_SCHEMA_ERROR', 'Поиск вернул повторяющиеся идентификаторы источников.')
    return EvidenceBundle(items=items)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        http: httpx.Client,
        *,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        verifier: Verifier | None = None,
        search_workflow: SearchWorkflow | None = None,
        planner: Planner | None = None,
        finalizer: MechanicFinalizer | None = None,
    ):
        self.settings = settings
        self.vehicles = VehicleDirectory.load(settings)
        self.services = MockServices(http, settings)
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.search_workflow = search_workflow
        self.finalizer = finalizer
        self.planner = planner if planner is not None else (
            OfflinePlanner() if settings.planner_mode == 'offline' else LLMPlanner(OpenRouterClient(http, settings))
        )

    @cached_property
    def tickets(self) -> TicketStore:
        return TicketStore(
            self.settings.artifacts_dir / 'agent-tickets.sqlite3', self.settings.http_timeout_seconds * 4 + 10,
        )

    def run(self, request: AgentRequest, *, context: ConversationContext | None = None) -> AgentResponse:
        vehicle, _ = self.vehicles.get(request.vehicle_id, request.session_id)
        if context is not None and (
            context.actor_id, context.vehicle_id, context.session_id, context.conversation_id, context.engine_model,
        ) != (request.actor_id, request.vehicle_id, request.session_id, request.conversation_id, vehicle.engine_model):
            raise AgentError('CONTEXT_MISMATCH', 'Контекст не соответствует пользователю, диалогу или выбранной сессии ТС.')
        message = request.message or {
            'diagnose': 'Запусти диагностику выбранной сессии.',
            'draft_ticket': 'Подготовь черновик заявки на дефектовку.',
        }[request.command]

        diagnostic = context.diagnostic if context is not None else None
        recent_messages = context.recent_messages if context is not None else ()
        if diagnostic and (self.measurement_issue(diagnostic) or self.ticket_mismatch(vehicle, diagnostic)):
            diagnostic = None
            recent_messages = ()
        if request.command == 'auto':
            if recent_messages:
                decision = self.planner.plan(message, vehicle.engine_model, diagnostic, recent_messages=recent_messages)
            else:
                decision = self.planner.plan(message, vehicle.engine_model, diagnostic)
        else:
            decision = Decision(request.command)
        response = AgentResponse(
            request_id=uuid4().hex, vehicle_id=request.vehicle_id, session_id=request.session_id,
            engine_model=vehicle.engine_model, conversation_id=request.conversation_id,
            action=decision.action,
            message='Уточните, что проверить или какой норматив найти.',
        )

        if decision.action in ('diagnose', 'draft_ticket'):
            self.run_diagnosis(request, vehicle, message, response)
        elif decision.action == 'search_manuals':
            response.message = 'Вопрос по руководству выбранного двигателя.'
            self.search(response, SearchPlan(
                engine_model=vehicle.engine_model, original_query=message,
                rewritten_query=decision.rewritten_query or message, subqueries=decision.subqueries,
            ), diagnostic)
        elif decision.action == 'get_diagnostic_history':
            response.message = 'Последние проверки выбранного ТС.'
            response.history = sorted(vehicle.sessions, key=lambda session: session.checked_at)[-5:]
        response.limitations.extend(note for note in conflict_notes(response) if note not in response.limitations)
        if self.finalizer is not None:
            response.message = self.finalizer.compose(response)
        return response

    def measurement_issue(self, result: DiagnosticResult) -> str | None:
        if result.status == 'check_failed' or result.signal_quality == 'bad':
            return 'Запись непригодна для заключения. Повторите замер.'
        if result.confidence is None or result.confidence < self.settings.min_diagnostic_confidence:
            return 'Уверенность модели ниже установленного порога. Повторите замер.'
        if result.signal_quality == 'warning' and self.settings.repeat_on_warning:
            return 'В записи есть помехи. Повторите замер.'
        return None

    @staticmethod
    def ticket_mismatch(vehicle: Vehicle, result: DiagnosticResult) -> str | None:
        fuel = vehicle.fuel.casefold()
        if (result.anomaly_type == 'F08' and fuel == 'бензин') or (
            result.anomaly_type in {'F12', 'F13'} and fuel == 'дизель'
        ):
            return 'Гипотеза ML несовместима с типом топлива в карточке ТС. Уточните исходные данные.'
        return None

    def run_diagnosis(self, request: AgentRequest, vehicle: Vehicle, message: str, response: AgentResponse) -> None:
        result = self.services.diagnose(request.vehicle_id, request.session_id)
        response.diagnostic = result
        if problem := self.measurement_issue(result):
            response.message = problem
            response.requires_remeasurement = True
            return
        if result.status == 'normal':
            response.message = 'Значимых отклонений в этой акустической проверке не обнаружено.'
            return
        if mismatch := self.ticket_mismatch(vehicle, result):
            response.message = mismatch
            return
        response.message = (f'Внешний ML-сервис обнаружил отклонение «{result.anomaly_title}». '
                            f'Уверенность модели {result.confidence:.0%}. Причину нужно проверить.')
        self.search(response, diagnostic_search(vehicle.engine_model, message, result), result)
        operations = [claim for claim in response.claims if claim.kind == 'operation']
        response.can_create_ticket = bool(operations)
        if response.action == 'draft_ticket':
            if operations:
                response.ticket_draft = self.save_draft(request.actor_id, response, result, operations)
            else:
                response.limitations.append('Для заявки нужна хотя бы одна подтверждённая операция.')

    def search(self, response: AgentResponse, plan: SearchPlan, diagnostic: DiagnosticResult | None) -> None:
        response.search_plan = plan
        if self.search_workflow is not None:
            outcome = self.search_workflow.run(plan, diagnostic)
            if outcome.plan.engine_model != plan.engine_model:
                raise AgentError('RAG_SCHEMA_ERROR', 'Поисковая стратегия изменила модель двигателя.')
            response.search_plan = outcome.plan
            response.evidence = scope_evidence(outcome.evidence, plan.engine_model)
            source_ids = {item.evidence_id for item in response.evidence.items}
            response.claims = [claim for claim in outcome.claims if all(key in source_ids for key in claim.evidence_ids)]
            if not response.claims:
                response.limitations.append('Недостаточно доказательств для технических выводов.')
            elif outcome.summary and len(response.claims) == len(outcome.claims):
                response.summary = outcome.summary
                if response.action == 'search_manuals':
                    response.message = outcome.summary
            return
        if self.retriever is None:
            response.limitations.append('Retriever ещё не подключён. Доступен поисковый план.')
            return

        response.evidence = scope_evidence(self.retriever.search(plan), plan.engine_model)
        sources = {item.evidence_id for item in response.evidence.items}
        if not response.evidence.items:
            response.limitations.append('Применимых доказательств для выбранного двигателя не найдено.')
            return
        if self.generator is None or self.verifier is None:
            response.limitations.append('Для технических рекомендаций нужны Generator и Verifier. Возвращены фрагменты.')
            return

        draft = self.generator.generate(plan, response.evidence, diagnostic)
        response.claims = [
            claim for claim in self.verifier.verify(draft, response.evidence).claims
            if claim in draft.claims and all(key in sources for key in claim.evidence_ids)
        ]
        if not response.claims:
            response.limitations.append('По найденным фрагментам не удалось подтвердить технические утверждения.')

    def save_draft(
        self, actor_id: str, response: AgentResponse, result: DiagnosticResult, operations: list[TechnicalClaim],
    ) -> TicketDraft:
        parts = [claim for claim in response.claims if claim.kind == 'spare_part']
        cited = {key for claim in operations + parts for key in claim.evidence_ids}
        references = dict.fromkeys((item.manual, item.pdf_page) for item in response.evidence.items if item.evidence_id in cited)
        conclusion, urgency = ticket_fields(result, response.claims)
        draft = TicketDraft(
            draft_id=uuid4().hex, engine_model=response.engine_model,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            payload=TicketPayload(
                vehicle_id=result.vehicle_id, session_id=result.session_id, anomaly_type=result.anomaly_type,
                conclusion=conclusion,
                urgency=urgency, operations=list(dict.fromkeys(claim.text for claim in operations)),
                spare_parts=list(dict.fromkeys(claim.text for claim in parts)),
                references=[TicketReference(manual=manual, pdf_page=page) for manual, page in references],
            ),
        )
        self.tickets.save(actor_id, draft, result)
        return draft

    def confirm_ticket(self, confirmation: TicketConfirmation) -> TicketReceipt:
        if not confirmation.confirmed:
            raise AgentError('CONFIRMATION_REQUIRED', 'Нужно явное подтверждение механика.')
        draft, original, previous_receipt = self.tickets.load(confirmation)
        if previous_receipt:
            return previous_receipt
        vehicle, _ = self.vehicles.get(confirmation.vehicle_id, confirmation.session_id)
        if draft.engine_model != vehicle.engine_model:
            raise AgentError('DRAFT_STALE', 'Модель двигателя не совпадает с черновиком. Подготовьте новый.')
        result = self.services.diagnose(confirmation.vehicle_id, confirmation.session_id)
        if self.measurement_issue(result):
            raise AgentError('REMEASUREMENT_REQUIRED', 'Сначала повторите замер.')
        if result != original or result.status != 'anomaly' or self.ticket_mismatch(vehicle, result):
            raise AgentError('DRAFT_STALE', 'Диагностика изменилась. Подготовьте новый черновик.')

        if receipt := self.tickets.reserve(confirmation, draft):
            return receipt
        try:
            receipt = self.services.create_ticket(draft.payload)
        except AgentError as error:
            try:
                self.tickets.finish(draft.draft_id, rejected=error.code in {'TICKET_REJECTED', 'TOIR_UNAVAILABLE'})
            except AgentError as store_error:
                raise AgentError('TICKET_OUTCOME_UNKNOWN', 'Не удалось сохранить статус отправки. Нужна сверка.') from store_error
            raise
        try:
            self.tickets.finish(draft.draft_id, receipt=receipt)
        except AgentError as error:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'ТОиР ответил, но квитанцию не удалось сохранить. Нужна сверка.') from error
        return receipt
