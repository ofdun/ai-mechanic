import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import cached_property
from uuid import uuid4

import httpx

from ml.config import Settings
from ml.contracts import (
    AgentError,
    AgentRequest,
    AgentResponse,
    ConversationContext,
    DiagnosticResult,
    EvidenceBundle,
    Generator,
    Retriever,
    SearchPlan,
    SearchWorkflow,
    TechnicalClaim,
    TicketConfirmation,
    TicketDraft,
    TicketPayload,
    TicketReceipt,
    TicketReference,
    Vehicle,
    Verifier,
)
from ml.generation.finalizer import MechanicFinalizer, conflict_notes, ticket_fields
from ml.generation.translate import Localizer
from ml.planner import Decision, LLMPlanner, OfflinePlanner, Planner, diagnostic_search
from ml.planner.local import LocalPlanner
from ml.providers.openrouter import OpenRouterClient
from ml.tools.services import FleetServices
from ml.tools.inspections import InspectionStore
from ml.tools.tickets import TicketStore
from ml.vehicles import VehicleDirectory


def scope_evidence(evidence: EvidenceBundle, engine_model: str) -> EvidenceBundle:
    items = [item for item in evidence.items if engine_model in item.engine_models]
    if len({item.evidence_id for item in items}) != len(items):
        raise AgentError('RAG_SCHEMA_ERROR', 'Поиск вернул повторяющиеся идентификаторы источников.')
    return EvidenceBundle(items=items)


def percentage(value: float) -> str:
    return format((Decimal(str(value)) * 100).normalize(), 'f').replace('.', ',') + '%'


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
        self.services = FleetServices(http, settings)
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.search_workflow = search_workflow
        self.finalizer = finalizer
        # Translation sends manual text to the provider, so it follows the same switch as generation.
        external = settings.allow_external_evidence and bool(settings.llm_api_key.get_secret_value())
        self.localizer = Localizer(OpenRouterClient(http, settings) if external else None, settings.a_generator_model)
        self.planner = planner or {
            'llm': lambda: LLMPlanner(OpenRouterClient(http, settings)),
            'local': LocalPlanner,
            'offline': OfflinePlanner,
        }[settings.planner_mode]()

    @cached_property
    def tickets(self) -> TicketStore:
        return TicketStore(
            self.settings.artifacts_dir / 'agent-tickets.sqlite3', self.settings.http_timeout_seconds * 4 + 10,
        )

    @cached_property
    def inspections(self) -> InspectionStore:
        return InspectionStore(self.settings.artifacts_dir / 'agent-inspections.sqlite3')

    def ticket_store_for(self, actor_id: str) -> TicketStore:
        return self.tickets

    def run(self, request: AgentRequest, *, context: ConversationContext | None = None) -> AgentResponse:
        vehicle, _ = self.vehicles.get(request.vehicle_id, request.session_id)
        if context is not None and (
            context.actor_id, context.vehicle_id, context.session_id, context.conversation_id, context.engine_model,
        ) != (request.actor_id, request.vehicle_id, request.session_id, request.conversation_id, vehicle.engine_model):
            raise AgentError('CONTEXT_MISMATCH', 'Контекст не соответствует пользователю, диалогу или выбранной сессии ТС.')
        if request.command == 'record_inspection':
            return self.record_inspection(request, vehicle, context.diagnostic if context else None)
        message = request.message or {
            'diagnose': 'Запусти диагностику выбранной сессии.',
            'draft_ticket': 'Подготовь черновик заявки на дефектовку.',
            'get_diagnostic_history': 'Покажи историю проверок.',
        }[request.command]

        diagnostic = context.diagnostic if context is not None else None
        recent_messages = context.recent_messages if context is not None else ()
        if diagnostic and (self.measurement_issue(diagnostic, request.accept_signal_warning)
                           or self.ticket_mismatch(vehicle, diagnostic)):
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
            message=decision.clarification or 'Уточните, что проверить или какой норматив найти.',
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
            response.inspections = self.inspections.for_vehicle(vehicle.vehicle_id)
        response.limitations.extend(note for note in conflict_notes(response) if note not in response.limitations)
        response.language = request.language
        if request.language != 'ru':
            response.localized = self.localizer.localize(response, request.language)
        if self.finalizer is not None:
            response.message = self.finalizer.compose(response)
        return response

    def warning_confirmable(self, result: DiagnosticResult) -> bool:
        """Noisy but confident: the only case the mechanic may override, never a bad or failed record."""
        return (result.status != 'check_failed' and result.signal_quality == 'warning'
                and self.settings.confirm_signal_warning
                and result.confidence is not None and result.confidence >= self.settings.min_diagnostic_confidence)

    def record_inspection(self, request: AgentRequest, vehicle: Vehicle, diagnostic: DiagnosticResult | None) -> AgentResponse:
        """Stores what the mechanic found. The diagnosis, its context and any ticket stay as they were."""
        if (diagnostic is None or diagnostic.status != 'anomaly' or self.measurement_issue(diagnostic, request.accept_signal_warning)
                or self.ticket_mismatch(vehicle, diagnostic)):
            raise AgentError('INSPECTION_UNAVAILABLE', 'Для записи осмотра получите достоверное отклонение в этом диалоге.')
        self.inspections.add(request.actor_id, diagnostic, request.inspection)
        response = AgentResponse(
            request_id=uuid4().hex, vehicle_id=request.vehicle_id, session_id=request.session_id,
            engine_model=vehicle.engine_model, conversation_id=request.conversation_id, action='record_inspection',
            message='Результат осмотра записан. Диагноз и заявка не меняются.',
            inspections=self.inspections.for_vehicle(vehicle.vehicle_id),
        )
        response.language = request.language
        return response

    def measurement_issue(self, result: DiagnosticResult, accepted_warning: bool = False, *, ticket: bool = False) -> str | None:
        if result.status == 'check_failed' or result.signal_quality == 'bad':
            return 'По этой записи нельзя оценить работу двигателя. Повторите замер.'
        threshold = self.settings.min_diagnostic_confidence
        noisy = result.signal_quality == 'warning'
        if accepted_warning and self.warning_confirmable(result):
            return None
        needs_consent = noisy and (ticket or self.settings.repeat_on_warning or self.settings.confirm_signal_warning)
        if result.confidence is not None and result.confidence >= threshold and not needs_consent:
            return None
        quality = 'В записи есть помехи. ' if noisy else ''
        confidence = (f'Уверенность ИИ {percentage(result.confidence)}' if result.confidence is not None
                      else 'Уверенность ИИ неизвестна')
        return f'{quality}{confidence}, порог для этой проверки {percentage(threshold)}. Повторите замер.'

    @staticmethod
    def ticket_mismatch(vehicle: Vehicle, result: DiagnosticResult) -> str | None:
        fuel = vehicle.fuel.casefold()
        if (result.anomaly_type == 'F08' and fuel == 'бензин') or (
            result.anomaly_type in {'F12', 'F13'} and fuel == 'дизель'
        ):
            engine = 'бензиновый' if fuel == 'бензин' else 'дизельный'
            return (f'На машине установлен {engine} двигатель {vehicle.engine_model}. '
                    f'ИИ указал «{result.anomaly_title}», но такая неисправность к этому двигателю неприменима. '
                    'Диагноз по записи поставить нельзя.')
        return None

    def run_diagnosis(self, request: AgentRequest, vehicle: Vehicle, message: str, response: AgentResponse) -> None:
        result = self.services.diagnose(request.vehicle_id, request.session_id)
        response.diagnostic = result
        response.min_confidence = self.settings.min_diagnostic_confidence
        problem = self.measurement_issue(result, request.accept_signal_warning, ticket=response.action == 'draft_ticket')
        mismatch = self.ticket_mismatch(vehicle, result)
        if mismatch and result.signal_quality != 'bad':
            next_step = problem or 'Повторите замер или направьте машину на дефектовку.'
            response.message = f'{mismatch} {next_step}'
            response.recognition_error = True
            return
        if problem:
            response.message = problem
            if result.status == 'anomaly' and result.signal_quality == 'warning':
                response.message = f'Предварительная гипотеза ИИ — «{result.anomaly_title}». {problem}'
            response.requires_remeasurement = True
            response.warning_confirmable = self.warning_confirmable(result)
            if response.warning_confirmable:
                response.message += ' Если повторить замер нельзя, подтвердите, что понимаете: запись с помехами.'
            return
        if result.signal_quality == 'warning':
            # Reached only after the mechanic's confirmation (or with REPEAT_ON_WARNING off): keep it visible.
            response.signal_warning_accepted = request.accept_signal_warning
            response.limitations.append('Запись с помехами: вывод сделан по ней после подтверждения механика. '
                                        'При сомнениях повторите замер.' if request.accept_signal_warning
                                        else 'Запись с помехами. При сомнениях повторите замер.')
        if result.status == 'normal':
            response.message = 'На записи звук двигателя в пределах нормы.'
            return
        response.message = (f'Искусственный интеллект обнаружил признаки отклонения «{result.anomaly_title}». '
                            'Причину нужно уточнить при осмотре.')
        self.search(response, diagnostic_search(vehicle.engine_model, message, result), result)
        if not any(claim.kind == 'risk' for claim in response.claims):
            response.limitations.append('Риск для этого замера по руководству не подтверждён. Рекомендуется дефектовка.')
        operations = [claim for claim in response.claims if claim.kind == 'operation']
        response.can_create_ticket = bool(operations) and (result.signal_quality != 'warning' or response.signal_warning_accepted)
        if not operations:
            response.manual_referral = (
                f'ТС {result.vehicle_id}, двигатель {vehicle.engine_model}, проверка {result.session_id}. '
                f'По записи предполагается отклонение «{result.anomaly_title}» ({result.anomaly_type}). '
                'Подтвердить порядок работ по руководству для этого двигателя не удалось. '
                'Рекомендуется оформить дефектовку по порядку автопарка.'
            )
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
            if len(response.claims) == len(outcome.claims):
                response.verification = outcome.verification
            if not response.claims:
                response.limitations.append(
                    'На найденных страницах не удалось подтвердить ответ на этот вопрос.'
                    if response.evidence.items else 'В руководстве по этому двигателю ответ не найден.'
                )
            elif outcome.summary and len(response.claims) == len(outcome.claims):
                response.summary = outcome.summary
                if response.action == 'search_manuals':
                    response.message = outcome.summary
            if response.action == 'search_manuals' and response.claims and re.search(
                r'\b(?:чинить|починить|отремонтировать|ремонтировать)\b', plan.original_query, re.I
            ) and not any(claim.kind == 'operation' for claim in response.claims):
                response.limitations.append(
                    f'Порядок ремонта для двигателя {plan.engine_model} подтвердить не удалось. '
                    'Уточните, какой признак неисправности вы наблюдаете.'
                )
            return
        if self.retriever is None:
            response.limitations.append('Сейчас поиск по руководствам недоступен.')
            return

        response.evidence = scope_evidence(self.retriever.search(plan), plan.engine_model)
        sources = {item.evidence_id for item in response.evidence.items}
        if not response.evidence.items:
            response.limitations.append('В руководстве по этому двигателю ответ не найден.')
            return
        if self.generator is None or self.verifier is None:
            response.limitations.append('Найдены страницы руководства, но подтвердить по ним порядок работ пока не удалось.')
            return

        draft = self.generator.generate(plan, response.evidence, diagnostic)
        verified = self.verifier.verify(draft, response.evidence)
        response.claims = [
            claim for claim in verified.claims
            if claim in draft.claims and all(key in sources for key in claim.evidence_ids)
        ]
        if len(response.claims) == len(verified.claims):
            response.verification = verified.verification
        if not response.claims:
            response.limitations.append('На найденных страницах руководства нет подтверждения для ответа.')

    def save_draft(
        self, actor_id: str, response: AgentResponse, result: DiagnosticResult, operations: list[TechnicalClaim],
    ) -> TicketDraft:
        parts = [claim for claim in response.claims if claim.kind == 'spare_part']
        cited = {key for claim in operations + parts for key in claim.evidence_ids}
        references = dict.fromkeys((item.manual, item.pdf_page) for item in response.evidence.items if item.evidence_id in cited)
        conclusion, urgency = ticket_fields(result, response.claims)
        draft = TicketDraft(
            draft_id=uuid4().hex, engine_model=response.engine_model,
            signal_warning_accepted=response.signal_warning_accepted,
            urgency_reason=(f'По правилу для кода {result.anomaly_type}'
                            if result.anomaly_type in self.settings.ticket_urgency_by_fault
                            else None),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            payload=TicketPayload(
                vehicle_id=result.vehicle_id, session_id=result.session_id, anomaly_type=result.anomaly_type,
                conclusion=conclusion,
                # A fleet rule in settings overrides the urgency computed from the fault, score and manual.
                urgency=self.settings.ticket_urgency_by_fault.get(result.anomaly_type, urgency),
                operations=list(dict.fromkeys(claim.text for claim in operations)),
                spare_parts=list(dict.fromkeys(claim.text for claim in parts)),
                references=[TicketReference(manual=manual, pdf_page=page) for manual, page in references],
            ),
        )
        self.ticket_store_for(actor_id).save(actor_id, draft, result)
        return draft

    def confirm_ticket(self, confirmation: TicketConfirmation) -> TicketReceipt:
        if not confirmation.confirmed:
            raise AgentError('CONFIRMATION_REQUIRED', 'Нужно явное подтверждение механика.')
        tickets = self.ticket_store_for(confirmation.actor_id)
        draft, original, previous_receipt = tickets.load(confirmation)
        if previous_receipt:
            return previous_receipt
        vehicle, _ = self.vehicles.get(confirmation.vehicle_id, confirmation.session_id)
        if draft.engine_model != vehicle.engine_model:
            raise AgentError('DRAFT_STALE', 'Модель двигателя не совпадает с черновиком. Подготовьте новый.')
        result = self.services.diagnose(confirmation.vehicle_id, confirmation.session_id)
        if problem := self.measurement_issue(result, draft.signal_warning_accepted, ticket=True):
            raise AgentError('REMEASUREMENT_REQUIRED', problem)
        if result != original or result.status != 'anomaly' or self.ticket_mismatch(vehicle, result):
            raise AgentError('DRAFT_STALE', 'Диагностика изменилась. Подготовьте новый черновик.')

        if confirmation.urgency is not None:
            draft.payload.urgency = confirmation.urgency
        if receipt := tickets.reserve(confirmation, draft):
            return receipt
        try:
            receipt = self.services.create_ticket(draft.payload)
        except AgentError as error:
            try:
                tickets.finish(draft.draft_id, rejected=error.code in {'TICKET_REJECTED', 'TOIR_UNAVAILABLE'})
            except AgentError as store_error:
                raise AgentError('TICKET_OUTCOME_UNKNOWN', 'Не удалось сохранить статус отправки. Нужна сверка.') from store_error
            raise
        try:
            tickets.finish(draft.draft_id, receipt=receipt)
        except AgentError as error:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'ТОиР ответил, но квитанцию не удалось сохранить. Нужна сверка.') from error
        return receipt
