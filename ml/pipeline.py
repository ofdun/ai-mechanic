import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx

from ml.audio import WhisperASR
from ml.config import Settings
from ml.contracts import (
    AgentError, AgentRequest, AgentResponse, DiagnosticResult, EvidenceBundle, Generator, Retriever,
    SearchPlan, TechnicalClaim, TicketConfirmation, TicketDraft, TicketPayload, TicketReceipt,
    TicketReference, Transcriber, Vehicle, Verifier,
)
from ml.planner import Decision, LLMPlanner, OfflinePlanner, Planner, diagnostic_search
from ml.providers.openrouter import OpenRouterClient
from ml.tools.mocks import MockServices, VehicleDirectory
from ml.tools.tickets import TicketStore


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        http: httpx.Client,
        *,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
        verifier: Verifier | None = None,
        planner: Planner | None = None,
        transcriber: Transcriber | None = None,
    ):
        self.settings = settings
        self.vehicles = VehicleDirectory.load(settings)
        self.services = MockServices(http, settings)
        self.tickets = TicketStore(settings.artifacts_dir / 'agent-tickets.sqlite3', settings.http_timeout_seconds * 4 + 10)
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.planner = planner if planner is not None else (
            OfflinePlanner() if settings.planner_mode == 'offline' else LLMPlanner(OpenRouterClient(http, settings))
        )
        self.transcriber = transcriber if transcriber is not None else WhisperASR(
            settings.asr_model, settings.asr_language, settings.artifacts_dir / 'asr',
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        vehicle, last_check = self.vehicles.get(request.vehicle_id, request.session_id)
        transcript = self.transcriber.transcribe(request.audio).strip() if request.audio is not None else None
        if transcript is not None and not 0 < len(transcript) <= 8000:
            raise AgentError('INVALID_TRANSCRIPT', 'Распознанный запрос пуст или слишком длинный.')
        message = request.message or transcript or {
            'diagnose': 'Запусти диагностику выбранной сессии.',
            'draft_ticket': 'Подготовь черновик заявки на дефектовку.',
        }[request.command]

        usable_context = last_check if not (self.measurement_issue(last_check) or self.ticket_mismatch(vehicle, last_check)) else None
        decision = self.planner.plan(message, vehicle.engine_model, usable_context) if request.command == 'auto' else Decision(request.command)
        response = AgentResponse(
            request_id=uuid4().hex, vehicle_id=request.vehicle_id, session_id=request.session_id,
            engine_model=vehicle.engine_model, action=decision.action, transcript=transcript,
            message='Уточните, что проверить или какой норматив найти.',
        )

        if decision.action in ('diagnose', 'draft_ticket'):
            self.run_diagnosis(request, vehicle, message, response)
        elif decision.action == 'search_manuals':
            response.message = 'Вопрос по руководству выбранного двигателя.'
            self.search(response, SearchPlan(
                engine_model=vehicle.engine_model, original_query=message,
                rewritten_query=decision.rewritten_query or message, subqueries=decision.subqueries,
            ), usable_context)
        elif decision.action == 'get_diagnostic_history':
            response.message = 'Последние проверки выбранного ТС.'
            response.history = sorted(vehicle.sessions, key=lambda session: session.checked_at)[-5:]
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
        if self.retriever is None:
            response.limitations.append('Retriever ещё не подключён. Доступен поисковый план.')
            return

        items = [item for item in self.retriever.search(plan).items if plan.engine_model in item.engine_models]
        sources = {item.evidence_id: item for item in items}
        if len(sources) != len(items):
            raise AgentError('RAG_SCHEMA_ERROR', 'Поиск вернул повторяющиеся идентификаторы источников.')
        response.evidence = EvidenceBundle(items=items)
        if not items:
            response.limitations.append('Применимых доказательств для выбранного двигателя не найдено.')
            return
        if self.generator is None:
            response.limitations.append('Generation ещё не подключён. Возвращены найденные фрагменты.')
            return

        draft = self.generator.generate(plan, response.evidence, diagnostic)
        if self.verifier is None:
            candidates = draft.claims
        else:
            candidates = [claim for claim in self.verifier.verify(draft, response.evidence).claims if claim in draft.claims]
        for claim in candidates:
            if not all(key in sources for key in claim.evidence_ids):
                continue
            if self.verifier is None:
                quote = ' '.join(claim.text.split())
                if not any(quote in ' '.join(sources[key].text.split()) for key in claim.evidence_ids):
                    continue
            response.claims.append(claim)
        if not response.claims:
            response.limitations.append('По найденным фрагментам не удалось подтвердить технические утверждения.')

    def save_draft(
        self, actor_id: str, response: AgentResponse, result: DiagnosticResult, operations: list[TechnicalClaim],
    ) -> TicketDraft:
        parts = [claim for claim in response.claims if claim.kind == 'spare_part']
        cited = {key for claim in operations + parts for key in claim.evidence_ids}
        references = dict.fromkeys((item.manual, item.pdf_page) for item in response.evidence.items if item.evidence_id in cited)
        draft = TicketDraft(
            draft_id=uuid4().hex, expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            payload=TicketPayload(
                vehicle_id=result.vehicle_id, session_id=result.session_id, anomaly_type=result.anomaly_type,
                conclusion=f'Предполагаемое отклонение {result.anomaly_type} · {result.anomaly_title}. Требуется дефектовка.',
                urgency='medium', operations=list(dict.fromkeys(claim.text for claim in operations)),
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
            except sqlite3.Error as store_error:
                raise AgentError('TICKET_OUTCOME_UNKNOWN', 'Не удалось сохранить статус отправки. Нужна сверка.') from store_error
            raise
        try:
            self.tickets.finish(draft.draft_id, receipt=receipt)
        except sqlite3.Error as error:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'ТОиР ответил, но квитанцию не удалось сохранить. Нужна сверка.') from error
        return receipt
