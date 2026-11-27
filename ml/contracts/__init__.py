from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FaultCode = Annotated[str, StringConstraints(pattern=r'^F(0[1-9]|1[0-4])$')]
TicketUrgency = Literal['low', 'medium', 'high']
Score = Annotated[float, Field(ge=0, le=1)]
PageNumber = Annotated[int, Field(ge=1)]
Action = Literal['diagnose', 'search_manuals', 'get_diagnostic_history', 'draft_ticket', 'clarify', 'record_inspection']
# Answer language for the mechanic: Russian, Uzbek (Latin), Tajik, Kyrgyz, Kazakh. Manuals and tickets stay Russian.
Language = Literal['ru', 'uz', 'tg', 'ky', 'kk', 'hi']


class AgentError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class DiagnosticResult(BaseModel):
    vehicle_id: Text
    session_id: Text
    status: Literal['normal', 'anomaly', 'check_failed']
    anomaly_type: FaultCode | None
    anomaly_title: Text | None
    anomaly_score: Score | None
    confidence: Score | None
    signal_quality: Literal['good', 'warning', 'bad']
    description: str
    model_version: Text

    @model_validator(mode='after')
    def consistent_status(self):
        if self.status == 'anomaly':
            if self.anomaly_type is None or self.anomaly_title is None or self.anomaly_score is None:
                raise ValueError('An anomaly must provide its type, title and score')
        elif self.anomaly_type is not None or self.anomaly_title is not None:
            raise ValueError('Fault fields are only valid for an anomaly')
        if self.status == 'check_failed' and (self.anomaly_score is not None or self.confidence is not None):
            raise ValueError('A failed measurement must have null scores')
        return self


class SessionRecord(DiagnosticResult):
    checked_at: AwareDatetime


class Vehicle(BaseModel):
    vehicle_id: Text
    make: str | None = None
    model: str | None = None
    engine_model: Text
    fuel: str = ''
    sessions: list[SessionRecord] = Field(default_factory=list)


class DocumentPage(BaseModel):
    pdf_page: PageNumber
    printed_page: str | None = None
    section: str | None = None
    text: str
    engine_models: list[Text] | None = Field(default=None, min_length=1)


class ParsedDocument(BaseModel):
    document_id: Text
    manual: Text
    engine_models: list[Text] = Field(min_length=1)
    pages: list[DocumentPage]


class SearchPlan(BaseModel):
    engine_model: Text
    original_query: Text
    rewritten_query: Text
    subqueries: list[Text] = Field(default_factory=list, max_length=5)
    anomaly_type: FaultCode | None = None


class Evidence(BaseModel):
    evidence_id: Text
    document_id: Text
    manual: Text
    engine_models: list[Text] = Field(min_length=1)
    pdf_page: PageNumber
    section: str | None = None
    text: Text

    @classmethod
    def from_page(
        cls, document: ParsedDocument, page: DocumentPage, *, evidence_id: str,
        excerpt: str | None = None, engine_models: list[str] | None = None,
    ) -> 'Evidence':
        page_models = document.engine_models if page.engine_models is None else page.engine_models
        models = page_models if engine_models is None else engine_models
        if not set(page_models).issubset(document.engine_models) or not set(models).issubset(page_models):
            raise ValueError('Page or fragment applicability cannot broaden its source scope')
        return cls(
            evidence_id=evidence_id, document_id=document.document_id, manual=document.manual,
            engine_models=models, pdf_page=page.pdf_page, section=page.section,
            text=page.text if excerpt is None else excerpt,
        )


class EvidenceBundle(BaseModel):
    items: list[Evidence] = Field(default_factory=list)


class TechnicalClaim(BaseModel):
    # Optional verbatim evidence excerpt for source navigation; old responses may omit it.
    quote: str = ''
    text: Text
    kind: Literal['observation', 'cause', 'risk', 'check', 'reference_value', 'operation', 'spare_part']
    evidence_ids: list[Text] = Field(min_length=1)


class TechnicalDraft(BaseModel):
    claims: list[TechnicalClaim] = Field(default_factory=list)


class CheckedPage(BaseModel):
    evidence_id: Text
    numbers: list[str] = Field(default_factory=list)


class AnswerVerification(BaseModel):
    engine_model: Text
    pages: list[CheckedPage] = Field(default_factory=list)
    rejected_claims: int | None = Field(default=None, ge=0)


class VerifiedResponse(TechnicalDraft):
    verification: AnswerVerification | None = None

    @classmethod
    def from_reviewed(cls, engine, draft, supported):
        approved = [claim for claim in draft.claims if claim in supported]
        keys = dict.fromkeys(key for claim in approved for key in claim.evidence_ids)
        audit = AnswerVerification(engine_model=engine, pages=[CheckedPage(evidence_id=k) for k in keys],
                                   rejected_claims=len(draft.claims) - len(approved)) if draft.claims else None
        return cls(claims=approved, verification=audit)


class SearchOutcome(BaseModel):
    verification: AnswerVerification | None = None
    plan: SearchPlan
    evidence: EvidenceBundle
    claims: list[TechnicalClaim]
    summary: str = ''


class TicketReference(BaseModel):
    manual: Text
    pdf_page: PageNumber


class TicketPayload(BaseModel):
    vehicle_id: Text
    session_id: Text
    anomaly_type: FaultCode
    conclusion: Text
    urgency: TicketUrgency
    operations: list[Text] = Field(min_length=1)
    spare_parts: list[Text]
    references: list[TicketReference] = Field(min_length=1)


class TicketDraft(BaseModel):
    draft_id: Text
    draft_version: Literal[1] = 1
    expires_at: datetime
    payload: TicketPayload
    engine_model: Text | None = None
    urgency_reason: Text | None = None
    # The mechanic confirmed a draft from a noisy recording (signal_quality=warning); checked again on send.
    signal_warning_accepted: bool = False


class TicketDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    vehicle_id: Text
    session_id: Text
    draft_id: Text
    draft_version: Literal[1]
    confirmed: StrictBool
    urgency: TicketUrgency | None = None


class TicketConfirmation(TicketDecision):
    actor_id: Text


class CallUsage(BaseModel):
    """One measured step of a turn. ``None`` means the provider did not report the value."""
    step: str
    kind: Literal['llm', 'retrieval', 'service']
    model: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int = 0
    retries: int = 0
    ok: bool = True


class TurnUsage(BaseModel):
    """Measured calls of one request to the agent, for the service view; not shown to the mechanic."""
    latency_ms: int
    steps: list[CallUsage] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    # Sum of the costs the provider returned; ``cost_complete`` is False when a paid call returned none.
    cost_usd: float | None = None
    cost_complete: bool = True
    retries: int = 0
    failed_calls: int = 0


class TicketReceipt(BaseModel):
    ticket_id: Text
    status: Literal['created']
    usage: TurnUsage | None = None


class PastTicket(BaseModel):
    """A ticket the agent created for this vehicle earlier (B1 vehicle memory)."""
    ticket_id: Text
    session_id: Text
    anomaly_type: Text


class PartNumber(BaseModel):
    """A part designation printed on a manual page of this engine; stock and compatibility are not known."""
    designation: Text
    context: str
    manual: Text
    document_id: Text
    pdf_page: PageNumber
    evidence_id: Text


class FigureReference(BaseModel):
    """A figure with a tightening order on a page of this engine's manual."""
    figure: Text
    caption: str | None = None
    manual: Text
    document_id: Text
    pdf_page: PageNumber
    # Numbered steps only from a sequence people checked against this page; empty means «open the figure».
    sequence: list[str] = Field(default_factory=list)


class InspectionInput(BaseModel):
    """What the mechanic found after the defect check. It never changes the diagnosis or the ticket."""
    model_config = ConfigDict(extra='forbid')

    # Did the acoustic hypothesis hold: confirmed / not confirmed / the cause was another one.
    outcome: Literal['confirmed', 'not_confirmed', 'other_cause']
    # For checks by marks: matched or not; no angle is entered. «matched» does not mean the engine is fine.
    marks: Literal['matched', 'not_matched'] | None = None
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ''


class InspectionRecord(InspectionInput):
    inspection_id: Text
    vehicle_id: Text
    session_id: Text
    model_version: Text
    anomaly_type: FaultCode | None = None
    recorded_at: AwareDatetime


class AgentTurn(BaseModel):
    model_config = ConfigDict(extra='forbid')

    vehicle_id: Text
    session_id: Text
    conversation_id: Text | None = None
    command: Literal['auto', 'diagnose', 'draft_ticket', 'get_diagnostic_history', 'record_inspection'] = 'auto'
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)] | None = None
    language: Language = 'ru'
    # «Понимаю, что запись с помехами»: lets a noisy but confident recording reach a verdict and a ticket.
    accept_signal_warning: bool = False
    inspection: InspectionInput | None = None


class AgentRequest(AgentTurn):
    actor_id: Text

    @model_validator(mode='after')
    def require_input(self):
        if self.command == 'auto' and self.message is None:
            raise ValueError('An automatic request requires text')
        if (self.command == 'record_inspection') != (self.inspection is not None):
            raise ValueError('An inspection is sent only with the record_inspection command')
        return self


class ConversationContext(BaseModel):
    actor_id: Text
    vehicle_id: Text
    session_id: Text
    conversation_id: Text
    engine_model: Text
    diagnostic: DiagnosticResult | None = None
    recent_messages: list[Annotated[Text, StringConstraints(max_length=1500)]] = Field(default_factory=list, max_length=4)

    @model_validator(mode='after')
    def match_diagnostic(self):
        if self.diagnostic and (self.diagnostic.vehicle_id, self.diagnostic.session_id) != (self.vehicle_id, self.session_id):
            raise ValueError('Diagnostic context must match the vehicle and session')
        return self


class AgentResponse(BaseModel):
    verification: AnswerVerification | None = None
    request_id: str
    vehicle_id: str
    session_id: str
    conversation_id: str | None = None
    engine_model: str
    action: Action
    message: str
    # Plain-language lead written only from the verified claims; empty when generation did not write one.
    summary: str = ''
    diagnostic: DiagnosticResult | None = None
    requires_remeasurement: bool = False
    # The record is noisy but confident: the mechanic may continue after an explicit confirmation.
    warning_confirmable: bool = False
    signal_warning_accepted: bool = False
    # The confidence threshold the verdict was checked against, shown to the mechanic.
    min_confidence: float | None = None
    recognition_error: bool = False
    manual_referral: str | None = None
    can_create_ticket: bool = False
    search_plan: SearchPlan | None = None
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claims: list[TechnicalClaim] = Field(default_factory=list)
    history: list[SessionRecord] = Field(default_factory=list)
    # B1: comparison with earlier checks of this vehicle and the tickets created for it.
    memory: list[str] = Field(default_factory=list)
    past_tickets: list[PastTicket] = Field(default_factory=list)
    # Part designations from the pages behind the answer; none is taken from external catalogues.
    part_numbers: list[PartNumber] = Field(default_factory=list)
    part_numbers_note: str | None = None
    figures: list[FigureReference] = Field(default_factory=list)
    # Inspection results recorded for this vehicle, newest last.
    inspections: list[InspectionRecord] = Field(default_factory=list)
    ticket_draft: TicketDraft | None = None
    limitations: list[str] = Field(default_factory=list)
    language: Language = 'ru'
    # The answer in ``language``; Russian fields above stay the source of truth for checks and the ticket.
    localized: 'LocalizedAnswer | None' = None
    usage: TurnUsage | None = None


class LocalizedText(BaseModel):
    text: str
    # The Russian source under a translated line that carries a prohibition, negation or condition,
    # so the mechanic can check the exact wording of the manual.
    original: str | None = None
    # False when the translation failed its checks and ``text`` is the Russian source.
    translated: bool = True


class LocalizedAnswer(BaseModel):
    language: Language
    message: LocalizedText
    summary: list[LocalizedText] = Field(default_factory=list)
    # Same order and length as AgentResponse.claims.
    claims: list[LocalizedText] = Field(default_factory=list)
    limitations: list[LocalizedText] = Field(default_factory=list)
    memory: list[LocalizedText] = Field(default_factory=list)
    # Fixed interface labels for this language: sections, conflict, urgency, notes.
    labels: dict[str, str] = Field(default_factory=dict)


AgentResponse.model_rebuild()


class Retriever(Protocol):
    def search(self, plan: SearchPlan) -> EvidenceBundle: ...


class Generator(Protocol):
    def generate(self, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None) -> TechnicalDraft: ...


class Verifier(Protocol):
    def verify(self, draft: TechnicalDraft, evidence: EvidenceBundle) -> VerifiedResponse: ...


class SearchWorkflow(Protocol):
    def run(self, plan: SearchPlan, diagnostic: DiagnosticResult | None) -> SearchOutcome: ...
