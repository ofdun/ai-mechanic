from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FaultCode = Annotated[str, StringConstraints(pattern=r'^F(0[1-9]|1[0-4])$')]
Score = Annotated[float, Field(ge=0, le=1)]
PageNumber = Annotated[int, Field(ge=1)]
Action = Literal['diagnose', 'search_manuals', 'get_diagnostic_history', 'draft_ticket', 'clarify']


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
    text: Text
    kind: Literal['observation', 'cause', 'risk', 'check', 'reference_value', 'operation', 'spare_part']
    evidence_ids: list[Text] = Field(min_length=1)


class TechnicalDraft(BaseModel):
    claims: list[TechnicalClaim] = Field(default_factory=list)


class VerifiedResponse(TechnicalDraft):
    pass


class SearchOutcome(BaseModel):
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
    urgency: Literal['low', 'medium', 'high']
    operations: list[Text] = Field(min_length=1)
    spare_parts: list[Text]
    references: list[TicketReference] = Field(min_length=1)


class TicketDraft(BaseModel):
    draft_id: Text
    draft_version: Literal[1] = 1
    expires_at: datetime
    payload: TicketPayload
    engine_model: Text | None = None


class TicketDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')

    vehicle_id: Text
    session_id: Text
    draft_id: Text
    draft_version: Literal[1]
    confirmed: StrictBool


class TicketConfirmation(TicketDecision):
    actor_id: Text


class TicketReceipt(BaseModel):
    ticket_id: Text
    status: Literal['created']


class AgentTurn(BaseModel):
    model_config = ConfigDict(extra='forbid')

    vehicle_id: Text
    session_id: Text
    conversation_id: Text | None = None
    command: Literal['auto', 'diagnose', 'draft_ticket'] = 'auto'
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)] | None = None


class AgentRequest(AgentTurn):
    actor_id: Text

    @model_validator(mode='after')
    def require_input(self):
        if self.command == 'auto' and self.message is None:
            raise ValueError('An automatic request requires text')
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
    can_create_ticket: bool = False
    search_plan: SearchPlan | None = None
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claims: list[TechnicalClaim] = Field(default_factory=list)
    history: list[SessionRecord] = Field(default_factory=list)
    ticket_draft: TicketDraft | None = None
    limitations: list[str] = Field(default_factory=list)


class Retriever(Protocol):
    def search(self, plan: SearchPlan) -> EvidenceBundle: ...


class Generator(Protocol):
    def generate(self, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None) -> TechnicalDraft: ...


class Verifier(Protocol):
    def verify(self, draft: TechnicalDraft, evidence: EvidenceBundle) -> VerifiedResponse: ...


class SearchWorkflow(Protocol):
    def run(self, plan: SearchPlan, diagnostic: DiagnosticResult | None) -> SearchOutcome: ...
