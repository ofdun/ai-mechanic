from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FaultCode = Annotated[str, StringConstraints(pattern=r'^F(0[1-9]|1[0-4])$')]
Score = Annotated[float, Field(ge=0, le=1)]
PageNumber = Annotated[int, Field(ge=1)]
Action = Literal['diagnose', 'search_manuals', 'get_diagnostic_history', 'draft_ticket', 'clarify']
MAX_AUDIO_BYTES = 20 * 1024 * 1024


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


class TicketConfirmation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    actor_id: Text
    vehicle_id: Text
    session_id: Text
    draft_id: Text
    draft_version: Literal[1]
    confirmed: StrictBool


class TicketReceipt(BaseModel):
    ticket_id: Text
    status: Literal['created']


class AudioInput(BaseModel):
    content: Annotated[bytes, Field(min_length=1, max_length=MAX_AUDIO_BYTES)]


class AgentRequest(BaseModel):
    actor_id: Text
    vehicle_id: Text
    session_id: Text
    command: Literal['auto', 'diagnose', 'draft_ticket'] = 'auto'
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)] | None = None
    audio: AudioInput | None = None

    @model_validator(mode='after')
    def require_input(self):
        if self.message is not None and self.audio is not None:
            raise ValueError('Pass either text or audio')
        if self.command == 'auto' and self.message is None and self.audio is None:
            raise ValueError('An automatic request requires text or audio')
        return self


class AgentResponse(BaseModel):
    request_id: str
    vehicle_id: str
    session_id: str
    engine_model: str
    action: Action
    message: str
    transcript: str | None = None
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


class Transcriber(Protocol):
    def transcribe(self, audio: AudioInput) -> str: ...
