"""Audit log: who asked what about which check, what the agent answered and on which pages.

Kept apart from the service logs, in its own SQLite file, without keys or tokens. Records older than the
retention period are removed on every write. Only the agent worker reads it (GET /audit); the web
backend does not expose it."""
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ml.contracts import (
    AgentError,
    AgentRequest,
    AgentResponse,
    TicketConfirmation,
    TicketReceipt,
)

COLUMNS = ('at', 'kind', 'actor_id', 'vehicle_id', 'session_id', 'entry')


class AuditLog:
    def __init__(self, path: Path, retention_days: int):
        self.path = path
        self.retention = timedelta(days=retention_days)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AgentError('AUDIT_STORAGE_UNAVAILABLE', 'Каталог журнала аудита недоступен.') from error
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS audit (
                              at TEXT NOT NULL, kind TEXT NOT NULL, actor_id TEXT NOT NULL,
                              vehicle_id TEXT NOT NULL, session_id TEXT NOT NULL, entry TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS audit_session ON audit(vehicle_id, session_id, at)')

    @contextmanager
    def connect(self):
        try:
            with closing(sqlite3.connect(self.path)) as db, db:
                yield db
        except sqlite3.Error as error:
            raise AgentError('AUDIT_STORAGE_UNAVAILABLE', 'Журнал аудита недоступен.') from error

    def write(self, kind: str, actor_id: str, vehicle_id: str, session_id: str, entry: dict) -> None:
        now = datetime.now(timezone.utc)
        with self.connect() as db:
            db.execute('DELETE FROM audit WHERE at < ?', ((now - self.retention).isoformat(),))
            db.execute('INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?)', (
                now.isoformat(), kind, actor_id, vehicle_id, session_id, json.dumps(entry, ensure_ascii=False)))

    def turn(self, request: AgentRequest, response: AgentResponse | None, error: AgentError | None = None) -> None:
        sources = {(item.manual, item.pdf_page) for item in response.evidence.items} if response else set()
        cited = {key for claim in response.claims for key in claim.evidence_ids} if response else set()
        entry = {
            'conversation_id': request.conversation_id, 'command': request.command, 'question': request.message,
            'language': request.language,
            'request_id': response.request_id if response else None,
            'action': response.action if response else None,
            'answer': response.message if response else None,
            'diagnostic': response.diagnostic.model_dump(include={'status', 'anomaly_type', 'anomaly_score',
                                                                   'confidence', 'signal_quality', 'model_version'})
            if response and response.diagnostic else None,
            # Pages that confirmed a claim, and all pages shown to the mechanic.
            'cited_pages': sorted([item.manual, item.pdf_page] for item in response.evidence.items
                                  if item.evidence_id in cited) if response else [],
            'shown_pages': sorted(list(page) for page in sources),
            'claims': len(response.claims) if response else 0,
            'draft_id': response.ticket_draft.draft_id if response and response.ticket_draft else None,
            'requires_remeasurement': response.requires_remeasurement if response else None,
            'error': error.code if error else None,
        }
        self.write('turn', request.actor_id, request.vehicle_id, request.session_id, entry)

    def confirmation(self, confirmation: TicketConfirmation, receipt: TicketReceipt | None,
                     error: AgentError | None = None) -> None:
        entry = {'draft_id': confirmation.draft_id, 'urgency': confirmation.urgency,
                 'ticket_id': receipt.ticket_id if receipt else None, 'error': error.code if error else None}
        self.write('ticket_confirm', confirmation.actor_id, confirmation.vehicle_id, confirmation.session_id, entry)

    def chain(self, vehicle_id: str, session_id: str) -> list[dict]:
        """Everything about one check, oldest first: questions, answers, drafts and the ticket."""
        with self.connect() as db:
            rows = db.execute(f'SELECT {", ".join(COLUMNS)} FROM audit WHERE vehicle_id=? AND session_id=? ORDER BY at, rowid',
                              (vehicle_id, session_id)).fetchall()
        return [{**dict(zip(COLUMNS[:-1], row[:-1])), **json.loads(row[-1])} for row in rows]
