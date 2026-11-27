import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import time

from ml.contracts import AgentError, DiagnosticResult, PastTicket, TicketConfirmation, TicketDraft, TicketReceipt


class TicketStore:
    def __init__(self, path: Path, pending_timeout: float):
        self.path = path
        self.pending_timeout = pending_timeout
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AgentError('TICKET_STORAGE_UNAVAILABLE', 'Каталог для хранения заявок недоступен.') from error
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS drafts (
                    draft_id TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    draft TEXT NOT NULL,
                    diagnostic TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'draft',
                    started_at REAL,
                    receipt TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_submission_per_fault
                ON drafts(vehicle_id, session_id, anomaly_type)
                WHERE state IN ('sending', 'created', 'unknown');
            ''')

    @contextmanager
    def connect(self):
        try:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.row_factory = sqlite3.Row
                yield db
        except sqlite3.Error as error:
            raise AgentError('TICKET_STORAGE_UNAVAILABLE', 'Хранилище заявок недоступно.') from error

    def save(self, actor_id: str, draft: TicketDraft, result: DiagnosticResult) -> None:
        with self.connect() as db:
            db.execute('''INSERT INTO drafts (draft_id, actor_id, vehicle_id, session_id, anomaly_type, draft, diagnostic)
                          VALUES (?, ?, ?, ?, ?, ?, ?)''', (
                draft.draft_id, actor_id, result.vehicle_id, result.session_id, result.anomaly_type,
                draft.model_dump_json(), result.model_dump_json(),
            ))

    def load(self, confirmation: TicketConfirmation) -> tuple[TicketDraft, DiagnosticResult, TicketReceipt | None]:
        with self.connect() as db:
            row = self.find(db, confirmation)
            draft = TicketDraft.model_validate_json(row['draft'])
            return draft, DiagnosticResult.model_validate_json(row['diagnostic']), self.receipt(row, draft, confirmation)

    @staticmethod
    def find(db, confirmation: TicketConfirmation):
        row = db.execute('''SELECT * FROM drafts
                            WHERE draft_id=? AND actor_id=? AND vehicle_id=? AND session_id=?''', (
            confirmation.draft_id, confirmation.actor_id, confirmation.vehicle_id, confirmation.session_id,
        )).fetchone()
        if row is None:
            raise AgentError('DRAFT_NOT_FOUND', 'Черновик не найден в контексте пользователя и сессии.')
        return row

    def receipt(self, row, draft: TicketDraft, confirmation: TicketConfirmation) -> TicketReceipt | None:
        if row['state'] == 'created':
            if confirmation.urgency is not None and confirmation.urgency != draft.payload.urgency:
                raise AgentError('DUPLICATE_TICKET', 'Заявка уже создана с другим приоритетом.')
            return TicketReceipt.model_validate_json(row['receipt'])
        if row['state'] == 'sending' and time() - row['started_at'] <= self.pending_timeout:
            raise AgentError('TICKET_IN_PROGRESS', 'Заявка уже отправляется.', retryable=True)
        if row['state'] in {'unknown', 'sending'}:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'Исход предыдущей отправки неизвестен. Нужна сверка с ТОиР.')
        if draft.expires_at <= datetime.now(timezone.utc):
            raise AgentError('DRAFT_EXPIRED', 'Черновик устарел. Подготовьте новый.')
        return None

    def reserve(self, confirmation: TicketConfirmation, draft: TicketDraft) -> TicketReceipt | None:
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.find(db, confirmation)
            receipt = self.receipt(row, TicketDraft.model_validate_json(row['draft']), confirmation)
            if receipt:
                return receipt
            try:
                db.execute("UPDATE drafts SET state='sending', started_at=?, draft=? WHERE draft_id=?",
                           (time(), draft.model_dump_json(), draft.draft_id))
            except sqlite3.IntegrityError as error:
                raise AgentError('DUPLICATE_TICKET', 'Для этой сессии уже есть заявка или попытка отправки.') from error
            return None

    def finish(self, draft_id: str, *, receipt: TicketReceipt | None = None, rejected: bool = False) -> None:
        if receipt:
            state = 'created'
        else:
            state = 'draft' if rejected else 'unknown'
        with self.connect() as db:
            db.execute('UPDATE drafts SET state=?, receipt=? WHERE draft_id=? AND state=?', (
                state, receipt.model_dump_json() if receipt else None, draft_id, 'sending',
            ))

    def created(self, vehicle_id: str) -> list[PastTicket]:
        """Tickets created in TOiR for this vehicle, in the order they were saved."""
        with self.connect() as db:
            rows = db.execute("SELECT session_id, anomaly_type, receipt FROM drafts WHERE vehicle_id=? AND state='created' "
                              'ORDER BY rowid', (vehicle_id,)).fetchall()
        return [PastTicket(ticket_id=TicketReceipt.model_validate_json(row['receipt']).ticket_id,
                           session_id=row['session_id'], anomaly_type=row['anomaly_type']) for row in rows]
