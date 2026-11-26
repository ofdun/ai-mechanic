import hashlib
import secrets
import sqlite3
import time
from contextlib import contextmanager, closing
from pathlib import Path
from uuid import uuid4

from ml.contracts import AgentError, AgentRequest, AgentResponse, ConversationContext
from ml.pipeline import Pipeline


SESSION_SECONDS = 7 * 24 * 3600
# Allow recovery if a worker dies mid-turn. All upstream HTTP calls have timeouts.
TURN_LEASE_SECONDS = 30 * 60


class WebSessions:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS actors (
                    actor TEXT PRIMARY KEY, expires INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    actor TEXT NOT NULL REFERENCES actors ON DELETE CASCADE,
                    conversation TEXT NOT NULL, vehicle TEXT NOT NULL, session TEXT NOT NULL,
                    context TEXT, turn TEXT, lease_until INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (actor, conversation)
                );
            ''')

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute('PRAGMA foreign_keys = ON')
            yield db

    def resolve(self, token: str | None) -> tuple[str, str | None]:
        now = int(time.time())
        with self.connection() as db:
            if token is not None:
                actor = hashlib.sha256(token.encode()).hexdigest()
                if db.execute('SELECT 1 FROM actors WHERE actor=? AND expires>?', (actor, now)).fetchone():
                    return actor, None
            token = secrets.token_urlsafe(32)
            actor = hashlib.sha256(token.encode()).hexdigest()
            db.execute('DELETE FROM actors WHERE expires <= ?', (now,))
            db.execute('INSERT INTO actors VALUES (?,?)', (actor, now + SESSION_SECONDS))
            return actor, token

    def run(self, pipeline: Pipeline, request: AgentRequest) -> AgentResponse:
        if request.conversation_id is None:
            request = request.model_copy(update={'conversation_id': uuid4().hex})
        pipeline.vehicles.get(request.vehicle_id, request.session_id)
        key = (request.actor_id, request.conversation_id)
        turn = uuid4().hex
        now = int(time.time())
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO conversations (actor,conversation,vehicle,session) VALUES (?,?,?,?)',
                       (*key, request.vehicle_id, request.session_id))
            vehicle, session, saved, lease = db.execute(
                'SELECT vehicle,session,context,lease_until FROM conversations WHERE actor=? AND conversation=?',
                key,
            ).fetchone()
            if (vehicle, session) != (request.vehicle_id, request.session_id):
                raise AgentError('CONTEXT_MISMATCH', 'Для выбранной машины или проверки нужен новый диалог.')
            if lease > now:
                raise AgentError('CONVERSATION_BUSY', 'В этом диалоге ещё готовится ответ.', retryable=True)
            db.execute('UPDATE conversations SET turn=?, lease_until=? WHERE actor=? AND conversation=?',
                       (turn, now + TURN_LEASE_SECONDS, *key))

        try:
            context = ConversationContext.model_validate_json(saved) if saved else None
            response = pipeline.run(request, context=context)
            messages = list(context.recent_messages) if context else []
            if request.message:
                messages.append(request.message[:1500])
            messages.append(response.message[:1500])
            updated = ConversationContext(
                actor_id=request.actor_id, conversation_id=request.conversation_id,
                vehicle_id=response.vehicle_id, session_id=response.session_id, engine_model=response.engine_model,
                diagnostic=response.diagnostic or (context.diagnostic if context else None),
                recent_messages=messages[-4:],
            )
            with self.connection() as db:
                stored = db.execute(
                    'UPDATE conversations SET context=?, turn=NULL, lease_until=0 '
                    'WHERE actor=? AND conversation=? AND turn=?',
                    (updated.model_dump_json(), *key, turn),
                )
                if not stored.rowcount:
                    raise AgentError('CONTEXT_MISMATCH', 'Контекст диалога изменился. Откройте новый диалог.')
            return response
        except BaseException:
            with self.connection() as db:
                db.execute('UPDATE conversations SET turn=NULL, lease_until=0 WHERE actor=? AND conversation=? AND turn=?',
                           (*key, turn))
            raise
