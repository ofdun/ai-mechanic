"""Inspection results after a defect check: material to check the acoustic model against, never a verdict."""
import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ml.contracts import AgentError, InspectionInput, InspectionRecord, SessionRecord


class InspectionStore:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AgentError('INSPECTION_STORAGE_UNAVAILABLE', 'Каталог для результатов осмотра недоступен.') from error
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS inspections (
                              inspection_id TEXT PRIMARY KEY,
                              actor_id TEXT NOT NULL,
                              vehicle_id TEXT NOT NULL,
                              session_id TEXT NOT NULL,
                              record TEXT NOT NULL
                          )''')

    @contextmanager
    def connect(self):
        try:
            with closing(sqlite3.connect(self.path)) as db, db:
                yield db
        except sqlite3.Error as error:
            raise AgentError('INSPECTION_STORAGE_UNAVAILABLE', 'Хранилище результатов осмотра недоступно.') from error

    def add(self, actor_id: str, session: SessionRecord, found: InspectionInput) -> InspectionRecord:
        record = InspectionRecord(
            **found.model_dump(), inspection_id=uuid4().hex, vehicle_id=session.vehicle_id,
            session_id=session.session_id, model_version=session.model_version, anomaly_type=session.anomaly_type,
            recorded_at=datetime.now(timezone.utc),
        )
        with self.connect() as db:
            db.execute('INSERT INTO inspections VALUES (?, ?, ?, ?, ?)', (
                record.inspection_id, actor_id, record.vehicle_id, record.session_id, record.model_dump_json()))
        return record

    def for_vehicle(self, vehicle_id: str) -> list[InspectionRecord]:
        with self.connect() as db:
            rows = db.execute('SELECT record FROM inspections WHERE vehicle_id=? ORDER BY rowid', (vehicle_id,)).fetchall()
        return [InspectionRecord.model_validate(json.loads(row[0])) for row in rows]

    def export(self) -> list[InspectionRecord]:
        """All records for the model team; the mechanic's id is not exported."""
        with self.connect() as db:
            rows = db.execute('SELECT record FROM inspections ORDER BY rowid').fetchall()
        return [InspectionRecord.model_validate(json.loads(row[0])) for row in rows]
