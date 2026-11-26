from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
from pydantic import BaseModel, ValidationError

from ml.config import Settings
from ml.contracts import AgentError, DiagnosticResult, SessionRecord, TicketPayload, TicketReceipt, Vehicle


class VehiclesFile(BaseModel):
    vehicles: list[Vehicle]


class VehicleDirectory:
    def __init__(self, vehicles: list[Vehicle]):
        self.vehicles = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
        self.sessions = {(vehicle.vehicle_id, session.session_id): session
                         for vehicle in vehicles for session in vehicle.sessions}
        if len(self.vehicles) != len(vehicles) or len(self.sessions) != sum(len(v.sessions) for v in vehicles):
            raise AgentError('DATA_SCHEMA_ERROR', 'Справочник содержит повторяющиеся идентификаторы.')
        if any(key[0] != session.vehicle_id for key, session in self.sessions.items()):
            raise AgentError('DATA_SCHEMA_ERROR', 'Проверка не принадлежит указанному ТС.')

    @classmethod
    def load(cls, settings: Settings):
        try:
            path = settings.vehicles_path or settings.data_dir / 'vehicles.json'
            if settings.vehicles_path is not None or path.is_file():
                data = path.read_bytes()
            else:
                data = cls.read_archive(settings.data_dir)
            return cls(VehiclesFile.model_validate_json(data).vehicles)
        except (OSError, BadZipFile, ValidationError) as error:
            raise AgentError('DATA_UNAVAILABLE', 'Не удалось прочитать vehicles.json. Проверьте DATA_DIR или VEHICLES_PATH.') from error

    @staticmethod
    def read_archive(data_dir: Path) -> bytes:
        archives = list(data_dir.glob('neurodvs_hackathon_mock_*.zip'))
        if len(archives) != 1:
            raise AgentError('DATA_UNAVAILABLE', 'Нужен vehicles.json или один архив Mock API в DATA_DIR.')
        with ZipFile(archives[0]) as archive:
            paths = [name for name in archive.namelist() if name.endswith('/data/vehicles.json') or name == 'data/vehicles.json']
            if len(paths) != 1:
                raise AgentError('DATA_SCHEMA_ERROR', 'Ожидался один data/vehicles.json внутри архива Mock API.')
            return archive.read(paths[0])

    def get(self, vehicle_id: str, session_id: str) -> tuple[Vehicle, SessionRecord]:
        session = self.sessions.get((vehicle_id, session_id))
        if session is None:
            raise AgentError('VEHICLE_OR_SESSION_NOT_FOUND', 'Сессия не найдена для выбранного ТС.')
        return self.vehicles[vehicle_id], session


class MockServices:
    def __init__(self, http: httpx.Client, settings: Settings):
        self.http = http
        self.ml_url = f'{str(settings.ml_api_url).rstrip("/")}/diagnose'
        self.toir_url = f'{str(settings.maintenance_api_url).rstrip("/")}/tickets'

    def diagnose(self, vehicle_id: str, session_id: str) -> DiagnosticResult:
        try:
            response = self.http.post(self.ml_url, json={'vehicle_id': vehicle_id, 'session_id': session_id})
        except httpx.RequestError as error:
            raise AgentError('ML_UPSTREAM_ERROR', 'Сервис диагностики недоступен.', retryable=True) from error
        if response.status_code == 404:
            raise AgentError('VEHICLE_OR_SESSION_NOT_FOUND', 'Mock ML не нашёл сессию ТС.')
        if response.status_code != 200:
            raise AgentError('ML_UPSTREAM_ERROR', 'Сервис диагностики отклонил запрос.', retryable=response.status_code >= 500)
        try:
            result = DiagnosticResult.model_validate_json(response.content)
        except ValidationError as error:
            raise AgentError('ML_SCHEMA_ERROR', 'Mock ML прислал результат в неверном формате.') from error
        if (result.vehicle_id, result.session_id) != (vehicle_id, session_id):
            raise AgentError('ML_SCHEMA_ERROR', 'Mock ML прислал результат другой сессии.')
        return result

    def create_ticket(self, payload: TicketPayload) -> TicketReceipt:
        try:
            response = self.http.post(self.toir_url, json=payload.model_dump(mode='json'))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as error:
            raise AgentError('TOIR_UNAVAILABLE', 'Не удалось установить соединение с ТОиР.', retryable=True) from error
        except httpx.RequestError as error:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'ТОиР не подтвердил исход отправки. Уточните его перед повтором.') from error
        if response.status_code in {404, 418, 422}:
            raise AgentError('TICKET_REJECTED', 'ТОиР отклонил заявку.')
        if response.status_code != 201:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'ТОиР не подтвердил исход отправки. Уточните его перед повтором.')
        try:
            return TicketReceipt.model_validate_json(response.content)
        except ValidationError as error:
            raise AgentError('TICKET_OUTCOME_UNKNOWN', 'ТОиР прислал некорректную квитанцию. Уточните исход отправки.') from error
