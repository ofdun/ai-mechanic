import httpx
from pydantic import ValidationError

from ml.config import Settings
from ml.contracts import AgentError, DiagnosticResult, TicketPayload, TicketReceipt


class FleetServices:
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
            raise AgentError('VEHICLE_OR_SESSION_NOT_FOUND', 'Выбранный замер этой машины не найден.')
        if response.status_code != 200:
            raise AgentError('ML_UPSTREAM_ERROR', 'Сервис диагностики отклонил запрос.', retryable=response.status_code >= 500)
        try:
            result = DiagnosticResult.model_validate_json(response.content)
        except ValidationError as error:
            raise AgentError('ML_SCHEMA_ERROR', 'Не удалось прочитать результат замера. Повторите запрос.') from error
        if (result.vehicle_id, result.session_id) != (vehicle_id, session_id):
            raise AgentError('ML_SCHEMA_ERROR', 'Получен результат другого замера. Запросите выбранную проверку ещё раз.')
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
