from pathlib import Path
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel, ValidationError

from ml.config import Settings
from ml.contracts import AgentError, SessionRecord, Vehicle


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
