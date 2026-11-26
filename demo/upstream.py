"""Two contract-compatible HTTP mocks containing synthetic measurements only."""
from itertools import count

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from demo.fixtures import training_fleet
from ml.contracts import DiagnosticResult, TicketPayload, TicketReceipt

app = FastAPI(title='Учебные сервисы диагностики и ТОиР')
measurements = {(v.vehicle_id, s.session_id): s for v in training_fleet() for s in v.sessions}
sequence = count(1)


class MeasurementId(BaseModel):
    vehicle_id: str
    session_id: str


@app.post('/diagnose', response_model=DiagnosticResult)
def diagnose(ids: MeasurementId):
    measurement = measurements.get((ids.vehicle_id, ids.session_id))
    if not measurement:
        raise HTTPException(404, 'Учебная проверка не найдена.')
    return measurement


@app.post('/tickets', status_code=201, response_model=TicketReceipt)
def ticket(payload: TicketPayload):
    if (payload.vehicle_id, payload.session_id) not in measurements:
        raise HTTPException(422, 'Неизвестная учебная проверка.')
    return TicketReceipt(ticket_id=f'DEMO-{next(sequence):04d}', status='created')
