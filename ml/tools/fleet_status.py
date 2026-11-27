"""B3: status of every vehicle for the chief mechanic, from its latest check.

The rules are the ones the agent already uses, so the fleet screen and the ticket never disagree:
- quarantine: a reliable anomaly whose ticket urgency is «high» (defect check before the next trip);
- attention: a reliable anomaly of lower urgency, or a check that cannot be trusted (failed record, noise,
  low confidence, a fault impossible for this engine) — that one needs a repeat measurement and is neither
  normal nor a proven fault;
- normal: the latest check found no deviation.
Nothing here estimates repair cost or downtime."""
from typing import Literal, Protocol

from pydantic import BaseModel

from ml.config import Settings
from ml.contracts import DiagnosticResult, SessionRecord, Vehicle
from ml.generation.profiles import urgency_for

State = Literal['quarantine', 'attention', 'normal']
Kind = Literal['fault', 'remeasure', 'normal', 'no_checks']
URGENCY_STEPS = {
    'high': 'дефектовка до выпуска в рейс',
    'medium': 'дефектовка на ближайшем ТО, эксплуатация под наблюдением',
    'low': 'проверить узел при следующей проверке',
}


class Checks(Protocol):
    settings: Settings

    def measurement_issue(self, result: DiagnosticResult) -> str | None: ...

    @staticmethod
    def ticket_mismatch(vehicle: Vehicle, result: DiagnosticResult) -> str | None: ...


class FleetStatus(BaseModel):
    state: State
    kind: Kind
    reason: str
    session_id: str | None = None
    checked_at: str | None = None
    anomaly_score: float | None = None
    # For a fault: what the web needs to write the reason in the mechanic's language.
    anomaly_type: str | None = None
    anomaly_title: str | None = None
    urgency: Literal['low', 'medium', 'high'] | None = None
    rank: int = 0


def score_text(value: float) -> str:
    return f'{value:.2f}'.replace('.', ',')


def latest(vehicle: Vehicle) -> SessionRecord | None:
    return max(vehicle.sessions, key=lambda session: session.checked_at, default=None)


def vehicle_status(vehicle: Vehicle, checks: Checks) -> tuple[FleetStatus, tuple]:
    """The status and a sort key: the most urgent vehicle has the smallest key."""
    session = latest(vehicle)
    if session is None:
        return FleetStatus(state='attention', kind='no_checks', reason='Проверок нет: нужен первый замер.'), (2, 0)
    base = {'session_id': session.session_id, 'checked_at': session.checked_at.isoformat(),
            'anomaly_score': session.anomaly_score}
    score = session.anomaly_score or 0
    problem = checks.measurement_issue(session) or checks.ticket_mismatch(vehicle, session)
    if problem:
        reason = f'{problem} Это не норма и не доказанная неисправность.'
        return FleetStatus(state='attention', kind='remeasure', reason=reason, **base), (2, -score)
    if session.status == 'normal':
        return FleetStatus(state='normal', kind='normal', reason='Последняя проверка без отклонений.', **base), (3, -score)
    urgency = checks.settings.ticket_urgency_by_fault.get(session.anomaly_type) or urgency_for(
        session.anomaly_type, session.anomaly_score)
    reason = (f'{session.anomaly_type} «{session.anomaly_title}», оценка аномалии {score_text(score)}: '
              f'{URGENCY_STEPS[urgency]}.')
    state = 'quarantine' if urgency == 'high' else 'attention'
    fault = {'anomaly_type': session.anomaly_type, 'anomaly_title': session.anomaly_title, 'urgency': urgency}
    return FleetStatus(state=state, kind='fault', reason=reason, **base, **fault), (0 if state == 'quarantine' else 1,
                                                                            -['low', 'medium', 'high'].index(urgency), -score)


def fleet_statuses(vehicles: list[Vehicle], checks: Checks) -> dict[str, FleetStatus]:
    """Statuses with ``rank`` 1 for the most urgent vehicle; ties keep the vehicle id order."""
    rated = sorted(((vehicle_status(vehicle, checks), vehicle.vehicle_id) for vehicle in vehicles),
                   key=lambda item: (item[0][1], item[1]))
    return {vehicle_id: status.model_copy(update={'rank': rank})
            for rank, ((status, _), vehicle_id) in enumerate(rated, 1)}
