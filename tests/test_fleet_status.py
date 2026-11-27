from pathlib import Path

import httpx
import pytest

from ml.config import Settings
from ml.contracts import Vehicle
from ml.pipeline import Pipeline
from ml.tools.fleet_status import fleet_statuses


def check(day, status, score=None, *, quality='good', confidence=0.9, code=None, title=None):
    return {'vehicle_id': 'x', 'session_id': f'S-{day}', 'status': status, 'anomaly_score': score,
            'confidence': None if status == 'check_failed' else confidence, 'signal_quality': quality,
            'description': '', 'model_version': 't', 'anomaly_type': code, 'anomaly_title': title,
            'checked_at': f'2026-11-{day:02d}T10:00:00+03:00'}


def vehicle(vehicle_id, *sessions, fuel='дизель'):
    return Vehicle(vehicle_id=vehicle_id, engine_model='ЯМЗ-236М2', fuel=fuel,
                   sessions=[{**session, 'vehicle_id': vehicle_id} for session in sessions])


@pytest.fixture
def checks(tmp_path: Path):
    path = tmp_path / 'vehicles.json'
    path.write_text('{"vehicles": []}', encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=path, artifacts_dir=tmp_path, planner_mode='offline',
                        ticket_urgency_by_fault={'F05': 'high'})
    return Pipeline(settings, httpx.Client())


def test_statuses_follow_the_latest_check_and_ticket_urgency(checks):
    fleet = [
        vehicle('NORMAL', check(1, 'anomaly', 0.9, code='F01', title='Клапаны'), check(9, 'normal', 0.1)),
        vehicle('HIGH', check(9, 'anomaly', 0.9, code='F01', title='Клапанный механизм')),
        vehicle('MEDIUM', check(9, 'anomaly', 0.6, code='F01', title='Клапанный механизм')),
        vehicle('RULE', check(9, 'anomaly', 0.5, code='F05', title='Генератор')),       # fleet rule: high
        vehicle('FAILED', check(9, 'check_failed')),
        vehicle('NOISY', check(9, 'anomaly', 0.95, quality='warning', code='F01', title='Клапанный механизм')),
        vehicle('WRONG', check(9, 'anomaly', 0.9, code='F12', title='Зажигание')),       # petrol fault on a diesel
        vehicle('EMPTY'),
    ]
    statuses = fleet_statuses(fleet, checks)
    state = {key: (value.state, value.kind) for key, value in statuses.items()}
    assert state == {
        'NORMAL': ('normal', 'normal'), 'HIGH': ('quarantine', 'fault'), 'MEDIUM': ('attention', 'fault'),
        'RULE': ('quarantine', 'fault'), 'FAILED': ('attention', 'remeasure'), 'NOISY': ('attention', 'remeasure'),
        'WRONG': ('attention', 'remeasure'), 'EMPTY': ('attention', 'no_checks'),
    }
    order = [key for key, _ in sorted(statuses.items(), key=lambda item: item[1].rank)]
    assert order[:3] == ['HIGH', 'RULE', 'MEDIUM'] and order[-1] == 'NORMAL'
    assert 'дефектовка до выпуска в рейс' in statuses['HIGH'].reason
    assert 'не норма и не доказанная неисправность' in statuses['FAILED'].reason
    assert statuses['FAILED'].anomaly_score is None
    assert (statuses['HIGH'].anomaly_type, statuses['HIGH'].urgency) == ('F01', 'high')
