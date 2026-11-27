import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from ml.config import Settings
from ml.contracts import AgentRequest, InspectionInput
from ml.generation import MechanicFinalizer
from ml.pipeline import Pipeline

SESSION = {'vehicle_id': 'M1', 'session_id': 'S-1', 'status': 'anomaly', 'anomaly_type': 'F14',
           'anomaly_title': 'Цепь ГРМ', 'anomaly_score': 0.8, 'confidence': 0.9, 'signal_quality': 'good',
           'description': '', 'model_version': 'neuro-1.2', 'checked_at': '2026-11-19T10:00:00+03:00'}


@pytest.fixture
def pipeline(tmp_path: Path):
    vehicles = tmp_path / 'vehicles.json'
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M1', 'engine_model': 'ЗМЗ-40905.10',
                                                  'fuel': 'бензин', 'sessions': [SESSION]}]}), encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'artifacts', planner_mode='offline')
    calls = []

    def upstream(request):
        calls.append(request.url.path)
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        pipeline = Pipeline(settings, http, finalizer=MechanicFinalizer())
        pipeline.upstream_calls = calls
        yield pipeline


def record(pipeline, **found):
    return pipeline.run(AgentRequest(actor_id='mechanic', vehicle_id='M1', session_id='S-1',
                                     command='record_inspection', inspection=InspectionInput(**found)))


def test_record_is_stored_with_the_check_and_model_version(pipeline):
    result = record(pipeline, outcome='other_cause', marks='matched', comment='  Натяжитель исправен, шум от генератора ')
    assert result.action == 'record_inspection' and result.diagnostic is None and result.ticket_draft is None
    assert result.message == 'Результат осмотра записан. Диагноз и заявка не меняются.'
    [saved] = result.inspections
    assert (saved.vehicle_id, saved.session_id, saved.model_version, saved.anomaly_type) == ('M1', 'S-1', 'neuro-1.2', 'F14')
    assert (saved.outcome, saved.marks, saved.comment) == ('other_cause', 'matched', 'Натяжитель исправен, шум от генератора')
    assert saved.recorded_at.tzinfo is not None
    # Nothing is asked from the acoustic model or TOiR, so the diagnosis and the ticket cannot change.
    assert pipeline.upstream_calls == []
    assert [item.model_dump() for item in pipeline.inspections.export()] == [saved.model_dump()]


def test_matched_marks_do_not_turn_the_check_into_normal(pipeline):
    record(pipeline, outcome='not_confirmed', marks='matched')
    history = pipeline.run(AgentRequest(actor_id='mechanic', vehicle_id='M1', session_id='S-1',
                                        command='get_diagnostic_history'))
    assert history.history[0].status == 'anomaly'
    assert '— гипотеза ИИ не подтвердилась, метки совпали' in history.message
    assert 'Результаты осмотра:' in history.message


def test_inspection_goes_only_with_its_command():
    with pytest.raises(ValidationError):
        AgentRequest(actor_id='a', vehicle_id='M1', session_id='S-1', command='record_inspection')
    with pytest.raises(ValidationError):
        AgentRequest(actor_id='a', vehicle_id='M1', session_id='S-1', command='diagnose',
                     inspection=InspectionInput(outcome='confirmed'))
    with pytest.raises(ValidationError):
        InspectionInput(outcome='confirmed', angle=12)                          # no angle is entered
