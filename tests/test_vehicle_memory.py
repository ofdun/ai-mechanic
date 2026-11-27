import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from ml.config import Settings
from ml.contracts import (
    AgentRequest,
    DiagnosticResult,
    PastTicket,
    TicketDraft,
    TicketPayload,
    TicketReceipt,
    TicketReference,
    Vehicle,
)
from ml.generation import MechanicFinalizer
from ml.pipeline import Pipeline
from ml.tools.tickets import TicketStore
from ml.tools.vehicle_memory import diagnosis_memory, history_memory

ENGINE = 'ЯМЗ-236М2'
F08 = {'anomaly_type': 'F08', 'anomaly_title': 'Дизельная топливоподача'}


def record(session_id, status, score, **fault):
    return {'vehicle_id': 'M1', 'session_id': session_id, 'status': status, 'anomaly_score': score,
            'confidence': None if status == 'check_failed' else 0.9, 'signal_quality': 'good', 'description': '',
            'model_version': 'test', 'anomaly_type': None, 'anomaly_title': None, **fault}


SESSIONS = [
    {**record('S-1', 'anomaly', 0.7, **F08), 'checked_at': '2026-10-08T10:00:00+03:00'},
    {**record('S-2', 'normal', 0.2), 'checked_at': '2026-10-20T10:00:00+03:00'},
    {**record('S-3', 'check_failed', None), 'checked_at': '2026-11-01T10:00:00+03:00'},
    {**record('S-4', 'anomaly', 0.9, **F08), 'checked_at': '2026-11-17T10:00:00+03:00'},
]
VEHICLE = Vehicle(vehicle_id='M1', engine_model=ENGINE, fuel='дизель', sessions=SESSIONS)
RELIABLE = [item for item in VEHICLE.sessions if item.status != 'check_failed']


def session(session_id):
    return next(item for item in VEHICLE.sessions if item.session_id == session_id)


def test_repeat_of_the_same_fault_names_the_earlier_date():
    lines = diagnosis_memory(VEHICLE, session('S-4'), session('S-4'), [], RELIABLE)
    assert lines == ['Повтор: «Дизельная топливоподача» (F08) уже находили на этой машине 08.10.2026 (S-1).',
                     'Заявок по этой машине агент ещё не создавал.']


def test_first_check_says_there_is_nothing_to_compare():
    assert diagnosis_memory(VEHICLE, session('S-1'), session('S-1'), None, RELIABLE) == [
        'Это первая проверка этой машины в системе: сравнить не с чем.',
        'Журнал заявок сейчас недоступен: прошлые заявки не учтены.']


def test_normal_check_lists_earlier_faults_and_tickets():
    tickets = [PastTicket(ticket_id='T-7', session_id='S-1', anomaly_type='F08')]
    assert diagnosis_memory(VEHICLE, session('S-2'), session('S-2'), tickets, RELIABLE) == [
        'Раньше на этой машине находили: F08 «Дизельная топливоподача» — 08.10.2026.',
        'Заявки по этой машине: T-7 — проверка S-1, F08.']


def test_history_counts_checks_and_repeats():
    assert history_memory(VEHICLE, [], RELIABLE) == [
        'Всего проверок: 4 — норма 1, отклонение 2, сбой 1.',
        'Повторялось: «Дизельная топливоподача» (F08) — 2 раза: 08.10.2026 (S-1), 17.11.2026 (S-4).',
        'Заявок по этой машине агент ещё не создавал.',
    ]
    single = VEHICLE.model_copy(update={'sessions': VEHICLE.sessions[:2]})
    assert history_memory(single, [], single.sessions)[1] == 'На пригодных проверках повторов одной и той же неисправности не было.'


def test_ticket_store_lists_only_created_tickets_of_the_vehicle(tmp_path):
    store = TicketStore(tmp_path / 't.sqlite3', 10)
    result = DiagnosticResult.model_validate(record('S-1', 'anomaly', 0.7, **F08))
    for draft_id, vehicle in (('d1', 'M1'), ('d2', 'M1'), ('d3', 'M2')):
        draft = TicketDraft(draft_id=draft_id, engine_model=ENGINE, expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
                            payload=TicketPayload(vehicle_id=vehicle, session_id='S-1', anomaly_type='F08', conclusion='x',
                                                  urgency='high', operations=['x'], spare_parts=[],
                                                  references=[TicketReference(manual='РЭ', pdf_page=1)]))
        store.save('mechanic', draft, result.model_copy(update={'vehicle_id': vehicle, 'session_id': draft_id}))
    with store.connect() as db:
        db.execute("UPDATE drafts SET state='created', receipt=? WHERE draft_id IN ('d1', 'd3')",
                   (TicketReceipt(ticket_id='T-1', status='created').model_dump_json(),))
    assert store.created('M1') == [PastTicket(ticket_id='T-1', session_id='d1', anomaly_type='F08')]


@pytest.fixture
def pipeline(tmp_path: Path):
    vehicles = tmp_path / 'vehicles.json'
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M1', 'engine_model': ENGINE, 'fuel': 'дизель',
                                                  'sessions': SESSIONS}]}), encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'artifacts',
                        planner_mode='offline')

    def upstream(request):
        session_id = json.loads(request.content)['session_id']
        return httpx.Response(200, json=next(item for item in SESSIONS if item['session_id'] == session_id))

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        yield Pipeline(settings, http, finalizer=MechanicFinalizer())


def ask(pipeline, session_id, **turn):
    return pipeline.run(AgentRequest(actor_id='mechanic', vehicle_id='M1', session_id=session_id, **turn))


def test_diagnosis_shows_the_repeat_in_the_answer(pipeline):
    result = ask(pipeline, 'S-4', command='diagnose')
    assert result.memory[0].startswith('Повтор: «Дизельная топливоподача» (F08) уже находили на этой машине 08.10.2026')
    assert 'По этой машине:' in result.message and '08.10.2026 (S-1)' in result.message


def test_failed_check_is_not_compared_with_the_past(pipeline):
    result = ask(pipeline, 'S-3', command='diagnose')
    assert result.requires_remeasurement and result.memory == []


@pytest.mark.parametrize('question', ['Какие заявки уже были по этой машине?', 'Повторялась ли эта неисправность?'])
def test_questions_about_the_past_get_history_not_a_new_draft(pipeline, question):
    result = ask(pipeline, 'S-4', message=question)
    assert result.action == 'get_diagnostic_history' and result.ticket_draft is None
    assert result.memory[-1] == 'Заявок по этой машине агент ещё не создавал.'


@pytest.mark.parametrize('invalid', [{'confidence': 0.1}, {'signal_quality': 'warning'}])
def test_unreliable_anomaly_does_not_count_as_a_past_repeat(pipeline, invalid):
    vehicle = pipeline.vehicles.vehicles['M1']
    vehicle.sessions[0] = vehicle.sessions[0].model_copy(update=invalid)
    result = ask(pipeline, 'S-4', command='diagnose')
    assert 'Повтор:' not in result.memory[0] and 'S-1' not in result.memory[0]
    history = ask(pipeline, 'S-4', message='Повторялась ли эта неисправность?')
    assert 'норма 1, отклонение 2, сбой 1' in history.memory[0]
    assert not any('Повторялось:' in item or 'S-1' in item for item in history.memory)
