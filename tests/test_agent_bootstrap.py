import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import httpx
import pytest

from ml.config import Settings
from ml.contracts import (
    AgentError, AgentRequest, AudioInput, Evidence, EvidenceBundle, TechnicalClaim, TechnicalDraft,
    TicketConfirmation,
)
from ml.pipeline import Pipeline


VEHICLE = 'X001XX77'
SESSION = 'S-TEST'
ENGINE = 'ЯМЗ-236М2'
OPERATION = 'Проверить герметичность топливопровода.'


def diagnostic(status='anomaly', *, signal='good', confidence=0.83):
    return {
        'vehicle_id': VEHICLE, 'session_id': SESSION, 'status': status,
        'anomaly_type': 'F08' if status == 'anomaly' else None,
        'anomaly_title': 'Дизельная топливоподача' if status == 'anomaly' else None,
        'anomaly_score': None if status == 'check_failed' else 0.76 if status == 'anomaly' else 0.08,
        'confidence': None if status == 'check_failed' else confidence,
        'signal_quality': signal, 'description': 'Синтетическая запись', 'model_version': 'test',
    }


def utterance(**fields):
    return AgentRequest(actor_id='mechanic', vehicle_id=VEHICLE, session_id=SESSION, **fields)


class TestUpstreams:
    __test__ = False

    def __init__(self):
        self.result = diagnostic()
        self.planner_call = ('diagnose', {})
        self.ticket_timeout = False
        self.ticket_started = None
        self.release_ticket = None
        self.requests = []

    def __call__(self, request: httpx.Request):
        data = json.loads(request.content)
        self.requests.append((request.url.path, data))
        if request.url.path == '/diagnose':
            return httpx.Response(200, json=self.result)
        if request.url.path == '/tickets':
            if self.ticket_started is not None:
                self.ticket_started.set()
                assert self.release_ticket.wait(timeout=5)
            if self.ticket_timeout:
                raise httpx.ReadTimeout('mock lost response', request=request)
            return httpx.Response(201, json={'ticket_id': 'TKT-0001', 'status': 'created'})
        name, arguments = self.planner_call
        return httpx.Response(200, json={'choices': [{'message': {'tool_calls': [{
            'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)},
        }]}}]})

    def calls(self, path):
        return [data for url, data in self.requests if url == path]


class FixtureRetriever:
    def search(self, plan):
        assert plan.engine_model == ENGINE
        return EvidenceBundle(items=[
            Evidence(evidence_id='verified', document_id='fixture', manual='Fixture', engine_models=[ENGINE],
                     pdf_page=3, text=OPERATION),
            Evidence(evidence_id='wrong', document_id='other', manual='Other', engine_models=['ЯМЗ-536'],
                     pdf_page=1, text='Действие из другого руководства.'),
        ])


class FixtureGenerator:
    def generate(self, plan, evidence, diagnostic):
        assert [item.evidence_id for item in evidence.items] == ['verified']
        return TechnicalDraft(claims=[
            TechnicalClaim(kind='operation', text=OPERATION, evidence_ids=['verified']),
            TechnicalClaim(kind='operation', text='Затянуть соединение с моментом 100 Н·м.', evidence_ids=['verified']),
            TechnicalClaim(kind='operation', text='Действие из другого руководства.', evidence_ids=['wrong']),
        ])


class FixtureASR:
    def transcribe(self, audio):
        assert audio.content == b'voice sample'
        return 'Запусти диагностику двигателя.'


@pytest.fixture
def setup(tmp_path: Path):
    vehicle_path = tmp_path / 'vehicles.json'
    vehicle_path.write_text(json.dumps({'vehicles': [{
        'vehicle_id': VEHICLE, 'engine_model': ENGINE, 'fuel': 'дизель',
        'sessions': [{**diagnostic(), 'checked_at': '2026-11-19T10:00:00+03:00'}],
    }]}), encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicle_path, artifacts_dir=tmp_path / 'artifacts',
                        llm_base_url='https://llm.test/api/v1', llm_api_key='test', planner_mode='llm')
    upstreams = TestUpstreams()
    with httpx.Client(transport=httpx.MockTransport(upstreams)) as http:
        yield upstreams, Pipeline(settings, http, retriever=FixtureRetriever(), generator=FixtureGenerator(),
                                  transcriber=FixtureASR())


def test_voice_uses_native_tool_call_and_diagnoses(setup):
    upstreams, pipeline = setup
    response = pipeline.run(utterance(audio=AudioInput(content=b'voice sample')))
    assert response.transcript.startswith('Запусти диагностику')
    assert response.diagnostic.status == 'anomaly'
    assert response.search_plan.anomaly_type == 'F08'
    assert response.can_create_ticket
    assert len(response.claims) == 1
    tool_call, diagnose = upstreams.requests
    assert tool_call[0] == '/api/v1/chat/completions' and diagnose[0] == '/diagnose'
    assert {tool['function']['name'] for tool in tool_call[1]['tools']} == {
        'diagnose', 'search_manuals', 'get_diagnostic_history', 'draft_ticket', 'clarify',
    }
    assert VEHICLE not in json.dumps(tool_call[1]) and SESSION not in json.dumps(tool_call[1])


def test_search_builds_rewritten_plan_for_selected_engine(setup):
    upstreams, pipeline = setup
    upstreams.planner_call = ('search_manuals', {'rewritten_query': 'Проверка форсунок', 'subqueries': ['давление впрыска']})
    response = pipeline.run(utterance(message='Как их проверить?'))
    assert response.search_plan.original_query == 'Как их проверить?'
    assert response.search_plan.rewritten_query == 'Проверка форсунок'
    assert response.search_plan.subqueries == ['давление впрыска']
    assert response.search_plan.engine_model == ENGINE
    assert not upstreams.calls('/diagnose')


def test_unknown_tool_never_posts_a_ticket(setup):
    upstreams, pipeline = setup
    upstreams.planner_call = ('create_ticket', {})
    with pytest.raises(AgentError, match='неизвестный инструмент'):
        pipeline.run(utterance(message='Создай заявку'))
    assert not upstreams.calls('/tickets')


@pytest.mark.parametrize(('status', 'signal', 'confidence', 'repeat'), [
    ('normal', 'good', 0.9, False),
    ('check_failed', 'bad', None, True),
    ('anomaly', 'good', 0.65, True),
    ('anomaly', 'warning', 0.9, True),
    ('anomaly', 'bad', 0.9, True),
])
def test_no_draft_from_normal_or_unusable_measurement(setup, status, signal, confidence, repeat):
    upstreams, pipeline = setup
    upstreams.result = diagnostic(status, signal=signal, confidence=confidence)
    response = pipeline.run(utterance(command='draft_ticket'))
    assert response.requires_remeasurement is repeat
    assert response.diagnostic.status == status
    assert not response.can_create_ticket
    assert response.ticket_draft is None
    assert not upstreams.calls('/tickets')


def confirmation(draft, *, actor='mechanic', confirmed=True):
    return TicketConfirmation(actor_id=actor, vehicle_id=VEHICLE, session_id=SESSION,
                              draft_id=draft.draft_id, draft_version=1, confirmed=confirmed)


def test_ticket_requires_confirmation_and_is_idempotent(setup):
    upstreams, pipeline = setup
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    assert draft.payload.operations == [OPERATION]
    assert draft.payload.references[0].pdf_page == 3
    for request, code in [(confirmation(draft, confirmed=False), 'CONFIRMATION_REQUIRED'),
                          (confirmation(draft, actor='another'), 'DRAFT_NOT_FOUND')]:
        with pytest.raises(AgentError) as error:
            pipeline.confirm_ticket(request)
        assert error.value.code == code
    assert not upstreams.calls('/tickets')
    first = pipeline.confirm_ticket(confirmation(draft))
    assert pipeline.confirm_ticket(confirmation(draft)) == first
    other_draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    with pytest.raises(AgentError) as error:
        pipeline.confirm_ticket(confirmation(other_draft))
    assert error.value.code == 'DUPLICATE_TICKET'
    assert upstreams.calls('/tickets') == [draft.payload.model_dump()]


def test_confirm_rechecks_the_measurement(setup):
    upstreams, pipeline = setup
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    upstreams.result = diagnostic('check_failed', signal='bad')
    with pytest.raises(AgentError) as error:
        pipeline.confirm_ticket(confirmation(draft))
    assert error.value.code == 'REMEASUREMENT_REQUIRED'
    assert not upstreams.calls('/tickets')


def test_concurrent_confirmation_posts_once(setup):
    upstreams, pipeline = setup
    upstreams.ticket_started = Event()
    upstreams.release_ticket = Event()
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    request = confirmation(draft)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(pipeline.confirm_ticket, request)
        try:
            assert upstreams.ticket_started.wait(timeout=5)
            second = pool.submit(pipeline.confirm_ticket, request)
            with pytest.raises(AgentError) as error:
                second.result(timeout=3)
            assert error.value.code == 'TICKET_IN_PROGRESS'
        finally:
            upstreams.release_ticket.set()
        assert first.result(timeout=3).ticket_id == 'TKT-0001'
    assert len(upstreams.calls('/tickets')) == 1


def test_timeout_does_not_resend_the_ticket(setup):
    upstreams, pipeline = setup
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    upstreams.ticket_timeout = True
    for _ in range(2):
        with pytest.raises(AgentError) as error:
            pipeline.confirm_ticket(confirmation(draft))
        assert error.value.code == 'TICKET_OUTCOME_UNKNOWN'
    assert len(upstreams.calls('/tickets')) == 1
