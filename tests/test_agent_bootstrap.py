import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import httpx
import pytest
from pydantic import ValidationError

from ml.config import Settings
from ml.contracts import (
    AgentError, AgentRequest, ConversationContext, DocumentPage, Evidence, EvidenceBundle,
    ParsedDocument, TechnicalClaim, TechnicalDraft, TicketConfirmation, VerifiedResponse,
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
        manual = ParsedDocument(document_id='fixture', manual='Fixture', engine_models=[ENGINE, 'ЯМЗ-536'], pages=[
            DocumentPage(pdf_page=3, text=OPERATION),
            DocumentPage(pdf_page=1, text='Действие для другой модели.', engine_models=['ЯМЗ-536']),
        ])
        # Exercise the ingestion-to-retrieval hand-off, including the narrower page metadata.
        manual = ParsedDocument.model_validate_json(manual.model_dump_json())
        return EvidenceBundle(items=[Evidence.from_page(manual, page, evidence_id=key)
                                     for key, page in zip(('verified', 'wrong'), manual.pages)])


class FixtureGenerator:
    def generate(self, plan, evidence, diagnostic):
        assert [item.evidence_id for item in evidence.items] == ['verified']
        return TechnicalDraft(claims=[
            TechnicalClaim(kind='operation', text=OPERATION, evidence_ids=['verified']),
            TechnicalClaim(kind='operation', text='Затянуть соединение с моментом 100 Н·м.', evidence_ids=['verified']),
            TechnicalClaim(kind='operation', text='Действие из другого руководства.', evidence_ids=['wrong']),
        ])


class FixtureVerifier:
    def verify(self, draft, evidence):
        return VerifiedResponse(claims=[claim for claim in draft.claims if claim.text == OPERATION])


@pytest.fixture
def setup(tmp_path: Path):
    vehicle_path = tmp_path / 'vehicles.json'
    vehicle_path.write_text(json.dumps({'vehicles': [{
        'vehicle_id': VEHICLE, 'engine_model': ENGINE, 'fuel': 'дизель',
        'sessions': [{**diagnostic(), 'checked_at': '2026-11-19T10:00:00+03:00'}],
    }]}), encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicle_path, artifacts_dir=tmp_path / 'artifacts',
                        llm_base_url='https://llm.test/api/v1', llm_api_key='test', planner_mode='llm',
                        min_diagnostic_confidence=0.70, repeat_on_warning=True)
    upstreams = TestUpstreams()
    with httpx.Client(transport=httpx.MockTransport(upstreams)) as http:
        yield upstreams, Pipeline(settings, http, retriever=FixtureRetriever(), generator=FixtureGenerator(),
                                  verifier=FixtureVerifier())


def test_text_uses_native_tool_call_and_diagnoses(setup):
    upstreams, pipeline = setup
    response = pipeline.run(utterance(message='Запусти диагностику двигателя.'))
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


def test_ticket_history_does_not_call_the_planner_or_prepare_a_draft(setup):
    upstreams, pipeline = setup
    upstreams.planner_call = ('draft_ticket', {})
    result = pipeline.run(utterance(message='Покажи прошлые заявки на эту машину'))
    assert result.action == 'get_diagnostic_history'
    assert result.memory[-1] == 'Заявок по этой машине агент ещё не создавал.'
    assert result.ticket_draft is None and upstreams.requests == []


def test_search_builds_rewritten_plan_for_selected_engine(setup):
    upstreams, pipeline = setup
    upstreams.planner_call = ('search_manuals', {'rewritten_query': 'Проверка форсунок', 'subqueries': ['давление впрыска']})
    response = pipeline.run(utterance(message='Как их проверить?'))
    assert response.search_plan.original_query == 'Как их проверить?'
    assert response.search_plan.rewritten_query == 'Проверка форсунок'
    assert response.search_plan.subqueries == ['давление впрыска']
    assert response.search_plan.engine_model == ENGINE
    assert not upstreams.calls('/diagnose')


def test_follow_up_uses_backend_context_instead_of_archived_anomaly(setup):
    upstreams, pipeline = setup
    upstreams.result = diagnostic('check_failed', signal='bad')
    fresh = pipeline.run(utterance(command='diagnose', conversation_id='chat'))
    context = ConversationContext(
        actor_id='mechanic', vehicle_id=VEHICLE, session_id=SESSION, conversation_id='chat',
        engine_model=ENGINE, diagnostic=fresh.diagnostic, recent_messages=['Ранее подозревали неисправность форсунок.'],
    )
    upstreams.planner_call = ('search_manuals', {'rewritten_query': 'Контроль записи'})
    follow_up = pipeline.run(utterance(message='Что теперь проверить?', conversation_id='chat'), context=context)
    prompt = json.loads(upstreams.calls('/api/v1/chat/completions')[-1]['messages'][-1]['content'])
    assert follow_up.conversation_id == 'chat'
    assert 'fault' not in prompt and 'observed' not in prompt
    assert 'recent_messages' not in prompt
    pipeline.run(utterance(message='Что проверить?'))
    standalone_prompt = json.loads(upstreams.calls('/api/v1/chat/completions')[-1]['messages'][-1]['content'])
    assert 'fault' not in standalone_prompt
    with pytest.raises(AgentError) as error:
        pipeline.run(utterance(message='Что проверить?', conversation_id='another'), context=context)
    assert error.value.code == 'CONTEXT_MISMATCH'
    upstreams.result = diagnostic()
    fresh = pipeline.run(utterance(command='diagnose', conversation_id='chat'))
    context = context.model_copy(update={'diagnostic': fresh.diagnostic})
    pipeline.run(utterance(message='Что проверить?', conversation_id='chat'), context=context)
    fresh_prompt = json.loads(upstreams.calls('/api/v1/chat/completions')[-1]['messages'][-1]['content'])
    assert fresh_prompt['fault'] == fresh.diagnostic.anomaly_title
    assert fresh_prompt['recent_messages'] == context.recent_messages


def test_unknown_tool_never_posts_a_ticket(setup):
    upstreams, pipeline = setup
    upstreams.planner_call = ('create_ticket', {})
    with pytest.raises(AgentError, match='неизвестный инструмент'):
        pipeline.run(utterance(message='Создай заявку'))
    assert not upstreams.calls('/tickets')


@pytest.mark.parametrize(('source_text', 'operation'), [
    ('Запрещено запускать двигатель.', 'запускать двигатель.'),
    ('Только после остывания открыть крышку.', 'открыть крышку.'),
    ('Затянуть с моментом 110 Н·м.', '10 Н·м.'),
])
def test_even_verbatim_substrings_cannot_create_unverified_ticket(setup, source_text, operation):
    upstreams, pipeline = setup

    class Source:
        def search(self, plan):
            return EvidenceBundle(items=[Evidence(evidence_id='p', document_id='d', manual='Fixture',
                                                  pdf_page=3, engine_models=[ENGINE], text=source_text)])

    class DraftGenerator:
        def generate(self, plan, evidence, diagnostic):
            return TechnicalDraft(claims=[TechnicalClaim(kind='operation', text=operation, evidence_ids=['p'])])

    pipeline.retriever = Source()
    pipeline.generator = DraftGenerator()
    pipeline.verifier = None
    response = pipeline.run(utterance(command='draft_ticket'))
    assert response.evidence.items[0].text == source_text
    assert not response.claims and not response.can_create_ticket and response.ticket_draft is None
    assert all(value in response.manual_referral for value in (VEHICLE, ENGINE, SESSION, 'F08', 'дефектовку'))
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
    assert response.manual_referral is None
    assert not upstreams.calls('/tickets')


def test_wrong_fuel_is_recognition_error_without_manual_or_draft(setup):
    _, pipeline = setup
    vehicle, _ = pipeline.vehicles.get(VEHICLE, SESSION)
    vehicle.fuel = 'бензин'
    vehicle.engine_model = 'ЗМЗ-5234.10'
    response = pipeline.run(utterance(command='draft_ticket'))
    assert response.recognition_error
    assert 'бензиновый двигатель ЗМЗ-5234.10' in response.message
    assert not response.can_create_ticket and response.ticket_draft is None
    assert response.search_plan is None and response.manual_referral is None


@pytest.mark.parametrize('signal', ['warning', 'good'])
def test_low_confidence_explains_configured_boundary(setup, signal):
    upstreams, pipeline = setup
    pipeline.settings.min_diagnostic_confidence = 0.875
    upstreams.result = diagnostic(signal=signal, confidence=0.873)
    response = pipeline.run(utterance(command='diagnose'))
    assert response.requires_remeasurement
    assert '87,3%' in response.message and '87,5%' in response.message
    assert ('помехи' in response.message) == (signal == 'warning')


def confirmation(draft, *, actor='mechanic', confirmed=True, urgency=None):
    return TicketConfirmation(actor_id=actor, vehicle_id=VEHICLE, session_id=SESSION,
                              draft_id=draft.draft_id, draft_version=1, confirmed=confirmed, urgency=urgency)


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


def test_mechanic_priority_is_sent_and_preserved_on_replay(setup):
    upstreams, pipeline = setup
    pipeline.settings.ticket_urgency_by_fault = {'F08': 'low'}
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    assert draft.payload.urgency == 'low' and 'F08' in draft.urgency_reason
    with pytest.raises(ValidationError):
        confirmation(draft, urgency='critical')
    decision = confirmation(draft, urgency='high')
    receipt = pipeline.confirm_ticket(decision)
    stored, _, saved_receipt = pipeline.tickets.load(decision)
    assert stored.payload.urgency == 'high' and saved_receipt == receipt
    assert pipeline.confirm_ticket(decision) == receipt
    with pytest.raises(AgentError) as rejected:
        pipeline.confirm_ticket(confirmation(draft, urgency='medium'))
    assert rejected.value.code == 'DUPLICATE_TICKET'
    assert upstreams.calls('/tickets') == [{**draft.payload.model_dump(), 'urgency': 'high'}]


def test_confirm_rechecks_the_measurement(setup):
    upstreams, pipeline = setup
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    upstreams.result = diagnostic('check_failed', signal='bad')
    with pytest.raises(AgentError) as error:
        pipeline.confirm_ticket(confirmation(draft))
    assert error.value.code == 'REMEASUREMENT_REQUIRED'
    assert not upstreams.calls('/tickets')


@pytest.mark.parametrize('old_draft', [False, True])
def test_confirm_checks_engine_after_restart(setup, old_draft):
    upstreams, pipeline = setup
    response = pipeline.run(utterance(command='draft_ticket'))
    draft = response.ticket_draft
    if old_draft:
        draft = draft.model_copy(update={'draft_id': 'legacy', 'engine_model': None})
        pipeline.tickets.save('mechanic', draft, response.diagnostic)
    else:
        data_path = pipeline.settings.vehicles_path
        data = json.loads(data_path.read_text())
        data['vehicles'][0]['engine_model'] = 'ЯМЗ-536'
        data_path.write_text(json.dumps(data), encoding='utf-8')
    restarted = Pipeline(pipeline.settings, pipeline.services.http)
    with pytest.raises(AgentError) as error:
        restarted.confirm_ticket(confirmation(draft))
    assert error.value.code == 'DRAFT_STALE'
    assert not upstreams.calls('/tickets')


@pytest.mark.parametrize('bad_url', ['127.0.0.1:8002', 'http://localhost:bad', 'https://toir.test/?q=bad'])
def test_invalid_url_rejected_before_ticket_state_changes(setup, bad_url):
    upstreams, pipeline = setup
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    with pytest.raises(ValidationError, match='maintenance_api_url'):
        Settings(_env_file=None, maintenance_api_url=bad_url)
    assert not upstreams.calls('/tickets')
    assert pipeline.confirm_ticket(confirmation(draft)).ticket_id == 'TKT-0001'


def test_diagnosis_works_with_unavailable_ticket_storage(setup, tmp_path):
    upstreams, original = setup
    unavailable = tmp_path / 'not-a-directory'
    unavailable.touch()
    settings = original.settings.model_copy(update={'artifacts_dir': unavailable})
    pipeline = Pipeline(settings, original.services.http, retriever=FixtureRetriever(),
                        generator=FixtureGenerator(), verifier=FixtureVerifier())
    assert pipeline.run(utterance(command='diagnose')).diagnostic.status == 'anomaly'
    with pytest.raises(AgentError) as error:
        pipeline.run(utterance(command='draft_ticket'))
    assert error.value.code == 'TICKET_STORAGE_UNAVAILABLE'
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
    restarted = Pipeline(pipeline.settings, pipeline.services.http, retriever=FixtureRetriever(),
                         generator=FixtureGenerator(), verifier=FixtureVerifier())
    another_draft = restarted.run(utterance(command='draft_ticket')).ticket_draft
    with pytest.raises(AgentError) as error:
        restarted.confirm_ticket(confirmation(another_draft))
    assert error.value.code == 'DUPLICATE_TICKET'
    assert len(upstreams.calls('/tickets')) == 1
