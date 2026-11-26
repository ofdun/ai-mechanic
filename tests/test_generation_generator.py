import json
from pathlib import Path

import httpx
import pytest

from ml.config import Settings
from ml.contracts import AgentRequest, Evidence, EvidenceBundle, SearchPlan
from ml.generation import ClaimGenerator, EvidenceVerifier, GroundedClaim, TechnicalReasoner
from ml.pipeline import Pipeline
from ml.providers.openrouter import OpenRouterClient


ENGINE = 'ЯМЗ-236М2'
PAGE_TEXT = (
    'Нарушение регулировки форсунки    Отрегулировать форсунку и, если\n'
    'или засорение распылителя         необходимо, промыть распылитель\n'
    'Давление начала впрыскивания форсунки 22-23 МПа.\n'
)
PAGE = Evidence(evidence_id='p52', document_id='re', manual='Fixture РЭ', engine_models=[ENGINE], pdf_page=52,
                text=PAGE_TEXT)
PLAN = SearchPlan(engine_model=ENGINE, original_query='Что проверить?', rewritten_query='Форсунки', anomaly_type='F08')
OPERATION = {
    'kind': 'operation', 'text': 'Отрегулировать форсунку и, если необходимо, промыть распылитель.',
    'evidence_ids': ['p52'], 'quote': 'Отрегулировать форсунку и, если необходимо, промыть распылитель',
    'conflict_group': '',
}
VALUE = {
    'kind': 'reference_value', 'text': 'Давление начала впрыскивания форсунки — 22–23 МПа.', 'evidence_ids': ['p52'],
    'quote': 'Давление начала впрыскивания форсунки 22-23 МПа.', 'conflict_group': '',
}
INVENTED = {**VALUE, 'text': 'Давление начала впрыскивания форсунки — 25 МПа.'}


def llm_client(replies, requests):
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(next(replies))}}]})

    settings = Settings(_env_file=None, llm_api_key='test', llm_base_url='https://llm.test/api/v1')
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenRouterClient(http, settings), http


def test_generator_requests_quotes_with_a_strict_schema():
    requests = []
    client, http = llm_client(iter([{'claims': [OPERATION, {**VALUE, 'text': '  '}, {**VALUE, 'evidence_ids': []}]}]),
                              requests)
    with http:
        claims = ClaimGenerator(client, 'deepseek/deepseek-v4.1-flash').draft(PLAN, EvidenceBundle(items=[PAGE]), None)
    assert [claim.text for claim in claims] == [OPERATION['text']] and isinstance(claims[0], GroundedClaim)
    request, = requests
    schema = request['response_format']['json_schema']
    assert schema['strict'] is True
    item = schema['schema']['$defs']['ClaimOutput']
    assert set(item['required']) == {'kind', 'text', 'evidence_ids', 'quote', 'conflict_group'}
    assert item['additionalProperties'] is False
    payload = json.loads(request['messages'][1]['content'])
    assert payload['engine_model'] == ENGINE and payload['question'] == 'Что проверить?'
    assert payload['sources'][0]['text'] == PAGE.text and 'document_id' not in payload['sources'][0]
    assert request['reasoning'] == {'enabled': False}
    assert 'дословная выдержка' in request['messages'][0]['content']


def test_generator_skips_the_call_without_sources():
    requests = []
    client, http = llm_client(iter([]), requests)
    with http:
        assert ClaimGenerator(client, 'model').draft(PLAN, EvidenceBundle(), None) == []
    assert requests == []


def test_reasoner_uses_the_reasoning_prompt():
    requests = []
    client, http = llm_client(iter([{'claims': [VALUE]}]), requests)
    with http:
        claims = TechnicalReasoner(client, 'z-ai/glm-5.3').draft(PLAN, EvidenceBundle(items=[PAGE]), None)
    assert claims[0].quote == VALUE['quote']
    assert requests[0]['reasoning'] == {'effort': 'low'} and requests[0]['max_tokens'] == TechnicalReasoner.max_tokens
    assert 'признак → возможная причина' in requests[0]['messages'][0]['content']


@pytest.fixture
def pipeline(tmp_path: Path):
    vehicles = tmp_path / 'vehicles.json'
    record = {'vehicle_id': 'M333YK47', 'session_id': 'S-0005', 'status': 'anomaly', 'anomaly_type': 'F08',
              'anomaly_title': 'Дизельная топливоподача', 'anomaly_score': 0.9, 'confidence': 0.8,
              'signal_quality': 'good', 'description': 'Стук', 'model_version': 'test'}
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M333YK47', 'engine_model': ENGINE, 'fuel': 'дизель',
                                                  'sessions': [{**record, 'checked_at': '2026-11-19T10:00:00+03:00'}]}]}),
                        encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'artifacts',
                        llm_api_key='test', llm_base_url='https://llm.test/api/v1', planner_mode='offline')
    llm_replies = iter([{'claims': [OPERATION, VALUE, INVENTED]}])

    def upstream(request):
        if request.url.path == '/diagnose':
            return httpx.Response(200, json=record)
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(next(llm_replies))}}]})

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=[PAGE])

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        generator = ClaimGenerator(OpenRouterClient(http, settings), settings.a_generator_model)
        yield Pipeline(settings, http, retriever=Retriever(), generator=generator, verifier=EvidenceVerifier(ENGINE))


def test_pipeline_drafts_ticket_from_verified_claims(pipeline):
    response = pipeline.run(AgentRequest(actor_id='mechanic', vehicle_id='M333YK47', session_id='S-0005',
                                         command='draft_ticket'))
    assert [claim.text for claim in response.claims] == [OPERATION['text'], VALUE['text']]
    payload = response.ticket_draft.payload
    assert payload.operations == [OPERATION['text']] and payload.urgency == 'high'
    assert payload.conclusion.startswith('Предполагаемая неисправность: Дизельная топливоподача (F08)')
    assert payload.references[0].pdf_page == 52
    assert response.message.startswith('⚠️ Отклонение: Дизельная топливоподача (F08)')
    assert '25 МПа' not in response.message and '22–23 МПа' in response.message
    assert 'Черновик заявки' in response.message
