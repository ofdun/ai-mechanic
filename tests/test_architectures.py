import json
from types import SimpleNamespace

import httpx
import pytest

from ml.architectures import LLMEvidenceWorkflow
from ml.config import Settings
from ml.contracts import AgentError, Evidence, EvidenceBundle, SearchPlan, TechnicalDraft
from ml.providers.openrouter import OpenRouterClient


PLAN = SearchPlan(engine_model='ЯМЗ-236М2', original_query='Форсунки', rewritten_query='Форсунки')
PAGE = Evidence(evidence_id='p52', document_id='guide', manual='РЭ', engine_models=[PLAN.engine_model],
                pdf_page=52, text='Проверить регулировку форсунки. Условное значение 110 Н·м.')
EXTRA = PAGE.model_copy(update={'evidence_id': 'p53', 'pdf_page': 53, 'text': 'Проверить герметичность соединения.'})
FOREIGN = PAGE.model_copy(update={'evidence_id': 'wrong-engine', 'engine_models': ['ЯМЗ-536']})


def claim(text, reference='p52', kind='check'):
    return {'text': text, 'kind': kind, 'evidence_ids': [reference], 'quote': text, 'conflict_group': ''}


def configured_workflow(architecture, replies, *, corrective_page=False):
    settings = Settings(_env_file=None, llm_api_key='test', agent_architecture=architecture,
                        llm_base_url='https://example.test/api/v1', allow_external_evidence=True)
    requests, queries = [], []

    def llm(request):
        payload = json.loads(request.content)
        requests.append(payload)
        response_format = payload['response_format']
        assert response_format['type'] == 'json_schema'
        assert response_format['json_schema']['strict'] is True
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(next(replies))}}]})

    def search(plan):
        assert plan.engine_model == PLAN.engine_model
        queries.append(plan.rewritten_query)
        return EvidenceBundle(items=[PAGE, EXTRA, FOREIGN] if corrective_page and len(queries) > 1 else [PAGE, FOREIGN])

    reviewed = SimpleNamespace(generate=lambda *args: TechnicalDraft())
    http = httpx.Client(transport=httpx.MockTransport(llm))
    return LLMEvidenceWorkflow(settings, OpenRouterClient(http, settings), SimpleNamespace(search=search), reviewed), http, requests, queries


def test_a_uses_planned_models_and_filters_unsupported_claims():
    good = claim('Проверить регулировку форсунки.')
    replies = iter([{'claims': [good, claim('10 Н·м.'), claim('110 бар.'), claim('Чужой двигатель.', 'wrong-engine'),
                                claim('Самостоятельная операция.', kind='operation')]}, {'indexes': [0, 1, 2]}])
    workflow, http, requests, _ = configured_workflow('a', replies)
    with http:
        result = workflow.run(PLAN, None)
    assert [r['model'] for r in requests] == ['deepseek/deepseek-v4.1-flash', 'z-ai/glm-5.3-flash']
    assert requests[0]['reasoning'] == {'enabled': False} and 'reasoning' not in requests[1]
    assert [item.evidence_id for item in result.evidence.items] == ['p52']
    assert [item.model_dump() for item in result.claims] == [good]
    verifier_input = json.loads(requests[-1]['messages'][-1]['content'])
    assert verifier_input['claims'] == [good]


def test_b_performs_one_corrective_lookup_and_finalizer_can_only_select_claims():
    original = claim('Проверить регулировку форсунки.')
    extra = claim('Проверить герметичность соединения.', 'p53')
    replies = iter([
        {'rewritten_query': 'Проверка форсунок', 'subqueries': []},
        {'claims': [original]},
        {'indexes': [0], 'corrective_query': 'Герметичность соединения'},
        {'claims': [original, extra]},
        {'indexes': [1, 0, 99, -1, 1]},
    ])
    workflow, http, requests, queries = configured_workflow('b', replies, corrective_page=True)
    with http:
        result = workflow.run(PLAN, None)
    assert queries == ['Проверка форсунок', 'Герметичность соединения']
    assert [r['model'] for r in requests] == [
        'z-ai/glm-5.3-flash', 'z-ai/glm-5.3', 'deepseek/deepseek-v4.1-flash',
        'deepseek/deepseek-v4.1-flash', 'z-ai/glm-5.3-flash',
    ]
    assert requests[1]['reasoning'] == {'effort': 'low'}
    assert [c.model_dump() for c in result.claims] == [extra, original]
    assert [item.evidence_id for item in result.evidence.items] == ['p52', 'p53']


def test_b_reconsiders_earlier_instruction_when_corrective_lookup_finds_conflict():
    manual_a = PAGE.model_copy(update={'text': 'Начать проверку раньше 10 минут.'})
    manual_b = EXTRA.model_copy(update={
        'document_id': 'other-guide', 'manual': 'РК', 'text': 'Начать проверку не раньше 10 минут.',
    })
    first = claim('Начать проверку раньше 10 минут.')
    # A disagreement is two claims, each with its own quote and a shared conflict group.
    earlier = {**first, 'conflict_group': 'c1'}
    later = {**claim('Начать проверку не раньше 10 минут.', 'p53'), 'conflict_group': 'c1'}
    replies = iter([
        {'rewritten_query': 'Условия проверки', 'subqueries': []},
        {'claims': [first]},
        {'indexes': [0], 'corrective_query': 'Условия в ремонтном руководстве'},
        {'claims': [earlier, later]},
        {'indexes': [0, 1]},
    ])
    workflow, http, requests, _ = configured_workflow('b', replies)
    results = iter([EvidenceBundle(items=[manual_a]), EvidenceBundle(items=[manual_b])])
    workflow.retriever = SimpleNamespace(search=lambda plan: next(results))
    with http:
        result = workflow.run(PLAN, None)

    repair_input = json.loads(requests[-2]['messages'][-1]['content'])
    assert [source['evidence_id'] for source in repair_input['sources']] == ['p52', 'p53']
    assert [item.model_dump() for item in result.claims] == [earlier, later]


def test_architectures_require_permission_to_send_evidence():
    workflow, http, requests, queries = configured_workflow('b', iter([]))
    workflow.settings.allow_external_evidence = False
    with http, pytest.raises(AgentError) as error:
        workflow.run(PLAN, None)
    assert error.value.code == 'LLM_EVIDENCE_NOT_ALLOWED'
    assert requests == queries == []
