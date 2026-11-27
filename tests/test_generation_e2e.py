import json

import httpx
import pytest

from evaluation.generation import e2e_trace as e2e
from evaluation.generation.oracle_context import Case, Usage
from ml.config import Settings
from ml.contracts import Evidence, EvidenceBundle
from ml.providers.openrouter import OpenRouterClient

ENGINE = 'ЯМЗ-236М2'


class FakeCorpus:
    def __init__(self, pages):
        self.pages = {(code, number): type('Page', (), {'document_id': doc, 'pdf_page': number, 'text': text})()
                      for code, doc, number, text in pages}


class FakeEngine:
    def __init__(self, items):
        self.items = items

    def search(self, plan):
        return EvidenceBundle(items=self.items)

    def rank_pages(self, plan, limit):
        return [type('Hit', (), {'chunk': item})() for item in self.items][:limit]


def page(doc, number, text):
    return Evidence(evidence_id=f'{doc}:{number}', document_id=doc, manual='РЭ', engine_models=[ENGINE],
                    pdf_page=number, text=text)


@pytest.fixture
def client():
    replies = {
        'ClaimsOutput': {'claims': [{'kind': 'reference_value', 'text': 'Зазор клапанов 0,25-0,30 мм.',
                                     'evidence_ids': ['re:31'], 'quote': 'Зазор клапанов 0,25-0,30 мм.',
                                     'conflict_group': ''}]},
        'ApprovalOutput': {'indexes': [0]},
        'PlainOutput': {'lead': 'Зазор клапанов 0,25-0,30 мм.', 'lines': [], 'assessment': ''},
        'JudgeOutput': {'verdict': 'partial', 'reason': 'Нет условий.'},
    }

    def llm(request):
        name = json.loads(request.content)['response_format']['json_schema']['name']
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(replies[name])}}]})

    settings = Settings(_env_file=None, llm_api_key='t', allow_external_evidence=True,
                        llm_base_url='https://example.test/api/v1')
    with httpx.Client(transport=httpx.MockTransport(llm)) as http:
        yield settings, OpenRouterClient(http, settings)


def case(references, answer='Зазор 0,25-0,30 мм на холодном двигателе.'):
    return Case('Q1', 'Какой зазор клапанов?', ENGINE, 'answer', answer, 'dev', references)


def test_trace_names_the_stage_that_lost_the_answer(client):
    settings, llm = client
    text = 'Зазор клапанов 0,25-0,30 мм. Регулировать на холодном двигателе.'
    corpus = FakeCorpus([('y236re', 're', 31, text), ('y236re', 're', 40, 'Другое.')])
    found = e2e.run_case(case([{'document': 'y236re', 'pdf_pages': [31]}]), corpus, FakeEngine([page('re', 31, text)]),
                         settings, llm, Usage(), pipeline='a', model=None, judge_model='judge')
    assert found.trace['stage'] == 'completeness'
    assert found.trace['retrieval']['expected_in_context'] == [['re', 31]]
    assert [step['stage'] for step in found.trace['steps']] == ['a_generator', 'a_verifier_finalizer', 'plain_answer']
    assert found.trace['verifier'][0][0]['rejected'] == []

    lost = e2e.run_case(case([{'document': 'y236re', 'pdf_pages': [31]}]), corpus,
                        FakeEngine([page('re', 40, 'Другое.')]), settings, llm, Usage(), pipeline='a', model=None,
                        judge_model='judge')
    assert lost.trace['stage'] == 'retrieval'
    assert e2e.breakdown([found, lost])['stages'] == {'retrieval': 1, 'completeness': 1}
