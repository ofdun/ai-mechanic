import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

from evaluation.generation import oracle_context as oracle
from ml.config import Settings
from ml.providers.openrouter import OpenRouterClient


ENGINE = 'ЯМЗ-236М2'
# Synthetic pages; the customer's manuals are not part of the repository.
PAGES = {
    1: 'Тепловой зазор впускных и выпускных клапанов регулировать в пределах 0,25-0,30 мм. ' * 5,
    2: 'Затяжку гаек крепления головки цилиндров выполнять моментом 196-216 Н·м в два приёма. ' * 5,
    3: 'Масло в картере проверять ежедневно по указателю уровня перед пуском двигателя. ' * 5,
}


@pytest.fixture
def corpus(tmp_path: Path) -> oracle.Corpus:
    manifest = tmp_path / 'manifest.json'
    documents = [{'document_id': f'doc-{code}', 'file': name, 'models': [ENGINE]}
                 for code, name in oracle.DOCUMENT_FILES.items()]
    manifest.write_text(json.dumps({'documents': documents}, ensure_ascii=False), encoding='utf-8')
    pages = tmp_path / 'pages.jsonl'
    pages.write_text(''.join(json.dumps({'document_id': 'doc-y236re', 'models': [ENGINE], 'pdf_page': number,
                                         'text': text}, ensure_ascii=False) + '\n'
                             for number, text in PAGES.items()), encoding='utf-8')
    return oracle.Corpus(pages, manifest)


def case(identifier='Q1', outcome='answer', answer='Зазор 0,25–0,30 мм, РЭ, физ. стр. 1.', pages=(1,)):
    return oracle.Case(identifier, 'Какой зазор клапанов?', ENGINE, outcome, answer, 'dev', [
        {'document': 'y236re', 'manual': 'Fixture РЭ', 'pdf_pages': list(pages), 'scope': 'passage'}])


def claim(text, page, quote=None, kind='reference_value'):
    return {'kind': kind, 'text': text, 'evidence_ids': [f'y236re-p{page}'], 'quote': quote or text,
            'conflict_group': ''}


GOOD = claim('Тепловой зазор впускных и выпускных клапанов — 0,25–0,30 мм.', 1,
             'Тепловой зазор впускных и выпускных клапанов регулировать в пределах 0,25-0,30 мм.')
DISTRACTOR = claim('Затяжка гаек крепления головки цилиндров — 196–216 Н·м в два приёма.', 2,
                   'Затяжку гаек крепления головки цилиндров выполнять моментом 196-216 Н·м в два приёма.')
INVENTED = claim('Тепловой зазор клапанов — 0,35 мм.', 1)


def client_for(replies, usage=None):
    def handler(request):
        reply = next(replies)
        return httpx.Response(200, json={'model': 'm', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5,
                                                                 'cost': 0.001},
                                         'choices': [{'message': {'content': json.dumps(reply, ensure_ascii=False)}}]})

    hooks = {'request': [usage.on_request], 'response': [usage.on_response]} if usage else {}
    http = httpx.Client(transport=httpx.MockTransport(handler), event_hooks=hooks)
    settings = Settings(_env_file=None, llm_api_key='test', llm_base_url='https://llm.test/api/v1',
                        allow_external_evidence=True)
    return OpenRouterClient(http, settings), settings, http


def test_oracle_bundle_mixes_reference_pages_with_distractors(corpus):
    bundle, expected, noise = oracle.oracle_bundle(case(), corpus, distractors=2, seed=7)
    assert expected == {('y236re', 1)} and noise == {('y236re', 2), ('y236re', 3)}
    assert sorted(item.evidence_id for item in bundle.items) == ['y236re-p1', 'y236re-p2', 'y236re-p3']
    again, _, _ = oracle.oracle_bundle(case(), corpus, distractors=2, seed=7)
    assert [item.evidence_id for item in again.items] == [item.evidence_id for item in bundle.items]


def test_reference_numbers_skip_pages_and_figures():
    answer = 'Рисунок А.2 на физической странице PDF 272 (печатная 271): моментом 7,84–9,8 Н·м, РК, физ. стр. 269 и 276.'
    assert oracle.reference_numbers(answer) == {'7.84', '9.8'}


def test_generator_case_metrics(corpus):
    usage = oracle.Usage()
    client, settings, http = client_for(iter([{'claims': [GOOD, DISTRACTOR, INVENTED]},
                                              {'verdict': 'partial', 'reason': 'Неполно.'}]), usage)
    with http:
        result = oracle.run_case(case(), corpus, settings, client, usage, pipeline='generator', model='m',
                                 distractors=2, seed=7, judge_model='judge')
    assert result.error is None and result.drafted == 3 and result.rejected_drafts == 1
    assert result.rejected == {'quote_not_found': 1}
    assert [entry['pages'] for entry in result.claims] == [[('y236re', 1)], [('y236re', 2)]]
    assert result.cited_reference_claims == 1 and result.cited_expected == 1
    assert (result.reference_numbers, result.found_numbers) == (2, 2)
    assert result.verdict == 'partial' and result.calls == 1 and result.cost == pytest.approx(0.001)
    assert result.answer.startswith('📘 Ответ по руководствам для ЯМЗ-236М2')

    summary = oracle.summarize([result])
    assert summary['citation_accuracy'] == 0.5 and summary['unsupported_claim_rate'] == 0.3333
    assert summary['numeric_accuracy'] == 1.0 and summary['answer_correctness'] == 0.5
    assert summary['abstention_accuracy'] == 1.0 and summary['cost_per_cycle_usd'] == 0.001


def test_workflow_a_and_abstention(corpus):
    usage = oracle.Usage()
    replies = iter([{'claims': [GOOD, INVENTED]}, {'indexes': [0]}, {'claims': []}])
    client, settings, http = client_for(replies, usage)
    with http:
        answered = oracle.run_case(case(), corpus, settings, client, usage, pipeline='a', model=None,
                                   distractors=1, seed=7, judge_model=None)
        abstained = oracle.run_case(case('Q2', 'abstain', 'Данных нет.'), corpus, settings, client, usage,
                                    pipeline='a', model=None, distractors=1, seed=7, judge_model=None)
    assert answered.drafted == 2 and answered.rejected_drafts == 1 and len(answered.claims) == 1
    assert answered.calls == 2 and abstained.calls == 1 and abstained.claims == []
    summary = oracle.summarize([answered, abstained])
    assert summary['abstention_accuracy'] == 1.0 and summary['answer_correctness'] is None
    assert summary['by_outcome'] == {'answer': {'cases': 1, 'answered': 1}, 'abstain': {'cases': 1, 'answered': 0}}


def test_errors_are_reported_not_scored(corpus):
    usage = oracle.Usage()
    client, settings, http = client_for(iter([{'unexpected': True}]), usage)
    with http:
        failed = oracle.run_case(case(), corpus, settings, client, usage, pipeline='generator', model='m',
                                 distractors=0, seed=7, judge_model=None)
    assert failed.error.startswith('LLM_SCHEMA_ERROR (')
    summary = oracle.summarize([failed])
    assert summary['completed'] == 0 and summary['errors'] == ['LLM_SCHEMA_ERROR (a_generator вернул данные вне ожидаемой схемы.)']


def load_context_script():
    path = Path(__file__).resolve().parents[1] / '.agents' / 'skills' / 'project-context' / 'scripts' / 'context.py'
    if not path.exists():
        pytest.skip('project-context skill is not on this branch')
    spec = importlib.util.spec_from_file_location('project_context', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_record_passes_project_context_validation(corpus):
    usage = oracle.Usage()
    client, settings, http = client_for(iter([{'claims': [GOOD]}]), usage)
    with http:
        result = oracle.run_case(case(), corpus, settings, client, usage, pipeline='generator', model='m',
                                 distractors=1, seed=7, judge_model=None)
    args = Namespace(pipeline='generator', model='m', judge_model=None, run_id=None, split='dev', distractors=1,
                     seed=7, dataset=oracle.DATASET, hypothesis='GEN-01')
    record = oracle.run_record(oracle.summarize([result]), args, settings, 'abcdef1234')
    assert record['run_id'] == 'oracle-generator-dev-abcdef1'
    assert record['citation_accuracy'] == 1.0 and record['currency'] == 'USD'
    assert 'answer_correctness' not in record
    try:
        context = load_context_script()
    except ModuleNotFoundError:
        pytest.skip('project-context dependencies are missing')
    context.validate_record(record)


def test_main_refuses_to_send_manuals_without_permission(corpus, tmp_path, monkeypatch, capsys):
    pages = tmp_path / 'pages.jsonl'
    monkeypatch.setattr(sys, 'argv', ['oracle', '--pages', str(pages), '--limit', '1'])
    with pytest.raises(SystemExit) as stopped:
        oracle.main()
    assert stopped.value.code == 2 and '--allow-external-evidence' in capsys.readouterr().err
