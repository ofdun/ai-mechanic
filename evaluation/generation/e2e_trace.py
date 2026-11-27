"""End-to-end trace: planner → team retrieval → workflow A/B → judge, and where each answer goes wrong.

Oracle Context hands the generator the reference pages, so it cannot show query or retrieval errors. This run
uses the planner and the retrieval engine as the stand does (by default the web budget: 3 pages, 1100 words),
logs every stage for every question and names the first stage that lost the answer:

    planner      the question was routed away from the manuals
    query        the reference page is found by the question with its dialogue context, not by the planner query
    retrieval    no query finds the reference page within the context budget
    pdf_text     the page was found, but numbers of the reference answer are missing from its extracted text
    generation   the page was in context, but no drafted claim cites it
    verification every drafted claim citing the page was rejected by EvidenceVerifier
    review       verified claims citing the page were dropped by the review or final selection
    completeness the answer cites the right page but misses or misstates details (or the judge disagrees)
    abstention   the reference expects a refusal and the agent answered

    uv run python -m evaluation.generation.e2e_trace --pages DATA/pages.jsonl --pipeline a --split dev \
        --allow-external-evidence --judge-model z-ai/glm-5.3-flash --output e2e.json --traces e2e-traces.jsonl

Traces quote manual text: keep them local or share them privately, never in the public repository.
"""

import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from evaluation.generation.oracle_context import (
    DATASET, Case, Corpus, Pipeline, Usage, answer_case, git_commit, reference_numbers, summarize,
)
from ml.config import Settings
from ml.contracts import Evidence, SearchPlan
from ml.generation.text import numbers
from ml.planner.local import LocalPlanner
from ml.providers.openrouter import OpenRouterClient
from ml.retrieval.engine import RetrievalConfig, RetrievalEngine, build_index

STAGES = ('planner', 'query', 'retrieval', 'pdf_text', 'generation', 'verification', 'review', 'completeness',
          'abstention')
DEPTH = 30


def page_of(item: Evidence) -> tuple[str, int]:
    return item.document_id, item.pdf_page


def expected_pages(case: Case, corpus: Corpus) -> set[tuple[str, int]]:
    found = set()
    for reference in case.references:
        for number in reference['pdf_pages']:
            page = corpus.pages.get((reference['document'], number))
            if page is not None:
                found.add((page.document_id, page.pdf_page))
    return found


_TEXTS: dict[int, dict] = {}


def page_texts(corpus: Corpus) -> dict[tuple[str, int], str]:
    if id(corpus) not in _TEXTS:
        _TEXTS[id(corpus)] = {(page.document_id, page.pdf_page): page.text for page in corpus.pages.values()}
    return _TEXTS[id(corpus)]


def rank_of(engine: RetrievalEngine, plan: SearchPlan, expected: set) -> int | None:
    """1-based position of the first reference page among ranked pages, before the context budget."""
    for position, hit in enumerate(engine.rank_pages(plan, DEPTH), 1):
        if (hit.chunk.document_id, hit.chunk.pdf_page) in expected:
            return position
    return None


def attribute(case: Case, result, expected: set, retrieved: set, with_context: set, routed: bool,
              texts: dict, pages: dict) -> str:
    """The first stage that lost the reference answer; ``ok`` for a correct answer."""
    if result.verdict == 'correct':
        return 'ok'
    if not routed:
        return 'planner'
    if case.outcome == 'abstain':
        return 'abstention'
    if not expected:
        return 'completeness'
    if not expected & retrieved:
        return 'query' if expected & with_context else 'retrieval'
    missing = reference_numbers(case.answer) - numbers(' '.join(texts[key] for key in expected & retrieved))
    if missing:
        # Numbers that sit on a reference page left out of the context are a retrieval loss, not a text one.
        elsewhere = numbers(' '.join(pages[key] for key in expected - retrieved if key in pages))
        return 'retrieval' if missing & elsewhere else 'pdf_text'
    checks = [check for stage in result.trace.get('verifier', []) for check in stage]
    citing = [check for check in checks if {tuple(page) for page in check['pages']} & expected]
    if not citing:
        return 'generation'
    if all(check['rejected'] for check in citing):
        return 'verification'
    if not any({tuple(page) for page in claim['pages']} & expected for claim in result.claims):
        return 'review'
    return 'completeness'


def run_case(case: Case, corpus: Corpus, engine: RetrievalEngine, settings: Settings, client: OpenRouterClient,
             usage: Usage, *, pipeline: Pipeline, model: str | None, judge_model: str | None):
    usage.start(case.id)
    decision = LocalPlanner().plan(case.question, case.engine_model, None, recent_messages=case.prior_facts)
    routed = decision.action == 'search_manuals'
    plan = SearchPlan(engine_model=case.engine_model, original_query=case.question,
                      rewritten_query=decision.rewritten_query or case.question, subqueries=decision.subqueries)
    # The same question with its dialogue context, as a planner that reads history would ask it.
    contextual = plan.model_copy(update={'rewritten_query': ' '.join([case.question, *case.prior_facts]),
                                         'subqueries': []})
    bundle = engine.search(plan)
    expected = expected_pages(case, corpus)
    retrieved = {page_of(item) for item in bundle.items}
    with_context = {page_of(item) for item in engine.search(contextual).items}
    result = answer_case(case, plan, bundle, expected, settings, client, usage, pipeline=pipeline, model=model,
                         judge_model=judge_model, page_key=page_of)
    texts = {page_of(item): item.text for item in bundle.items}
    result.trace['planner'] = {'action': decision.action, 'rewritten_query': plan.rewritten_query,
                               'subqueries': list(plan.subqueries)}
    result.trace['retrieval'] = {
        'context_pages': len(bundle.items), 'expected_in_context': sorted(map(list, expected & retrieved)),
        'expected_missing': sorted(map(list, expected - retrieved)),
        'rank_planner_query': rank_of(engine, plan, expected), 'rank_with_context': rank_of(engine, contextual, expected),
    }
    pages = page_texts(corpus)
    result.trace['stage'] = attribute(case, result, expected, retrieved, with_context, routed, texts, pages)
    return result


def breakdown(results) -> dict:
    stages = Counter(result.trace.get('stage', 'error') for result in results)
    examples: dict[str, list[str]] = {}
    for result in results:
        stage = result.trace.get('stage', 'error')
        if stage != 'ok' and len(examples.setdefault(stage, [])) < 5:
            examples[stage].append(result.id)
    hits = [result for result in results if result.trace.get('retrieval', {}).get('expected_in_context')]
    with_reference = [result for result in results if result.expected_pages]
    return {
        'stages': {stage: stages.get(stage, 0) for stage in ('ok', *STAGES, 'error') if stages.get(stage)},
        'examples': examples,
        'retrieval_hit_rate': round(len(hits) / len(with_reference), 4) if with_reference else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Сквозной прогон с поиском и разбором ошибок по этапам.')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--pages', type=Path, required=True, help='pages.jsonl из ml/ingestion/export_pages.py')
    parser.add_argument('--manifest', type=Path, help='manifest.json; по умолчанию рядом с pages.jsonl')
    parser.add_argument('--index', type=Path, help='Готовый индекс; без него индекс собирается во временный каталог.')
    parser.add_argument('--retrieval-config', type=Path, help='YAML для RetrievalConfig, по умолчанию configs/retrieval.yaml')
    parser.add_argument('--top-k', type=int, default=3, help='Страниц в контексте; 3 — как на веб-стенде.')
    parser.add_argument('--max-words', type=int, default=1100, help='Бюджет слов контекста; 1100 — как на стенде.')
    parser.add_argument('--pipeline', choices=('generator', 'reasoner', 'a', 'b'), default='a')
    parser.add_argument('--model')
    parser.add_argument('--judge-model')
    parser.add_argument('--split', choices=('dev', 'holdout', 'all'), default='dev')
    parser.add_argument('--ids')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--allow-external-evidence', action='store_true')
    parser.add_argument('--output', type=Path, help='Сводка без текста руководств.')
    parser.add_argument('--traces', type=Path, help='Трассы по кейсам с цитатами руководств, только локально.')
    args = parser.parse_args()
    if not args.allow_external_evidence:
        parser.error('Передача страниц руководств во внешний LLM требует --allow-external-evidence.')
    settings = Settings(allow_external_evidence=True)
    if not settings.llm_api_key.get_secret_value():
        parser.error('Укажите LLM_API_KEY в окружении или .env.')

    manifest = args.manifest or args.pages.with_name('manifest.json')
    corpus = Corpus(args.pages, manifest)
    config = RetrievalConfig.load(args.retrieval_config).model_copy(
        update={'top_k': args.top_k, 'max_evidence_words': args.max_words})
    with tempfile.TemporaryDirectory() as scratch:
        index = args.index or Path(scratch) / 'index'
        if args.index is None:
            build_index(args.pages, index, config, manifest=manifest)
        engine = RetrievalEngine.load(index, config)
        cases = Case.load(args.dataset, split=args.split, ids=set(args.ids.split(',')) if args.ids else None)
        cases = cases[:args.limit] if args.limit else cases
        usage = Usage()
        with httpx.Client(event_hooks={'request': [usage.on_request], 'response': [usage.on_response]}) as http:
            client = OpenRouterClient(http, settings)
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                results = list(pool.map(lambda case: run_case(
                    case, corpus, engine, settings, client, usage, pipeline=args.pipeline, model=args.model,
                    judge_model=args.judge_model), cases))

    summary = {**summarize(results), 'breakdown': breakdown(results), 'commit': git_commit(),
               'retrieval': {'mode': config.mode, 'top_k': config.top_k, 'max_evidence_words': config.max_evidence_words},
               'pipeline': args.pipeline, 'judge_model': args.judge_model, 'split': args.split}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    if args.traces:
        with args.traces.open('w', encoding='utf-8') as stream:
            for result in results:
                stream.write(json.dumps({'id': result.id, 'verdict': result.verdict, 'error': result.error,
                                         **result.trace}, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
