"""Oracle Context evaluation of the generation stage (GEN-01 … GEN-07).

The generator receives the reference pages of each benchmark question (``expected_evidence``), mixed with a
few distractor pages of the same manuals, so the result measures generation and verification rather than
retrieval. Metrics follow ``docs/shared-project-context.md``:

* ``answer_correctness`` — LLM judge against the reference answer (correct 1, partial 0.5), only with
  ``--judge-model``;
* ``numeric_accuracy`` — share of reference numbers (without page and figure numbers) present in the answer;
* ``citation_accuracy`` — share of answer claims citing a reference page rather than a distractor;
* ``unsupported_claim_rate`` — share of drafted claims rejected by ``EvidenceVerifier``; rejected claims
  never reach the mechanic;
* ``abstention_accuracy`` — no answer exactly where the reference expects abstention.

Manual text is sent to the LLM provider only with ``--allow-external-evidence``. The summary and the run
record contain counts, never manual text; ``--cases-output`` writes per-case details for local analysis.

    uv run python -m evaluation.generation.oracle_context --pages DATA/pages.jsonl --manifest DATA/manifest.json \\
        --pipeline a --split dev --allow-external-evidence --judge-model z-ai/glm-5.3 \\
        --output oracle-a.json --run-file run-a.json
"""
import argparse
import json
import random
import re
import statistics
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from ml.architectures import LLMEvidenceWorkflow
from ml.config import Settings
from ml.contracts import AgentError, AgentResponse, Evidence, EvidenceBundle, SearchPlan, TechnicalDraft
from ml.generation.claims import GroundedClaim
from ml.generation.finalizer import MechanicFinalizer, find_conflicts
from ml.generation.generator import ClaimGenerator
from ml.generation.reasoner import TechnicalReasoner
from ml.generation.text import claim_numbers
from ml.generation.verifier import EvidenceVerifier
from ml.providers.openrouter import OpenRouterClient


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'evaluation' / 'datasets' / 'customer_questions.jsonl'
# Source codes of the benchmark (evaluation/datasets/README.md) → PDF files of the customer's kit.
DOCUMENT_FILES = {
    'y236re': 'ЯМЗ_236_руководство по эксплуатации.pdf',
    'y236rk': 'ЯМЗ_236_руководство по РЕМОНТУ.pdf',
    'y536re': 'yamz-536_re_2016.pdf',
    'y536rk': 'yamz-536_rk_2013.pdf',
    'z5234': 'zmz-5234_rk_2025.pdf',
    'z409': 'zmz-40904-40905_rk_2025.pdf',
}
ANSWERED = {'answer', 'partially_supported', 'conflicting_sources'}
REFERENCE_SCOPES = {'passage', 'module'}
MAX_PAGES_PER_REFERENCE = 6
_REFERENCE_NUMBERS = re.compile(
    r'(?:физ(?:ическ[а-я]*)?\.?\s*)?(?:стр(?:\.|аниц[а-я]*)|с\.)\s*(?:PDF\s*)?\d+(?:\s*(?:[-–,]|и)\s*\d+)*'
    r'|печатн[а-я]*\s*\d+|\bPDF\s*\d+'
    r'|(?:рис(?:\.|ун[а-я]*)|табл(?:\.|иц[а-я]*)|прил(?:\.|ожени[а-я]*))\s*[А-ЯA-Z]?\.?\d+(?:\.\d+)*',
    re.IGNORECASE,
)


def nfc(value: str) -> str:
    return unicodedata.normalize('NFC', value)


def reference_numbers(answer: str) -> set[str]:
    """Numbers a correct answer must state: no page, figure, table or appendix numbers."""
    return claim_numbers(_REFERENCE_NUMBERS.sub(' ', answer))


# Data -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Page:
    code: str
    document_id: str
    engine_models: tuple[str, ...]
    pdf_page: int
    section: str | None
    text: str


class Corpus:
    """Pages exported by ``ml/ingestion/export_pages.py`` (``pages.jsonl`` plus its ``manifest.json``)."""

    def __init__(self, pages_path: Path, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))['documents']
        by_file = {nfc(item['file']): item['document_id'] for item in manifest}
        missing = [name for name in DOCUMENT_FILES.values() if nfc(name) not in by_file]
        if missing:
            raise ValueError(f'В manifest нет файлов: {missing}')
        code_of = {by_file[nfc(name)]: code for code, name in DOCUMENT_FILES.items()}
        self.pages: dict[tuple[str, int], Page] = {}
        for line in pages_path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            code = code_of.get(record['document_id'])
            if code and record['text'].strip():
                self.pages[code, record['pdf_page']] = Page(
                    code, record['document_id'], tuple(record['models']), record['pdf_page'], record.get('section'),
                    record['text'],
                )
        self.by_code: dict[str, list[Page]] = {}
        for page in self.pages.values():
            self.by_code.setdefault(page.code, []).append(page)


@dataclass
class Case:
    id: str
    question: str
    engine_model: str
    outcome: str
    answer: str
    split: str
    references: list[dict]
    prior_facts: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path, *, split: str, ids: set[str] | None = None) -> list['Case']:
        cases = []
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('suite') != 'rag' or 'expected_answer' not in row:
                continue
            if split != 'all' and row['split'] != split or ids and row['id'] not in ids:
                continue
            cases.append(cls(row['id'], row['text'], row['context']['engine_model'], row['answer_outcome'],
                             row['expected_answer'], row['split'], row.get('expected_evidence') or [],
                             tuple(row['context'].get('prior_facts') or ())))
        return cases


def oracle_bundle(case: Case, corpus: Corpus, distractors: int, seed: int) -> tuple[EvidenceBundle, set, set]:
    """Reference pages plus distractors of the same manuals, in a shuffled but reproducible order."""
    rng = random.Random(f'{seed}:{case.id}')
    chosen, expected, manuals = [], set(), {}
    for reference in case.references:
        manuals[reference['document']] = reference['manual']
        for number in reference['pdf_pages'][:MAX_PAGES_PER_REFERENCE]:
            page = corpus.pages.get((reference['document'], number))
            if page is None or page in chosen:
                continue
            chosen.append(page)
            if reference['scope'] in REFERENCE_SCOPES:
                expected.add((reference['document'], number))
    taken = {(page.code, page.pdf_page) for page in chosen}
    pool = [page for code in manuals for page in corpus.by_code.get(code, [])
            if (page.code, page.pdf_page) not in taken and len(page.text) > 300]
    noise = rng.sample(pool, min(distractors, len(pool)))
    pages = chosen + noise
    rng.shuffle(pages)
    items = [Evidence(evidence_id=f'{page.code}-p{page.pdf_page}', document_id=page.document_id,
                      manual=manuals[page.code], engine_models=list(page.engine_models), pdf_page=page.pdf_page,
                      section=page.section, text=page.text) for page in pages]
    return EvidenceBundle(items=items), expected, {(page.code, page.pdf_page) for page in noise}


# LLM usage ------------------------------------------------------------------------------------


class Usage:
    """Token, cost and latency accounting from OpenRouter responses, per case."""

    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.calls: dict[str, list[dict]] = {}

    def start(self, case_id: str):
        self.local.case = case_id
        with self.lock:
            self.calls.setdefault(case_id, [])

    def on_request(self, request: httpx.Request):
        request.extensions['started'] = time.monotonic()

    def on_response(self, response: httpx.Response):
        response.read()
        try:
            data = response.json()
        except ValueError:
            data = {}
        usage = data.get('usage') if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        record = {
            'model': data.get('model') if isinstance(data, dict) else None,
            'input_tokens': usage.get('prompt_tokens') or 0, 'output_tokens': usage.get('completion_tokens') or 0,
            'cost': usage.get('cost') or 0.0,
            'latency_ms': (time.monotonic() - response.request.extensions.get('started', time.monotonic())) * 1000,
        }
        with self.lock:
            self.calls.setdefault(getattr(self.local, 'case', '?'), []).append(record)


# Running ----------------------------------------------------------------------------------------

Pipeline = Literal['generator', 'reasoner', 'a', 'b']


class OracleRetriever:
    def __init__(self, bundle: EvidenceBundle):
        self.bundle = bundle

    def search(self, plan: SearchPlan) -> EvidenceBundle:
        return self.bundle


class NoReviewedClaims:
    def generate(self, *args) -> TechnicalDraft:
        return TechnicalDraft()


class TracingClient:
    """Delegates to the OpenRouter client and keeps every structured reply of one case, in call order."""

    def __init__(self, client: OpenRouterClient):
        self.client = client
        self.steps: list[dict] = []

    def complete_json(self, role, model, messages, response_type, **options):
        result = self.client.complete_json(role, model, messages, response_type, **options)
        self.steps.append({'stage': role, 'model': model, 'output': result.model_dump()})
        return result

    def __getattr__(self, name):
        return getattr(self.client, name)


def claim_record(claim, pages: dict[str, tuple[str, int]]) -> dict:
    return {'kind': claim.kind, 'text': claim.text, 'quote': getattr(claim, 'quote', ''),
            'conflict_group': getattr(claim, 'conflict_group', ''),
            'pages': sorted({pages[key] for key in claim.evidence_ids if key in pages})}


class RecordingWorkflow(LLMEvidenceWorkflow):
    """Team workflow A/B that also keeps the first draft and every verifier decision, for metrics and traces."""

    drafts: list | None = None

    def generate(self, role, model, plan, evidence, diagnostic, *, note=''):
        claims = super().generate(role, model, plan, evidence, diagnostic, note=note)
        if self.drafts is None:
            self.drafts = list(claims)
        return claims

    def valid_claims(self, claims, plan, evidence, diagnostic, *, include_reviewed=False):
        report = EvidenceVerifier(plan.engine_model, plan.anomaly_type).inspect(TechnicalDraft(claims=claims), evidence)
        pages = {item.evidence_id: (item.document_id, item.pdf_page) for item in evidence.items}
        if not hasattr(self, 'verifications'):
            self.verifications = []
        self.verifications.append([{**claim_record(check.claim, pages), 'rejected': list(check.reasons)}
                                   for check in report.checks])
        return super().valid_claims(claims, plan, evidence, diagnostic, include_reviewed=include_reviewed)


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    verdict: Literal['correct', 'partial', 'incorrect', 'abstained']
    reason: str


JUDGE = (
    'Ты оцениваешь ответ помощника механика по эталону. Сравни смысл, числа, единицы, необходимый порядок действий '
    'и условия применимости. Дословное совпадение не требуется. correct — ответ передаёт всё существенное из эталона '
    'без ошибок; partial — верная часть без ошибок, но существенное пропущено; incorrect — есть ошибка в фактах, '
    'числах или условиях; abstained — ответ сообщает, что данных нет. Если эталон сам ожидает отказ, а ответ '
    'отказывается, выбери correct. В reason одно предложение.'
)


@dataclass
class CaseResult:
    id: str
    outcome: str
    drafted: int = 0
    rejected_drafts: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    claims: list[dict] = field(default_factory=list)
    expected_pages: int = 0
    cited_expected: int = 0
    cited_reference_claims: int = 0
    reference_numbers: int = 0
    found_numbers: int = 0
    conflict_shown: bool = False
    verdict: str | None = None
    verdict_reason: str | None = None
    error: str | None = None
    seconds: float = 0.0
    cost: float = 0.0
    calls: int = 0
    answer: str = ''
    trace: dict = field(default_factory=dict)


def run_case(case: Case, corpus: Corpus, settings: Settings, client: OpenRouterClient, usage: Usage, *,
             pipeline: Pipeline, model: str | None, distractors: int, seed: int,
             judge_model: str | None) -> CaseResult:
    usage.start(case.id)
    bundle, expected, _ = oracle_bundle(case, corpus, distractors, seed)
    # Prior facts stand in for the dialogue history that the planner folds into its query.
    query = ' '.join([case.question, *case.prior_facts]) if case.prior_facts else case.question
    plan = SearchPlan(engine_model=case.engine_model, original_query=case.question, rewritten_query=query)
    return answer_case(case, plan, bundle, expected, settings, client, usage, pipeline=pipeline, model=model,
                       judge_model=judge_model, page_key=oracle_page)


def oracle_page(item: Evidence) -> tuple[str, int]:
    return item.evidence_id.rsplit('-p', 1)[0], item.pdf_page


def answer_case(case: Case, plan: SearchPlan, bundle: EvidenceBundle, expected: set, settings: Settings,
                client: OpenRouterClient, usage: Usage, *, pipeline: Pipeline, model: str | None,
                judge_model: str | None, page_key=oracle_page) -> CaseResult:
    """Generation, verification and judging of one case on a given evidence bundle, with a full trace."""
    result = CaseResult(case.id, case.outcome)
    result.expected_pages = len(expected)
    tracer = TracingClient(client)
    workflow = None
    started = time.monotonic()
    summary = ''
    try:
        if pipeline in {'generator', 'reasoner'}:
            drafter = TechnicalReasoner if pipeline == 'reasoner' else ClaimGenerator
            default = settings.b_reasoner_model if pipeline == 'reasoner' else settings.a_generator_model
            drafts = drafter(tracer, model or default).draft(plan, bundle, None)
            final = EvidenceVerifier(case.engine_model).verify(TechnicalDraft(claims=drafts), bundle).claims
        else:
            workflow_settings = settings.model_copy(update={'agent_architecture': pipeline})
            workflow = RecordingWorkflow(workflow_settings, tracer, OracleRetriever(bundle), NoReviewedClaims())
            outcome = workflow.run(plan, None)
            final, summary = outcome.claims, outcome.summary
            drafts = workflow.drafts or []
    except AgentError as error:
        result.error = f'{error.code} ({error})'
        result.seconds = time.monotonic() - started
        result.trace = {'steps': tracer.steps}
        return finish_usage(result, usage)
    result.seconds = time.monotonic() - started

    report = EvidenceVerifier(case.engine_model).inspect(TechnicalDraft(claims=drafts), bundle)
    result.drafted = len(drafts)
    result.rejected_drafts = len(report.rejected)
    for check in report.rejected:
        for reason in check.reasons:
            result.rejected[reason] = result.rejected.get(reason, 0) + 1

    sources = {item.evidence_id: item for item in bundle.items}
    page_of = {item.evidence_id: page_key(item) for item in bundle.items}
    cited = set()
    for claim in final:
        pages = {page_of[key] for key in claim.evidence_ids if key in page_of}
        cited |= pages
        result.cited_reference_claims += bool(pages & expected)
        result.claims.append({'kind': claim.kind, 'text': claim.text, 'pages': sorted(pages),
                              'quote': claim.quote if isinstance(claim, GroundedClaim) else ''})
    result.cited_expected = len(cited & expected)
    stated = set().union(*(claim_numbers(claim.text) for claim in final)) if final else set()
    wanted = reference_numbers(case.answer)
    result.reference_numbers, result.found_numbers = len(wanted), len(wanted & stated)
    result.conflict_shown = bool(find_conflicts(final, sources))

    response = AgentResponse(request_id=case.id, vehicle_id='oracle', session_id='oracle', engine_model=case.engine_model,
                             action='search_manuals', message=summary, summary=summary, search_plan=plan,
                             evidence=bundle, claims=final)
    result.answer = MechanicFinalizer().compose(response)
    result.trace = {
        'query': {'question': case.question, 'prior_facts': list(case.prior_facts),
                  'rewritten_query': plan.rewritten_query, 'subqueries': list(plan.subqueries)},
        'evidence': [{'page': list(page_key(item)), 'manual': item.manual, 'expected': page_key(item) in expected}
                     for item in bundle.items],
        'steps': tracer.steps,
        'verifier': getattr(workflow, 'verifications', []) if workflow else [
            [{**claim_record(check.claim, page_of), 'rejected': list(check.reasons)} for check in report.checks]],
        'summary': summary,
        'final': result.claims,
        'answer': result.answer,
        'reference': {'outcome': case.outcome, 'answer': case.answer, 'pages': sorted(map(list, expected))},
    }
    generation_calls = len(usage.calls.get(case.id, []))
    if judge_model:
        try:
            verdict = client.complete_json('oracle_judge', judge_model, [
                {'role': 'system', 'content': JUDGE},
                {'role': 'user', 'content': json.dumps({
                    'question': case.question, 'reference_answer': case.answer,
                    'reference_outcome': case.outcome, 'answer': result.answer,
                }, ensure_ascii=False)},
            ], JudgeOutput, max_tokens=1500)
            result.verdict, result.verdict_reason = verdict.verdict, verdict.reason
        except AgentError as error:
            result.error = f'judge:{error.code} ({error})'
    result.trace['judge'] = {'model': judge_model, 'verdict': result.verdict, 'reason': result.verdict_reason}
    return finish_usage(result, usage, calls=generation_calls)


def finish_usage(result: CaseResult, usage: Usage, *, calls: int | None = None) -> CaseResult:
    """Cost and number of calls of the answer itself; judge calls are not part of the product cycle."""
    calls = usage.calls.get(result.id, [])[:calls]
    result.calls = len(calls)
    result.cost = sum(call['cost'] for call in calls)
    return result


# Metrics ----------------------------------------------------------------------------------------


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered) + 0.5) - 1))]


def summarize(results: list[CaseResult]) -> dict:
    """Metrics over completed cases; a failed judge call keeps the case but leaves it unjudged."""
    done = [result for result in results if result.error is None or result.error.startswith('judge:')]
    claims = sum(len(result.claims) for result in done)
    drafted = sum(result.drafted for result in done)
    reasons: dict[str, int] = {}
    for result in done:
        for reason, count in result.rejected.items():
            reasons[reason] = reasons.get(reason, 0) + count
    judged = [result for result in done if result.verdict is not None]
    score = {'correct': 1.0, 'partial': 0.5}
    conflicts = [result for result in done if result.outcome == 'conflicting_sources']
    with_pages = [result for result in done if result.expected_pages]
    milliseconds = [result.seconds * 1000 for result in done]
    by_outcome: dict[str, dict] = {}
    for result in done:
        bucket = by_outcome.setdefault(result.outcome, {'cases': 0, 'answered': 0})
        bucket['cases'] += 1
        bucket['answered'] += bool(result.claims)
    return {
        'cases': len(results), 'completed': len(done),
        'errors': sorted({result.error for result in results if result.error}),
        'answer_correctness': ratio(sum(score.get(result.verdict, 0.0) for result in judged), len(judged)),
        'verdicts': {verdict: sum(result.verdict == verdict for result in judged)
                     for verdict in ('correct', 'partial', 'incorrect', 'abstained')} if judged else None,
        'numeric_accuracy': ratio(sum(result.found_numbers for result in done),
                                  sum(result.reference_numbers for result in done)),
        'citation_accuracy': ratio(sum(result.cited_reference_claims for result in done), claims),
        'reference_page_coverage': ratio(sum(result.cited_expected for result in with_pages),
                                         sum(result.expected_pages for result in with_pages)),
        'unsupported_claim_rate': ratio(sum(result.rejected_drafts for result in done), drafted),
        'abstention_accuracy': ratio(sum((not result.claims) == (result.outcome == 'abstain') for result in done),
                                     len(done)),
        'conflicts_shown': ratio(sum(result.conflict_shown for result in conflicts), len(conflicts)),
        'drafted_claims': drafted, 'answer_claims': claims, 'rejection_reasons': dict(sorted(reasons.items())),
        'by_outcome': by_outcome,
        'latency_p50_ms': round(percentile(milliseconds, 0.5), 1) if milliseconds else None,
        'latency_p95_ms': round(percentile(milliseconds, 0.95), 1) if milliseconds else None,
        'cost_per_cycle_usd': round(statistics.fmean(result.cost for result in done), 6) if done else None,
        'llm_calls': sum(result.calls for result in done),
    }


METRIC_FIELDS = ('answer_correctness', 'numeric_accuracy', 'citation_accuracy', 'unsupported_claim_rate',
                 'abstention_accuracy')


def run_record(summary: dict, args: argparse.Namespace, settings: Settings, commit: str) -> dict:
    """Experiment record in the project-context format (docs/project-context.md)."""
    models = {
        'generator': args.model or settings.a_generator_model,
        'reasoner': args.model or settings.b_reasoner_model,
        'a': f'{settings.a_generator_model}; {settings.a_verifier_model}',
        'b': '; '.join(dict.fromkeys((settings.b_query_model, settings.b_reasoner_model, settings.b_critic_model,
                                     settings.b_repair_model, settings.b_finalizer_model))),
    }[args.pipeline]
    if args.judge_model:
        models += f'; judge {args.judge_model}'
    record = {
        'run_id': args.run_id or f'oracle-{args.pipeline}-{args.split}-{commit[:7]}',
        'status': 'completed' if summary['completed'] else 'failed',
        'hypothesis': args.hypothesis,
        'configuration': (f'Oracle Context: эталонные страницы expected_evidence (до {MAX_PAGES_PER_REFERENCE} на ссылку) '
                          f'и {args.distractors} отвлекающие страницы тех же руководств, seed {args.seed}. '
                          f'Конвейер {args.pipeline}, EvidenceVerifier, temperature 0.'),
        'models': models,
        'dataset': f'{args.dataset.name}, suite rag, {summary["cases"]} кейсов',
        'split': args.split,
        'commit': commit,
        'result': '; '.join(f'{name} {summary[name]}' for name in METRIC_FIELDS if summary[name] is not None)
                  + f'; завершено {summary["completed"]} из {summary["cases"]}',
        'notes': (f'unsupported_claim_rate — доля черновых утверждений, отклонённых EvidenceVerifier; до механика они '
                  f'не доходят. citation_accuracy — доля утверждений ответа со ссылкой на эталонную, а не отвлекающую '
                  f'страницу; покрытие эталонных страниц {summary["reference_page_coverage"]}. numeric_accuracy — доля '
                  f'чисел эталона без номеров страниц и рисунков. Расхождения показаны в {summary["conflicts_shown"]} '
                  f'кейсов conflicting_sources. Ошибки: {", ".join(summary["errors"]) or "нет"}. '
                  f'Вызовов LLM {summary["llm_calls"]}, стоимость и задержка — только стадии генерации и проверки.'),
    }
    for name in METRIC_FIELDS:
        if summary[name] is not None:
            record[name] = summary[name]
    for name in ('latency_p50_ms', 'latency_p95_ms'):
        if summary[name] is not None:
            record[name] = summary[name]
    if summary['cost_per_cycle_usd'] is not None:
        record['cost_per_cycle'], record['currency'] = summary['cost_per_cycle_usd'], 'USD'
    return record


def git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def dry_run(cases: list[Case], corpus: Corpus, args: argparse.Namespace) -> dict:
    """Check the reference pages without calling an LLM and estimate the prompt size."""
    sizes, missing, without_numbers = [], [], 0
    for case in cases:
        bundle, expected, _ = oracle_bundle(case, corpus, args.distractors, args.seed)
        wanted = {(reference['document'], page) for reference in case.references
                  if reference['scope'] in REFERENCE_SCOPES for page in reference['pdf_pages'][:MAX_PAGES_PER_REFERENCE]}
        if wanted - expected:
            missing.append(case.id)
        without_numbers += not reference_numbers(case.answer)
        sizes.append(sum(len(item.text) for item in bundle.items))
    return {
        'cases': len(cases), 'outcomes': {outcome: sum(case.outcome == outcome for case in cases)
                                          for outcome in sorted({case.outcome for case in cases})},
        'cases_with_reference_pages_missing_text': missing,
        'cases_without_reference_numbers': without_numbers,
        'source_chars_p50': percentile(sizes, 0.5), 'source_chars_max': max(sizes, default=0),
        'source_chars_total': sum(sizes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Oracle Context: оценка генерации на эталонных страницах.')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--pages', type=Path, required=True, help='pages.jsonl из ml/ingestion/export_pages.py')
    parser.add_argument('--manifest', type=Path, help='manifest.json; по умолчанию рядом с pages.jsonl')
    parser.add_argument('--pipeline', choices=('generator', 'reasoner', 'a', 'b'), default='a')
    parser.add_argument('--model', help='Модель для generator или reasoner вместо настроек.')
    parser.add_argument('--judge-model', help='Модель-судья для answer_correctness.')
    parser.add_argument('--split', choices=('dev', 'holdout', 'all'), default='dev')
    parser.add_argument('--ids', help='Список ID через запятую.')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--distractors', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--allow-external-evidence', action='store_true',
                        help='Разрешить передачу страниц руководств провайдеру LLM.')
    parser.add_argument('--dry-run', action='store_true', help='Проверить данные без вызовов LLM.')
    parser.add_argument('--output', type=Path, help='Сводка метрик без текста руководств.')
    parser.add_argument('--run-file', type=Path, help='Запись прогона для project-context finish --run-file.')
    parser.add_argument('--run-id')
    parser.add_argument('--hypothesis', default='GEN-01/GEN-07: качество генерации и проверки на эталонном контексте')
    parser.add_argument('--cases-output', type=Path, help='Подробности по кейсам с текстом руководств, только локально.')
    args = parser.parse_args()

    corpus = Corpus(args.pages, args.manifest or args.pages.with_name('manifest.json'))
    cases = Case.load(args.dataset, split=args.split, ids=set(args.ids.split(',')) if args.ids else None)
    cases = cases[:args.limit] if args.limit else cases
    if args.dry_run:
        print(json.dumps(dry_run(cases, corpus, args), ensure_ascii=False, indent=2))
        return 0
    if not args.allow_external_evidence:
        parser.error('Передача страниц руководств во внешний LLM требует --allow-external-evidence.')
    settings = Settings(allow_external_evidence=True)
    if not settings.llm_api_key.get_secret_value():
        parser.error('Укажите LLM_API_KEY в окружении или .env.')

    usage = Usage()
    with httpx.Client(event_hooks={'request': [usage.on_request], 'response': [usage.on_response]}) as http:
        client = OpenRouterClient(http, settings)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = list(pool.map(lambda case: run_case(
                case, corpus, settings, client, usage, pipeline=args.pipeline, model=args.model,
                distractors=args.distractors, seed=args.seed, judge_model=args.judge_model), cases))

    summary = summarize(results)
    record = run_record(summary, args, settings, git_commit())
    text = json.dumps({'summary': summary, 'run': record}, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    if args.run_file:
        args.run_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.cases_output:
        with args.cases_output.open('w', encoding='utf-8') as stream:
            for result in results:
                stream.write(json.dumps(result.__dict__, ensure_ascii=False) + '\n')
    return 0 if summary['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
