"""Page-level retrieval metrics on a golden set: Hit@K, Recall@K, MRR, nDCG@K and latency."""

import math
import statistics
import time
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ml.contracts import AgentError, PageNumber, SearchPlan, Text
from ml.retrieval.engine import RetrievalConfig, RetrievalEngine


class PageRef(BaseModel):
    document_id: Text
    pdf_page: PageNumber


class GoldenQuestion(BaseModel):
    """One JSONL line of the golden set. An empty `relevant` list marks a question without an answer in the corpus."""

    id: Text
    engine_model: Text
    question: Text
    relevant: list[PageRef] = Field(default_factory=list)
    split: str = 'dev'
    tags: list[str] = Field(default_factory=list)


class QuestionResult(BaseModel):
    id: str
    engine_model: str
    retrieved: list[PageRef]
    first_relevant_rank: int | None
    relevant_found: int
    relevant_total: int
    latency_ms: float
    top_score: float | None


class Report(BaseModel):
    corpus_version: str
    config: RetrievalConfig
    split: str | None
    questions: int
    answerable: int
    unanswerable: int
    ks: list[int]
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    latency_p50_ms: float
    latency_p95_ms: float
    results: list[QuestionResult]


def load_golden(path: Path, split: str | None = None) -> list[GoldenQuestion]:
    questions = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise AgentError('RAG_GOLDEN_INVALID', f'Не удалось прочитать {path}.') from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            question = GoldenQuestion.model_validate_json(line)
        except ValidationError as error:
            raise AgentError('RAG_GOLDEN_INVALID', f'{path.name}:{number} не соответствует формату вопроса.') from error
        if split is None or question.split == split:
            questions.append(question)
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise AgentError('RAG_GOLDEN_INVALID', f'{path.name}: id вопросов повторяются.')
    return questions


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)] if ordered else 0.0


def dcg(gains: list[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def evaluate(
    engine: RetrievalEngine, questions: list[GoldenQuestion], ks: tuple[int, ...] = (1, 3, 5, 10),
    *, split: str | None = None,
) -> Report:
    depth = max(ks)
    results = []
    for question in questions:
        plan = SearchPlan(engine_model=question.engine_model, original_query=question.question,
                          rewritten_query=question.question)
        started = time.perf_counter()
        hits = engine.rank_pages(plan, depth)
        latency = (time.perf_counter() - started) * 1000
        retrieved = [PageRef(document_id=hit.chunk.document_id, pdf_page=hit.chunk.pdf_page) for hit in hits]
        relevant = {(ref.document_id, ref.pdf_page) for ref in question.relevant}
        keys = [(ref.document_id, ref.pdf_page) for ref in retrieved]
        flags = [key in relevant for key in keys]
        results.append(QuestionResult(
            id=question.id, engine_model=question.engine_model, retrieved=retrieved,
            first_relevant_rank=flags.index(True) + 1 if any(flags) else None,
            relevant_found=len(set(keys) & relevant),
            relevant_total=len(relevant), latency_ms=round(latency, 3),
            top_score=round(hits[0].score, 6) if hits else None,
        ))

    answerable = [(question, result) for question, result in zip(questions, results) if question.relevant]

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    def recall(question: GoldenQuestion, result: QuestionResult, k: int) -> float:
        relevant = {(ref.document_id, ref.pdf_page) for ref in question.relevant}
        found = {(ref.document_id, ref.pdf_page) for ref in result.retrieved[:k]} & relevant
        return len(found) / len(relevant)

    def ndcg(question: GoldenQuestion, result: QuestionResult, k: int) -> float:
        relevant = {(ref.document_id, ref.pdf_page) for ref in question.relevant}
        gains = [int((ref.document_id, ref.pdf_page) in relevant) for ref in result.retrieved[:k]]
        return dcg(gains) / dcg([1] * min(k, len(relevant)))

    latencies = [result.latency_ms for result in results]
    return Report(
        corpus_version=engine.manifest.corpus_version,
        config=engine.config.model_copy(update={
            'chunker': engine.manifest.chunker, 'analyzer': engine.manifest.analyzer,
            'chunk_words': engine.manifest.chunk_words, 'chunk_overlap': engine.manifest.chunk_overlap,
        }),
        split=split, questions=len(questions), answerable=len(answerable),
        unanswerable=len(questions) - len(answerable), ks=list(ks),
        hit_at_k={k: mean([float(bool(r.first_relevant_rank and r.first_relevant_rank <= k)) for _, r in answerable]) for k in ks},
        recall_at_k={k: mean([recall(q, r, k) for q, r in answerable]) for k in ks},
        mrr=mean([1 / r.first_relevant_rank if r.first_relevant_rank else 0.0 for _, r in answerable]),
        ndcg_at_k={k: mean([ndcg(q, r, k) for q, r in answerable]) for k in ks},
        latency_p50_ms=round(nearest_rank(latencies, 50), 3), latency_p95_ms=round(nearest_rank(latencies, 95), 3),
        results=results,
    )


def run_record(
    report: Report, *, run_id: str, commit: str, dataset: str, k: int, hypothesis: str, issue_url: str | None = None,
) -> dict:
    """Experiment-log JSON for `context.py experiment-log --file` or `finish --run-file`."""
    if k not in report.ks:
        raise AgentError('RAG_CONFIG_INVALID', f'k={k} не входит в измеренные {report.ks}.')
    config = report.config
    hits = ', '.join(f'Hit@{key} {value:.3f}' for key, value in report.hit_at_k.items())
    ndcg = ', '.join(f'nDCG@{key} {value:.3f}' for key, value in report.ndcg_at_k.items())
    record = {
        'run_id': run_id,
        'status': 'completed',
        'hypothesis': hypothesis,
        'configuration': f'{config.describe()}, фильтр модели до top-k, корпус {report.corpus_version}',
        'models': ', '.join(filter(None, (
            config.embedder if config.mode != 'sparse' else None, config.reranker,
        ))) or 'none',
        'dataset': dataset,
        'commit': commit,
        'k': k,
        'recall_at_k': report.recall_at_k[k],
        'mrr': report.mrr,
        'latency_p50_ms': report.latency_p50_ms,
        'latency_p95_ms': report.latency_p95_ms,
        'result': f'{report.answerable} вопросов с ответом: {hits}; {ndcg}; MRR {report.mrr:.3f}.',
        'notes': (
            f'Вопросов без ответа в корпусе: {report.unanswerable}, они не входят в метрики ранжирования. '
            'Релевантность считается по физическим страницам, фрагменты одной страницы схлопываются. '
            'Задержка только поиска в процессе (с эмбеддингом запроса), без загрузки индекса и LLM.'
        ),
    }
    if report.split:
        record['split'] = report.split
    if issue_url:
        record['issue_url'] = issue_url
    return record
