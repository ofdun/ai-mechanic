"""Parameter sweep for retrieval precision on one index: many configurations, one load of index and models.

    uv run --extra dense python -m evaluation.retrieval.sweep --grid configs/retrieval-sweep.yaml \
        --golden evaluation/datasets/customer_questions.jsonl --manifest ../data/manifest.json --split dev \
        --output evaluation/results/sweep-dev.json --misses evaluation/results/sweep-dev-misses.jsonl

Reranker scores are cached on disk by model, query and passage (`--cache`), so a repeated or extended
sweep only scores new pairs. Latency is therefore not reported here: measure it with `evaluation.retrieval`
on the chosen configuration. Tune on dev only and check the winner once on holdout.

Besides Hit@K the sweep reports diagnostics for the misses at rank 1: whether the first page is a neighbour
(±1) of an expected page in the same manual, and whether it comes from an expected manual at all. They tell
a ranking error from a labelling question (a procedure that spans pages, the same norm in РЭ and РК).
"""

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from evaluation.retrieval.customer import is_customer_format, load_customer_questions, load_rewrites
from evaluation.retrieval.metrics import GoldenQuestion, QueryMode, Report, evaluate, load_golden
from ml.config import Settings
from ml.contracts import AgentError
from ml.retrieval.dense import load_dense
from ml.retrieval.engine import RetrievalConfig, RetrievalEngine, load_embedder, load_reranker, read_index
from ml.retrieval.reranker import CachedReranker


class Variant(BaseModel):
    name: str
    query_mode: QueryMode = 'combined'
    overrides: dict = Field(default_factory=dict)


class Grid(BaseModel):
    base: dict = Field(default_factory=dict)
    variants: list[Variant]


class QueryCache:
    """Query embeddings are reused across variants with the same embedder."""

    def __init__(self, inner):
        self.inner, self.name = inner, inner.name
        self.embed_query = lru_cache(maxsize=None)(inner.embed_query)

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)


def diagnostics(questions: list[GoldenQuestion], report: Report) -> dict:
    by_id = {question.id: question for question in questions}
    neighbour = same_manual = misses = 0
    for result in report.results:
        question = by_id[result.id]
        if not question.relevant or result.first_relevant_rank == 1 or not result.retrieved:
            continue
        misses += 1
        top = result.retrieved[0]
        documents = {ref.document_id for ref in question.relevant}
        same_manual += top.document_id in documents
        neighbour += any(ref.document_id == top.document_id and abs(ref.pdf_page - top.pdf_page) == 1
                         for ref in question.relevant)
    answerable = report.answerable or 1
    return {
        'misses_at_1': misses,
        'hit_at_1_with_neighbour': round(report.hit_at_k[1] + neighbour / answerable, 4),
        'miss_top1_neighbour': neighbour,
        'miss_top1_same_manual': same_manual,
    }


def misses(questions: list[GoldenQuestion], report: Report, variant: str) -> list[dict]:
    by_id = {question.id: question for question in questions}
    rows = []
    for result in report.results:
        question = by_id[result.id]
        if question.relevant and result.first_relevant_rank != 1:
            rows.append({
                'variant': variant, 'id': result.id, 'engine_model': result.engine_model,
                'question': question.question, 'rank': result.first_relevant_rank,
                'expected': [f'{ref.document_id}:{ref.pdf_page}' for ref in question.relevant],
                'top3': [f'{ref.document_id}:{ref.pdf_page}' for ref in result.retrieved[:3]],
            })
    return rows


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    settings = Settings()
    parser = argparse.ArgumentParser(description='Подбор параметров retrieval на одном индексе')
    parser.add_argument('--grid', type=Path, default=Path('configs/retrieval-sweep.yaml'))
    parser.add_argument('--index', type=Path, default=settings.artifacts_dir / 'indexes' / 'manuals')
    parser.add_argument('--golden', type=Path, default=Path('evaluation/datasets/customer_questions.jsonl'))
    parser.add_argument('--manifest', type=Path, default=settings.data_dir / 'manifest.json')
    parser.add_argument('--rewrites', type=Path)
    parser.add_argument('--split', default='dev')
    parser.add_argument('--only', help='Имена вариантов через запятую')
    parser.add_argument('--cache', type=Path, default=settings.artifacts_dir / 'cache' / 'rerank')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--misses', type=Path, help='JSONL промахов Hit@1 по каждому варианту')
    args = parser.parse_args()

    try:
        grid = Grid.model_validate(yaml.safe_load(args.grid.read_text(encoding='utf-8')))
        if is_customer_format(args.golden):
            rewrites = load_rewrites(args.rewrites) if args.rewrites else None
            questions = load_customer_questions(args.golden, args.manifest, args.split, rewrites).questions
        else:
            questions = load_golden(args.golden, args.split)
        manifest, chunks, pages = read_index(args.index)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        dense, embedders, rerankers = {}, {}, {}
        variants = grid.variants
        if args.only:
            wanted = set(args.only.split(','))
            variants = [variant for variant in variants if variant.name in wanted]
        results, missed = {}, []
        for variant in variants:
            config = RetrievalConfig.model_validate({**grid.base, **variant.overrides})
            vectors = embedder = reranker = None
            if config.mode != 'sparse':
                if config.embedder not in dense:
                    dense[config.embedder] = load_dense(args.index, config.embedder, chunk_ids)
                    embedders[config.embedder] = QueryCache(load_embedder(config))
                vectors, embedder = dense[config.embedder], embedders[config.embedder]
            if config.reranker:
                key = (config.reranker, config.rerank_max_length)
                if key not in rerankers:
                    slug = f'{config.reranker}-{config.rerank_max_length}'.replace('/', '__').replace(':', '_')
                    rerankers[key] = CachedReranker(load_reranker(config), args.cache / f'{slug}.json')
                reranker = rerankers[key]
            engine = RetrievalEngine(manifest, chunks, pages, config, dense=vectors, embedder=embedder,
                                     reranker=reranker)
            report = evaluate(engine, questions, split=args.split, query_mode=variant.query_mode)
            if reranker is not None:
                reranker.save()
            row = {
                'query_mode': variant.query_mode, 'config': config.describe(),
                'hit_at_k': report.hit_at_k, 'recall_at_k': report.recall_at_k, 'mrr': report.mrr,
                'answerable': report.answerable, 'rewrites': report.rewrites, **diagnostics(questions, report),
            }
            results[variant.name] = row
            missed.extend(misses(questions, report, variant.name))
            hits = ' '.join(f'@{k} {v:.3f}' for k, v in report.hit_at_k.items())
            print(f'{variant.name:<28} Hit {hits}  MRR {report.mrr:.3f}  '
                  f'±1 {row["hit_at_1_with_neighbour"]:.3f}', flush=True)
    except AgentError as error:
        print(json.dumps({'error': {'code': error.code, 'message': str(error)}}, ensure_ascii=False))
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'split': args.split, 'variants': results}, ensure_ascii=False, indent=2),
                               encoding='utf-8')
    if args.misses:
        args.misses.parent.mkdir(parents=True, exist_ok=True)
        args.misses.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in missed), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
