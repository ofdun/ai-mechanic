"""CLI: retrieval metrics on a golden set and a run JSON for the experiment log.

Team benchmark, three query modes on one index (rewrites from `python -m evaluation.retrieval.rewrite`):

    uv run python -m evaluation.retrieval --golden evaluation/datasets/customer_questions.jsonl \
        --manifest DATA/manifest.json --rewrites evaluation/results/rewrites-dev.jsonl \
        --split dev --query-mode all --output evaluation/results/ret-01.json \
        --run-file evaluation/results/ret-01-run.json --run-id ret-01-bm25-page-<commit> --commit <commit> \
        --dataset "customer_questions 8e7a6e6, dev"

The simplified format (`question`, `relevant`) from evaluation/datasets/README.md is read the same way.
"""

import argparse
import json
import sys
from pathlib import Path

from evaluation.retrieval.customer import is_customer_format, load_customer_questions, load_rewrites
from evaluation.retrieval.metrics import QUERY_MODES, evaluate, load_golden, run_record
from ml.config import Settings
from ml.contracts import AgentError
from ml.retrieval.engine import RetrievalConfig, RetrievalEngine


def suffixed(path: Path, mode: str) -> Path:
    return path.with_name(f'{path.stem}-{mode}{path.suffix}')


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    settings = Settings()
    parser = argparse.ArgumentParser(description='Hit@K, Recall@K, MRR и nDCG retrieval на golden set')
    parser.add_argument('--config', type=Path, help='YAML с параметрами, по умолчанию configs/retrieval.yaml')
    parser.add_argument('--index', type=Path, default=settings.artifacts_dir / 'indexes' / 'manuals')
    parser.add_argument('--golden', type=Path, required=True,
                        help='customer_questions.jsonl или упрощённый формат retrieval')
    parser.add_argument('--manifest', type=Path, default=settings.data_dir / 'manifest.json',
                        help='Каталог руководств ingestion для customer_questions.jsonl')
    parser.add_argument('--rewrites', type=Path, help='JSONL переформулировок Planner для customer_questions.jsonl')
    parser.add_argument('--split')
    parser.add_argument('--query-mode', choices=(*QUERY_MODES, 'all'), default='combined',
                        help='original — вопрос, rewritten — переформулировка и подзапросы, combined — всё через RRF')
    parser.add_argument('--output', type=Path, help='Полный отчёт с результатами по вопросам')
    parser.add_argument('--run-file', type=Path, help='JSON прогона для журнала экспериментов')
    parser.add_argument('--run-id')
    parser.add_argument('--commit')
    parser.add_argument('--dataset')
    parser.add_argument('--hypothesis', default='RET-01 Page-level retrieval достаточен для baseline')
    parser.add_argument('--issue-url', default='https://github.com/ofdun/ai-mechanic/issues/4')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    if args.run_file and not (args.run_id and args.commit and args.dataset):
        parser.error('Для --run-file нужны --run-id, --commit и --dataset.')
    modes = QUERY_MODES if args.query_mode == 'all' else (args.query_mode,)

    try:
        skipped = {}
        if is_customer_format(args.golden):
            rewrites = load_rewrites(args.rewrites) if args.rewrites else None
            adapted = load_customer_questions(args.golden, args.manifest, args.split, rewrites)
            questions, skipped = adapted.questions, adapted.skipped
        else:
            questions = load_golden(args.golden, args.split)
        engine = RetrievalEngine.load(args.index, RetrievalConfig.load(args.config))
        reports = {mode: evaluate(engine, questions, split=args.split, query_mode=mode) for mode in modes}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {mode: report.model_dump(mode='json') for mode, report in reports.items()}
            if len(modes) == 1:
                payload = payload[modes[0]]
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.run_file:
            args.run_file.parent.mkdir(parents=True, exist_ok=True)
            for mode, report in reports.items():
                single = len(modes) == 1
                record = run_record(report, run_id=args.run_id if single else f'{args.run_id}-{mode}',
                                    commit=args.commit, dataset=args.dataset, k=args.k,
                                    hypothesis=args.hypothesis, issue_url=args.issue_url)
                target = args.run_file if single else suffixed(args.run_file, mode)
                target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    except AgentError as error:
        print(json.dumps({'error': {'code': error.code, 'message': str(error)}}, ensure_ascii=False))
        return 1
    summary = {mode: report.model_dump(mode='json', exclude={'results', 'config'}) for mode, report in reports.items()}
    if skipped:
        summary['skipped'] = {reason: len(ids) for reason, ids in skipped.items()}
    print(json.dumps(summary[modes[0]] if len(modes) == 1 and not skipped else summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
