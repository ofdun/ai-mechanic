"""CLI: retrieval metrics on a golden set and a run JSON for the experiment log.

    uv run python -m evaluation.retrieval --golden evaluation/datasets/retrieval.jsonl --split dev \
        --output evaluation/results/ret-01.json --run-file evaluation/results/ret-01-run.json \
        --run-id ret-01-bm25-page-<commit> --commit <commit> --dataset "golden v1, dev"
"""

import argparse
import json
import sys
from pathlib import Path

from evaluation.retrieval.metrics import evaluate, load_golden, run_record
from ml.config import Settings
from ml.contracts import AgentError
from ml.retrieval.engine import RetrievalConfig, RetrievalEngine


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Hit@K, Recall@K, MRR и nDCG retrieval на golden set')
    parser.add_argument('--config', type=Path, help='YAML с параметрами, по умолчанию configs/retrieval.yaml')
    parser.add_argument('--index', type=Path, default=Settings().artifacts_dir / 'indexes' / 'manuals')
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--split')
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

    try:
        engine = RetrievalEngine.load(args.index, RetrievalConfig.load(args.config))
        report = evaluate(engine, load_golden(args.golden, args.split), split=args.split)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report.model_dump_json(indent=2), encoding='utf-8')
        if args.run_file:
            record = run_record(report, run_id=args.run_id, commit=args.commit, dataset=args.dataset, k=args.k,
                                hypothesis=args.hypothesis, issue_url=args.issue_url)
            args.run_file.parent.mkdir(parents=True, exist_ok=True)
            args.run_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    except AgentError as error:
        print(json.dumps({'error': {'code': error.code, 'message': str(error)}}, ensure_ascii=False))
        return 1
    print(json.dumps(report.model_dump(mode='json', exclude={'results'}), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
