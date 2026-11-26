"""Planner rewrites for the retrieval benchmark: one LLM call per question, saved for every run.

    LLM_API_KEY=... uv run python -m evaluation.retrieval.rewrite \
        --golden evaluation/datasets/customer_questions.jsonl --split dev --output evaluation/results/rewrites-dev.jsonl

The Planner sees the question, the engine model and `prior_facts` as recent messages, exactly as in the
agent. A question for which it picks another tool gets no rewrite, and the runner then uses the question.
Only the question text goes to the provider, not the manuals.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from evaluation.retrieval.customer import is_search_case, read_jsonl
from ml.config import Settings
from ml.contracts import AgentError
from ml.planner import LLMPlanner
from ml.providers.openrouter import OpenRouterClient


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Переформулировки Planner для замера retrieval')
    parser.add_argument('--golden', type=Path, default=Path('evaluation/datasets/customer_questions.jsonl'))
    parser.add_argument('--split')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    settings = Settings()
    if not settings.llm_api_key.get_secret_value():
        parser.error('Нужен LLM_API_KEY.')
    rows = [row for row in read_jsonl(args.golden, 'RAG_GOLDEN_INVALID')
            if is_search_case(row) and (args.split is None or row.get('split') == args.split)]
    done = {}
    if args.output.exists():
        done = {row['id']: row for row in read_jsonl(args.output, 'RAG_GOLDEN_INVALID')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {'rewritten': 0, 'other_tool': 0, 'error': 0, 'cached': 0}
    with httpx.Client(timeout=settings.http_timeout_seconds) as http, args.output.open('a', encoding='utf-8') as out:
        planner = LLMPlanner(OpenRouterClient(http, settings))
        for row in rows:
            if row['id'] in done:
                counts['cached'] += 1
                continue
            context = row['context']
            try:
                decision = planner.plan(row['text'], context['engine_model'], None,
                                        recent_messages=context.get('prior_facts', []))
            except AgentError as error:
                counts['error'] += 1
                print(f'{row["id"]}: {error.code}', file=sys.stderr)
                continue
            record = {'id': row['id'], 'action': decision.action, 'model': settings.llm_model}
            if decision.action == 'search_manuals' and decision.rewritten_query:
                record.update(rewritten_query=decision.rewritten_query, subqueries=decision.subqueries[:5])
                counts['rewritten'] += 1
            else:
                counts['other_tool'] += 1
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
    print(json.dumps({'output': str(args.output), **counts}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
