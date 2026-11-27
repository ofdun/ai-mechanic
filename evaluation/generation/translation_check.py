"""Live check of answer translation on real answers from an Oracle Context trace.

For each answer and language it records how many lines passed the checks (numbers kept, not Russian) and
how many show the Russian original as a safety-critical line, and writes the translated answers for review
by a native speaker.

    uv run python -m evaluation.generation.translation_check --traces oracle-a-traces.jsonl --limit 20 \
        --languages uz,tg,ky,kk --allow-external-evidence --output translation.json --answers translation.jsonl
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from ml.config import Settings
from ml.contracts import AgentResponse, TechnicalClaim
from ml.generation.translate import Localizer, assemble, rejection, source_items
from ml.providers.openrouter import OpenRouterClient


def answer_from_trace(row: dict) -> AgentResponse | None:
    trace = row.get('trace') or {}
    final = trace.get('final') or []
    if not final:
        return None
    return AgentResponse(
        request_id=row['id'], vehicle_id='trace', session_id='trace', engine_model='', action='search_manuals',
        message=trace.get('summary', '').splitlines()[0] if trace.get('summary') else 'Ответ по руководству.',
        summary='\n'.join(trace.get('summary', '').splitlines()[1:]),
        claims=[TechnicalClaim(kind=claim['kind'], text=claim['text'], evidence_ids=['trace']) for claim in final],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Проверка перевода ответов на языки СНГ.')
    parser.add_argument('--traces', type=Path, required=True, help='--cases-output из oracle_context')
    parser.add_argument('--languages', default='uz,tg,ky,kk')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--model')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--allow-external-evidence', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--answers', type=Path, help='Переводы для проверки носителем; содержат текст руководств.')
    args = parser.parse_args()
    if not args.allow_external_evidence:
        parser.error('Передача ответов во внешний LLM требует --allow-external-evidence.')
    settings = Settings(allow_external_evidence=True)
    rows = [json.loads(line) for line in args.traces.read_text(encoding='utf-8').splitlines() if line.strip()]
    answers = [answer for answer in map(answer_from_trace, rows) if answer is not None][:args.limit]
    languages = args.languages.split(',')
    with httpx.Client(timeout=settings.http_timeout_seconds) as http:
        localizer = Localizer(OpenRouterClient(http, settings), args.model or settings.a_generator_model)
        jobs = [(answer, language) for answer in answers for language in languages]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = list(pool.map(lambda job: (job, localizer.translate(source_items(job[0]), job[1])), jobs))

    stats: dict[str, dict] = defaultdict(lambda: {'lines': 0, 'translated': 0, 'with_original': 0, 'answers': 0,
                                                  'rejected': defaultdict(int)})
    with (args.answers.open('w', encoding='utf-8') if args.answers else open(os.devnull, 'w')) as stream:
        for (answer, language), raw in results:
            localized = assemble(answer, language, raw)
            lines = [localized.message, *localized.summary, *localized.claims]
            entry = stats[language]
            entry['answers'] += 1
            entry['lines'] += len(lines)
            entry['translated'] += sum(line.translated for line in lines)
            entry['with_original'] += sum(bool(line.original) for line in lines)
            items = source_items(answer)
            rejected = {key: {'source': items[key], 'raw': raw.get(key, ''),
                              'reason': rejection(' '.join(items[key].split()), ' '.join(raw.get(key, '').split()),
                                                  language)}
                        for key in items}
            rejected = {key: value for key, value in rejected.items() if value['reason']}
            for value in rejected.values():
                entry['rejected'][value['reason']] += 1
            stream.write(json.dumps({'id': answer.request_id, 'language': language,
                                     'source': [answer.message, *answer.summary.splitlines(),
                                                *[claim.text for claim in answer.claims]],
                                     'localized': [line.model_dump() for line in lines],
                                     'rejected': rejected}, ensure_ascii=False) + '\n')

    summary = {language: {**entry, 'translated_share': round(entry['translated'] / entry['lines'], 4)}
               for language, entry in stats.items() if entry['lines']}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
