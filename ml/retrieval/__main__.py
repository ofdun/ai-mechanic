"""CLI: build a retrieval index, add dense vectors and search it.

    uv run python -m ml.retrieval build                               # DATA_DIR/pages.jsonl + manifest.json
    uv run --extra dense python -m ml.retrieval embed                 # vectors for config.embedder
    uv run python -m ml.retrieval search --engine ЯМЗ-536 --query "момент затяжки болтов головки"
"""

import argparse
import json
import sys
import time
from pathlib import Path

from ml.config import Settings
from ml.contracts import AgentError, SearchPlan
from ml.retrieval.engine import RetrievalConfig, RetrievalEngine, build_index, embed_index, load_embedder


def uncovered_models(index_models: list[str]) -> list[str]:
    """Vehicle engine models that have no manual in the index. Empty when vehicles.json is unavailable."""
    from ml.tools.mocks import VehicleDirectory

    try:
        vehicles = VehicleDirectory.load(Settings()).vehicles.values()
    except AgentError:
        return []
    return sorted({vehicle.engine_model for vehicle in vehicles} - set(index_models))


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    settings = Settings()
    default_index = settings.artifacts_dir / 'indexes' / 'manuals'
    parser = argparse.ArgumentParser(description='Retrieval по руководствам: индекс, векторы и поиск')
    parser.add_argument('--config', type=Path, help='YAML с параметрами, по умолчанию configs/retrieval.yaml')
    commands = parser.add_subparsers(dest='command', required=True)

    build = commands.add_parser('build', help='Собрать индекс из выгрузки ingestion')
    build.add_argument('--source', '--parsed-dir', type=Path, default=settings.data_dir / 'pages.jsonl',
                       help='pages.jsonl из ml/ingestion/export_pages.py или каталог *.json ParsedDocument')
    build.add_argument('--manifest', type=Path, help='Каталог руководств, по умолчанию manifest.json рядом с pages.jsonl')
    build.add_argument('--output', type=Path, default=default_index)
    build.add_argument('--embed', action='store_true', help='Сразу посчитать векторы config.embedder')

    embed = commands.add_parser('embed', help='Посчитать dense-векторы для готового индекса')
    embed.add_argument('--index', type=Path, default=default_index)
    embed.add_argument('--model', help='Переопределить config.embedder')

    search = commands.add_parser('search', help='Найти фрагменты для модели двигателя')
    search.add_argument('--index', type=Path, default=default_index)
    search.add_argument('--engine', required=True)
    search.add_argument('--query', required=True)
    search.add_argument('--subquery', action='append', default=[])
    search.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    try:
        config = RetrievalConfig.load(args.config)
        if args.command == 'build':
            manifest = build_index(args.source, args.output, config, manifest=args.manifest)
            summary = manifest.model_dump(mode='json')
            summary['index'] = str(args.output)
            if missing := uncovered_models(manifest.engine_models):
                summary['warning'] = f'Модели из vehicles.json без руководств в индексе: {", ".join(missing)}'
            if args.embed:
                summary['dense'] = {config.embedder: embed_index(args.output, load_embedder(config), config)}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == 'embed':
            if args.model:
                config = config.model_copy(update={'embedder': args.model})
            started = time.perf_counter()
            rows = embed_index(args.index, load_embedder(config), config)
            print(json.dumps({'model': config.embedder, 'chunks': rows,
                              'seconds': round(time.perf_counter() - started, 1)}, ensure_ascii=False))
        else:
            engine = RetrievalEngine.load(args.index, config)
            plan = SearchPlan(engine_model=args.engine, original_query=args.query, rewritten_query=args.query,
                              subqueries=args.subquery)
            for hit in engine.rank(plan, args.top_k):
                chunk = hit.chunk
                printed = f', печатная {chunk.printed_page}' if chunk.printed_page else ''
                print(f'{hit.score:.4f}  {chunk.chunk_id} · {chunk.manual} · PDF {chunk.pdf_page}{printed}'
                      f' · {chunk.section or "без раздела"}')
                print('    ' + ' '.join(chunk.text.split())[:240])
    except AgentError as error:
        print(json.dumps({'error': {'code': error.code, 'message': str(error)}}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
