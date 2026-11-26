import argparse
import json
import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from ml.config import Settings
from ml.contracts import AgentError, AgentRequest, TicketConfirmation
from ml.pdf_baseline import build_pdf_pipeline, build_retrieval_pipeline
from ml.pipeline import Pipeline
from ml.retrieval import RetrievalConfig, RetrievalEngine


def main() -> int:
    parser = argparse.ArgumentParser(description='Локальный запуск Digital Mechanic без веб-интерфейса')
    parser.add_argument('--verbose', action='store_true')
    indexes = parser.add_mutually_exclusive_group()
    indexes.add_argument('--pdf-index', type=Path)
    indexes.add_argument('--retrieval-index', type=Path, help='Индекс из python -m ml.retrieval build')
    commands = parser.add_subparsers(dest='operation', required=True)
    run = commands.add_parser('run')
    confirm = commands.add_parser('confirm')
    for command in (run, confirm):
        command.add_argument('--actor', default='local-mechanic')
        command.add_argument('--vehicle', required=True)
        command.add_argument('--session', required=True)

    run.add_argument('--command', choices=('auto', 'diagnose', 'draft_ticket'), default='auto')
    run.add_argument('--message')
    confirm.add_argument('--draft-id', required=True)
    confirm.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger('httpx').setLevel(logging.WARNING)
    try:
        settings = Settings()
        context = {'actor_id': args.actor, 'vehicle_id': args.vehicle, 'session_id': args.session}
        with httpx.Client(timeout=settings.http_timeout_seconds) as http:
            if args.pdf_index:
                pipeline = build_pdf_pipeline(settings, http, args.pdf_index)
            elif args.retrieval_index:
                pipeline = build_retrieval_pipeline(
                    settings, http, RetrievalEngine.load(args.retrieval_index, RetrievalConfig.load()))
            else:
                pipeline = Pipeline(settings, http)
            if args.operation == 'run':
                result = pipeline.run(AgentRequest(**context, command=args.command, message=args.message))
            else:
                result = pipeline.confirm_ticket(TicketConfirmation(
                    **context, draft_id=args.draft_id, draft_version=1, confirmed=args.confirm,
                ))
        print(result.model_dump_json(indent=2))
        return 0
    except AgentError as error:
        print(json.dumps({'error': {'code': error.code, 'message': str(error), 'retryable': error.retryable}}, ensure_ascii=False))
        return 1
    except (OSError, ValidationError) as error:
        if isinstance(error, ValidationError):
            errors = error.errors(include_input=False, include_context=False, include_url=False)
            message = '; '.join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in errors)
        else:
            message = 'Не удалось прочитать локальный файл или подготовить артефакты.'
        print(json.dumps({'error': {'code': 'INVALID_INPUT', 'message': message}}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
