import argparse
import json
import logging
from pathlib import Path

import httpx
from pydantic import ValidationError

from ml.config import Settings
from ml.contracts import AgentError, AgentRequest, AudioInput, MAX_AUDIO_BYTES, TicketConfirmation
from ml.pdf_baseline import build_pdf_pipeline
from ml.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description='Локальный запуск Digital Mechanic без веб-интерфейса')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--pdf-index', type=Path)
    commands = parser.add_subparsers(dest='operation', required=True)
    run = commands.add_parser('run')
    confirm = commands.add_parser('confirm')
    for command in (run, confirm):
        command.add_argument('--actor', default='local-mechanic')
        command.add_argument('--vehicle', required=True)
        command.add_argument('--session', required=True)

    run.add_argument('--command', choices=('auto', 'diagnose', 'draft_ticket'), default='auto')
    inputs = run.add_mutually_exclusive_group()
    inputs.add_argument('--message')
    inputs.add_argument('--audio', type=Path)
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
            pipeline = build_pdf_pipeline(settings, http, args.pdf_index) if args.pdf_index else Pipeline(settings, http)
            if args.operation == 'run':
                audio = None
                if args.audio:
                    with args.audio.open('rb') as file:
                        audio = AudioInput(content=file.read(MAX_AUDIO_BYTES + 1))
                result = pipeline.run(AgentRequest(**context, command=args.command, message=args.message, audio=audio))
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
