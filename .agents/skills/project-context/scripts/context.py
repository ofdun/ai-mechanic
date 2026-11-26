#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


SKILL_DIR = Path(__file__).resolve().parents[1]
COMPONENTS = ('rag', 'agent', 'backend', 'frontend', 'evaluation')
STATUSES = ('todo', 'in-progress', 'blocked', 'review')
LABELS = {
    'project-context': ('1D76DB', 'Общая память команды'),
    'pc:reference': ('5319E7', 'Архитектура, решения или справка'),
    **{f'pc:{name}': ('BFDADC', name) for name in COMPONENTS},
    **{f'pc:{name}': ('FBCA04', name) for name in STATUSES},
}
METRICS = (
    'recall_at_k', 'mrr', 'citation_accuracy', 'numeric_accuracy',
    'answer_correctness', 'unsupported_claim_rate', 'abstention_accuracy',
    'ticket_schema_validity', 'tool_selection_accuracy',
)
FIELDS = (
    'run_id', 'status', 'recorded_at_utc', 'hypothesis', 'configuration', 'models',
    'dataset', 'split', 'commit', 'issue_url', 'artifact_url', 'k', *METRICS,
    'latency_p50_ms', 'latency_p95_ms', 'cost_per_cycle', 'currency', 'result', 'notes',
)
TAB = 'Experiments'
SHEETS_RANGE = f'{TAB}!A:AA'
SHEETS_API = 'https://sheets.googleapis.com/v4/spreadsheets'
OAUTH_TOKEN_URL = 'https://oauth2.googleapis.com/token'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


class ContextError(Exception):
    pass


def read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_cli(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as error:
        raise ContextError(f'Установите {args[0]}.') from error
    if result.returncode:
        detail = result.stderr.strip()
        raise ContextError(f'{args[0]} завершился с кодом {result.returncode} · {detail}')
    return result.stdout.strip()


class GitHub:
    def __init__(self, repository):
        self.repository = repository

    def gh(self, *args):
        return run_cli('gh', *args, '--repo', self.repository)

    def issue(self, number):
        fields = 'number,title,state,url,body,labels,assignees,comments,updatedAt'
        return json.loads(self.gh('issue', 'view', str(number), '--json', fields))

    def issues(self, fields, *args):
        return json.loads(self.gh('issue', 'list', '--json', fields, *args))

    def initialize(self, seed):
        existing_labels = json.loads(self.gh('label', 'list', '--limit', '500', '--json', 'name'))
        known = {label['name'] for label in existing_labels}
        for label, (color, description) in LABELS.items():
            if label not in known:
                self.gh('label', 'create', label, '--color', color, '--description', description)
        issues = self.issues('number,body,url', '--state', 'all', '--limit', '500', '--label', 'project-context')
        by_marker = {}
        for issue in issues:
            match = re.search(r'<!-- project-context:([a-z-]+) -->', issue['body'])
            if match:
                key = match.group(1)
                if key in by_marker:
                    raise ContextError(f'Найдены дубли начальной карточки {key}. Проверьте Issues перед повтором.')
                by_marker[key] = issue['url']
        for item in seed:
            key = item['key']
            if key not in by_marker:
                body = f"<!-- project-context:{key} -->\n\n{item['body']}\n"
                by_marker[key] = self.gh(
                    'issue', 'create', '--title', item['title'], '--body-file', self.body_file(body),
                    '--label', ','.join(['project-context', *item['labels']]),
                )
            print(by_marker[key])
        return int(by_marker['hub'].rsplit('/', 1)[-1])

    @staticmethod
    def body_file(body):
        path = Path(OPERATIONS) / 'github-body.md'
        path.write_text(body, encoding='utf-8')
        return str(path)

    def finish(self, issue, status, body):
        number = str(issue['number'])
        digest = hashlib.sha256(f'{status}\n{body}'.encode()).hexdigest()[:20]
        marker = f'<!-- project-context-update:{digest} -->'
        if all(marker not in comment['body'] for comment in issue['comments']):
            self.gh('issue', 'comment', number, '--body-file', self.body_file(f'{body}\n\n{marker}\n'))
        current = {label['name'] for label in issue['labels']}
        desired = {'project-context'}
        if status != 'done':
            desired.add(f'pc:{status}')
        obsolete = (current & {f'pc:{state}' for state in STATUSES}) - desired
        options = []
        if missing := desired - current:
            options.extend(('--add-label', ','.join(sorted(missing))))
        if obsolete:
            options.extend(('--remove-label', ','.join(sorted(obsolete))))
        if options:
            self.gh('issue', 'edit', number, *options)
        if status == 'done' and issue['state'] != 'CLOSED':
            self.gh('issue', 'close', number, '--reason', 'completed')
        print(issue['url'])


def google_auth_file():
    default = Path.home() / '.config' / 'ai-mechanic' / 'google-oauth.json'
    path = Path(os.environ.get('PROJECT_CONTEXT_GOOGLE_AUTH_FILE', default)).expanduser().resolve()
    if path.is_relative_to(SKILL_DIR.parents[2]):
        raise ContextError('Файл OAuth должен находиться вне checkout.')
    return path


def google_login(client_secrets):
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as error:
        raise ContextError('Для sheet-login запустите этот скрипт через uv run --with google-auth-oauthlib.') from error
    path = google_auth_file()
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES, autogenerate_code_verifier=True)
    credentials = flow.run_local_server(port=0, prompt='consent select_account')
    if not credentials.refresh_token:
        raise ContextError('Google не выдал refresh token. Повторите вход с подтверждением доступа.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
        json.dump({
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'refresh_token': credentials.refresh_token,
        }, file)
    Path(file.name).replace(path)
    print('Google OAuth настроен. Проверьте таблицу командой sheet-list или sheet-init.')


def google_token():
    if token := os.environ.get('GOOGLE_SHEETS_ACCESS_TOKEN'):
        return token
    path = google_auth_file()
    if not path.is_file():
        raise ContextError('Google OAuth не настроен. Выполните sheet-login по docs/project-context.md.')
    credentials = read_json(path)
    keys = ('client_id', 'client_secret', 'refresh_token')
    if not isinstance(credentials, dict) or not all(credentials.get(key) for key in keys):
        raise ContextError('Локальный файл OAuth неполный. Повторите sheet-login.')
    data = urlencode({'grant_type': 'refresh_token', **{key: credentials[key] for key in keys}}).encode()
    try:
        with urlopen(Request(OAUTH_TOKEN_URL, data=data), timeout=20) as response:
            return json.load(response)['access_token']
    except HTTPError as error:
        raise ContextError(f'Google OAuth HTTP {error.code}. Если доступ истёк, повторите sheet-login.') from error
    except (URLError, TimeoutError) as error:
        raise ContextError('Не удалось подключиться к Google OAuth.') from error


class Google:
    def __init__(self, spreadsheet_id):
        if spreadsheet_id and not re.fullmatch(r'[A-Za-z0-9_-]+', spreadsheet_id):
            raise ContextError('Передайте ID таблицы без URL или пробелов.')
        self.spreadsheet_id = spreadsheet_id
        self.token = google_token()

    def request(self, method, url, payload=None):
        headers = {'Authorization': f'Bearer {self.token}', 'Content-Type': 'application/json'}
        if quota_project := os.environ.get('PROJECT_CONTEXT_GOOGLE_QUOTA_PROJECT'):
            headers['X-Goog-User-Project'] = quota_project
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        try:
            with urlopen(Request(url, method=method, headers=headers, data=data), timeout=20) as response:
                return json.load(response)
        except HTTPError as error:
            detail = json.loads(error.read()).get('error', {})
            reasons = ', '.join(item.get('reason', '') for item in detail.get('details', []) if item.get('reason'))
            raise ContextError(f'Google API HTTP {error.code} · {detail.get("status", "")} {reasons}'.rstrip()) from error
        except (URLError, TimeoutError) as error:
            raise ContextError('Google API недоступен. После ошибки записи проверьте ID перед повтором.') from error

    @property
    def url(self):
        return f'https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit'

    def values_url(self):
        if not self.spreadsheet_id:
            raise ContextError('ID общей таблицы ещё не задан. См. docs/project-context.md.')
        return f'{SHEETS_API}/{self.spreadsheet_id}/values/{quote(SHEETS_RANGE, safe="")}'

    @staticmethod
    def tab_properties():
        return {'title': TAB, 'gridProperties': {
            'rowCount': 1000, 'columnCount': len(FIELDS), 'frozenRowCount': 1, 'frozenColumnCount': 2,
        }}

    def values(self):
        return self.request('GET', f'{self.values_url()}?valueRenderOption=UNFORMATTED_VALUE').get('values', [])

    def create(self):
        response = self.request('POST', SHEETS_API, {
            'properties': {'title': 'Цифровой механик · эксперименты', 'timeZone': 'Etc/UTC'},
            'sheets': [{'properties': self.tab_properties()}],
        })
        self.spreadsheet_id = response['spreadsheetId']

    def initialize(self, plans):
        endpoint = f'{SHEETS_API}/{self.spreadsheet_id}'
        info = self.request('GET', f'{endpoint}?fields=sheets.properties.title')
        if not any(sheet['properties']['title'] == TAB for sheet in info.get('sheets', [])):
            self.request('POST', f'{endpoint}:batchUpdate', {
                'requests': [{'addSheet': {'properties': self.tab_properties()}}],
            })
        values = self.values()
        if not values:
            self.request('PUT', f'{SHEETS_API}/{self.spreadsheet_id}/values/{TAB}!A1?valueInputOption=RAW', {
                'values': [list(FIELDS)],
            })
        for plan in plans:
            self.log({**plan, 'status': 'planned'})
        print(self.url)

    def records(self):
        values = self.values()
        if not values or values[0] != list(FIELDS):
            raise ContextError('Заголовки Experiments не соответствуют схеме skill. Проверьте таблицу перед записью.')
        return [dict(zip(FIELDS, row)) for row in values[1:] if row and row[0]]

    def log(self, record):
        validate_record(record)
        run_id = record['run_id']
        old = [row for row in self.records() if row['run_id'] == run_id]
        if old:
            if len(old) != 1 or any(old[0].get(key, '') != record.get(key, '')
                                    for key in FIELDS if key != 'recorded_at_utc'):
                raise ContextError(f'Конфликт run_id {run_id}. Проверьте существующие строки и выберите новый ID.')
            return run_id
        record = {**record, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        self.request('POST', f'{self.values_url()}:append?valueInputOption=RAW&insertDataOption=INSERT_ROWS', {
            'values': [[record.get(key, '') for key in FIELDS]],
        })
        found = [row for row in self.records() if row['run_id'] == run_id]
        if len(found) != 1 or any(found[0].get(key, '') != val for key, val in record.items()):
            raise ContextError(f'Проверка записи {run_id} не прошла. Проверьте таблицу перед повторной отправкой.')
        return run_id

def validate_record(record):
    if not isinstance(record, dict) or record.keys() - set(FIELDS):
        raise ContextError('Ожидается JSON-объект прогона с полями из docs/project-context.md.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', str(record.get('run_id', ''))):
        raise ContextError('run_id должен быть уникальной строкой без пробелов.')
    if not isinstance(record.get('run_id'), str) or record.get('status') not in {'planned', 'completed', 'failed'}:
        raise ContextError('Нужны строковый run_id и status planned, completed или failed.')
    required = ['hypothesis', 'configuration', 'dataset']
    if record['status'] != 'planned':
        required.extend(('models', 'commit', 'result'))
    for key in required:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ContextError(f'Нужно заполнить {key}.')
    for key, value in record.items():
        if key in {'k', *METRICS, 'latency_p50_ms', 'latency_p95_ms', 'cost_per_cycle'}:
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ContextError(f'{key} должен быть конечным неотрицательным числом.')
            if key in METRICS and value > 1:
                raise ContextError(f'{key} указывайте долей от 0 до 1.')
        elif not isinstance(value, str):
            raise ContextError(f'{key} должен быть строкой.')
    if ('k' in record and (type(record['k']) is not int or record['k'] < 1)) or ('recall_at_k' in record and 'k' not in record):
        raise ContextError('Для Recall@K нужен целый положительный k.')
    if 'cost_per_cycle' in record and not re.fullmatch(r'[A-Z]{3}', record.get('currency', '')):
        raise ContextError('Для стоимости нужна трёхбуквенная валюта, например RUB.')
    if 'recorded_at_utc' in record:
        raise ContextError('recorded_at_utc заполняется автоматически.')


def write_config(settings):
    path = SKILL_DIR / 'context.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def start(args, config, github):
    had_error = False
    view = {'repository': config['repository']}
    if config['hub_issue']:
        view['hub_url'] = f'https://github.com/{config["repository"]}/issues/{config["hub_issue"]}'
    if config['spreadsheet_id']:
        view['spreadsheet_url'] = f'https://docs.google.com/spreadsheets/d/{config["spreadsheet_id"]}/edit'
    try:
        fields = 'number,title,url,labels,assignees,updatedAt'
        filters = ['--label', 'project-context', '--limit', str(args.limit)]
        if args.component:
            filters.extend(('--label', f'pc:{args.component}'))
        view['active_issues'] = github.issues(fields, *filters)
        view['references'] = github.issues(
            'number,title,url,body,updatedAt', '--label', 'project-context', '--label', 'pc:reference', '--limit', '12',
        )
        if args.issue:
            issue = github.issue(args.issue)
            issue['comments'] = issue['comments'][-5:]
            view['task'] = issue
    except ContextError as error:
        view['github_error'] = str(error)
        had_error = True
    if args.experiments:
        try:
            view['recent_experiments'] = Google(config['spreadsheet_id']).records()[-args.limit:]
        except ContextError as error:
            view['sheets_error'] = str(error)
            had_error = True
    dump(view)
    return int(had_error)


def main():
    parser = argparse.ArgumentParser(description='Синхронизация контекста ai-mechanic')
    commands = parser.add_subparsers(dest='command', required=True)
    startup = commands.add_parser('start')
    startup.add_argument('--issue', type=int)
    startup.add_argument('--component', choices=COMPONENTS)
    startup.add_argument('--experiments', action='store_true')
    startup.add_argument('--limit', type=int, choices=range(1, 51), default=12, metavar='1..50')
    finish = commands.add_parser('finish')
    finish.add_argument('issue', type=int)
    finish.add_argument('--status', choices=(*STATUSES[1:], 'done'), required=True)
    finish.add_argument('--body-file', type=Path, required=True)
    finish.add_argument('--run-file', type=Path)
    commands.add_parser('github-init')
    login = commands.add_parser('sheet-login')
    login.add_argument('--client-secrets', type=Path, required=True)
    sheet_init = commands.add_parser('sheet-init')
    sheet_init.add_argument('--spreadsheet-id')
    log = commands.add_parser('sheet-log')
    log.add_argument('--file', type=Path, required=True)
    listing = commands.add_parser('sheet-list')
    listing.add_argument('--limit', type=int, choices=range(1, 51), default=12, metavar='1..50')
    args = parser.parse_args()
    config = read_json(SKILL_DIR / 'context.json')
    github = GitHub(config['repository'])

    if args.command == 'start':
        return start(args, config, github)
    if args.command == 'github-init':
        config['hub_issue'] = github.initialize(read_json(SKILL_DIR / 'seed.json')['issues'])
        write_config(config)
    elif args.command == 'finish':
        body = args.body_file.read_text(encoding='utf-8').strip()
        if not body:
            raise ContextError('Файл результата задачи пуст.')
        issue = github.issue(args.issue)
        if issue['state'] == 'CLOSED' and args.status != 'done':
            raise ContextError('Issue закрыта. Согласуйте повторное открытие перед изменением статуса.')
        if args.run_file:
            google = Google(config['spreadsheet_id'])
            record = read_json(args.run_file)
            validate_record(record)
            if record.get('issue_url', issue['url']) != issue['url']:
                raise ContextError('issue_url прогона не совпадает с Issue.')
            run_id = google.log({**record, 'issue_url': issue['url']})
            body += f'\n\nПрогон `{run_id}` · [Google Sheets]({google.url})'
        github.finish(issue, args.status, body)
    elif args.command == 'sheet-login':
        google_login(args.client_secrets)
    elif args.command == 'sheet-init':
        existing = config['spreadsheet_id']
        if existing and args.spreadsheet_id and args.spreadsheet_id != existing:
            raise ContextError('В репозитории настроена другая таблица. Согласуйте миграцию перед заменой ID.')
        google = Google(existing or args.spreadsheet_id)
        if not google.spreadsheet_id:
            google.create()
            config['spreadsheet_id'] = google.spreadsheet_id
            write_config(config)
            print(google.url, flush=True)
        google.initialize(read_json(SKILL_DIR / 'seed.json')['plans'])
        config['spreadsheet_id'] = google.spreadsheet_id
        write_config(config)
    elif args.command == 'sheet-list':
        dump(Google(config['spreadsheet_id']).records()[-args.limit:])
    elif args.command == 'sheet-log':
        google = Google(config['spreadsheet_id'])
        print(google.log(read_json(args.file)), google.url)
    return 0


if __name__ == '__main__':
    try:
        with tempfile.TemporaryDirectory(prefix='ai-mechanic-context-') as OPERATIONS:
            sys.exit(main())
    except (ContextError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'project-context: {error}', file=sys.stderr)
        sys.exit(1)
