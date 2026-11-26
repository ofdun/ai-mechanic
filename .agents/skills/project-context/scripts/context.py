#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


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
    'run_id', 'status', 'hypothesis', 'configuration', 'models',
    'dataset', 'split', 'commit', 'issue_url', 'artifact_url', 'k', *METRICS,
    'latency_p50_ms', 'latency_p95_ms', 'cost_per_cycle', 'currency', 'result', 'notes',
)


class ContextError(Exception):
    pass


def read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def seed_body(item, plans):
    key = item['key']
    body = f"<!-- project-context:{key} -->\n\n{item['body']}\n"
    if key == 'experiments':
        body += '\n## Запланированные эксперименты\n\n'
        for plan in plans:
            body += (f"- **{plan['run_id']}** · {plan['hypothesis']} "
                     f"{plan['configuration']}. Данные: {plan['dataset']}. {plan['notes']}.\n")
    return body


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
        for item in seed['issues']:
            key = item['key']
            if key not in by_marker:
                body = seed_body(item, seed['plans'])
                by_marker[key] = self.gh(
                    'issue', 'create', '--title', item['title'], '--body-file', self.body_file(body),
                    '--label', ','.join(['project-context', *item['labels']]),
                )
            print(by_marker[key])
        return {f'{key}_issue': int(by_marker[key].rsplit('/', 1)[-1])
                for key in ('hub', 'experiments')}

    @staticmethod
    def body_file(body):
        path = Path(OPERATIONS) / 'github-body.md'
        path.write_text(body, encoding='utf-8')
        return str(path)

    def log_experiment(self, number, record):
        validate_record(record)
        run_id = record['run_id']
        payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:20]
        prefix = f'<!-- project-context-run:{run_id}:'
        marker = f'{prefix}{digest} -->'
        issue = self.issue(number)
        matches = [comment for comment in issue['comments'] if comment['body'].startswith(prefix)]
        if matches:
            if len(matches) != 1 or not matches[0]['body'].startswith(marker):
                raise ContextError(f'Конфликт run_id {run_id}. Проверьте журнал и выберите новый ID.')
            return matches[0]['url']
        body = f"{marker}\n\n### Прогон {run_id} · {record['status']}\n\n```json\n{payload}\n```\n"
        return self.gh('issue', 'comment', str(number), '--body-file', self.body_file(body))

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


def validate_record(record):
    if not isinstance(record, dict) or record.keys() - set(FIELDS):
        raise ContextError('Ожидается JSON-объект прогона с полями из docs/project-context.md.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', str(record.get('run_id', ''))):
        raise ContextError('run_id должен быть уникальной строкой без пробелов.')
    if not isinstance(record.get('run_id'), str) or record.get('status') not in ('planned', 'completed', 'failed'):
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
    view['experiments_url'] = f'https://github.com/{config["repository"]}/issues/{config["experiments_issue"]}'
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
        if args.experiments:
            journal = github.issue(config['experiments_issue'])
            journal['comments'] = [comment for comment in journal['comments']
                                   if '<!-- project-context-update:' not in comment['body']][-args.limit:]
            view['experiments'] = journal
    except ContextError as error:
        view['github_error'] = str(error)
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
    log = commands.add_parser('experiment-log')
    log.add_argument('--file', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(SKILL_DIR / 'context.json')
    github = GitHub(config['repository'])

    if args.command == 'start':
        return start(args, config, github)
    if args.command == 'github-init':
        config.update(github.initialize(read_json(SKILL_DIR / 'seed.json')))
        write_config(config)
    elif args.command == 'finish':
        if args.issue == config['experiments_issue']:
            raise ContextError('Журнал экспериментов принимает комментарии через experiment-log.')
        body = args.body_file.read_text(encoding='utf-8').strip()
        if not body:
            raise ContextError('Файл результата задачи пуст.')
        issue = github.issue(args.issue)
        if issue['state'] == 'CLOSED' and args.status != 'done':
            raise ContextError('Issue закрыта. Согласуйте повторное открытие перед изменением статуса.')
        if args.run_file:
            record = read_json(args.run_file)
            validate_record(record)
            if record.get('issue_url', issue['url']) != issue['url']:
                raise ContextError('issue_url прогона не совпадает с Issue.')
            run_url = github.log_experiment(config['experiments_issue'], {**record, 'issue_url': issue['url']})
            body += f"\n\n[Прогон {record['run_id']}]({run_url})"
        github.finish(issue, args.status, body)
    elif args.command == 'experiment-log':
        print(github.log_experiment(config['experiments_issue'], read_json(args.file)))
    return 0


if __name__ == '__main__':
    try:
        with tempfile.TemporaryDirectory(prefix='ai-mechanic-context-') as OPERATIONS:
            sys.exit(main())
    except (ContextError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'project-context: {error}', file=sys.stderr)
        sys.exit(1)
