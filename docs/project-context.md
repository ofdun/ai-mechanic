# Подключение агентов команды

Общий контекст доступен в [GitHub Issue #1](https://github.com/ofdun/ai-mechanic/issues/1). Архитектура, контракты, задачи, статусы и блокеры находятся в [карточках с меткой project-context](https://github.com/ofdun/ai-mechanic/issues?q=label%3Aproject-context). Ответственных назначайте в Assignees после согласования в команде.

## Быстрый старт

Откройте checkout ветки `feat/project-context` в coding-агенте. После включения изменений в основную ветку достаточно обновить обычный checkout. Установите Python 3.10+ и GitHub CLI. Для записи в Issues авторизуйте `gh` через `gh auth login --web` с доступом к репозиторию.

В Codex запустите `$project-context`. Агент применяет его и по правилу в корневом `AGENTS.md`. Если ваш клиент запускает навыки иначе, попросите его прочитать `.agents/skills/project-context/SKILL.md`. Один и тот же helper работает из терминала, дополнительных скриптов для участников не требуется. [Как Codex обнаруживает навыки в репозитории](https://developers.openai.com/blog/skills-agents-sdk)

```bash
python3 .agents/skills/project-context/scripts/context.py start --issue 3 --component backend
```

`start` читает актуальные справочные Issues, очередь по компоненту и выбранную карточку. Для работы с измерениями добавьте `--experiments`. Перед завершением агент пишет краткий итог в связанную Issue, например со статусом `review`. Добавьте результат проверки, ссылки на код и оставшиеся блокеры. Смена статуса меняет только метки `pc:*`, относящиеся к этапу задачи. `done` закрывает Issue.

```bash
python3 .agents/skills/project-context/scripts/context.py finish 3 \
  --status review --body-file /tmp/ai-mechanic-update.md
```

Для эксперимента добавьте `--run-file /tmp/ai-mechanic-run.json`. Helper сначала записывает строку и проверяет её чтением, потом обновляет Issue. Для независимого прогона используйте `sheet-log --file`. Если запись прервалась, проверьте Issue и `run_id` перед повтором. Повтор с тем же содержанием пропускает уже сохранённую запись. Не запускайте параллельно запись того же `run_id`: Sheets не гарантирует уникальность столбца при конкурентных добавлениях.

## Google Sheets без gcloud

Таблицу подключает один владелец. Для доступа напрямую к Sheets API нужен OAuth-клиент Google; он выпускается через браузер. Подойдёт существующий клиент команды с типом **Desktop app**. `gcloud`, сервисный аккаунт, сервер и Drive API для этого процесса не требуются.

1. В нужном Google-аккаунте откройте Cloud Console, выберите проект команды и включите **Google Sheets API**. В **Google Auth Platform** настройте аудиторию, создайте OAuth-клиент **Desktop app** и скачайте JSON. Для внешнего приложения в Testing добавьте участников как Test users. Google может ограничивать refresh token в режиме Testing семью днями. Храните клиентский JSON вне checkout. [Настройка Google OAuth для Sheets](https://developers.google.com/workspace/sheets/api/quickstart/python) · [Правила срока действия токенов](https://developers.google.com/identity/protocols/oauth2#expiration)
2. Выполните команду ниже на машине с браузером. Выберите нужный Google-аккаунт в окне входа. Helper сохранит refresh token локально в `~/.config/ai-mechanic/google-oauth.json` и будет обновлять короткоживущий access token при обращениях к Sheets. Участники авторизуют свои аккаунты. OAuth-файлы и токены хранятся вне Git.

   ```bash
   uv run --with google-auth-oauthlib python .agents/skills/project-context/scripts/context.py \
     sheet-login --client-secrets /путь/вне/repo/desktop-client.json
   ```

3. Владелец один раз запускает `sheet-init`. Можно присоединить существующую общую таблицу через `--spreadsheet-id ID` из адреса. Без ID команда создаст новую таблицу. Она подготовит вкладку `Experiments` и восемь строк со статусом `planned`, проверяя запись чтением. Имеющиеся строки сохраняются, несовпадение заголовков останавливает инициализацию.

   ```bash
   python3 .agents/skills/project-context/scripts/context.py sheet-init
   python3 .agents/skills/project-context/scripts/context.py sheet-list
   ```

4. В Google Sheets выдайте редакторский доступ участникам или группе. Опубликуйте обновлённый `.agents/skills/project-context/context.json` с ID таблицы в ветке проекта. Файл содержит только общие идентификаторы и не включает OAuth. Участники обновляют checkout и проверяют доступ через `sheet-list`. Среда с уже настроенным поставщиком OAuth может передать `GOOGLE_SHEETS_ACCESS_TOKEN` со scope `https://www.googleapis.com/auth/spreadsheets` через собственное хранилище секретов.

**Ссылка на таблицу.** После `sheet-init` скрипт выведет ссылку, а `start` будет показывать её в поле `spreadsheet_url`.

## Запись эксперимента

Минимальный формат для измеренного прогона включает `run_id`, `status`, `hypothesis`, `configuration`, `models`, `dataset`, `commit`, `result`. `completed` и `failed` относятся к фактическим прогонам. Привязывайте прогон к задаче и артефактам по `issue_url`, `artifact_url` при наличии. Измеренные поля передавайте числами, неизвестные опускайте. Доли задаются от 0 до 1. Для `recall_at_k` обязателен `k`. Для `cost_per_cycle` нужна валюта, например `RUB`. `recorded_at_utc` генерируется автоматически.

| Поля | Что записывать |
|---|---|
| `run_id`, `status` | Уникальный ID, статус `planned`, `completed` или `failed` |
| `hypothesis`, `configuration`, `models` | Гипотеза, параметры, имена и версии моделей или `none` для лексического поиска |
| `dataset`, `split`, `commit`, `issue_url`, `artifact_url` | Версия набора, сборка кода, связь с задачей и артефактами |
| `k`, `recall_at_k`, `mrr`, `citation_accuracy`, `numeric_accuracy` | Retrieval и качество ссылок и нормативов |
| `answer_correctness`, `unsupported_claim_rate`, `abstention_accuracy`, `ticket_schema_validity`, `tool_selection_accuracy` | Ответы, абстенция и поведение агента |
| `latency_p50_ms`, `latency_p95_ms`, `cost_per_cycle`, `currency` | Задержка в миллисекундах и стоимость полного цикла |
| `result`, `notes` | Итог измерения, критические ошибки и интерпретация |

Гипотезы для начального заполнения лежат в `.agents/skills/project-context/seed.json`. Планы остаются отдельными строками, а измеренный результат записывается с новым ID. Согласованные к публикации выводы фиксируйте в GitHub Issue.
