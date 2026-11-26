# Подключение агентов команды

[Общий контекст проекта](shared-project-context.md) содержит требования, роли, архитектуру, контракты и гипотезы. Задачи, текущие статусы и блокеры доступны в [индексе GitHub](https://github.com/ofdun/ai-mechanic/issues/1) и [по метке project-context](https://github.com/ofdun/ai-mechanic/issues?q=label%3Aproject-context). Эксперименты записывайте в комментариях к [общему журналу](https://github.com/ofdun/ai-mechanic/issues/9). Один комментарий на прогон.

## Быстрый старт

Общий контекст находится в `main`. Для новой задачи создайте рабочую ветку от актуальной `main` и откройте её в coding-агенте. В Codex запустите `$project-context`. Правило в корневом `AGENTS.md` применяется и при обычной работе над задачей. В других клиентах попросите агента прочитать `.agents/skills/project-context/SKILL.md`. Для команд нужны Python 3.10+, GitHub CLI и вход через `gh auth login --web`. [Как Codex обнаруживает навыки](https://developers.openai.com/blog/skills-agents-sdk)

```bash
ctx=.agents/skills/project-context/scripts/context.py
python3 "$ctx" start --issue 3 --component backend --experiments
python3 "$ctx" finish 3 --status review --body-file /tmp/ai-mechanic-update.md
```

`start` читает справочные Issues, очередь, выбранную задачу и, с флагом `--experiments`, журнал. Перед завершением добавьте в связанную задачу результат, проверки, ссылки на код и оставшиеся блокеры. `done` закрывает задачу. Ответственных назначайте в Assignees по договорённости в команде.

## Запись эксперимента

Оставьте комментарий в [журнале #9](https://github.com/ofdun/ai-mechanic/issues/9) или передайте агенту JSON прогона. Укажите уникальный ID, гипотезу, конфигурацию и версии моделей, данные, коммит, метрики, задержку, стоимость и результат. Если метрика не измерялась, опустите её. Привяжите запись к задаче или артефакту, если есть.

```bash
python3 "$ctx" experiment-log --file /tmp/ai-mechanic-run.json
# Если прогон связан с задачей, эта команда запишет его и добавит ссылку в задачу.
python3 "$ctx" finish 3 --status review --body-file /tmp/ai-mechanic-update.md \
  --run-file /tmp/ai-mechanic-run.json
```

Для одного прогона выберите одну из этих команд. Последовательный повтор с тем же содержанием вернёт ссылку на существующий комментарий. Для нового прогона выберите новый ID.

| Поля JSON | Что записывать |
|---|---|
| `run_id`, `status` | Уникальный ID, статус `planned`, `completed` или `failed` |
| `hypothesis`, `configuration`, `models` | Гипотеза, настройки и версии моделей. Для чисто лексического поиска подойдёт `none` |
| `dataset`, `split`, `commit`, `issue_url`, `artifact_url` | Версия данных, версия кода, связь с задачей и артефактами |
| `k`, `recall_at_k`, `mrr`, `citation_accuracy`, `numeric_accuracy` | Метрики поиска, ссылок и нормативов |
| `answer_correctness`, `unsupported_claim_rate`, `abstention_accuracy`, `ticket_schema_validity`, `tool_selection_accuracy` | Метрики ответов и поведения агента |
| `latency_p50_ms`, `latency_p95_ms`, `cost_per_cycle`, `currency` | Задержка в миллисекундах и стоимость полного цикла |
| `result`, `notes` | Итог измерения, ошибки и интерпретация |

Для измеренного прогона обязательны `run_id`, `status`, `hypothesis`, `configuration`, `models`, `dataset`, `commit` и `result`. Доли передавайте от 0 до 1, с `recall_at_k` указывайте `k`, со стоимостью указывайте валюту. Связывайте прогон с ID из [каталога гипотез](shared-project-context.md#11-гипотезы-и-владельцы-экспериментов) или указывайте собственную гипотезу. В шапке журнала перечислены первые сквозные эксперименты.
