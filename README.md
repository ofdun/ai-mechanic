# Цифровой механик

ML-ядро для диагностики, поиска по техническим руководствам и подготовки заявки. [Общий контекст и владельцы компонентов](docs/shared-project-context.md) · [Работа coding-агентов](docs/project-context.md)

## Запуск ядра

Нужен Python 3.10+ и `uv`. Пакет моков заказчика уже содержит приложения и свои зависимости. Из распакованной папки `neurodvs_hackathon_mock_2026-11-16` запустите команды в разных терминалах.

```bash
uv run --no-project --with-requirements requirements.txt uvicorn mock_ml.app:app --port 8001
uv run --no-project --with-requirements requirements.txt uvicorn mock_toir.app:app --port 8002
```

Из корня этого репозитория:

```bash
uv sync
cp .env.example .env
```

Задайте `DATA_DIR` с `vehicles.json` или архивом `neurodvs_hackathon_mock_*.zip`. Можно передать конкретный путь к распакованному JSON через `VEHICLES_PATH`. Клиент использует `ML_API_URL` и `MAINTENANCE_API_URL`.

Mock ML, Mock ТОиР и OpenAPI-спецификации предоставляет заказчик. Запись, обработка звука двигателя и акустическая диагностика относятся к его сервису. Ядро отправляет `vehicle_id` и `session_id` в `POST /diagnose`, получает готовый JSON и использует его для интерпретации и поиска по руководству. Для создания заявки ядро отправляет подтверждённый черновик в `POST /tickets`.

```bash
# Получение результата Mock ML по явной команде доступно без ключа LLM.
uv run python -m ml run --vehicle M333YK47 --session S-0005 --command diagnose
# Запрос на естественном языке. Задайте LLM_API_KEY в окружении или в .env.
uv run python -m ml run --vehicle M333YK47 --session S-0005 --message 'Как проверить форсунки?'
# Детерминированный режим для локальной проверки проводки компонентов.
PLANNER_MODE=offline uv run python -m ml run --vehicle M333YK47 --session S-0005 \
  --message 'Запусти диагностику двигателя'
```

В режиме `llm` один вызов OpenRouter выбирает инструмент. Для поиска он отдаёт переформулировку и до пяти подзапросов. `SearchPlan` сохраняет исходный текст, а доверенная модель двигателя берётся из карточки ТС. `--verbose` включает метрики вызова LLM без текста запроса. `PLANNER_MODE=offline` переключает на ограниченный набор текстовых команд для локального запуска.

## Стыки для команды

```python
import httpx

from ml.config import Settings
from ml.contracts import AgentRequest
from ml.pipeline import Pipeline

settings = Settings()
with httpx.Client(timeout=settings.http_timeout_seconds) as http:
    pipeline = Pipeline(settings, http)
    response = pipeline.run(AgentRequest(
        actor_id='mechanic-1', vehicle_id='M333YK47', session_id='S-0005', command='diagnose',
    ))
```

Веб-сервис управляет сроком жизни `httpx.Client` и `Pipeline`. `actor_id` передаётся из проверенного контекста пользователя. Для уточнения передайте `AgentRequest.conversation_id` и `context=ConversationContext(...)` с последним результатом диагностики, моделью двигателя и, при необходимости, последними сообщениями из серверной сессии. `AgentError` содержит машиночитаемые `code` и `retryable`.

| Кто подключается | Точка входа |
|---|---|
| Роман | `ParsedDocument` с физическими страницами `DocumentPage`. Для страницы с более узкой применимостью заполните `DocumentPage.engine_models` |
| Валерий | `Retriever.search(SearchPlan) -> EvidenceBundle`, передаётся как `retriever=` |
| Алексей | `ml.generation`: `ClaimGenerator` и `TechnicalReasoner` реализуют `Generator`, `EvidenceVerifier` реализует `Verifier`, `MechanicFinalizer` собирает ответ механику. Описание в разделе «Generation и Verification» |
| Елисей | `Pipeline.run(AgentRequest) -> AgentResponse` и `Pipeline.confirm_ticket(TicketConfirmation) -> TicketReceipt` |

Для создания фрагмента из страницы используйте `Evidence.from_page`. Он сохраняет применимость страницы и позволяет сузить её для отдельного фрагмента. Пайплайн возвращает поисковый план и доступные фрагменты. Технические рекомендации и черновик требуют Generator и Verifier. В ответ попадают утверждения из исходного черновика, которые одобрил Verifier и у которых есть применимые ссылки.

### Первый сквозной сценарий по PDF

Временный BM25-baseline использует физические страницы 52–54 проверенного руководства 236-3902150-Б РЭ для ЯМЗ-236М2. Он читает PDF через Poppler (`pdftotext`), строит SQLite FTS5 и проверяет одну полную строку по F08, сохраняя условие применимости операции. Для других моделей и операций нужны командные адаптеры.

```bash
# Замените ../data на путь к комплекту заказчика.
DATA_DIR=../data uv run python -m ml.pdf_baseline
DATA_DIR=../data AGENT_ARCHITECTURE=baseline uv run python -m ml --pdf-index artifacts/manuals.sqlite3 run \
  --vehicle M333YK47 --session S-0005 --command draft_ticket
DATA_DIR=../data uv run python -m ml confirm \
  --vehicle M333YK47 --session S-0005 --draft-id ID_ИЗ_ЧЕРНОВИКА --confirm
```

### Архитектуры A и B через OpenRouter

С `--pdf-index` и `PLANNER_MODE=llm` по умолчанию выбрана архитектура A. Для B задайте `AGENT_ARCHITECTURE=b`. Для отправки фрагментов в OpenRouter используйте `ALLOW_EXTERNAL_EVIDENCE=true` после разрешения владельца данных. `--verbose` логирует этапы, модели, число токенов и стоимость. Ключ и текст запроса в лог не попадают.

| Цепочка | Модели по порядку |
|---|---|
| A | Planner — GLM 5.3 Flash → Generator — DeepSeek V4.1 Flash → Verifier и Finalizer — GLM 5.3 Flash |
| B | Planner и Query Strategist — GLM 5.3 Flash → Reasoner — GLM 5.3 → Critic и условный Repair — DeepSeek V4.1 Flash → Finalizer — GLM 5.3 Flash |

Для запроса по руководству ниже доступна вся цепочка. Явная диагностическая команда пропускает Planner. B выполняет максимум один дополнительный поиск по запросу Critic, а Repair вызывается при появлении новых фрагментов. Если поиск пуст, ядро возвращает ограничение. Модели задаются в `.env`, см. `.env.example`.

```bash
ALLOW_EXTERNAL_EVIDENCE=true AGENT_ARCHITECTURE=a uv run python -m ml --verbose --pdf-index artifacts/manuals.sqlite3 run \
  --vehicle M333YK47 --session S-0005 --message 'Найди в руководстве, что проверить при дымном выхлопе'
ALLOW_EXTERNAL_EVIDENCE=true AGENT_ARCHITECTURE=b uv run python -m ml --verbose --pdf-index artifacts/manuals.sqlite3 run \
  --vehicle M333YK47 --session S-0005 --message 'Найди в руководстве, что проверить при дымном выхлопе'
```

Обе версии используют одинаковый индекс и проверяют применимость источников, допустимые ссылки и численные значения. LLM может предложить диагностические проверки с источниками. Операция для заявки должна совпадать с проверенным правилом для F08 и получить одобрение стадии верификации. `AGENT_ARCHITECTURE=baseline` возвращает детерминированный PDF-режим. `PLANNER_MODE=offline` включает его и для естественно-языкового запроса.

### Подключение к FastAPI

После `uv sync --extra api` зарегистрируйте router в приложении команды. `current_actor_id` должен получать ID из существующей серверной аутентификации. Для сценария выше создайте `pipeline` через `build_pdf_pipeline(settings, shared_http_client, index_path)`.

```python
from ml.api import create_agent_router

app.include_router(create_agent_router(pipeline, current_actor_id))
```

| Маршрут | Что передавать |
|---|---|
| `POST /agent/run` | JSON с `vehicle_id`, `session_id`, опциональным `conversation_id`. Для запуска диагностики и черновика передайте `command=diagnose` или `command=draft_ticket`, для уточнения — `message` |
| `POST /agent/tickets/confirm` | JSON с `vehicle_id`, `session_id`, `draft_id`, `draft_version` и `confirmed=true` |

Известные ошибки ядра возвращаются как `detail` с полями `code`, `message` и `retryable`. Для серверной памяти можно передать `load_context(actor_id, conversation_id)` при создании router. Приложение команды хранит результаты предыдущих ходов и управляет сроком жизни `Pipeline` и HTTP-клиента.

### Generation и Verification

Модуль `ml/generation/` превращает вердикт Mock ML и найденные страницы руководств в проверенный ответ механику.

| Файл | Что делает |
|---|---|
| `generator.py` | `ClaimGenerator`, архитектура A. LLM возвращает атомарные утверждения. К каждому прикладывается дословная выдержка со страницы в поле `quote` |
| `reasoner.py` | `TechnicalReasoner`, архитектура B. Сильная модель строит цепочку признак → причина → проверка → норматив → ремонт |
| `verifier.py` | `EvidenceVerifier` проверяет каждое утверждение кодом, без LLM. Для каждого отказа возвращает код причины |
| `finalizer.py` | `MechanicFinalizer` собирает ответ по порядку ТЗ: вердикт → признаки → гипотезы → проверки → нормативы → ремонт → источники → ограничения. Фактов от себя не добавляет |
| `profiles.py` | Словарь узлов F01–F14, оценка риска агентом и базовая срочность |

Утверждение проходит verifier, только если выполнено всё сразу:

- выдержка найдена на указанной странице руководства для двигателя из карточки ТС, а не в оглавлении. Поиск выдерживает переносы, склеенные ячейки и перемешанные колонки таблиц PDF, но не допускает пропуска букв внутри слова или слов «не», «если», «при» и подобных;
- каждое число утверждения есть в выдержке, стоит в том же порядке, диапазон не обрезан, единицы не пересчитаны. Число из соседней строки таблицы нельзя приклеить к чужой подписи;
- сохранены запреты и условия источника. Запреты, разрешения на эксплуатацию и модели двигателя, которых нет в источнике, не добавлены;
- после диагностики операция и запчасть относятся к узлу из диагноза.

`Pipeline` всегда считает вывод и срочность заявки. Срочность: базовая критичность типа неисправности плюс поправка на один уровень по `anomaly_score` (≥ 0,85 повышает, < 0,40 понижает). Прямой запрет эксплуатации в проверенной выдержке всегда даёт `high`. Если источники расходятся, `limitations` получает строку с обеими формулировками и страницами и советом уточнить у специалиста. Веб-клиент сам рисует утверждения, поэтому `message` остаётся коротким. Для текстового канала передайте `finalizer=MechanicFinalizer()`: тогда `message` содержит весь ответ по порядку ТЗ, расхождения показаны рядом. Структурированная версия ответа доступна через `MechanicFinalizer().answer(response)`.

Бенчмарк проверки на руководствах заказчика (гипотеза GEN-07). Материалы и их текст остаются локально, в отчёт попадают только счётчики:

```bash
uv run python -m evaluation.generation.verifier_benchmark --pages DATA/pages.jsonl --pdf-dir DATA --output run.json
```

### Черновик и подтверждение

Вызов `Pipeline.run` с `command='draft_ticket'` сохраняет черновик только после пригодной аномалии и хотя бы одной подтверждённой операции. Для кнопки подтверждения используйте отдельный `Pipeline.confirm_ticket`. Передайте идентичность, пару ТС и сессии, `draft_id`, `draft_version` и явное `confirmed=True`. Ядро сверяет модель двигателя и свежую диагностику перед отправкой сохранённого payload. Черновик действует 30 минут. Черновик без сохранённой модели требуется сформировать заново.

SQLite открывается при первой операции с заявкой. База `ARTIFACTS_DIR/agent-tickets.sqlite3` хранит черновики, статус отправки и квитанции. Повтор того же подтверждения возвращает сохранённый `ticket_id`. При `TICKET_OUTCOME_UNKNOWN` нужна ручная сверка с ТОиР. В этой версии повторная отправка для той же сессии и аномалии остаётся заблокированной даже после создания нового черновика или перезапуска. Порог `MIN_DIAGNOSTIC_CONFIDENCE=0.70` и `REPEAT_ON_WARNING=true` — конфигурируемые стартовые правила.

## Проверки

```bash
uv run pytest -q
uv run --extra api pytest -q
```

Тесты проверяют tool calling и переформулирование текстового запроса, контекст уточнений, фильтрацию источников, защиту подтверждения и отправки. Для прогона с материалами заказчика используйте `M333YK47/S-0005` для аномалии, `P456AT178/S-0001` для нормы и `P456AT178/S-0003` для сбоя записи.
