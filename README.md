# Цифровой механик

ML-ядро для диагностики, поиска по техническим руководствам и подготовки заявки. [Общий контекст и владельцы компонентов](docs/shared-project-context.md) · [Работа coding-агентов](docs/project-context.md)

## Быстрый запуск через Docker

Нужен Docker с Compose 2.24+. Для нового checkout скопируйте `.env.example` в `.env`. Если `.env` уже есть, перенесите нужные настройки вручную. Укажите в `FLEET_DIR` папку с шестью PDF и исходным ZIP заказчика, в `MOCK_DIR` — распакованную папку, в которой лежат `mock_ml/`, `mock_toir/` и `data/`.

```bash
docker compose up --build -d
```

Откройте [интерфейс механика](http://localhost:8080). Compose собирает React, индексирует руководства и запускает веб после успешной сборки индекса. Прогресс первого запуска виден в `docker compose logs -f index web`. Каталог с PDF подключён для чтения. Индекс и квитанции сохраняются в томе `state` при `docker compose down`. Штатный мок ТОиР хранит собственные заявки в памяти процесса.

Для подключения двух внешних сервисов укажите их базовые URL в `.env`, очистите `COMPOSE_PROFILES` и выполните `docker compose up -d web`. При замене одного сервиса достаточно изменить соответствующий URL и выполнить ту же команду. На Docker Desktop сервис на вашем компьютере доступен контейнеру через `host.docker.internal`. Ранее запущенные локальные моки можно остановить командой `docker compose --profile mocks stop ml maintenance`, когда оба адреса переключены.

### Публичный стенд для жюри

Локальный запуск слушает `127.0.0.1`. Для показа жюри укажите публичный IP сервера и откройте входящие TCP-порты 80 и 443. Вариант для стенда публикует Caddy, который выпускает и обновляет сертификат, проверяет пароль и передаёт запрос приложению. Тот же вход защищает файлы руководств и отдельные страницы.

В серверном `.env` задайте `COMPOSE_FILE=compose.yaml:compose.stand.yaml`, `STAND_HOST`, `STAND_USER` и `STAND_PASSWORD`. В шаблоне приведён учебный пароль. Для стенда выберите свой и заключите его в одинарные кавычки. Шлюз вычислит bcrypt-хеш при запуске. Чтобы хранить в `.env` только хеш, используйте вместо `STAND_PASSWORD` переменную `STAND_PASSWORD_HASH` с результатом команды `docker run --rm -it caddy:2-alpine caddy hash-password` в одинарных кавычках.

Выполните `docker compose up --build -d`. Проверьте `docker compose ps` и откройте `https://<STAND_HOST>`. Браузер запросит логин и пароль. Передайте их жюри отдельно от ссылки. Команда `docker compose logs -f gateway web agent` показывает состояние сервисов. Для хранения состояния в существующем каталоге задайте `STATE_DIR` и укажите в `FLEET_USER` числовые UID:GID пользователя с доступом на запись.

## Устройство веб-приложения

В `app/frontend` находится React, в `app/backend` — публичный сервер на Go. Пакет `internal/httpapi` регистрирует маршруты Gin и проверяет cookie через middleware, `internal/mechanic` ведёт диалог и проверяет подтверждение механика, `internal/sqlite` хранит браузерные сессии и контекст через GORM. Адаптер `internal/agent` вызывает внутренний сервис из `ml/` для диагностики, поиска и операций с заявкой. Физические страницы обрабатывает тот же PDF-движок, которым построен индекс. Порт агента доступен внутри сети Compose.

Параметры запуска Go-сервера собраны в `internal/app/config.go`. В Compose `WEB_PORT` задаёт порт публикации, `WEB_LISTEN_PORT` — порт внутри контейнера. Проверка готовности обращается к адресу прослушивания. Для подключения другого экземпляра агента задайте `AGENT_URL`. время ожидания ответа меняется через `WEB_AGENT_TIMEOUT` (по умолчанию `20m`).

## Разработка ядра без контейнеров

Нужен Python 3.10+ и `uv`. Пакет моков заказчика уже содержит приложения и свои зависимости. Из распакованной папки `neurodvs_hackathon_mock_2026-11-16` запустите команды в разных терминалах.

```bash
uv run --no-project --with-requirements requirements.txt uvicorn mock_ml.app:app --port 8001
uv run --no-project --with-requirements requirements.txt uvicorn mock_toir.app:app --port 8002
```

Из корня этого репозитория:

```bash
uv sync
cp .env.example .env
export ML_API_URL=http://127.0.0.1:8001 MAINTENANCE_API_URL=http://127.0.0.1:8002
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

### Retrieval по всем руководствам

Состояние, контракты и что учесть другим участникам — в [docs/retrieval.md](docs/retrieval.md).

`ml.retrieval.RetrievalEngine` реализует `Retriever`. Путь запроса: `SearchPlan → фильтр модели → BM25 и exact dense на каждый запрос плана → RRF → cross-encoder → родительские страницы и соседи → бюджет → EvidenceBundle`. Фильтр модели двигателя применяется до любого top-k, `DocumentPage.engine_models` сужает применимость страницы. Параметры и гипотезы RET-01…RET-10 лежат в `configs/retrieval.yaml`.

| Модуль | Что делает |
|---|---|
| `chunking` | Загрузка `ParsedDocument`, `page` (страница целиком) и `structural` (абзацы внутри страницы, строки таблиц сохраняются). Фрагмент не пересекает физическую страницу |
| `sparse` | Snowball для русского, `0,25` = `0.25`, BM25 |
| `embeddings`, `dense` | sentence-transformers (`BAAI/bge-m3` по умолчанию, пресеты Qwen3-Embedding и e5-instruct), точный косинус на NumPy. Векторы привязаны к фрагментам индекса по хешу ID |
| `fusion` | Reciprocal Rank Fusion запросов плана и методов |
| `reranker` | Cross-encoder, например `BAAI/bge-reranker-v2-m3` |
| `engine` | Сборка и чтение индекса, режимы `sparse`, `dense`, `hybrid`, контекст `page` или `chunk`, соседние страницы, бюджет слов |

Источник корпуса — выгрузка ingestion (`ml/ingestion/export_pages.py`): `DATA_DIR/pages.jsonl` с записью на физическую страницу и каталог `DATA_DIR/manifest.json`. Из каталога берётся `manual` для ссылок и заявки — обозначение руководства, а если его нет, название. `build --source` также принимает каталог файлов `ParsedDocument`. Строки `models` должны совпадать с `vehicles.json`, `build` предупреждает о моделях ТС без руководств.

```bash
DATA_DIR=../data uv run python -m ml.retrieval build                 # → artifacts/indexes/manuals
uv sync --extra dense
uv run --extra dense python -m ml.retrieval embed                    # векторы config.embedder, на CPU десятки минут
uv run python -m ml.retrieval search --engine ЯМЗ-536 --query 'момент затяжки болтов головки цилиндров'
uv run python -m ml --retrieval-index artifacts/indexes/manuals run --vehicle M333YK47 --session S-0005 \
  --message 'Как проверить форсунки?'
```

Для `mode: dense` или `hybrid` и для `reranker` нужен `--extra dense`. Сравнение эмбеддеров: `embed --model Qwen/Qwen3-Embedding-0.6B` кладёт векторы рядом, выбор — полем `embedder` в конфиге. Смена `chunker` требует `build` и повторного `embed`, устаревшие векторы отклоняются.

Формат golden set описан в [evaluation/datasets](evaluation/datasets/README.md). `evaluation.retrieval` считает Hit@K, Recall@K, MRR, nDCG@K по физическим страницам и задержку поиска, `--run-file` готовит JSON прогона для журнала экспериментов. Для сравнения конфигураций передайте разные `--config`.

```bash
uv run python -m evaluation.retrieval --golden evaluation/datasets/retrieval.jsonl --split dev \
  --output evaluation/results/ret-01.json --run-file evaluation/results/ret-01-run.json \
  --run-id ret-01-bm25-page-<commit> --commit <commit> --dataset 'golden v1, dev'
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
| `POST /agent/tickets/confirm` | JSON с `vehicle_id`, `session_id`, `draft_id`, `draft_version`, `confirmed=true` и выбранным `urgency` |

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

Метрики генерации на эталонном наборе считает Oracle Context. Генератор получает эталонные страницы вопроса из `evaluation/datasets/customer_questions.jsonl` и две отвлекающие страницы тех же руководств. Поиск в замере не участвует.

| Метрика | Как считается |
|---|---|
| `answer_correctness` | Модель-судья сравнивает ответ с эталоном: верно — 1, частично — 0,5. Считается только с `--judge-model` |
| `numeric_accuracy` | Доля чисел эталона, которые есть в ответе. Номера страниц, рисунков и приложений не учитываются |
| `citation_accuracy` | Доля утверждений ответа со ссылкой на эталонную страницу, а не на отвлекающую |
| `unsupported_claim_rate` | Доля черновых утверждений, отклонённых `EvidenceVerifier`. До механика они не доходят |
| `abstention_accuracy` | Ответа нет ровно там, где эталон ожидает отказ |

Также выводятся покрытие эталонных страниц, доля показанных расхождений источников, причины отказов verifier, задержка p50/p95 и стоимость цикла. `pages.jsonl` и `manifest.json` выпускает `ml/ingestion/export_pages.py`. Без `--allow-external-evidence` страницы руководств не уходят провайдеру. `--dry-run` проверяет данные без вызовов LLM:

```bash
uv run python -m evaluation.generation.oracle_context --pages DATA/pages.jsonl --dry-run
LLM_API_KEY=... uv run python -m evaluation.generation.oracle_context --pages DATA/pages.jsonl --pipeline a \
  --split dev --allow-external-evidence --judge-model z-ai/glm-5.3 --output oracle-a.json --run-file run-a.json
```

`--pipeline` принимает `generator`, `reasoner`, `a` или `b`. Файл `--run-file` готов для `context.py finish --run-file`. `--cases-output` сохраняет ответы по кейсам с цитатами руководств, поэтому храните его только локально.

### Черновик и подтверждение

Вызов `Pipeline.run` с `command='draft_ticket'` сохраняет черновик только после пригодной аномалии и хотя бы одной подтверждённой операции. Для кнопки подтверждения используйте отдельный `Pipeline.confirm_ticket`. Передайте идентичность, пару ТС и сессии, `draft_id`, `draft_version` и явное `confirmed=True`. Ядро сверяет модель двигателя и свежую диагностику перед отправкой сохранённого payload. Поле подтверждения `urgency` принимает `low`, `medium` или `high`. Клиенты, которые опускают это поле, подтверждают приоритет черновика. Рекомендация в черновике берётся из `TICKET_URGENCY_BY_FAULT` для кода отклонения, для остальных кодов действует `DEFAULT_TICKET_URGENCY` (по умолчанию `medium`). Механик выбирает окончательный приоритет перед отправкой. Черновик действует 30 минут. Черновик без сохранённой модели требуется сформировать заново.

SQLite открывается при первой операции с заявкой. База `ARTIFACTS_DIR/agent-tickets.sqlite3` хранит черновики, статус отправки и квитанции. Повтор того же подтверждения возвращает сохранённый `ticket_id`. При `TICKET_OUTCOME_UNKNOWN` нужна ручная сверка с ТОиР. В этой версии повторная отправка для той же сессии и аномалии остаётся заблокированной даже после создания нового черновика или перезапуска. Порог `MIN_DIAGNOSTIC_CONFIDENCE=0.70` и `REPEAT_ON_WARNING=true` — конфигурируемые стартовые правила.

## Проверки

```bash
(cd app/backend && go test ./...)
uv run pytest -q
uv run --extra api pytest -q
```

Тесты проверяют tool calling и переформулирование текстового запроса, контекст уточнений, фильтрацию источников, защиту подтверждения и отправки. Для прогона с материалами заказчика используйте `M333YK47/S-0005` для аномалии, `P456AT178/S-0001` для нормы и `P456AT178/S-0003` для сбоя записи.

## Веб-приложение механика

Пользователь выбирает ТС и проверку, запрашивает диагностику и задаёт вопросы в чате. У подтверждённых утверждений есть ссылки на физические страницы PDF. Команда «Создать заявку» готовит черновик. Кнопка подтверждения отправляет его в ТОиР и показывает номер квитанции.

### Локальный учебный стенд

Нужны Python 3.10+, uv, Node.js 22 и npm. Запускайте из корня checkout.

```bash
uv sync --locked --extra api
npm --prefix app/frontend ci
npm --prefix app/frontend run build
uv run python -m demo.prepare --directory ./artifacts/training
uv run uvicorn demo.upstream:app --host 127.0.0.1 --port 8001
```

В другом терминале из той же папки запустите веб-сервис.

```bash
DATA_DIR=./artifacts/training ARTIFACTS_DIR=./artifacts/training/artifacts \
ML_API_URL=http://127.0.0.1:8001 MAINTENANCE_API_URL=http://127.0.0.1:8001 \
PLANNER_MODE=offline uv run uvicorn demo.app:create_app --factory --port 8080
```

Откройте `http://localhost:8080`. На ДЕМО-01 доступно отклонение с учебной инструкцией и созданием заявки. У ДЕМО-02 есть норма и непригодная запись. Каждая браузерная сессия проходит упражнение со своими квитанциями. Учебный корпус содержит одну инструкцию. Offline Planner предназначен для проверки интеграции и поддерживает ограниченный набор команд.

### Стенд с комплектом заказчика

Используйте [быстрый запуск через Docker](#быстрый-запуск-через-docker). Compose запускает индексатор, внутренний сервис агента, Go-бэкенд и выбранные моки.

По умолчанию Compose задаёт `WEB_RETRIEVAL_MODE=team`. Сервис `index` читает шесть PDF и готовит в общем томе `retrieval/` для поиска и `manuals.sqlite3` для читалки. Go передаёт запрос в Python-worker по существующему API. Для свободных вопросов worker ищет страницы по модели двигателя через BM25 с русским стеммингом. Для нормативов и работ использует проверенные записи со ссылками на конкретные страницы. Штатный профиль выполняется на CPU без ключа LLM.

Чтобы воспроизвести предыдущий поиск через SQLite, задайте в `.env` параметр `WEB_RETRIEVAL_MODE=legacy` и выполните `docker compose up -d index agent`.

Архив содержит 50 карточек ТС и 195 замеров. Список можно отфильтровать по двигателю. Поиск руководства использует модель двигателя из выбранной карточки. Ссылки в отчёте, чате и заявке открывают физическую страницу PDF внутри приложения. Индекс, контекст и квитанции хранятся в `ARTIFACTS_DIR`.

Числовой норматив появляется в ответе, когда удалось подтвердить строку, единицы и условия её применения. Черновик доступен после проверки хотя бы одной работы по руководству для выбранного двигателя. Если подходящую работу подтвердить не удалось, во вкладке «Заявка» появляется текст для ручной дефектовки с машиной, двигателем, проверкой и предполагаемым отклонением. При низкой уверенности приложение показывает полученную оценку и действующий порог. При несовместимости результата с топливом двигателя механик видит ошибку распознавания и предложение повторить замер или провести дефектовку.

Для проверки отклонения и заявки выберите `M333YK47 / S-0005`. У `P456AT178` замер `S-0001` демонстрирует норму, а `S-0003` — отказ записи. На машине с ЗМЗ-5234.10 спросите про момент затяжки крышек коренных подшипников и откройте источник. [Браузерная проверка на комплекте](tests/e2e/README.md)

### Переключение диагностики и ТОиР (F6)

На стенде настройки находятся в `.env` рядом с `compose.yaml`. После смены базового адреса примените конфигурацию командой `docker compose up -d web`. Встроенные сервисы доступны по именам `ml` и `maintenance`. В адресах внешних сервисов допускается префикс пути. Приложение добавляет к нему `/diagnose` или `/tickets`.

### Руководства в локальном профиле

`MANUAL_MANIFEST_PATH` задаёт каталог руководств для сборки индекса и просмотра PDF, `REVIEWED_EVIDENCE_PATH` — каталог проверенных выводов для локального агента. По умолчанию используются [редакции руководств](ml/ingestion/manifest.json) и [проверенные выводы](ml/ingestion/reviewed_evidence.json) из репозитория. Для новой редакции обновите каталог руководств и пересоберите индекс.

В каталоге выводов каждая запись связывает код неисправности или тему вопроса с утверждениями и физическими страницами. Для каждой страницы указан SHA-256 её текста после нормализации пробелов (`' '.join(text.split())`, UTF-8). Поля в фигурных скобках получают значения из первой захватывающей группы соответствующего выражения на этой странице. Если изменился текст или отсутствует часть многостраничной процедуры, локальный агент пропускает всю запись. Полнотекстовый поиск по выбранной модели остаётся доступен и для тем без проверенного норматива.
