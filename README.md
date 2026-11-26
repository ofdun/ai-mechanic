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
| Алексей | `Generator.generate(SearchPlan, EvidenceBundle, DiagnosticResult | None) -> TechnicalDraft`, передаётся как `generator=`. `Verifier.verify(TechnicalDraft, EvidenceBundle) -> VerifiedResponse` подключается через `verifier=` |
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

### Черновик и подтверждение

Вызов `Pipeline.run` с `command='draft_ticket'` сохраняет черновик только после пригодной аномалии и хотя бы одной подтверждённой операции. Для кнопки подтверждения используйте отдельный `Pipeline.confirm_ticket`. Передайте идентичность, пару ТС и сессии, `draft_id`, `draft_version` и явное `confirmed=True`. Ядро сверяет модель двигателя и свежую диагностику перед отправкой сохранённого payload. Черновик действует 30 минут. Черновик без сохранённой модели требуется сформировать заново.

SQLite открывается при первой операции с заявкой. База `ARTIFACTS_DIR/agent-tickets.sqlite3` хранит черновики, статус отправки и квитанции. Повтор того же подтверждения возвращает сохранённый `ticket_id`. При `TICKET_OUTCOME_UNKNOWN` нужна ручная сверка с ТОиР. В этой версии повторная отправка для той же сессии и аномалии остаётся заблокированной даже после создания нового черновика или перезапуска. Порог `MIN_DIAGNOSTIC_CONFIDENCE=0.70` и `REPEAT_ON_WARNING=true` — конфигурируемые стартовые правила.

## Проверки

```bash
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
npm --prefix frontend ci
npm --prefix frontend run build
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

Положите шесть PDF и архив `neurodvs_hackathon_mock_*.zip` в один каталог. Распакуйте архив и запустите Mock ML и Mock ТОиР по разделу «Запуск ядра». Из корня проекта выполните:

```bash
uv sync --locked --extra api
npm --prefix frontend ci
npm --prefix frontend run build
export DATA_DIR="/путь/к/комплекту"
export ARTIFACTS_DIR="$PWD/artifacts/fleet"
export ML_API_URL=http://127.0.0.1:8001
export MAINTENANCE_API_URL=http://127.0.0.1:8002
export PLANNER_MODE=local AGENT_ARCHITECTURE=baseline ALLOW_EXTERNAL_EVIDENCE=false
uv run python -m ml.manual_corpus --data-dir "$DATA_DIR" --output "$ARTIFACTS_DIR/manuals.sqlite3"
uv run uvicorn web.app:create_app --factory --port 8080
```

В этом профиле выбор инструмента, чтение страниц и проверка источников выполняются локально. Архив содержит 50 карточек ТС и 195 замеров. Список можно отфильтровать по двигателю. Поиск руководства использует модель двигателя из выбранной карточки. Ссылки в отчёте, чате и заявке открывают физическую страницу PDF внутри приложения. Индекс, контекст и квитанции хранятся в `ARTIFACTS_DIR`.

Числовой норматив появляется в ответе, когда удалось подтвердить строку, единицы и условия её применения. Черновик доступен после проверки хотя бы одной работы по руководству для выбранного двигателя. При нехватке подтверждений приложение сообщает об ограничении. На записях с помехами и низкой уверенностью механик видит исходную гипотезу и просьбу повторить замер.

Для проверки отклонения и заявки выберите `M333YK47 / S-0005`. У `P456AT178` замер `S-0001` демонстрирует норму, а `S-0003` — отказ записи. На машине с ЗМЗ-5234.10 спросите про момент затяжки крышек коренных подшипников и откройте источник. [Браузерная проверка на комплекте](tests/e2e/README.md)

Для systemd в [deploy](deploy) есть три юнита `mechanic-fleet*.service`. Они используют каталог приложения `/opt/ai-mechanic/fleet/app` с собранным `frontend/dist`, каталог `data` рядом с ним и распакованные сервисы в `/opt/ai-mechanic/fleet/mocks`. Пользователь `mechanic-demo` читает комплект и пишет в `/var/lib/mechanic-fleet`. Соберите индекс в этом каталоге по команде выше. Веб-сервис занимает порт 8080, заглушки слушают только loopback на портах 8101 и 8102. Для этой конфигурации используется `systemctl enable --now mechanic-fleet.service`. При необходимости адреса сервисов можно задать в `/etc/ai-mechanic/fleet.env`.

### Учебное размещение под systemd

Разместите исходники и `frontend/dist` в `/opt/ai-mechanic/current`. Создайте виртуальное окружение в `/opt/ai-mechanic/venv` и установите туда проект с зависимостями `.[api]`. Под системным пользователем `mechanic-demo` работают два юнита из [deploy](deploy). Скопируйте их в `/etc/systemd/system/`, выполните `systemctl daemon-reload` и `systemctl enable --now mechanic-demo-upstream.service mechanic-demo.service`. Приложение слушает порт 8080, учебный upstream — loopback-порт 8001. SQLite и учебный каталог сохраняются в `/var/lib/mechanic-demo`. Для проверки состояния используйте `systemctl status mechanic-demo` и `curl -fsS http://127.0.0.1:8080/health`.
