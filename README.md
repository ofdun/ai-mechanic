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

Задайте `DATA_DIR` с `vehicles.json` или архивом `neurodvs_hackathon_mock_*.zip`. Можно передать конкретный путь к распакованному JSON через `VEHICLES_PATH`. Клиент моков использует `ML_API_URL` и `MAINTENANCE_API_URL`.

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

### Голос

```bash
uv sync --extra asr
PLANNER_MODE=offline uv run --extra asr python -m ml run \
  --vehicle M333YK47 --session S-0005 --audio /path/to/command.wav
```

Whisper извлекает речь локально на CPU. Первый запуск скачивает модель из `ASR_MODEL` в `ARTIFACTS_DIR/asr`. `ASR_LANGUAGE=ru` задаёт язык. Пустая транскрипция возвращает ошибку. Распознанная строка передаётся тому же Planner и возвращается как `transcript`. Входной лимит — 20 МиБ.

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

Веб-сервис управляет сроком жизни `httpx.Client` и `Pipeline`. `actor_id` передаётся из проверенного контекста пользователя. Для голосового сообщения передайте `AgentRequest.audio=AudioInput(content=...)` с байтами файла. `AgentError` содержит машиночитаемые `code` и `retryable`.

| Кто подключается | Точка входа |
|---|---|
| Роман | `ParsedDocument` с физическими страницами `DocumentPage` и применимыми моделями |
| Валерий | `Retriever.search(SearchPlan) -> EvidenceBundle`, передаётся как `retriever=` |
| Алексей | `Generator.generate(SearchPlan, EvidenceBundle, DiagnosticResult | None) -> TechnicalDraft`, передаётся как `generator=`. `Verifier.verify(TechnicalDraft, EvidenceBundle) -> VerifiedResponse` подключается через `verifier=` |
| Елисей | `Pipeline.run(AgentRequest) -> AgentResponse` и `Pipeline.confirm_ticket(TicketConfirmation) -> TicketReceipt` |

Пока Retriever или Generator не переданы, пайплайн возвращает вычисленный поисковый план и ограничение. Подключённые фрагменты проходят проверку модели и идентификаторов источников. Без Verifier временная проверка пропускает только дословные утверждения из применимого доказательства. С Verifier принимаются только подтверждённые claims из исходного черновика. Это стык для будущей технической верификации норм и применимости процедур.

### Черновик и подтверждение

Вызов `Pipeline.run` с `command='draft_ticket'` сохраняет черновик только после пригодной аномалии и хотя бы одной подтверждённой операции. Для кнопки подтверждения используйте отдельный `Pipeline.confirm_ticket`. Передайте идентичность, пару ТС и сессии, `draft_id`, `draft_version` и явное `confirmed=True`. Ядро заново сверяет диагностику и отправляет сохранённый payload в ТОиР. Черновик действует 30 минут.

Одна SQLite-база в `ARTIFACTS_DIR/agent-tickets.sqlite3` хранит черновики, статус отправки и квитанции. Повтор того же подтверждения возвращает сохранённый `ticket_id`. Новый черновик для уже отправляемой или созданной заявки не создаёт дубль. При `TICKET_OUTCOME_UNKNOWN` нужна сверка с ТОиР перед новой отправкой. Порог `MIN_DIAGNOSTIC_CONFIDENCE=0.70` и `REPEAT_ON_WARNING=true` — конфигурируемые стартовые правила.

## Проверки

```bash
uv run pytest -q
```

Тесты проверяют протокол tool calling и переписывание запроса, путь транскрипции, статусы, фильтрацию источников, запрет отправки без подтверждения, повторы и неизвестный исход. Они используют небольшие фикстуры и HTTPX MockTransport. Для прогона с материалами заказчика удобно взять `M333YK47/S-0005` для аномалии, `P456AT178/S-0001` для нормы и `P456AT178/S-0003` для сбоя записи.
