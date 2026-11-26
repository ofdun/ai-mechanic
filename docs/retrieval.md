# Retrieval — контекст компонента

Владелец · Валерий. Ветка `feat/retrieval`. Задача — [#4 «RAG · собрать индекс и измерить базовый поиск»](https://github.com/ofdun/ai-mechanic/issues/4), разметка — [#7](https://github.com/ofdun/ai-mechanic/issues/7), прогоны — [журнал #9](https://github.com/ofdun/ai-mechanic/issues/9). Общие требования — в [общем контексте проекта](shared-project-context.md), разделы 7, 10–12.

Документ фиксирует состояние на 26.11.2026: что готово, какие контракты retrieval ожидает от соседних компонентов и что нужно учесть другим участникам. Измеренных результатов пока нет.

## 1. Состояние

| Что | Статус |
|---|---|
| BM25 по страницам (RET-01) | Готово, проверено на синтетических руководствах |
| Структурный chunking (RET-02, RET-03) | Готово |
| Dense на `BAAI/bge-m3` (RET-04) | Готово, модель проверена на CPU. Векторы для всего корпуса не считались |
| Hybrid BM25 + dense через RRF (RET-06, RET-07) | Готово |
| Cross-encoder reranker (RET-08) | Код готов, с реальной моделью не запускался |
| Родительские страницы и соседи (RET-09), бюджет контекста (RET-10) | Готово |
| Чтение выгрузки ingestion (`pages.jsonl` и `manifest.json`) | Готово, на реальной выгрузке не проверено |
| Подключение к агенту, baseline и архитектуры A/B | Готово через `build_retrieval_pipeline`. Сквозной прогон на моках оргов пройден (раздел 5) |
| Совпадение моделей с `vehicles.json` | Проверено: 5 моделей в 50 ТС, все есть в каталоге ingestion, расхождений нет |
| Оценка Hit@K, Recall@K, MRR, nDCG@K, latency и JSON прогона | Готово |
| Golden set | Есть у Вадима: `evaluation/datasets/customer_questions.jsonl`, 241 вход, 202 со страницами `passage` или `module`. `evaluation.retrieval` пока читает только упрощённый формат |
| Прогоны на корпусе заказчика | Не выполнялись: нет комплекта PDF на машине владельца |

Тесты: `uv run pytest -q`, 49 passed, 1 skipped. Retrieval-тесты используют вымышленные руководства, материалы заказчика в репозиторий не попадают.

## 2. Путь запроса

```text
SearchPlan
  → фильтр модели двигателя (до любого top-k)
  → BM25 и exact dense на каждый запрос плана: original, rewritten, subqueries
  → Reciprocal Rank Fusion
  → опциональный cross-encoder
  → родительские страницы, соседние страницы того же руководства
  → бюджет слов
  → EvidenceBundle
```

| Модуль `ml/retrieval` | Назначение |
|---|---|
| `chunking.py` | Загрузка корпуса, `page` (страница целиком) и `structural` (абзацы внутри страницы, строки таблиц сохраняются). Фрагмент не пересекает физическую страницу, у каждого есть `parent_id = document_id:pdf_page` |
| `sparse.py` | Токенизация русского текста (Snowball, `ё`→`е`, `0,25` = `0.25`, стоп-слова) и BM25 в памяти |
| `embeddings.py` | sentence-transformers. `BAAI/bge-m3` по умолчанию, пресеты Qwen3-Embedding 0.6B/4B/8B и `multilingual-e5-large-instruct`. Векторы нормализованы |
| `dense.py` | Точный косинус на NumPy. Векторы `dense/<model>.npy` привязаны к фрагментам индекса хешем их ID, устаревшие отклоняются |
| `fusion.py` | RRF, `k=60` по умолчанию |
| `reranker.py` | Cross-encoder, например `BAAI/bge-reranker-v2-m3`. Qwen3-Reranker требует отдельного адаптера |
| `engine.py` | `RetrievalConfig`, сборка и чтение индекса, `RetrievalEngine` |

Оценка лежит в `evaluation/retrieval`, параметры — в `configs/retrieval.yaml`, наборы и их формат — в `evaluation/datasets/README.md`.

## 3. Контракты

### Вход от ingestion

Retrieval читает выгрузку `ml/ingestion/export_pages.py` из ветки `codex/pdf-ingestion`:

- `DATA_DIR/pages.jsonl` — запись на каждую физическую страницу: `document_id`, `models`, `pdf_page` с 1, `printed_page`, `section`, `text`. Пустые страницы остаются в выгрузке, нумерация не сдвигается;
- `DATA_DIR/manifest.json` — каталог руководств. Из него берётся `manual` для ссылок и заявки: `designation`, а если его нет — `title`.

Также поддерживается каталог файлов `ParsedDocument` из `ml.contracts`, по одному JSON на руководство. Повторяющийся `document_id` с одинаковым содержимым учитывается один раз, с разным — ошибка сборки. Страница не может расширить применимость документа.

### Выход для агента

`RetrievalEngine` реализует `Retriever.search(SearchPlan) -> EvidenceBundle`.

- `evidence_id`: `document_id:pdf_page` для страницы, `document_id:pdf_page#n` для фрагмента при `context: chunk`.
- `engine_models` каждого элемента содержит модель из плана. Фильтр строгий: строка модели должна совпадать посимвольно (после NFC).
- `manual`, `pdf_page` и `section` пригодны для ссылки в ответе и заявке. Печатный номер хранится в индексе, но в `Evidence` его пока нет: в контракте нет такого поля.

### Конфигурация

`configs/retrieval.yaml`, ключевые поля:

| Поле | По умолчанию | Гипотеза |
|---|---|---|
| `mode` | `sparse` | RET-06, RET-07: `sparse`, `dense`, `hybrid` |
| `chunker`, `chunk_words`, `chunk_overlap` | `page`, 300, 50 | RET-01–RET-03. Размер в словах, на русское слово приходится ~1.5–2 токена |
| `embedder`, `embed_max_seq_length` | `BAAI/bge-m3`, 512 | RET-04, RET-05 |
| `per_query_k`, `rrf_k` | 30, 60 | RET-07 |
| `reranker`, `rerank_candidates` | `null`, 30 | RET-08 |
| `context`, `neighbor_pages` | `page`, 0 | RET-09 |
| `top_k`, `max_evidence_words` | 8, 6000 | RET-10. 6000 слов ≈ 10K токенов |

Смена `chunker`, `chunk_words` или `analyzer` требует `build` и повторного `embed`.

## 4. Запуск

```bash
DATA_DIR=../data uv run python -m ml.retrieval build             # → artifacts/indexes/manuals
uv sync --extra dense
uv run --extra dense python -m ml.retrieval embed                # BGE-M3 для всего корпуса
uv run python -m ml.retrieval search --engine ЯМЗ-536 --query 'момент затяжки болтов головки цилиндров'
uv run python -m ml --retrieval-index artifacts/indexes/manuals run --vehicle M333YK47 --session S-0005 \
  --message 'Как проверить форсунки?'
uv run python -m evaluation.retrieval --golden evaluation/datasets/retrieval.jsonl --split dev \
  --output evaluation/results/ret-01.json --run-file evaluation/results/ret-01-run.json \
  --run-id ret-01-bm25-page-<commit> --commit <commit> --dataset 'golden v1, dev'
```

`--run-file` готовит JSON в формате журнала #9. Релевантность считается по физическим страницам, фрагменты одной страницы схлопываются. Вопросы без ответа в корпусе в метрики ранжирования не входят.

## 5. Замеры и ресурсы

Проверено на ноутбуке без GPU: 8 логических ядер, 7.3 ГБ RAM.

- BGE-M3 загружается на CPU около 35 секунд, эмбеддинг одного запроса — около 250 мс. В плане до 7 запросов, поэтому `hybrid` добавляет до ~2 секунд на поиск.
- Разговорные формулировки dense находит без общих слов с текстом: «сколько антифриза» → охлаждающая жидкость, «тянуть головку» → момент затяжки. Это довод в пользу `hybrid`, его нужно подтвердить на golden set.
- Векторы для 1743 страниц на CPU — по оценке 15–30 минут. Не измерено.
- Модели скачиваются в кеш Hugging Face, веса BGE-M3 весят около 2.3 ГБ.
- **Сквозной прогон на моках оргов** (`neurodvs_hackathon_mock_2026-11-16`: Mock ML, Mock ТОиР, `vehicles.json`). Индекс собран из синтетической выгрузки в формате ingestion, но с реальными `document_id` и моделями, планировщик офлайн, архитектура baseline:
  - C1, M333YK47 / S-0005: Mock ML вернул `anomaly` F08 с уверенностью 0.73. Retrieval выдал только страницы ЯМЗ-236М2 (`yamz-236_re_2010:52`, `:53`), страница ЯМЗ-536 отсечена фильтром. Правило F08 дало операцию, черновик ссылается на «236-3902150-Б РЭ», стр. 52. Подтверждение создало `TKT-0001`, повторное вернуло ту же заявку;
  - C3, P456AT178 / S-0003: `check_failed` → `requires_remeasurement: true`, без черновика и заявки;
  - `build` предупредил о трёх моделях ЗМЗ без руководств в синтетическом индексе: проверка покрытия работает.
  Это проверка связки, а не качества поиска: текст страниц выдуманный.
- Snowball не сводит формы с беглой гласной: «форсунок» и «форсунки» дают разные основы. Вариант без стемминга сравнивается через `analyzer: plain`.

## 6. Что учесть участникам

### Роман · ingestion

- **Формат выгрузки принят как есть.** Retrieval читает `pages.jsonl` и `manifest.json` без изменений.
- **`manual` прямо в записях страниц.** Желательно добавить, чтобы ссылки не зависели от манифеста. Сейчас для двух руководств ЗМЗ `designation` пустой, в ссылку попадает `title`.
- **Строки `models` должны посимвольно совпадать с `vehicles[].engine_model`.** Сейчас совпадают: ЯМЗ-236М2, ЯМЗ-536, ЗМЗ-5234.10, ЗМЗ-40904.10, ЗМЗ-40905.10, все в NFC. При расхождении фильтр вернёт пустой результат без ошибки. `ml.retrieval build` предупреждает о моделях ТС без руководств, если видит `vehicles.json`.
- **`document_id` из `data/manifest.json` считаются каноническими.** Их использует golden set, и на один из них опирается правило F08 у Вадима.
- **Зависимость и путь каталога.** `pypdfium2` стоит в отдельном `requirements.txt`, его стоит добавить в `pyproject.toml`. В `.gitignore` основной ветки есть `/data/`, поэтому `data/manifest.json` при слиянии окажется в игнорируемой папке: нужен `git add -f` или другое место каталога.
- **Для улучшений RET-02 и ING-03** полезны разметка разделов по всем руководствам (сейчас только приложение А в 536 РЭ), таблицы со связью строки, колонки и единицы, OCR шести растровых страниц 236 РК.

### Вадим · агент и интеграция

- **В `feat/retrieval` влита `feat/ml-agent-bootstrap` до `d170b04`.** Конфликты были в README, `pyproject.toml` и `uv.lock`.
- **Изменения в общих файлах:**
  - в `pyproject.toml` добавлены `numpy`, `pyyaml`, `snowballstemmer`, extra `dense` (sentence-transformers и torch) и `pythonpath = ["."]` для pytest;
  - `ml/__main__.py` получил флаг `--retrieval-index`;
  - в `ml/pdf_baseline.py` выделен `build_retrieval_pipeline(settings, http, retriever)`, и `build_pdf_pipeline` вызывает его.
- **`DOCUMENT_ID` в `ml/pdf_baseline.py` заменён на `yamz-236_re_2010`** из каталога ingestion. Правило F08 ищет строку таблицы регулярным выражением под `pdftotext -layout`. Ingestion извлекает текст через pypdfium2, поэтому раскладка может не совпасть и C1 не дойдёт до заявки. Проверяется только на реальном PDF.
- **Planner влияет на поиск напрямую.** Retrieval объединяет `original_query`, `rewritten_query` и `subqueries` через RRF. Профили F01–F14 (AG-08) лучше добавлять в `rewritten_query` или `subqueries` — тогда они пройдут через тот же путь.
- **Задержка в `hybrid`:** каждый запрос плана — это эмбеддинг на CPU, около 250 мс. Если latency важна, ограничивать число подзапросов.

### Алексей · generation и verification

- **По умолчанию `context: page`:** в `EvidenceBundle` приходят целые физические страницы, а не фрагменты. `evidence_id` устойчив: `document_id:pdf_page`.
- **Бюджет по умолчанию** — 6000 слов (~10K токенов), до 8 страниц. Для Oracle Context можно задать другой бюджет в конфиге.
- **Многостраничные процедуры:** `neighbor_pages: 1` добавляет соседние страницы того же руководства, если они применимы к той же модели.

### Елисей · веб-сервис

- Retrieval не имеет собственного HTTP API, он работает внутри пайплайна агента.
- Для показа источника в UI есть `manual`, `pdf_page` и `section`. Печатного номера страницы в ответе агента пока нет.
- Индекс — файлы в `artifacts/indexes/manuals`. Для `hybrid` на сервере нужны extra `dense`, модель в кеше Hugging Face и, по оценке, около 3 ГБ RAM.

### Все · golden set

- **Набор уже есть:** `customer_questions.jsonl` от Вадима, 241 вход, split dev 170 / holdout 71. Страницы указаны в `expected_evidence` кодами документов (`y236re`, `y536rk`, …), а не `document_id` ingestion. Для retrieval-метрик нужно соответствие кодов и `document_id`. Holdout не трогать до финальной оценки.
- **Метрики считает Вадим.** Retrieval отдаёт индекс, конфигурации для сравнения и разбор промахов.

## 7. Открытые вопросы и блокеры

1. Нет комплекта PDF и `vehicles.json` на машине владельца retrieval.
2. Ни одна гипотеза RET не измерена: нет индекса по реальным PDF. `evaluation.retrieval` ещё не читает `customer_questions.jsonl`.
3. Нужно проверить правило F08 на тексте pypdfium2. На тексте с раскладкой `pdftotext -layout` оно срабатывает.
4. `git fetch` и `gh` на машине владельца не работают (SSL), ветка не опубликована, Issue #4 не обновлена.
5. Векторы и reranker на всём корпусе не замерены по времени и памяти.

## 8. Следующие шаги

1. Выгрузка ingestion → `build` → первый прогон RET-01 на dev-части `customer_questions.jsonl` (202 кейса со страницами).
2. `embed` BGE-M3 → сравнение `sparse`, `dense` и `hybrid` (RET-04, RET-06, RET-07).
3. `structural` с 256/512/1024 токенами, reranker, `neighbor_pages`, подбор `top_k` и бюджета (RET-02, RET-03, RET-08–RET-10).
4. По результатам — Qwen3-Embedding (RET-04, RET-05).
5. Каждый прогон — комментарием в журнал #9, итог — в #4.
