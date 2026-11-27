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
| Чтение выгрузки ingestion (`pages.jsonl`, `manifest.json`, каталог ParsedDocument) | Готово. Вадим собрал BM25 по актуальному ParsedDocument Романа: 6 руководств, 1743 страницы |
| Подключение к агенту, baseline и архитектуры A/B | Готово через `build_retrieval_pipeline`. Сквозной прогон на моках оргов пройден (раздел 5) |
| Совпадение моделей с `vehicles.json` | Проверено: 5 моделей в 50 ТС, все есть в каталоге ingestion, расхождений нет |
| Оценка Hit@K, Recall@K, MRR, nDCG@K, latency и JSON прогона | Готово |
| Golden set | `evaluation.retrieval` читает `customer_questions.jsonl` напрямую: 204 кейса поиска, 202 со страницами (dev 140, holdout 62), 2 `abstain`. Три режима запроса на одном индексе: `original`, `rewritten`, `combined` |
| Расширение запроса по коду неисправности F01–F14 (plan-expansion) | Готово, включено по умолчанию (`expand_anomaly`). На реальном корпусе C1 не перепроверен |
| Защита сборки | `build` не заменяет каталог с входными данными и каталог, который не является индексом |
| Прогоны на корпусе заказчика | Не выполнялись: нет комплекта PDF на машине владельца |

Тесты: `uv run pytest -q`, 60 passed, 1 skipped. Retrieval-тесты используют вымышленные руководства, материалы заказчика в репозиторий не попадают.

## 2. Путь запроса

```text
SearchPlan
  → фильтр модели двигателя (до любого top-k)
  → запросы плана: original, rewritten, subqueries; при диагнозе F01–F14 — словарь узла в rewritten и отдельным запросом
  → BM25 и exact dense на каждый запрос
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
| `fusion.py` | RRF с весами списков, `k=60` по умолчанию |
| `reranker.py` | Cross-encoder, например `BAAI/bge-reranker-v2-m3`, окна по словам для MaxP, кэш баллов. Qwen3-Reranker требует отдельного адаптера |
| `cloudflare.py` | Workers AI: `cf:@cf/baai/bge-m3` для векторов и `cf:@cf/baai/bge-reranker-base` для reranking |
| `openrouter.py` | OpenRouter: `or:baai/bge-m3` для векторов, `or:cohere/rerank-5-pro`, `or:mongodb/rerank-3` и другие модели `/rerank` для reranking |
| `expansion.py` | Запросы по кодам Mock ML F01–F14 словами руководств: узел, детали, «неисправности, причины и методы устранения» |
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
| `expand_anomaly` | `true` | plan-expansion: словарь узла по коду F01–F14 для диагностических планов |
| `per_query_k`, `rrf_k`, `sparse_weight`, `dense_weight` | 30, 60, 1, 1 | RET-07 |
| `reranker`, `rerank_candidates` | `null`, 30 | RET-08 |
| `rerank_passage_words`, `rerank_passage_overlap` | `null`, 40 | MaxP: страница оценивается лучшим окном, а не первыми 512 токенами |
| `rerank_query`, `rerank_context`, `rerank_prior_weight` | `rewritten`, `false`, 0 | Запрос реранкера, обозначение руководства перед текстом, вес порядка слияния рядом с реранкером |
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
# переформулировки Planner: один вызов LLM на вопрос, файл переиспользуется всеми прогонами
LLM_API_KEY=... uv run python -m evaluation.retrieval.rewrite --split dev --output evaluation/results/rewrites-dev.jsonl
uv run python -m evaluation.retrieval --golden evaluation/datasets/customer_questions.jsonl \
  --manifest ../data/manifest.json --rewrites evaluation/results/rewrites-dev.jsonl --split dev --query-mode all \
  --output evaluation/results/ret-01.json --run-file evaluation/results/ret-01-run.json \
  --run-id ret-01-bm25-page-<commit> --commit <commit> --dataset 'customer_questions 8e7a6e6, dev'
```

Режимы запроса: `original` — только вопрос, `rewritten` — переформулировка Planner и подзапросы, `combined` — всё вместе через RRF, `all` — три прогона на одном индексе (`run-<mode>.json`). Без файла переформулировок `rewritten` берёт `reference_rewrite` из набора, иначе исходный вопрос; число переформулировок пишется в отчёт. Коды документов набора (`y236re`, …) сопоставляются с каноническими `document_id` через имя PDF в манифесте, как в Oracle Context Алексея. Страницы `context` в релевантные не входят.

`--run-file` готовит JSON в формате журнала #9. Релевантность считается по физическим страницам, фрагменты одной страницы схлопываются. Вопросы без ответа в корпусе в метрики ранжирования не входят.

## 5. Замеры и ресурсы

### Dev, 140 вопросов с эталонными страницами (Вадим, `ret-comparison-dev-70b1b35`, M3 Pro)

| Вариант | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | p50 / p95, мс |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.593 | 0.764 | 0.843 | 0.900 | 0.691 | 0.19 / 0.60 |
| BGE-M3 | 0.479 | 0.714 | 0.793 | 0.864 | 0.609 | 24 / 56 |
| BM25 + BGE-M3, RRF | 0.586 | 0.771 | 0.843 | 0.907 | 0.699 | 25 / 28 |
| + `bge-reranker-v2-m3` | 0.736 | 0.900 | 0.921 | 0.971 | 0.822 | 2778 / 3090 |

Исходные запросы, страница целиком, без переформулировки. Контроль на машине владельца retrieval (те же шесть PDF, SHA-256 совпадают с манифестом, 1743 страницы): BM25 — Hit@1 0.600, @3 0.771, @5 0.857, @10 0.907, MRR 0.699, расхождение с Вадимом в один вопрос. В сквозном прогоне Алексея (`e2e-a-dev-6c02153`, sparse, top_k 3) эталонная страница есть в топ-30 у 139 из 143 вопросов, но в топ-3 — у 107: кандидаты находятся, теряется порядок.

### Подбор на OpenRouter: Hit@1 0.893 на dev, 0.839 на holdout

`configs/retrieval-sweep-openrouter.yaml`, гибрид BM25 + `or:baai/bge-m3`, 30 кандидатов в реранкер, исходные вопросы, страница целиком. Dev — 140 вопросов:

| Вариант | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Гибрид без реранкера | 0.579 | 0.793 | 0.857 | 0.914 | 0.703 |
| Гибрид, `sparse_weight: 2` | 0.607 | 0.800 | 0.843 | 0.914 | 0.714 |
| + `cohere/rerank-v3.5` | 0.821 | 0.943 | 0.964 | 0.993 | 0.888 |
| + `cohere/rerank-5-fast` | 0.829 | 0.950 | 0.964 | 0.986 | 0.889 |
| + `cohere/rerank-5-pro` | 0.879 | 0.943 | 0.971 | 0.993 | 0.916 |
| + `cohere/rerank-5-pro`, `rerank_context` | 0.886 | 0.971 | 0.986 | 0.993 | 0.928 |
| + `cohere/rerank-5-pro`, `rerank_prior_weight: 0.5` | 0.814 | 0.907 | 0.936 | 0.993 | 0.871 |
| + `mongodb/rerank-2.5` | 0.893 | 0.971 | 0.979 | 0.993 | 0.934 |
| **+ `mongodb/rerank-2.5`, `rerank_context`** | **0.893** | **0.971** | **0.986** | 0.993 | 0.933 |

Holdout, 62 вопроса, один прогон победителя: Hit@1 **0.839**, @3 0.935, @5 0.984, @10 0.984, MRR 0.896. Гибрид без реранкера на тех же вопросах — Hit@1 0.581. У 4 из 10 промахов первой стоит соседняя с эталонной страница.

- Хостинговые реранкеры обходят `bge-reranker-v2-m3` (0.736) на 9–16 вопросов: страница входит целиком, без обрезки на 512 токенах.
- Обозначение руководства перед текстом поднимает Hit@3–5: оно помогает различать РЭ и РК одного двигателя.
- Подстраховка порядком слияния вредит: хостинговый реранкер уверен, а порядок слияния тянет вниз слабые лексические совпадения.
- `combined` на этом наборе равен `original`: в `customer_questions.jsonl` нет `reference_rewrite` для dev, а файла переформулировок Planner в прогоне не было.
- `mongodb/rerank-3` не уложился в таймаут 60 с на 30 страниц.
- Прогон 140 вопросов занимает около 4 минут: последовательные вызовы, примерно 1.8 с на вопрос вместе с эмбеддингом запроса.

### Как поднять Hit@1 до 0.80

Цель — +9 вопросов на первом месте (103 → 112 из 140). Рычаги в `configs/retrieval-sweep.yaml`, подбор одной командой:

```bash
uv run --extra dense python -m evaluation.retrieval.sweep --manifest ../data/manifest.json --split dev \
  --output evaluation/results/sweep-dev.json --misses evaluation/results/sweep-dev-misses.jsonl
```

1. **MaxP.** Cross-encoder видит 512 токенов, страница длиннее, нужная строка таблицы обрезается. `rerank_passage_words` режет страницу на окна, балл страницы — лучшее окно.
2. **Вес BM25 в RRF.** В одиночку BM25 сильнее dense, а при равных весах гибрид терял его первые места (Q050, Q111, Q148).
3. **Порядок слияния как подстраховка.** `rerank_prior_weight` объединяет порядок реранкера и слияния, если cross-encoder не уверен.
4. **Обозначение руководства перед текстом** (`rerank_context`) и запрос реранкера из вопроса и переформулировки (`rerank_query: both`).
5. **Переформулировка Planner** (`--rewrites`): в пилоте Вадима Hit@5 BM25 вырос с 26/32 до 29/32.

Баллы реранкера кэшируются в `artifacts/cache/rerank`, поэтому новые варианты считают только новые пары. Диагностика промахов: `hit_at_1_with_neighbour` (первая страница — соседняя с эталонной) и `miss_top1_same_manual` отделяют ошибку ранжирования от вопроса к разметке. Подбирать только на dev, победителя один раз проверить на holdout. Задержку выбранной конфигурации мерить `evaluation.retrieval`, в подборе её нет из-за кэша.

### Cloudflare Workers AI

- `@cf/baai/bge-m3` — те же векторы BGE-M3 без torch и 3 ГБ RAM на сервере: `embedder: cf:@cf/baai/bge-m3`, затем `python -m ml.retrieval embed --model cf:@cf/baai/bge-m3`. Векторы Cloudflare и локальной модели хранятся раздельно и не смешиваются.
- Из реранкеров в Workers AI есть только `@cf/baai/bge-reranker-base`. Прирост Hit@1 выше дал `bge-reranker-v2-m3`, его там нет. `configs/retrieval-sweep-cloudflare.yaml` сравнивает оба.
- Эмбеддинг корпуса и reranking отправляют текст руководств в Cloudflare. Без `ALLOW_EXTERNAL_EVIDENCE=true` они отказывают с `RAG_EXTERNAL_DISABLED`. Ключи: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`.

### OpenRouter

- `baai/bge-m3` через `/embeddings` — те же векторы BGE-M3, $0.01 за 1M токенов, вход до 8194 токенов: `embedder: or:baai/bge-m3`, затем `python -m ml.retrieval embed --model or:baai/bge-m3`.
- Реранкеры через `/rerank`: Cohere `rerank-v3.5` (4k токенов), `rerank-4-*`, `rerank-5-pro` и `rerank-5-fast`, `mongodb/rerank-2.5` и `rerank-3` (бывшие Voyage), всё с контекстом 32k. Страница помещается целиком, окна MaxP не нужны. `bge-reranker-v2-m3` там нет. Qwen3-Reranker в сетку не входит по решению команды.
- Ключ `OPENROUTER_API_KEY`; если он пуст, берётся `LLM_API_KEY` при `LLM_BASE_URL` на OpenRouter. Текст руководств уходит OpenRouter и провайдеру модели, нужен `ALLOW_EXTERNAL_EVIDENCE=true`. На 429 и 5xx — до трёх повторов с паузой 1, 2, 4 с.
- Сетка `configs/retrieval-sweep-openrouter.yaml`: гибрид на векторах OpenRouter, пять реранкеров, затем подстраховка, заголовок и переформулировка поверх лучшего.

### Ресурсы

Проверено на ноутбуке без GPU: 8 логических ядер, 7.3 ГБ RAM.

- BGE-M3 загружается на CPU около 35 секунд, эмбеддинг одного запроса — около 250 мс. В плане до 7 запросов, поэтому `hybrid` добавляет до ~2 секунд на поиск.
- Разговорные формулировки dense находит без общих слов с текстом: «сколько антифриза» → охлаждающая жидкость, «тянуть головку» → момент затяжки. Это довод в пользу `hybrid`, его нужно подтвердить на golden set.
- Векторы для 1743 страниц на CPU — по оценке 15–30 минут. Не измерено.
- Модели скачиваются в кеш Hugging Face, веса BGE-M3 весят около 2.3 ГБ.
- **Сквозной прогон на моках оргов** (`neurodvs_hackathon_mock_2026-11-16`: Mock ML, Mock ТОиР, `vehicles.json`). Индекс собран из синтетической выгрузки в формате ingestion, но с реальными `document_id` и моделями, планировщик офлайн, архитектура baseline:
  - C1, M333YK47 / S-0005: Mock ML вернул `anomaly` F08 с уверенностью 0.73. Retrieval выдал только страницы ЯМЗ-236М2 (`yamz-236_re_2010:52`, `:53`, тогда ещё старый ID), страница ЯМЗ-536 отсечена фильтром. Правило F08 дало операцию, черновик ссылается на «236-3902150-Б РЭ», стр. 52. Подтверждение создало `TKT-0001`, повторное вернуло ту же заявку;
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
- **`DOCUMENT_ID` в `ml/pdf_baseline.py` — канонический `yamz-236-operations-2010`** из актуального `data/manifest.json`, тот же ID использует SQLite-индекс веба.
- **Правило F08 не зависит от раскладки.** Оно находит строку «Нарушение регулировки или засорение форсунки → Отрегулировать форсунку и, если необходимо, промыть и прочистить её» в двух формах: колонки построчно (`pdftotext -layout`) и ячейки целиком (pypdfium2). Пробелы и переносы строк не важны. `ml.retrieval build` предупреждает, если на стр. 52 строка не найдена.
- **Диагностический план F08 находит стр. 52.** Переформулировка «<название неисправности>. Возможные причины…» одинаково похожа на все таблицы неисправностей, поэтому retrieval добавляет словарь узла по коду F01–F14 к `rewritten_query` и отдельным запросом.
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

- **Набор уже есть:** `customer_questions.jsonl` от Вадима, 241 вход, split dev 170 / holdout 71. `evaluation.retrieval` читает его напрямую, коды документов сопоставляются через имя PDF в манифесте. Holdout не трогать до финальной оценки.
- **Метрики считает Вадим.** Retrieval отдаёт индекс, конфигурации для сравнения и разбор промахов.

## 7. Открытые вопросы и блокеры

1. Корпус есть у владельца retrieval, но без GPU: `bge-reranker-v2-m3` на CPU — около часа на вариант подбора. Хостинговые реранкеры OpenRouter снимают это ограничение, если команда согласна отправлять им текст руководств.
2. Цель Hit@1 0.80 достигнута с `or:mongodb/rerank-2.5` и `rerank_context: true`: 0.893 на dev, 0.839 на holdout. Для веба нужно решить, принимает ли команда отправку текста руководств в OpenRouter и задержку около 1–2 с на запрос. Локальный `bge-reranker-v2-m3` даёт 0.736.
3. Словарь F01–F14 в retrieval дублирует профили `ml.generation.profiles` Алексея. После слияния веток свести к одному источнику.
4. Векторы и reranker на всём корпусе не замерены по времени и памяти.

## 8. Следующие шаги

1. `evaluation.retrieval.sweep` по `configs/retrieval-sweep.yaml` на dev, разбор промахов Hit@1, победитель — на holdout.
2. Тот же подбор на OpenRouter (`configs/retrieval-sweep-openrouter.yaml`) или Cloudflare (`configs/retrieval-sweep-cloudflare.yaml`) и решение, где работает реранкер.
3. `structural` с 256/512/1024 токенами, reranker, `neighbor_pages`, подбор `top_k` и бюджета (RET-02, RET-03, RET-08–RET-10).
4. По результатам — Qwen3-Embedding (RET-04, RET-05).
5. Каждый прогон — комментарием в журнал #9, итог — в #4.
