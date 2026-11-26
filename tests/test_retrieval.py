"""Retrieval tests on invented manuals. Customer PDFs are not committed to the public repository."""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.retrieval.__main__ import main as evaluation_cli
from evaluation.retrieval.metrics import GoldenQuestion, PageRef, evaluate, load_golden, run_record
from ml.contracts import AgentError, DocumentPage, ParsedDocument, SearchPlan
from ml.retrieval import RetrievalConfig, RetrievalEngine, build_index, load_documents
from ml.retrieval.__main__ import main as cli
from ml.retrieval.chunking import structural_chunks
from ml.retrieval.dense import load_dense
from ml.retrieval.engine import embed_index, read_index
from ml.retrieval.fusion import reciprocal_rank_fusion
from ml.retrieval.sparse import tokenizer

DIESEL = 'ДВС-100'
DIESEL_TURBO = 'ДВС-100Т'
PETROL = 'БДВ-7'


def diesel_manual() -> ParsedDocument:
    return ParsedDocument(document_id='dvs-100-re', manual='100.0001 РЭ', engine_models=[DIESEL, DIESEL_TURBO], pages=[
        DocumentPage(pdf_page=1, text=''),
        DocumentPage(pdf_page=2, printed_page='1', section='Топливная система',
                     text='Форсунку следует снять и проверить давление начала впрыскивания на стенде.'),
        DocumentPage(pdf_page=3, printed_page='2', section='Приложение А. Моменты затяжки',
                     text='Момент затяжки болтов крепления головки цилиндров 0,25 кН·м в три приёма.'),
        DocumentPage(pdf_page=4, printed_page='3', section='Турбокомпрессор', engine_models=[DIESEL_TURBO],
                     text='Проверить люфт ротора турбокомпрессора. Давление наддува по таблице.'),
        DocumentPage(pdf_page=5, printed_page='4', section='Система охлаждения',
                     text='Уровень охлаждающей жидкости проверять на холодном двигателе.'),
    ])


def petrol_manual() -> ParsedDocument:
    return ParsedDocument(document_id='bdv-7-rk', manual='7.0002 РК', engine_models=[PETROL], pages=[
        DocumentPage(pdf_page=10, section='Форсунки',
                     text='Форсунки форсунки форсунок: промыть форсунку, проверить форсунки и форсунку на стенде.'),
        DocumentPage(pdf_page=11, section='Зажигание', text='Зазор между электродами свечи зажигания.'),
    ])


@pytest.fixture
def parsed_dir(tmp_path):
    directory = tmp_path / 'parsed'
    directory.mkdir()
    for document in (diesel_manual(), petrol_manual()):
        (directory / f'{document.document_id}.json').write_text(document.model_dump_json(), encoding='utf-8')
    return directory


@pytest.fixture
def retriever(parsed_dir, tmp_path):
    build_index(parsed_dir, tmp_path / 'index')
    return RetrievalEngine.load(tmp_path / 'index', RetrievalConfig(top_k=3))


def plan(engine: str, query: str, *subqueries: str) -> SearchPlan:
    return SearchPlan(engine_model=engine, original_query=query, rewritten_query=query, subqueries=list(subqueries))


def test_analyzer_matches_inflections_and_decimal_separators():
    tokenize = tokenizer('stem')
    assert tokenize('форсунки') == tokenize('форсунку') == tokenize('Форсунка')
    assert tokenize('цилиндров') == tokenize('цилиндра')
    # Known Snowball limit: a fleeting vowel in the genitive plural keeps a separate stem.
    assert tokenize('форсунок') != tokenize('форсунки')
    assert tokenize('0,25 кН')[0] == tokenize('0.25')[0] == '0.25'
    assert tokenize('и в на') == []
    assert tokenizer('plain')('Форсунки') == ['форсунки']


def test_engine_filter_is_applied_before_top_k(retriever):
    evidence = retriever.search(plan(DIESEL, 'проверить форсунки'))
    assert evidence.items
    assert all(DIESEL in item.engine_models for item in evidence.items)
    assert evidence.items[0].evidence_id == 'dvs-100-re:2'
    assert evidence.items[0].manual == '100.0001 РЭ'
    assert retriever.search(plan('НЕИЗВЕСТНЫЙ-1', 'форсунки')).items == []


def test_page_applicability_narrows_document_models(retriever):
    query = 'люфт ротора турбокомпрессора'
    assert [item.pdf_page for item in retriever.search(plan(DIESEL_TURBO, query)).items][:1] == [4]
    assert 4 not in [item.pdf_page for item in retriever.search(plan(DIESEL, query)).items]


def test_numbers_and_section_titles_are_searchable(retriever):
    assert retriever.search(plan(DIESEL, 'момент затяжки головки цилиндров')).items[0].pdf_page == 3
    assert retriever.search(plan(DIESEL, '0.25')).items[0].pdf_page == 3
    assert retriever.search(plan(DIESEL, 'приложение А')).items[0].section.startswith('Приложение А')


def test_subqueries_are_fused_into_one_ranking(retriever):
    pages = [item.pdf_page for item in retriever.search(
        plan(DIESEL, 'давление впрыскивания форсунки', 'уровень охлаждающей жидкости')).items]
    assert {2, 5} <= set(pages)
    assert len(pages) == len(set(pages))


def test_empty_pages_are_not_indexed_but_keep_numbering(parsed_dir, tmp_path):
    manifest = build_index(parsed_dir, tmp_path / 'index')
    diesel = next(document for document in manifest.documents if document.document_id == 'dvs-100-re')
    assert (diesel.pages, diesel.chunks, diesel.empty_pages) == (5, 4, [1])
    assert manifest.engine_models == sorted([DIESEL, DIESEL_TURBO, PETROL])
    assert build_index(parsed_dir, tmp_path / 'again').corpus_version == manifest.corpus_version


def test_identical_copy_is_counted_once_and_conflicting_copy_fails(parsed_dir):
    (parsed_dir / 'copy-from-archive.json').write_text(diesel_manual().model_dump_json(), encoding='utf-8')
    assert [document.document_id for document in load_documents(parsed_dir)] == ['bdv-7-rk', 'dvs-100-re']
    changed = diesel_manual()
    changed.pages[1].text = 'Другой текст'
    (parsed_dir / 'copy-from-archive.json').write_text(changed.model_dump_json(), encoding='utf-8')
    with pytest.raises(AgentError, match='разным содержимым'):
        load_documents(parsed_dir)


def test_page_cannot_broaden_document_applicability(tmp_path):
    document = diesel_manual()
    document.pages[1].engine_models = [PETROL]
    (tmp_path / 'bad.json').write_text(document.model_dump_json(), encoding='utf-8')
    with pytest.raises(AgentError, match='расширяет применимость'):
        load_documents(tmp_path)


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(AgentError) as error:
        RetrievalEngine.load(tmp_path / 'missing')
    assert error.value.code == 'RAG_INDEX_UNAVAILABLE'


def test_metrics_on_known_rankings(retriever):
    questions = [
        GoldenQuestion(id='q1', engine_model=DIESEL, question='давление начала впрыскивания форсунки',
                       relevant=[PageRef(document_id='dvs-100-re', pdf_page=2)]),
        GoldenQuestion(id='q2', engine_model=DIESEL, question='момент затяжки головки',
                       relevant=[PageRef(document_id='dvs-100-re', pdf_page=3),
                                 PageRef(document_id='dvs-100-re', pdf_page=99)]),
        GoldenQuestion(id='q3', engine_model=PETROL, question='давление в шинах'),
    ]
    report = evaluate(retriever, questions, ks=(1, 3))
    assert (report.answerable, report.unanswerable) == (2, 1)
    assert report.hit_at_k[1] == 1.0
    assert report.recall_at_k[1] == 0.75
    assert report.mrr == 1.0
    assert report.ndcg_at_k[3] == round((1 + 1 / 1.6309297535714575) / 2, 4)

    record = run_record(report, run_id='ret-01-test', commit='abc', dataset='synthetic', k=3, hypothesis='RET-01')
    assert record['recall_at_k'] == report.recall_at_k[3] and record['models'] == 'none'
    with pytest.raises(AgentError):
        run_record(report, run_id='x', commit='abc', dataset='synthetic', k=10, hypothesis='RET-01')


def test_golden_file_rejects_duplicate_ids(tmp_path):
    line = json.dumps({'id': 'q1', 'engine_model': DIESEL, 'question': 'форсунки'}, ensure_ascii=False)
    path = tmp_path / 'golden.jsonl'
    path.write_text(f'{line}\n\n', encoding='utf-8')
    assert len(load_golden(path)) == 1
    path.write_text(f'{line}\n{line}\n', encoding='utf-8')
    with pytest.raises(AgentError, match='повторяются'):
        load_golden(path)


def test_cli_builds_searches_and_evaluates(parsed_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('DATA_DIR', str(tmp_path / 'no-data'))
    index = tmp_path / 'cli-index'
    monkeypatch.setattr('sys.argv', ['ml.retrieval', 'build', '--parsed-dir', str(parsed_dir), '--output', str(index)])
    assert cli() == 0
    assert json.loads(capsys.readouterr().out)['chunker'] == 'page'

    monkeypatch.setattr('sys.argv', ['ml.retrieval', 'search', '--index', str(index), '--engine', DIESEL,
                                     '--query', 'момент затяжки'])
    assert cli() == 0
    assert 'PDF 3, печатная 2' in capsys.readouterr().out

    golden = tmp_path / 'golden.jsonl'
    golden.write_text(json.dumps({'id': 'q1', 'engine_model': DIESEL, 'question': 'момент затяжки',
                                  'relevant': [{'document_id': 'dvs-100-re', 'pdf_page': 3}]},
                                 ensure_ascii=False) + '\n', encoding='utf-8')
    run_file = tmp_path / 'run.json'
    monkeypatch.setattr('sys.argv', ['evaluation.retrieval', '--index', str(index), '--golden', str(golden),
                                     '--run-file', str(run_file), '--run-id', 'ret-01-cli', '--commit', 'abc',
                                     '--dataset', 'synthetic'])
    assert evaluation_cli() == 0
    assert json.loads(capsys.readouterr().out)['mrr'] == 1.0
    record = json.loads(run_file.read_text(encoding='utf-8'))
    assert record['k'] == 5 and 'rrf_k=60' in record['configuration']


def test_config_file_overrides_defaults_and_rejects_unknown_keys(tmp_path):
    path = tmp_path / 'retrieval.yaml'
    path.write_text('top_k: 3\nanalyzer: plain\n', encoding='utf-8')
    config = RetrievalConfig.load(path)
    assert (config.top_k, config.analyzer, config.rrf_k) == (3, 'plain', 60)
    path.write_text('top_kk: 3\n', encoding='utf-8')
    with pytest.raises(AgentError):
        RetrievalConfig.load(path)
    assert RetrievalConfig.load(Path('configs/retrieval.yaml')) == RetrievalConfig()


def test_rrf_rewards_agreement_between_rankings():
    assert reciprocal_rank_fusion([[(7, 3.0), (8, 1.0)]]) == [(7, 3.0), (8, 1.0)]
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 5.0)], [(2, 4.0), (3, 1.0)], []], k=60)
    assert [index for index, _ in fused] == [2, 1, 3]


# --- structural chunking, dense, hybrid, reranking, parent context, control parser ---------------------------------

class HashEmbedder:
    """Deterministic bag-of-stems vectors: exercises dense plumbing without downloading a model."""

    name = 'test/hash-embedder'

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.tokenize = tokenizer('stem')

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self.tokenize(text):
            vector[sum(map(ord, token)) % self.dimension] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text):
        return self._vector(text)


class KeywordReranker:
    name = 'test/keyword-reranker'

    def __init__(self, keyword: str):
        self.keyword = keyword

    def score(self, query, passages):
        return [float(self.keyword in passage.casefold()) for passage in passages]


def engine_for(path, config, *, embedder=None, reranker=None):
    manifest, chunks, pages = read_index(path)
    dense = load_dense(path, embedder.name, [chunk.chunk_id for chunk in chunks]) if embedder else None
    return RetrievalEngine(manifest, chunks, pages, config, dense=dense, embedder=embedder, reranker=reranker)


def test_structural_chunks_stay_inside_a_page_and_keep_table_lines():
    table = 'Болт      Момент\nМ12       0,25\nМ14       0,32'
    long_paragraph = ' '.join(f'слово{number}' for number in range(50))
    document = ParsedDocument(document_id='doc', manual='M', engine_models=[DIESEL], pages=[
        DocumentPage(pdf_page=7, text=f'Заголовок раздела\n\n{table}\n\n{long_paragraph}'),
        DocumentPage(pdf_page=8, text='   '),
    ])
    chunks = structural_chunks(document, chunk_words=20, chunk_overlap=5)
    assert {chunk.pdf_page for chunk in chunks} == {7}
    assert {chunk.parent_id for chunk in chunks} == {'doc:7'}
    assert chunks[0].chunk_id == 'doc:7#1'
    assert table in chunks[0].text
    windows = [chunk.text.split() for chunk in chunks if chunk.text.startswith('слово')]
    assert windows[0][-5:] == windows[1][:5]
    assert all(len(words) <= 20 for words in windows)


def test_dense_and_hybrid_modes_respect_the_engine_filter(parsed_dir, tmp_path):
    index = tmp_path / 'index'
    build_index(parsed_dir, index)
    embedder = HashEmbedder()
    assert embed_index(index, embedder, RetrievalConfig()) == 6
    for mode in ('dense', 'hybrid'):
        engine = engine_for(index, RetrievalConfig(mode=mode, top_k=3, embedder=embedder.name), embedder=embedder)
        evidence = engine.search(plan(DIESEL, 'давление начала впрыскивания форсунки'))
        assert evidence.items[0].evidence_id == 'dvs-100-re:2'
        assert all(DIESEL in item.engine_models for item in evidence.items)
        assert engine.search(plan('НЕИЗВЕСТНЫЙ-1', 'форсунки')).items == []


def test_stale_vectors_are_rejected(parsed_dir, tmp_path):
    index = tmp_path / 'index'
    build_index(parsed_dir, index)
    embedder = HashEmbedder()
    embed_index(index, embedder, RetrievalConfig())
    build_index(parsed_dir, index, RetrievalConfig(chunker='structural', chunk_words=5, chunk_overlap=1))
    with pytest.raises(AgentError):
        load_dense(index, embedder.name, [chunk.chunk_id for chunk in read_index(index)[1]])


def test_dense_mode_requires_vectors(parsed_dir, tmp_path):
    build_index(parsed_dir, tmp_path / 'index')
    manifest, chunks, pages = read_index(tmp_path / 'index')
    with pytest.raises(AgentError):
        RetrievalEngine(manifest, chunks, pages, RetrievalConfig(mode='hybrid'))


def test_reranker_reorders_fused_candidates(parsed_dir, tmp_path):
    build_index(parsed_dir, tmp_path / 'index')
    config = RetrievalConfig(top_k=2, reranker='test/keyword-reranker')
    engine = engine_for(tmp_path / 'index', config, reranker=KeywordReranker('жидкост'))
    pages = [item.pdf_page for item in engine.search(plan(DIESEL, 'проверить давление уровень жидкости')).items]
    assert pages[0] == 5


def test_page_context_merges_chunks_adds_neighbours_and_respects_budget(parsed_dir, tmp_path):
    build_index(parsed_dir, tmp_path / 'index', RetrievalConfig(chunker='structural', chunk_words=5, chunk_overlap=1))
    page_engine = engine_for(tmp_path / 'index', RetrievalConfig(top_k=1, neighbor_pages=1))
    items = page_engine.search(plan(DIESEL, 'момент затяжки болтов головки цилиндров')).items
    assert [item.evidence_id for item in items] == ['dvs-100-re:3', 'dvs-100-re:2']
    assert 'в три приёма' in items[0].text

    cooling = page_engine.search(plan(DIESEL, 'уровень охлаждающей жидкости')).items
    assert [item.pdf_page for item in cooling] == [5]

    chunk_engine = engine_for(tmp_path / 'index', RetrievalConfig(top_k=3, context='chunk', max_evidence_words=6))
    chunk_items = chunk_engine.search(plan(DIESEL, 'момент затяжки болтов головки цилиндров')).items
    assert len(chunk_items) == 1 and '#' in chunk_items[0].evidence_id

    pages = [hit.chunk.chunk_id for hit in page_engine.rank_pages(plan(DIESEL, 'затяжки головки цилиндров'), 5)]
    assert len(pages) == len(set(pages)) and all('#' not in page for page in pages)


def write_export(directory: Path, records: list[dict], documents: list[dict]) -> Path:
    """Files in the format of ml/ingestion/export_pages.py: pages.jsonl next to manifest.json."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps({'documents': documents}, ensure_ascii=False), encoding='utf-8')
    pages = directory / 'pages.jsonl'
    pages.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records), encoding='utf-8')
    return pages


def test_ingestion_export_is_read_with_manual_names_from_the_manifest(tmp_path):
    record = {'document_id': 'dvs-100_re_2020', 'models': [DIESEL], 'printed_page': None, 'section': None}
    pages = write_export(tmp_path / 'data', [
        {**record, 'pdf_page': 1, 'text': ''},
        {**record, 'pdf_page': 2, 'printed_page': '1', 'section': 'Приложение А', 'text': 'Момент затяжки болтов'},
        {'document_id': 'bdv-7_rk_2025', 'models': [PETROL], 'pdf_page': 1, 'printed_page': None, 'section': None,
         'text': 'Зазор свечи зажигания'},
    ], [
        {'document_id': 'dvs-100_re_2020', 'title': 'Руководство по эксплуатации', 'designation': '100.0001 РЭ'},
        {'document_id': 'bdv-7_rk_2025', 'title': 'БДВ-7. Руководство по ремонту', 'designation': None},
    ])
    documents = {document.document_id: document for document in load_documents(pages)}
    assert documents['dvs-100_re_2020'].manual == '100.0001 РЭ'
    assert documents['bdv-7_rk_2025'].manual == 'БДВ-7. Руководство по ремонту'
    assert [page.pdf_page for page in documents['dvs-100_re_2020'].pages] == [1, 2]

    manifest = build_index(pages, tmp_path / 'index')
    assert next(item for item in manifest.documents if item.document_id == 'dvs-100_re_2020').empty_pages == [1]
    engine = RetrievalEngine.load(tmp_path / 'index')
    hit, = engine.search(plan(DIESEL, 'момент затяжки')).items
    assert (hit.evidence_id, hit.manual, hit.section) == ('dvs-100_re_2020:2', '100.0001 РЭ', 'Приложение А')

    unknown = write_export(tmp_path / 'other', [{**record, 'pdf_page': 1, 'text': 'x'}], [])
    with pytest.raises(AgentError, match='нет в каталоге'):
        load_documents(unknown)
    (tmp_path / 'other' / 'manifest.json').unlink()
    with pytest.raises(AgentError, match='каталог руководств'):
        load_documents(unknown)


def test_evaluation_counts_pages_not_chunks(parsed_dir, tmp_path):
    build_index(parsed_dir, tmp_path / 'index', RetrievalConfig(chunker='structural', chunk_words=5, chunk_overlap=1))
    engine = engine_for(tmp_path / 'index', RetrievalConfig())
    question = GoldenQuestion(id='q', engine_model=DIESEL, question='момент затяжки болтов головки цилиндров',
                              relevant=[PageRef(document_id='dvs-100-re', pdf_page=3)])
    report = evaluate(engine, [question], ks=(1, 3))
    retrieved = report.results[0].retrieved
    assert len(retrieved) == len({(ref.document_id, ref.pdf_page) for ref in retrieved})
    assert report.mrr == 1.0 and report.config.chunker == 'structural' and report.config.chunk_words == 5


def test_example_golden_set_is_valid():
    questions = load_golden(Path('evaluation/datasets/retrieval.example.jsonl'))
    assert [bool(question.relevant) for question in questions] == [True, True, False]
