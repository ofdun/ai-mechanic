import re

from ml.contracts import SearchPlan, TechnicalDraft
from ml.pdf_baseline import ENGINE, FIRST_PAGE, PdfPageRetriever, ReviewedOperations, build_index


def test_page_index_filters_engine_and_keeps_the_whole_approved_operation(monkeypatch, tmp_path):
    monkeypatch.setattr('ml.pdf_baseline.read_pages', lambda _: [
        'Нарушение регулировки или   Отрегулировать форсунку и, если\n'
        'засорение форсунки         необходимо, промыть и прочистить\n'
        '                           ее\n',
        'Давление топлива',
        'Система охлаждения',
    ])
    path = tmp_path / 'manuals.sqlite3'
    build_index(tmp_path, path)
    retriever = PdfPageRetriever(path)
    plan = SearchPlan(engine_model=ENGINE, original_query='Форсунки', rewritten_query='Форсунки', anomaly_type='F08')
    evidence = retriever.search(plan)
    assert [item.pdf_page for item in evidence.items] == [FIRST_PAGE, FIRST_PAGE + 1]
    assert retriever.search(plan.model_copy(update={'engine_model': 'ЯМЗ-536'})).items == []
    assert [item.pdf_page for item in retriever.figure_pages(ENGINE, re.compile('охлажден'))] == [FIRST_PAGE + 2]
    assert retriever.figure_pages('ЯМЗ-536', re.compile('охлажден')) == []

    operations = ReviewedOperations()
    approved, = operations.operations(evidence)
    assert approved.evidence_ids == [evidence.items[0].evidence_id]
    assert 'если подтверждено нарушение регулировки или засорение форсунки' in approved.text.lower()
    assert 'если необходимо' in approved.text
    shortened = approved.model_copy(update={'text': 'Отрегулировать форсунку.'})
    assert operations.verify(TechnicalDraft(claims=[shortened, approved]), evidence).claims == [approved]


def test_inflected_component_matches_only_the_selected_engines_manual(tmp_path):
    import sqlite3
    from contextlib import closing

    path = tmp_path / 'manuals.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE VIRTUAL TABLE manual_pages USING fts5('
                   'document_id UNINDEXED, manual UNINDEXED, engine UNINDEXED, pdf_page UNINDEXED, text)')
        db.executemany('INSERT INTO manual_pages VALUES (?, ?, ?, ?, ?)', [
            ('a', 'Инструкция А', 'Мотор А', 3, 'Схема крепления кронштейнов.'),
            ('b', 'Инструкция Б', 'Мотор Б', 8, 'Как снять кронштейны?'),
        ])
    plan = SearchPlan(engine_model='Мотор А', original_query='Как снять кронштейны?', rewritten_query='Кронштейны')
    result = PdfPageRetriever(path).search(plan)
    assert [(item.document_id, item.pdf_page) for item in result.items] == [('a', 3)]


def test_reviewed_row_is_found_in_both_extraction_layouts():
    from ml.pdf_baseline import injector_row

    layout = ('Нарушение регулировки или   Отрегулировать форсунку и, если\n'
              'засорение форсунки         необходимо, промыть и прочистить\n'
              '                           ее\n')
    # pypdfium2 emits whole cells in reading order, with its own line breaks.
    cells = ('Двигатель дымит\nНарушение регулировки или\nзасорение форсунки\n'
             'Отрегулировать форсунку и, если\nнеобходимо, промыть и прочистить\nеё\nСледующая строка')
    expected = ('Если подтверждено нарушение регулировки или засорение форсунки, '
                'отрегулировать форсунку и, если необходимо, промыть и прочистить ')
    assert injector_row(layout) == expected + 'ее.'
    assert injector_row(cells) == expected + 'её.'
    assert injector_row(cells.replace('засорение форсунки', 'износ форсунки')) is None
    assert injector_row('Отрегулировать форсунку и, если необходимо, промыть и прочистить еще раз') is None


def test_c1_operation_reaches_the_draft_through_the_retrieval_engine(tmp_path):
    from ml.contracts import DiagnosticResult, DocumentPage, ParsedDocument
    from ml.pdf_baseline import DOCUMENT_ID, MANUAL
    from ml.retrieval import RetrievalConfig, RetrievalEngine, build_index

    # Generic troubleshooting pages agree with the diagnostic rewrite, the reviewed row shares no words with it.
    filler = [DocumentPage(pdf_page=page, text=f'Возможные причины и проверки узла {page}, процедуры обслуживания.')
              for page in range(40, 52)]
    table = DocumentPage(pdf_page=52, text=(
        'Двигатель не развивает мощности, дымный выпуск\nНарушение регулировки или\nзасорение форсунки\n'
        'Отрегулировать форсунку и, если\nнеобходимо, промыть и прочистить\nее'))
    manuals = tmp_path / 'parsed'
    manuals.mkdir()
    (manuals / f'{DOCUMENT_ID}.json').write_text(ParsedDocument(
        document_id=DOCUMENT_ID, manual=MANUAL, engine_models=[ENGINE], pages=[*filler, table],
    ).model_dump_json(), encoding='utf-8')
    build_index(manuals, tmp_path / 'index')
    engine = RetrievalEngine.load(tmp_path / 'index')
    diagnostic = DiagnosticResult(
        vehicle_id='M333YK47', session_id='S-0005', status='anomaly', anomaly_type='F08',
        anomaly_title='Дизельная топливоподача', anomaly_score=0.73, confidence=0.73, signal_quality='good',
        description='Неравномерная работа на холостом ходу', model_version='mock')
    plan = SearchPlan(engine_model=ENGINE, original_query='Проведи диагностику', anomaly_type='F08',
                      rewritten_query='Дизельная топливоподача. Возможные причины, документированные проверки и процедуры.')
    unexpanded = RetrievalEngine.load(tmp_path / 'index', RetrievalConfig(expand_anomaly=False))
    assert f'{DOCUMENT_ID}:52' not in [item.evidence_id for item in unexpanded.search(plan).items]
    evidence = engine.search(plan)
    assert evidence.items[0].evidence_id == f'{DOCUMENT_ID}:52'
    operations = ReviewedOperations()
    draft = operations.generate(plan, evidence, diagnostic)
    assert [claim.evidence_ids for claim in draft.claims] == [[f'{DOCUMENT_ID}:52']]
    assert operations.verify(draft, evidence).claims == draft.claims
