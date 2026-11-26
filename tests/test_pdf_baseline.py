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
    assert [item.pdf_page for item in evidence.items] == [FIRST_PAGE]
    assert retriever.search(plan.model_copy(update={'engine_model': 'ЯМЗ-536'})).items == []

    operations = ReviewedOperations()
    approved, = operations.operations(evidence)
    assert 'если подтверждено нарушение регулировки или засорение форсунки' in approved.text.lower()
    assert 'если необходимо' in approved.text
    shortened = approved.model_copy(update={'text': 'Отрегулировать форсунку.'})
    assert operations.verify(TechnicalDraft(claims=[shortened, approved]), evidence).claims == [approved]
