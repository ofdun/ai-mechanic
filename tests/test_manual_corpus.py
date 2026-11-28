import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pypdfium2')

from ml import manual_corpus
from ml.contracts import AgentError, SearchPlan
from ml.pdf_baseline import PdfPageRetriever
from ml.retrieval import RetrievalEngine


def test_shared_manual_index_is_scoped_numbered_and_rolls_back_bad_rebuild(tmp_path):
    pdf = tmp_path / 'stand.pdf'
    pdf.write_bytes((Path(__file__).resolve().parents[1] / 'demo' / 'training-stand.pdf').read_bytes())
    guide = {
        'file': pdf.name, 'document_id': 'stand', 'title': 'Стенд', 'models': ['Мотор А', 'Мотор Б'],
        'expected_pages': 2, 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'documents': [guide]}))
    assert manual_corpus.document_catalog(manifest) == {'stand': guide}
    index = tmp_path / 'manuals.sqlite3'
    team = tmp_path / 'retrieval'
    assert manual_corpus.build_index(tmp_path, index, manifest, retrieval_dir=team) == 2
    plan = SearchPlan(engine_model='Мотор Б', original_query='маркировку', rewritten_query='маркировку')
    matches = RetrievalEngine.load(team).search(plan).items
    assert any((hit.document_id, hit.pdf_page) == ('stand', 2) for hit in matches)
    retriever = PdfPageRetriever(index)
    for engine in guide['models']:
        page = retriever.page(engine, 'stand', 2)
        assert (page.document_id, page.pdf_page, page.engine_models) == ('stand', 2, [engine])
        assert 'Сверить маркировку' in page.text
    assert retriever.page('Мотор В', 'stand', 2) is None
    assert retriever.page('Мотор А', 'stand', 3) is None
    second = tmp_path / 'second.pdf'
    second.write_bytes(b'other edition')
    manifest.write_text(json.dumps({'documents': [guide, {**guide, 'file': second.name, 'document_id': 'changed'}]}))
    with pytest.raises(ValueError, match='SHA-256'):
        manual_corpus.build_index(tmp_path, index, manifest, retrieval_dir=team)
    assert retriever.page('Мотор А', 'stand', 2) == page.model_copy(update={'engine_models': ['Мотор А']})
    assert retriever.page('Мотор А', 'changed', 2) is None
    assert RetrievalEngine.load(team).search(plan).items == matches


def test_web_hybrid_build_reuses_vectors_and_preserves_index_on_failed_refresh(tmp_path, monkeypatch):
    pdf = Path(__file__).resolve().parents[1] / 'demo' / 'training-stand.pdf'
    local_pdf = tmp_path / pdf.name
    local_pdf.write_bytes(pdf.read_bytes())
    guide = {
        'file': local_pdf.name, 'document_id': 'stand', 'title': 'Стенд', 'models': ['Мотор А'],
        'expected_pages': 2, 'sha256': hashlib.sha256(local_pdf.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'documents': [guide]}))
    config = manual_corpus.RetrievalConfig(mode='hybrid', embedder='test')
    index = tmp_path / 'manuals.sqlite3'
    team = tmp_path / 'retrieval'
    calls = []

    class Embedder:
        name = config.embedder

        def embed_documents(self, texts):
            calls.append(list(texts))
            return np.ones((len(texts), 1), dtype=np.float32)

    monkeypatch.setattr('ml.retrieval.engine.load_embedder', lambda _: Embedder())
    for _ in range(2):
        assert manual_corpus.build_index(tmp_path, index, manifest, retrieval_dir=team, retrieval_config=config) == 2
        assert RetrievalEngine.load(team, config).dense.vectors.shape == (2, 1)
    assert len(calls) == 1

    def unavailable(_):
        raise AgentError('RAG_BACKEND_UNAVAILABLE', 'Embedding provider unavailable')

    monkeypatch.setattr('ml.retrieval.engine.load_embedder', unavailable)
    guide['title'] = 'Новая редакция'
    manifest.write_text(json.dumps({'documents': [guide]}))
    with pytest.raises(AgentError, match='Embedding provider unavailable'):
        manual_corpus.build_index(tmp_path, index, manifest, retrieval_dir=team, retrieval_config=config)
    assert PdfPageRetriever(index).page('Мотор А', 'stand', 2).manual == 'Стенд'
    monkeypatch.setattr('ml.retrieval.engine.load_embedder', lambda _: Embedder())
    assert RetrievalEngine.load(team, config).manifest.documents[0].manual == 'Стенд'
    assert manual_corpus.build_index(tmp_path, index, manifest, retrieval_dir=team, retrieval_config=config) == 2
    assert len(calls) == 2
    assert RetrievalEngine.load(team, config).manifest.documents[0].manual == 'Новая редакция'
