import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip('pypdfium2')

from ml import manual_corpus
from ml.pdf_baseline import PdfPageRetriever


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
    assert manual_corpus.build_index(tmp_path, index, manifest) == 2
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
        manual_corpus.build_index(tmp_path, index, manifest)
    assert retriever.page('Мотор А', 'stand', 2) == page.model_copy(update={'engine_models': ['Мотор А']})
    assert retriever.page('Мотор А', 'changed', 2) is None
