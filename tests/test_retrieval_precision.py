"""Precision levers of retrieval: weighted fusion, MaxP reranking, fused prior, Cloudflare backends, sweep."""

import json

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from ml.config import Settings
from ml.contracts import AgentError, DocumentPage, ParsedDocument, SearchPlan
from ml.retrieval import RetrievalConfig, build_index
from ml.retrieval.cloudflare import CloudflareClient, CloudflareEmbedder, CloudflareReranker, is_cloudflare
from ml.retrieval.engine import RetrievalEngine, read_index, rerank_query
from ml.retrieval.fusion import reciprocal_rank_fusion
from ml.retrieval.reranker import CachedReranker, windows

ENGINE = 'ДВС-100'


class Spy:
    """Scores passages by a keyword and records what the cross-encoder saw."""

    name = 'test/spy-reranker'

    def __init__(self, keyword: str):
        self.keyword, self.calls = keyword, []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [float(self.keyword in passage.casefold()) for passage in passages]


def long_manual() -> ParsedDocument:
    filler = ' '.join(f'слово{i}' for i in range(200))
    return ParsedDocument(document_id='dvs-100-rk', manual='100.0002 РК', engine_models=[ENGINE], pages=[
        DocumentPage(pdf_page=1, section='Головка цилиндров', text='Затяжка головки цилиндров. ' * 3),
        # The answer sits at the end of a long page, beyond the first window.
        DocumentPage(pdf_page=2, section='Головка цилиндров', text=f'Затяжка головки. {filler} момент 250 Н·м'),
        DocumentPage(pdf_page=3, text='Система охлаждения, термостат.'),
    ])


@pytest.fixture
def index(tmp_path):
    parsed = tmp_path / 'parsed'
    parsed.mkdir()
    (parsed / 'dvs-100-rk.json').write_text(long_manual().model_dump_json(), encoding='utf-8')
    build_index(parsed, tmp_path / 'index')
    return tmp_path / 'index'


def engine(index, reranker=None, **overrides):
    manifest, chunks, pages = read_index(index)
    config = RetrievalConfig(**({'reranker': reranker.name} if reranker else {}), **overrides)
    return RetrievalEngine(manifest, chunks, pages, config, reranker=reranker)


def plan(query: str, original: str | None = None) -> SearchPlan:
    return SearchPlan(engine_model=ENGINE, original_query=original or query, rewritten_query=query)


def test_weighted_rrf_prefers_the_heavier_ranking():
    sparse, dense = [(1, 9.0), (2, 8.0)], [(2, 0.9), (1, 0.8)]
    assert reciprocal_rank_fusion([sparse, dense])[0][0] == 1  # tie broken by index
    assert reciprocal_rank_fusion([sparse, dense], weights=[1, 2])[0][0] == 2
    assert reciprocal_rank_fusion([sparse, dense], weights=[2, 1])[0][0] == 1
    assert reciprocal_rank_fusion([sparse, dense], weights=[1, 0]) == sparse  # a zero weight drops the ranking
    assert reciprocal_rank_fusion([sparse]) == sparse
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([sparse], weights=[1, 1])


def test_windows_cover_the_whole_text_with_overlap():
    text = ' '.join(str(i) for i in range(250))
    parts = windows(text, 120, 40)
    assert [part.split()[0] for part in parts] == ['0', '80', '160']
    assert parts[-1].split()[-1] == '249'
    assert windows('коротко', 120, 40) == ['коротко'] and windows(text, None) == [text]


def test_maxp_finds_an_answer_beyond_the_first_window(index):
    spy = Spy('250 н·м')
    whole = engine(index, spy, rerank_max_length=16)
    # Without windows the spy sees the full page too, so emulate truncation: only windows expose the tail.
    passages = engine(index, spy, rerank_passage_words=60, rerank_passage_overlap=10)
    hits = passages.rank_pages(plan('момент затяжки головки цилиндров'), 3)
    assert hits[0].chunk.pdf_page == 2
    query, seen = spy.calls[-1]
    assert query == 'момент затяжки головки цилиндров' and len(seen) > 3
    assert all(len(passage.split()) <= 60 + 3 for passage in seen)  # window plus the section header
    assert whole.config.rerank_passage_words is None


def test_rerank_context_and_query_modes(index):
    spy = Spy('100.0002 рк')
    ranked = engine(index, spy, rerank_context=True, rerank_query='both')
    ranked.rank(plan('момент затяжки', original='как тянуть головку'))
    query, seen = spy.calls[-1]
    assert query == 'как тянуть головку\nмомент затяжки'
    assert all(passage.startswith('100.0002 РК\n') for passage in seen)
    same = plan('момент затяжки')
    assert rerank_query(same, 'both') == 'момент затяжки' and rerank_query(same, 'original') == 'момент затяжки'


def test_fused_prior_keeps_lexical_first_place_when_reranker_is_flat(index):
    flat = Spy('нет такого слова')
    query = plan('термостат охлаждения головки')
    fused = [hit.chunk.pdf_page for hit in engine(index).rank(query)]
    assert fused[0] == 3 and set(fused) == {1, 2, 3}
    # An unsure (flat) reranker sorts ties by chunk index; the fused prior restores the lexical first place.
    assert engine(index, flat).rank(query)[0].chunk.pdf_page == 1
    assert engine(index, flat, rerank_prior_weight=3.0).rank(query)[0].chunk.pdf_page == 3


def test_cached_reranker_scores_each_pair_once_and_persists(tmp_path):
    spy = Spy('a')
    cached = CachedReranker(spy, tmp_path / 'cache.json')
    assert cached.score('q', ['a', 'b', 'a']) == [1.0, 0.0, 1.0]
    assert cached.score('q', ['b', 'a']) == [0.0, 1.0]
    assert len(spy.calls) == 1 and spy.calls[0][1] == ['a', 'b']
    cached.save()
    again = CachedReranker(Spy('a'), tmp_path / 'cache.json')
    assert again.score('q', ['a']) == [1.0] and again.inner.calls == []


def cloudflare(handler, *, allow_external=True) -> CloudflareClient:
    settings = Settings(cloudflare_account_id='acc', cloudflare_api_token=SecretStr('token'),
                        allow_external_evidence=allow_external)
    return CloudflareClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_cloudflare_embedder_batches_and_normalizes():
    requests = []

    def handler(request):
        requests.append(request)
        texts = json.loads(request.content)['text']
        return httpx.Response(200, json={'success': True, 'errors': [], 'result': {
            'shape': [len(texts), 2], 'data': [[3.0, 4.0] for _ in texts]}})

    embedder = CloudflareEmbedder('cf:@cf/baai/bge-m3', cloudflare(handler), batch_size=2)
    vectors = embedder.embed_documents(['a', 'b', 'c'])
    assert vectors.shape == (3, 2) and np.allclose(vectors[0], [0.6, 0.8])
    assert len(requests) == 2 and requests[0].url.path.endswith('/accounts/acc/ai/run/@cf/baai/bge-m3')
    assert requests[0].headers['Authorization'] == 'Bearer token'
    assert np.allclose(embedder.embed_query('q'), [0.6, 0.8])
    assert json.loads(requests[0].content)['truncate_inputs'] is True
    assert CloudflareEmbedder.vectors({'response': [[1.0]]}) == [[1.0]] and CloudflareEmbedder.vectors([[2.0]]) == [[2.0]]
    with pytest.raises(AgentError, match='неизвестном формате'):
        CloudflareEmbedder.vectors({'shape': [1, 1]})
    assert is_cloudflare(embedder.name) and not is_cloudflare('BAAI/bge-m3') and not is_cloudflare(None)


def test_cloudflare_reranker_maps_ids_and_guards_manual_text():
    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={'success': True, 'result': {'response': [
            {'id': i, 'score': float(len(item['text']))} for i, item in reversed(list(enumerate(body['contexts'])))]}})

    reranker = CloudflareReranker('cf:@cf/baai/bge-reranker-base', cloudflare(handler), batch_size=2)
    assert reranker.score('q', ['a', 'bbb', 'cc']) == [1.0, 3.0, 2.0]
    blocked = CloudflareReranker('cf:@cf/baai/bge-reranker-base', cloudflare(handler, allow_external=False))
    with pytest.raises(AgentError) as error:
        blocked.score('q', ['manual text'])
    assert error.value.code == 'RAG_EXTERNAL_DISABLED'
    blocked_embed = CloudflareEmbedder('cf:@cf/baai/bge-m3', cloudflare(handler, allow_external=False))
    with pytest.raises(AgentError, match='ALLOW_EXTERNAL_EVIDENCE'):
        blocked_embed.embed_documents(['manual text'])


def test_cloudflare_errors_and_missing_credentials():
    failing = cloudflare(lambda request: httpx.Response(400, json={'success': False, 'errors': [{'message': 'bad'}]}))
    with pytest.raises(AgentError, match='bad'):
        CloudflareEmbedder('cf:@cf/baai/bge-m3', failing).embed_query('q')
    with pytest.raises(AgentError, match='CLOUDFLARE_ACCOUNT_ID'):
        CloudflareClient(Settings(cloudflare_account_id='', cloudflare_api_token=SecretStr('')))


def test_sweep_runs_variants_on_one_index_and_reports_misses(index, tmp_path, monkeypatch, capsys):
    from evaluation.retrieval.sweep import main as sweep

    golden = tmp_path / 'golden.jsonl'
    golden.write_text(json.dumps({
        'id': 'q1', 'engine_model': ENGINE, 'question': 'момент затяжки головки цилиндров',
        'relevant': [{'document_id': 'dvs-100-rk', 'pdf_page': 2}]}, ensure_ascii=False) + '\n', encoding='utf-8')
    grid = tmp_path / 'grid.yaml'
    grid.write_text(
        'base: {mode: sparse}\n'
        'variants:\n'
        '  - {name: bm25, query_mode: original}\n'
        '  - {name: bm25-plain, query_mode: original, overrides: {bm25_b: 0.0}}\n', encoding='utf-8')
    output, missed = tmp_path / 'sweep.json', tmp_path / 'misses.jsonl'
    monkeypatch.setattr('sys.argv', ['sweep', '--grid', str(grid), '--index', str(index), '--golden', str(golden),
                                     '--output', str(output), '--misses', str(missed)])
    assert sweep() == 0
    result = json.loads(output.read_text(encoding='utf-8'))['variants']
    assert set(result) == {'bm25', 'bm25-plain'}
    row = result['bm25']
    assert row['answerable'] == 1 and 'hit_at_1_with_neighbour' in row
    if row['hit_at_k']['1'] < 1:
        first = json.loads(missed.read_text(encoding='utf-8').splitlines()[0])
        assert first['expected'] == ['dvs-100-rk:2'] and first['top3']
    assert 'bm25-plain' in capsys.readouterr().out
