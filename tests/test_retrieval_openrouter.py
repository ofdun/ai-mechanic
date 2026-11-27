"""OpenRouter retrieval backends over a mock transport: embeddings, rerank, guards, retries, config wiring."""

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from pydantic import SecretStr

from ml.config import Settings
from ml.contracts import AgentError
from ml.retrieval import RetrievalConfig
from ml.retrieval.engine import load_embedder, load_reranker
from ml.retrieval.openrouter import OpenRouterClient, OpenRouterEmbedder, OpenRouterReranker, api_key, is_openrouter


def client(handler, *, allow_external=True, key='or-key', sleeps=None, **settings) -> OpenRouterClient:
    config = Settings(openrouter_api_key=SecretStr(key), allow_external_evidence=allow_external, **settings)
    return OpenRouterClient(config, http=httpx.Client(transport=httpx.MockTransport(handler)),
                            sleep=(sleeps.append if sleeps is not None else lambda _: None))


def test_embedder_batches_orders_by_index_and_normalizes():
    requests = []

    def handler(request):
        requests.append(request)
        texts = json.loads(request.content)['input']
        data = [{'object': 'embedding', 'index': i, 'embedding': [3.0, 4.0] if t != 'z' else [0.0, 2.0]}
                for i, t in enumerate(texts)]
        return httpx.Response(200, json={'object': 'list', 'data': list(reversed(data)), 'model': 'baai/bge-m3'})

    embedder = OpenRouterEmbedder('or:baai/bge-m3', client(handler), batch_size=2)
    vectors = embedder.embed_documents(['a', 'b', 'z'])
    assert vectors.shape == (3, 2) and np.allclose(vectors[0], [0.6, 0.8]) and np.allclose(vectors[2], [0, 1])
    assert len(requests) == 2 and str(requests[0].url) == 'https://openrouter.ai/api/v1/embeddings'
    assert requests[0].headers['Authorization'] == 'Bearer or-key'
    body = json.loads(requests[0].content)
    assert body == {'model': 'baai/bge-m3', 'input': ['a', 'b'], 'encoding_format': 'float'}
    assert np.allclose(embedder.embed_query('q'), [0.6, 0.8])


def test_reranker_maps_indexes_and_guards_manual_text():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        results = [{'index': i, 'relevance_score': len(doc) / 10, 'document': {'text': doc}}
                   for i, doc in enumerate(body['documents'])]
        return httpx.Response(200, json={'model': body['model'], 'results': sorted(
            results, key=lambda item: -item['relevance_score'])})

    reranker = OpenRouterReranker('or:cohere/rerank-5-pro', client(handler), batch_size=2)
    assert reranker.score('q', ['a', 'bbb', 'cc']) == [0.1, 0.3, 0.2]
    assert seen[0] == {'model': 'cohere/rerank-5-pro', 'query': 'q', 'documents': ['a', 'bbb'], 'top_n': 2}
    assert reranker.score('q', []) == []
    with pytest.raises(AgentError) as error:
        OpenRouterReranker('or:cohere/rerank-5-pro', client(handler, allow_external=False)).score('q', ['manual'])
    assert error.value.code == 'RAG_EXTERNAL_DISABLED'
    with pytest.raises(AgentError, match='ALLOW_EXTERNAL_EVIDENCE'):
        OpenRouterEmbedder('or:baai/bge-m3', client(handler, allow_external=False)).embed_documents(['manual'])


def test_retries_rate_limits_then_reports_errors():
    calls, sleeps = [], []

    def flaky(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, json={'error': {'code': 429, 'message': 'Rate limit exceeded'}})
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0]}]})

    assert np.allclose(OpenRouterEmbedder('or:baai/bge-m3', client(flaky, sleeps=sleeps)).embed_query('q'), [1.0])
    assert len(calls) == 3 and sleeps == [1, 2]

    broke = client(lambda request: httpx.Response(402, json={'error': {'code': 402, 'message': 'Insufficient credits'}}))
    with pytest.raises(AgentError, match='Insufficient credits'):
        OpenRouterEmbedder('or:baai/bge-m3', broke).embed_query('q')
    short = client(lambda request: httpx.Response(200, json={'results': []}))
    with pytest.raises(AgentError, match='0 оценок на 1'):
        OpenRouterReranker('or:cohere/rerank-v3.5', short).score('q', ['x'])


def test_key_falls_back_to_openrouter_llm_key_only():
    assert api_key(Settings(openrouter_api_key=SecretStr(''), llm_api_key=SecretStr('llm'))) == 'llm'
    assert api_key(Settings(openrouter_api_key=SecretStr('own'), llm_api_key=SecretStr('llm'))) == 'own'
    other = Settings(openrouter_api_key=SecretStr(''), llm_api_key=SecretStr('llm'),
                     llm_base_url='https://api.example.com/v1')
    assert api_key(other) == ''
    with pytest.raises(AgentError, match='OPENROUTER_API_KEY'):
        OpenRouterClient(other)


def test_config_prefix_selects_openrouter_backends(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-key')
    config = RetrievalConfig(mode='hybrid', embedder='or:baai/bge-m3', reranker='or:mongodb/rerank-3')
    assert isinstance(load_embedder(config), OpenRouterEmbedder)
    assert isinstance(load_reranker(config), OpenRouterReranker)
    assert is_openrouter('or:x') and not is_openrouter('cf:@cf/baai/bge-m3') and not is_openrouter(None)


def test_openrouter_grid_validates():
    grid = yaml.safe_load(Path('configs/retrieval-sweep-openrouter.yaml').read_text(encoding='utf-8'))
    names = [variant['name'] for variant in grid['variants']]
    assert len(names) == len(set(names))
    for variant in grid['variants']:
        config = RetrievalConfig.model_validate({**grid['base'], **variant.get('overrides', {})})
        assert is_openrouter(config.embedder)
        assert config.reranker is None or (is_openrouter(config.reranker) and 'qwen' not in config.reranker)
