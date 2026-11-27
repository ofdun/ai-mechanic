"""OpenRouter backends: BGE-M3 embeddings and hosted rerankers (Cohere, Voyage) over the REST API.

Names in the retrieval config carry the `or:` prefix: `embedder: or:baai/bge-m3`,
`reranker: or:cohere/rerank-v3.5`. The key comes from `OPENROUTER_API_KEY`; when it is empty, `LLM_API_KEY` is
used if `LLM_BASE_URL` points at OpenRouter.

Document embeddings and reranking send manual text to OpenRouter and its providers, so both require
`ALLOW_EXTERNAL_EVIDENCE=true`. Query embeddings carry only the question. Vectors are stored per embedder name,
so OpenRouter and local BGE-M3 vectors are never mixed.

API (https://openrouter.ai/docs/api/api-reference):
- `POST /embeddings {model, input: [...], encoding_format: "float"}` → `data: [{index, embedding}]`;
- `POST /rerank {model, query, documents: [...], top_n}` → `results: [{index, relevance_score}]`.
"""

import time
from collections.abc import Callable, Sequence

import httpx
import numpy as np

from ml.config import Settings
from ml.contracts import AgentError

PREFIX = 'or:'
API = 'https://openrouter.ai/api/v1'
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def is_openrouter(name: str | None) -> bool:
    return bool(name) and name.startswith(PREFIX)


def api_key(settings: Settings) -> str:
    key = settings.openrouter_api_key.get_secret_value()
    if not key and 'openrouter.ai' in str(settings.llm_base_url):
        key = settings.llm_api_key.get_secret_value()
    return key


class OpenRouterClient:
    def __init__(self, settings: Settings, *, http: httpx.Client | None = None, timeout: float = 60,
                 retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.key = api_key(settings)
        if not self.key:
            raise AgentError('RAG_BACKEND_UNAVAILABLE', 'Для OpenRouter задайте OPENROUTER_API_KEY или LLM_API_KEY.')
        self.allow_external = settings.allow_external_evidence
        self.http = http or httpx.Client(timeout=timeout)
        self.retries, self.sleep = retries, sleep

    def require_external(self, what: str) -> None:
        if not self.allow_external:
            raise AgentError('RAG_EXTERNAL_DISABLED',
                             f'{what} отправляет текст руководств в OpenRouter. Разрешите ALLOW_EXTERNAL_EVIDENCE=true.')

    def post(self, path: str, payload: dict) -> dict:
        model = payload.get('model', path)
        for attempt in range(self.retries + 1):
            try:
                response = self.http.post(f'{API}/{path}', json=payload,
                                          headers={'Authorization': f'Bearer {self.key}'})
            except httpx.HTTPError as error:
                if attempt == self.retries:
                    raise AgentError('RAG_BACKEND_UNAVAILABLE', f'OpenRouter {model} недоступен.') from error
                self.sleep(2 ** attempt)
                continue
            if response.status_code in RETRY_STATUS and attempt < self.retries:
                self.sleep(2 ** attempt)
                continue
            try:
                body = response.json()
            except ValueError as error:
                raise AgentError('RAG_BACKEND_UNAVAILABLE', f'OpenRouter {model}: ответ не JSON.') from error
            if response.status_code != 200 or 'error' in body:
                error = body.get('error') if isinstance(body, dict) else None
                message = error.get('message', error) if isinstance(error, dict) else error or response.status_code
                raise AgentError('RAG_BACKEND_UNAVAILABLE', f'OpenRouter {model}: {message}.')
            return body
        raise AssertionError('unreachable')


class OpenRouterEmbedder:
    def __init__(self, name: str, client: OpenRouterClient, *, batch_size: int = 32):
        self.name, self.model, self.client, self.batch_size = name, name.removeprefix(PREFIX), client, batch_size

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            body = self.client.post('embeddings', {'model': self.model, 'input': batch, 'encoding_format': 'float'})
            data = sorted(body.get('data') or [], key=lambda item: item['index'])
            if len(data) != len(batch):
                raise AgentError('RAG_BACKEND_UNAVAILABLE',
                                 f'OpenRouter вернул {len(data)} векторов на {len(batch)} текстов.')
            rows.extend(item['embedding'] for item in data)
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.client.require_external('Эмбеддинг корпуса')
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class OpenRouterReranker:
    def __init__(self, name: str, client: OpenRouterClient, *, batch_size: int = 100):
        self.name, self.model, self.client, self.batch_size = name, name.removeprefix(PREFIX), client, batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self.client.require_external('Reranking')
        scores = [0.0] * len(passages)
        for start in range(0, len(passages), self.batch_size):
            batch = list(passages[start:start + self.batch_size])
            body = self.client.post('rerank', {'model': self.model, 'query': query, 'documents': batch,
                                               'top_n': len(batch)})
            results = body.get('results') or []
            if len(results) != len(batch):
                raise AgentError('RAG_BACKEND_UNAVAILABLE',
                                 f'OpenRouter вернул {len(results)} оценок на {len(batch)} фрагментов.')
            for item in results:
                scores[start + int(item['index'])] = float(item['relevance_score'])
        return scores
