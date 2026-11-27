"""Cloudflare Workers AI backends: BGE-M3 embeddings and the BGE reranker over the REST API.

Names in the retrieval config carry the `cf:` prefix: `embedder: cf:@cf/baai/bge-m3`,
`reranker: cf:@cf/baai/bge-reranker-base`. Credentials come from `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_API_TOKEN` (Settings or .env).

Document embeddings and reranking send manual text to Cloudflare, so both require
`ALLOW_EXTERNAL_EVIDENCE=true`. Query embeddings carry only the question. Vectors from Cloudflare and
from a local model are stored under different names and are never mixed: the index keeps one file per
embedder name, and a query is embedded by the same backend as the documents.
"""

from collections.abc import Sequence

import httpx
import numpy as np

from ml.config import Settings
from ml.contracts import AgentError

PREFIX = 'cf:'
API = 'https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}'


def is_cloudflare(name: str | None) -> bool:
    return bool(name) and name.startswith(PREFIX)


class CloudflareClient:
    def __init__(self, settings: Settings, *, http: httpx.Client | None = None, timeout: float = 60):
        account, token = settings.cloudflare_account_id, settings.cloudflare_api_token.get_secret_value()
        if not account or not token:
            raise AgentError('RAG_BACKEND_UNAVAILABLE', 'Для Cloudflare задайте CLOUDFLARE_ACCOUNT_ID и CLOUDFLARE_API_TOKEN.')
        self.account, self.token = account, token
        self.allow_external = settings.allow_external_evidence
        self.http = http or httpx.Client(timeout=timeout)

    def require_external(self, what: str) -> None:
        if not self.allow_external:
            raise AgentError('RAG_EXTERNAL_DISABLED',
                             f'{what} отправляет текст руководств в Cloudflare. Разрешите ALLOW_EXTERNAL_EVIDENCE=true.')

    def run(self, model: str, payload: dict) -> dict:
        try:
            response = self.http.post(API.format(account=self.account, model=model), json=payload,
                                      headers={'Authorization': f'Bearer {self.token}'})
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise AgentError('RAG_BACKEND_UNAVAILABLE', f'Cloudflare {model} недоступен.') from error
        if not isinstance(body, dict):
            body = {'result': body}
        if response.status_code != 200 or body.get('success') is False:
            errors = '; '.join(str(item.get('message', item)) if isinstance(item, dict) else str(item)
                               for item in body.get('errors') or []) or response.status_code
            raise AgentError('RAG_BACKEND_UNAVAILABLE', f'Cloudflare {model}: {errors}.')
        # The REST API wraps the model output in {"result", "success", "errors"}; the docs show only the inner part.
        return body['result'] if 'result' in body else body


class CloudflareEmbedder:
    def __init__(self, name: str, client: CloudflareClient, *, batch_size: int = 50):
        # The documented batch limit of bge-m3 is 100 texts per call.
        self.name, self.model, self.client, self.batch_size = name, name.removeprefix(PREFIX), client, min(batch_size, 100)

    @staticmethod
    def vectors(result) -> list:
        """Embedding rows from `{"data": [...]}` (documented examples), `{"response": [...]}` or a bare list."""
        if isinstance(result, dict):
            for key in ('data', 'response', 'embeddings'):
                if isinstance(result.get(key), list):
                    return result[key]
        if isinstance(result, list):
            return result
        raise AgentError('RAG_BACKEND_UNAVAILABLE', 'Cloudflare вернул эмбеддинги в неизвестном формате.')

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            rows.extend(self.vectors(self.client.run(self.model, {'text': batch, 'truncate_inputs': True})))
        if len(rows) != len(texts):
            raise AgentError('RAG_BACKEND_UNAVAILABLE', f'Cloudflare вернул {len(rows)} векторов на {len(texts)} текстов.')
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.client.require_external('Эмбеддинг корпуса')
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class CloudflareReranker:
    def __init__(self, name: str, client: CloudflareClient, *, batch_size: int = 32):
        self.name, self.model, self.client, self.batch_size = name, name.removeprefix(PREFIX), client, batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self.client.require_external('Reranking')
        scores = [0.0] * len(passages)
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            result = self.client.run(self.model, {
                'query': query, 'contexts': [{'text': text} for text in batch], 'top_k': len(batch),
            })
            # Scores are relevance logits (the docs map them to [0, 1] by sigmoid); only their order matters here.
            for item in result['response']:
                scores[start + int(item['id'])] = float(item['score'])
        return scores
