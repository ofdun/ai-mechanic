"""Embedding models for dense retrieval (RET-04, RET-05).

Candidates in order: BAAI/bge-m3, Qwen/Qwen3-Embedding-0.6B, intfloat/multilingual-e5-large-instruct,
Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B. Models run through sentence-transformers
(`uv sync --extra dense`). Vectors are L2-normalized, so the dot product equals cosine similarity.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ml.contracts import AgentError

E5_INSTRUCTION = 'Instruct: Given a question about an engine, retrieve passages of its service manual that answer it\nQuery: '


@dataclass(frozen=True)
class EmbedderPreset:
    query_prompt: str | None = None
    query_prompt_name: str | None = None
    document_prompt: str | None = None


PRESETS = {
    'BAAI/bge-m3': EmbedderPreset(),
    'Qwen/Qwen3-Embedding-0.6B': EmbedderPreset(query_prompt_name='query'),
    'Qwen/Qwen3-Embedding-4B': EmbedderPreset(query_prompt_name='query'),
    'Qwen/Qwen3-Embedding-8B': EmbedderPreset(query_prompt_name='query'),
    'intfloat/multilingual-e5-large-instruct': EmbedderPreset(query_prompt=E5_INSTRUCTION),
}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str, *, device: str = 'cpu', batch_size: int = 8, max_seq_length: int = 512):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise AgentError('RAG_DENSE_UNAVAILABLE', 'Для dense retrieval установите extra: uv sync --extra dense.') from error
        self.name = name
        self.preset = PRESETS.get(name, EmbedderPreset())
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(name, device=device)
        except OSError as error:
            raise AgentError('RAG_DENSE_UNAVAILABLE', f'Не удалось загрузить модель {name}.') from error
        self.model.max_seq_length = max_seq_length

    def _encode(self, texts: Sequence[str], prompt: str | None, prompt_name: str | None, progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=progress, prompt=prompt, prompt_name=prompt_name,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.preset.document_prompt, None, progress=len(texts) > 32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.preset.query_prompt, self.preset.query_prompt_name, progress=False)[0]


def model_slug(name: str) -> str:
    return name.replace('/', '__')
