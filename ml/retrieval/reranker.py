"""Cross-encoder reranking of fused candidates (RET-08).

Backends: BAAI/bge-reranker-v2-m3 through sentence-transformers CrossEncoder (local, `uv sync --extra dense`)
and Cloudflare Workers AI (`cf:@cf/baai/bge-reranker-base`, see ml.retrieval.cloudflare).

A page is usually longer than the 512-token window of the cross-encoder, so the engine can split it into
word windows and keep the best window (`rerank_passage_words`). Scores are cached by query and passage,
which makes a parameter sweep over one index cheap.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ml.contracts import AgentError


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, name: str, *, device: str = 'cpu', batch_size: int = 8, max_length: int = 512):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise AgentError('RAG_RERANK_UNAVAILABLE', 'Для reranking установите extra: uv sync --extra dense.') from error
        self.name = name
        self.batch_size = batch_size
        try:
            self.model = CrossEncoder(name, device=device, max_length=max_length)
        except OSError as error:
            raise AgentError('RAG_RERANK_UNAVAILABLE', f'Не удалось загрузить модель {name}.') from error

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, passage) for passage in passages], batch_size=self.batch_size,
                                    show_progress_bar=False)
        return [float(value) for value in scores]


def windows(text: str, words: int | None, overlap: int = 0) -> list[str]:
    """Word windows of a passage for MaxP reranking. Without a window size the text is one passage."""
    tokens = text.split()
    if not words or len(tokens) <= words:
        return [text]
    step = max(1, words - overlap)
    starts = range(0, max(len(tokens) - overlap, 1), step)
    return [' '.join(tokens[start:start + words]) for start in starts if tokens[start:start + words]]


class CachedReranker:
    """Memoizes scores by (model, query, passage). `path` keeps them between sweep runs."""

    def __init__(self, inner: Reranker, path: Path | None = None):
        self.inner, self.name, self.path = inner, inner.name, path
        self.cache: dict[str, float] = {}
        if path is not None and path.is_file():
            self.cache = json.loads(path.read_text(encoding='utf-8'))
        self.dirty = False

    def key(self, query: str, passage: str) -> str:
        return hashlib.sha1(f'{self.name}\x00{query}\x00{passage}'.encode()).hexdigest()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        keys = [self.key(query, passage) for passage in passages]
        first = {}
        for i, key in enumerate(keys):
            first.setdefault(key, i)
        missing = [i for key, i in first.items() if key not in self.cache]
        if missing:
            scores = self.inner.score(query, [passages[i] for i in missing])
            for i, value in zip(missing, scores):
                self.cache[keys[i]] = value
            self.dirty = True
        return [self.cache[key] for key in keys]

    def save(self) -> None:
        if self.path is not None and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache), encoding='utf-8')
            self.dirty = False
