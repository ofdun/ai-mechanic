"""Cross-encoder reranking of fused candidates (RET-08).

Candidates: no reranker (control), BAAI/bge-reranker-v2-m3 through sentence-transformers CrossEncoder.
Qwen3-Reranker scores with a causal LM yes/no prompt and needs its own adapter behind the same protocol.
"""

from collections.abc import Sequence
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
