"""Exact dense search over L2-normalized vectors with NumPy (the corpus is a few thousand chunks).

Vectors are stored next to the index as dense/<model>.npy with a metadata JSON. Rows follow chunks.jsonl,
which the metadata pins by a digest of chunk ids, so a stale file is rejected instead of silently misaligned.
"""

import hashlib
import json
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from ml.contracts import AgentError
from ml.retrieval.embeddings import Embedder, model_slug


class DenseMeta(BaseModel):
    model: str
    dimension: int
    rows: int
    chunk_ids_sha256: str
    max_seq_length: int | None = None


def chunk_ids_digest(chunk_ids: Sequence[str]) -> str:
    return hashlib.sha256('\n'.join(chunk_ids).encode('utf-8')).hexdigest()


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError('Dense vectors must be a 2-D matrix')
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def top(self, query: np.ndarray, limit: int, allowed: Collection[int] | None = None) -> list[tuple[int, float]]:
        if allowed is None:
            candidates = np.arange(len(self.vectors))
        else:
            candidates = np.fromiter(sorted(allowed), dtype=np.int64, count=len(allowed))
        if not len(candidates):
            return []
        scores = self.vectors[candidates] @ np.asarray(query, dtype=np.float32)
        count = min(limit, len(candidates))
        best = np.argpartition(-scores, count - 1)[:count]
        order = sorted(best, key=lambda position: (-float(scores[position]), int(candidates[position])))
        return [(int(candidates[position]), float(scores[position])) for position in order]


def vector_paths(index_dir: Path, model: str) -> tuple[Path, Path]:
    base = index_dir / 'dense' / model_slug(model)
    return base.with_suffix('.npy'), base.with_suffix('.json')


def embed_chunks(
    index_dir: Path, embedder: Embedder, chunk_ids: Sequence[str], texts: Sequence[str],
    *, max_seq_length: int | None = None,
) -> DenseMeta:
    vectors = embedder.embed_documents(texts)
    if vectors.shape[0] != len(chunk_ids):
        raise AgentError('RAG_DENSE_UNAVAILABLE', 'Число векторов не совпадает с числом фрагментов.')
    meta = DenseMeta(
        model=embedder.name, dimension=int(vectors.shape[1]), rows=len(chunk_ids),
        chunk_ids_sha256=chunk_ids_digest(chunk_ids), max_seq_length=max_seq_length,
    )
    vectors_path, meta_path = vector_paths(index_dir, embedder.name)
    try:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, vectors.astype(np.float32))
        meta_path.write_text(meta.model_dump_json(indent=2), encoding='utf-8')
    except OSError as error:
        raise AgentError('RAG_INDEX_UNAVAILABLE', f'Не удалось записать векторы в {vectors_path}.') from error
    return meta


def load_dense(
    index_dir: Path, model: str, chunk_ids: Sequence[str], *, max_seq_length: int | None = None,
) -> DenseIndex:
    vectors_path, meta_path = vector_paths(index_dir, model)
    try:
        meta = DenseMeta.model_validate_json(meta_path.read_bytes())
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise AgentError(
            'RAG_INDEX_UNAVAILABLE', f'Нет векторов {model} в {index_dir}. Запустите python -m ml.retrieval embed.',
        ) from error
    if (meta.model != model or meta.chunk_ids_sha256 != chunk_ids_digest(chunk_ids)
            or vectors.shape != (len(chunk_ids), meta.dimension) or meta.rows != len(chunk_ids)
            or (max_seq_length is not None and meta.max_seq_length != max_seq_length)):
        raise AgentError('RAG_INDEX_UNAVAILABLE', f'Векторы {model} не соответствуют фрагментам индекса. Пересчитайте их.')
    return DenseIndex(vectors)
