"""Retrieval engine.

SearchPlan → engine-model filter → BM25 and exact dense per query → RRF → optional cross-encoder →
parent pages and neighbours → evidence budget → EvidenceBundle.

The index is a directory with manifest.json, chunks.jsonl (ranked units), pages.jsonl (parents that
are cited) and dense/<model>.npy vectors built by `python -m ml.retrieval embed`.
"""

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml.contracts import AgentError, Evidence, EvidenceBundle, ParsedDocument, SearchPlan
from ml.document_ids import document_id_variants
from ml.retrieval.chunking import CHUNKERS, Chunk, corpus_version, load_documents, nfc, page_chunks
from ml.retrieval.dense import DenseIndex, embed_chunks, load_dense
from ml.retrieval.embeddings import Embedder
from ml.retrieval.expansion import fault_query
from ml.retrieval.fusion import reciprocal_rank_fusion
from ml.retrieval.reranker import Reranker
from ml.retrieval.sparse import BM25, Analyzer, tokenizer

FORMAT_VERSION = 2
MANIFEST = 'manifest.json'
CHUNKS = 'chunks.jsonl'
PAGES = 'pages.jsonl'
DEFAULT_CONFIG = Path('configs/retrieval.yaml')


class RetrievalConfig(BaseModel):
    """Experimental parameters. Chunking and the analyzer apply at build time, the rest at query time."""

    model_config = ConfigDict(extra='forbid')

    mode: Literal['sparse', 'dense', 'hybrid'] = 'sparse'
    chunker: str = 'page'
    chunk_words: int = Field(300, ge=2)
    chunk_overlap: int = Field(50, ge=0)
    analyzer: Analyzer = 'stem'
    bm25_k1: float = Field(1.2, gt=0)
    bm25_b: float = Field(0.75, ge=0, le=1)
    embedder: str = 'BAAI/bge-m3'
    embed_max_seq_length: int = Field(512, ge=16)
    embed_batch_size: int = Field(8, ge=1)
    device: str = 'cpu'
    expand_anomaly: bool = True
    per_query_k: int = Field(30, ge=1)
    rrf_k: int = Field(60, ge=1)
    reranker: str | None = None
    rerank_candidates: int = Field(30, ge=1)
    rerank_max_length: int = Field(512, ge=16)
    top_k: int = Field(8, ge=1)
    context: Literal['chunk', 'page'] = 'page'
    neighbor_pages: int = Field(0, ge=0, le=3)
    max_evidence_words: int | None = Field(6000, ge=1)

    @classmethod
    def load(cls, path: Path | None = None) -> 'RetrievalConfig':
        """Read YAML. Without an explicit path a missing default file means built-in defaults."""
        target = path or DEFAULT_CONFIG
        if path is None and not target.is_file():
            return cls()
        try:
            return cls.model_validate(yaml.safe_load(target.read_text(encoding='utf-8')) or {})
        except (OSError, yaml.YAMLError, ValidationError) as error:
            raise AgentError('RAG_CONFIG_INVALID', f'Не удалось прочитать конфигурацию retrieval {target}.') from error

    def describe(self) -> str:
        parts = [f'mode={self.mode}', f'chunker={self.chunker}']
        if self.chunker != 'page':
            parts.append(f'chunk_words={self.chunk_words}/{self.chunk_overlap}')
        if self.mode != 'dense':
            parts.append(f'BM25 k1={self.bm25_k1} b={self.bm25_b} analyzer={self.analyzer}')
        if self.mode != 'sparse':
            parts.append(f'dense={self.embedder} max_seq={self.embed_max_seq_length}')
        parts.append(f'expand_anomaly={self.expand_anomaly} per_query_k={self.per_query_k} rrf_k={self.rrf_k}')
        parts.append(f'rerank={self.reranker}/{self.rerank_candidates}' if self.reranker else 'rerank=none')
        parts.append(f'context={self.context}+{self.neighbor_pages} top_k={self.top_k} budget={self.max_evidence_words}')
        return ', '.join(parts)


class DocumentSummary(BaseModel):
    document_id: str
    manual: str
    engine_models: list[str]
    pages: int
    empty_pages: list[int]
    chunks: int


class Manifest(BaseModel):
    format_version: Literal[2] = FORMAT_VERSION
    corpus_version: str
    chunker: str
    chunk_words: int
    chunk_overlap: int
    analyzer: Analyzer
    created_at: datetime
    engine_models: list[str]
    documents: list[DocumentSummary]


def index_text(chunk: Chunk) -> str:
    """Text that BM25 and the embedder see: the section title helps both."""
    return '\n'.join(filter(None, (chunk.section, chunk.text)))


def summarize(document: ParsedDocument, chunks: list[Chunk]) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id, manual=document.manual, engine_models=document.engine_models,
        pages=len(document.pages), empty_pages=[page.pdf_page for page in document.pages if not page.text.strip()],
        chunks=len(chunks),
    )


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.write_text(''.join(chunk.model_dump_json() + '\n' for chunk in chunks), encoding='utf-8')


def _read_jsonl(path: Path) -> list[Chunk]:
    with path.open(encoding='utf-8') as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def build_index(
    source: Path, output: Path, config: RetrievalConfig | None = None, *, manifest: Path | None = None,
) -> Manifest:
    """Build from the ingestion export (pages.jsonl plus manifest.json) or a directory of ParsedDocument JSON."""
    config = config or RetrievalConfig()
    if config.chunker not in CHUNKERS:
        raise AgentError('RAG_CONFIG_INVALID', f'Неизвестный chunker {config.chunker}. Доступно: {", ".join(CHUNKERS)}.')
    if config.chunk_overlap >= config.chunk_words:
        raise AgentError('RAG_CONFIG_INVALID', 'chunk_overlap должен быть меньше chunk_words.')
    check_output(source, output, manifest)
    documents = load_documents(source, manifest)
    chunker = CHUNKERS[config.chunker]
    by_document = [
        (document, chunker(document, chunk_words=config.chunk_words, chunk_overlap=config.chunk_overlap))
        for document in documents
    ]
    chunks = [chunk for _, document_chunks in by_document for chunk in document_chunks]
    pages = [page for document in documents for page in page_chunks(document)]
    if not chunks:
        raise AgentError('RAG_CORPUS_INVALID', 'В корпусе нет страниц с текстом.')
    manifest = Manifest(
        corpus_version=corpus_version(documents), chunker=config.chunker, chunk_words=config.chunk_words,
        chunk_overlap=config.chunk_overlap, analyzer=config.analyzer, created_at=datetime.now(timezone.utc),
        engine_models=sorted({model for document in documents for model in document.engine_models}),
        documents=[summarize(document, document_chunks) for document, document_chunks in by_document],
    )
    write_index(output, chunks, pages, manifest)
    return manifest


def check_output(source: Path, output: Path, manifest: Path | None = None) -> None:
    """Refuse an output that overlaps the inputs or an existing directory that is not a retrieval index.

    Rebuilding replaces the whole output directory, so it must never hold the corpus or other files.
    """
    target = output.resolve()
    source = source.resolve()
    inputs = [source]
    if manifest is not None:
        inputs.append(manifest.resolve())
    elif source.is_file():
        inputs.append(source.with_name('manifest.json'))
    for path in inputs:
        if target == path or target in path.parents:
            raise AgentError('RAG_CONFIG_INVALID', f'Каталог индекса {output} содержит входные данные {path}.')
    if source.is_dir() and source in target.parents:
        raise AgentError('RAG_CONFIG_INVALID', f'Индекс нельзя собирать внутри каталога ParsedDocument {source}.')
    if output.exists() and not is_index(output):
        raise AgentError('RAG_CONFIG_INVALID', f'{output} существует и не является индексом retrieval, он не будет заменён.')


def is_index(path: Path) -> bool:
    """An existing directory may be replaced only if it is empty or holds a retrieval index."""
    if not path.is_dir() or path.is_symlink():
        return False
    entries = {entry.name for entry in path.iterdir()}
    if not entries:
        return True
    if not {MANIFEST, CHUNKS, PAGES} <= entries or not entries <= {MANIFEST, CHUNKS, PAGES, 'dense'}:
        return False
    try:
        Manifest.model_validate_json((path / MANIFEST).read_bytes())
    except (OSError, ValidationError):
        return False
    return True


def write_index(output: Path, chunks: list[Chunk], pages: list[Chunk], manifest: Manifest) -> None:
    """Write into a fresh sibling directory, then swap it in. Only a verified index is ever removed."""
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f'.{output.name}.build-'))
    except OSError as error:
        raise AgentError('RAG_INDEX_UNAVAILABLE', f'Не удалось записать индекс в {output}.') from error
    previous = None
    try:
        _write_jsonl(staging / CHUNKS, chunks)
        _write_jsonl(staging / PAGES, pages)
        (staging / MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
        if output.exists():
            if not is_index(output):
                raise AgentError('RAG_CONFIG_INVALID', f'{output} не является индексом retrieval, он не будет заменён.')
            previous = Path(tempfile.mkdtemp(dir=output.parent, prefix=f'.{output.name}.old-'))
            previous.rmdir()
            output.rename(previous)
        staging.rename(output)
    except OSError as error:
        if previous is not None and not output.exists():
            previous.rename(output)
            previous = None
        raise AgentError('RAG_INDEX_UNAVAILABLE', f'Не удалось записать индекс в {output}.') from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


def read_index(path: Path) -> tuple[Manifest, list[Chunk], list[Chunk]]:
    try:
        manifest = Manifest.model_validate_json((path / MANIFEST).read_bytes())
        chunks, pages = _read_jsonl(path / CHUNKS), _read_jsonl(path / PAGES)
    except (OSError, ValidationError, json.JSONDecodeError) as error:
        raise AgentError('RAG_INDEX_UNAVAILABLE', f'Индекс {path} недоступен или устарел. Пересоберите его.') from error
    return manifest, chunks, pages


def embed_index(path: Path, embedder: Embedder, config: RetrievalConfig) -> int:
    """Compute dense vectors for an existing index. Several models can live side by side (RET-04/05)."""
    _, chunks, _ = read_index(path)
    embed_chunks(path, embedder, [chunk.chunk_id for chunk in chunks], [index_text(chunk) for chunk in chunks],
                 max_seq_length=config.embed_max_seq_length)
    return len(chunks)


def load_embedder(config: RetrievalConfig) -> Embedder:
    from ml.retrieval.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(config.embedder, device=config.device, batch_size=config.embed_batch_size,
                                       max_seq_length=config.embed_max_seq_length)


def load_reranker(config: RetrievalConfig) -> Reranker:
    from ml.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker(config.reranker, device=config.device, batch_size=config.embed_batch_size,
                                max_length=config.rerank_max_length)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def queries(plan: SearchPlan, *, expand_anomaly: bool = True) -> list[str]:
    """Original, rewritten and sub-queries without repeats, plus the fault-code query of a diagnosis.

    The fault vocabulary also extends the rewritten query: a diagnostic rewrite such as "<fault title>.
    Possible causes, checks and procedures" otherwise matches every troubleshooting table equally, and
    under RRF a single expansion list loses to pages that several generic lists agree on.
    """
    expansion = fault_query(plan.anomaly_type) if expand_anomaly else None
    rewritten = f'{plan.rewritten_query} {expansion}' if expansion else plan.rewritten_query
    variants = (plan.original_query, rewritten, *plan.subqueries, expansion)
    return list(dict.fromkeys(query.strip() for query in variants if query and query.strip()))


class RetrievalEngine:
    """Implements ml.contracts.Retriever. The engine-model filter is applied before any top-k."""

    def __init__(
        self, manifest: Manifest, chunks: list[Chunk], pages: list[Chunk], config: RetrievalConfig | None = None,
        *, dense: DenseIndex | None = None, embedder: Embedder | None = None, reranker: Reranker | None = None,
    ):
        self.config = config or RetrievalConfig()
        if self.config.mode != 'sparse' and (dense is None or embedder is None):
            raise AgentError('RAG_CONFIG_INVALID', f'Режим {self.config.mode} требует dense-векторы и embedder.')
        if self.config.reranker and reranker is None:
            raise AgentError('RAG_CONFIG_INVALID', 'В конфигурации указан reranker, но он не загружен.')
        self.manifest, self.chunks, self.dense, self.embedder, self.reranker = manifest, chunks, dense, embedder, reranker
        self.pages = {page.chunk_id: page for page in pages}
        self.tokenize = tokenizer(manifest.analyzer)
        self.sparse = None
        if self.config.mode != 'dense':
            self.sparse = BM25([self.tokenize(index_text(chunk)) for chunk in chunks],
                               k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.by_model: dict[str, frozenset[int]] = {
            model: frozenset(i for i, chunk in enumerate(chunks) if model in chunk.engine_models)
            for model in {model for chunk in chunks for model in chunk.engine_models}
        }

    @classmethod
    def load(cls, path: Path, config: RetrievalConfig | None = None) -> 'RetrievalEngine':
        config = config or RetrievalConfig()
        manifest, chunks, pages = read_index(path)
        dense = embedder = reranker = None
        if config.mode != 'sparse':
            dense = load_dense(path, config.embedder, [chunk.chunk_id for chunk in chunks])
            embedder = load_embedder(config)
        if config.reranker:
            reranker = load_reranker(config)
        return cls(manifest, chunks, pages, config, dense=dense, embedder=embedder, reranker=reranker)

    def candidates(self, plan: SearchPlan) -> list[tuple[int, float]]:
        """Fused chunk candidates for all queries of the plan, already restricted to the engine model."""
        allowed = self.by_model.get(nfc(plan.engine_model))
        if not allowed:
            return []
        depth = self.config.per_query_k
        rankings = []
        for query in queries(plan, expand_anomaly=self.config.expand_anomaly):
            if self.sparse is not None and (tokens := self.tokenize(query)):
                rankings.append(self.sparse.top(tokens, depth, allowed))
            if self.dense is not None:
                rankings.append(self.dense.top(self.embedder.embed_query(query), depth, allowed))
        return reciprocal_rank_fusion(rankings, k=self.config.rrf_k)

    def rank(self, plan: SearchPlan, limit: int | None = None) -> list[Hit]:
        """Chunks in final order. `limit=None` returns every fused candidate (evaluation uses it)."""
        fused = self.candidates(plan)
        if self.reranker is not None and fused:
            pool = fused[:max(self.config.rerank_candidates, limit or 0)]
            scores = self.reranker.score(plan.rewritten_query, [index_text(self.chunks[i]) for i, _ in pool])
            fused = sorted(zip((i for i, _ in pool), scores), key=lambda item: (-item[1], item[0]))
        return [Hit(self.chunks[index], score) for index, score in fused[:limit]]

    def rank_pages(self, plan: SearchPlan, limit: int) -> list[Hit]:
        """Distinct parent pages in the order of their best chunk."""
        seen: dict[str, Hit] = {}
        for hit in self.rank(plan):
            if hit.chunk.parent_id not in seen:
                seen[hit.chunk.parent_id] = Hit(self.pages.get(hit.chunk.parent_id, hit.chunk), hit.score)
                if len(seen) == limit:
                    break
        return list(seen.values())

    def _neighbors(self, page: Chunk, model: str) -> list[Chunk]:
        found = []
        for offset in (*range(-self.config.neighbor_pages, 0), *range(1, self.config.neighbor_pages + 1)):
            neighbor = self.pages.get(f'{page.document_id}:{page.pdf_page + offset}')
            if neighbor is not None and model in neighbor.engine_models:
                found.append(neighbor)
        return found

    @staticmethod
    def _evidence(unit: Chunk) -> Evidence:
        document_id, _ = document_id_variants(unit.document_id)
        evidence_id = document_id + unit.chunk_id.removeprefix(unit.document_id)
        return Evidence(
            evidence_id=evidence_id, document_id=document_id, manual=unit.manual,
            engine_models=unit.engine_models, pdf_page=unit.pdf_page, section=unit.section, text=unit.text,
        )

    def page(self, engine_model: str, document_id: str, pdf_page: int) -> Evidence | None:
        canonical, legacy = document_id_variants(document_id)
        page = self.pages.get(f'{canonical}:{pdf_page}')
        if page is None and legacy is not None:
            page = self.pages.get(f'{legacy}:{pdf_page}')
        if page is None or nfc(engine_model) not in page.engine_models:
            return None
        return self._evidence(page)

    def search(self, plan: SearchPlan) -> EvidenceBundle:
        model = nfc(plan.engine_model)
        if self.config.context == 'chunk':
            units = [hit.chunk for hit in self.rank(plan, self.config.top_k)]
        else:
            units = []
            for hit in self.rank_pages(plan, self.config.top_k):
                units.append(hit.chunk)
                units.extend(self._neighbors(hit.chunk, model))
        items, ids, words = [], set(), 0
        for unit in units:
            if unit.chunk_id in ids:
                continue
            size = len(unit.text.split())
            budget = self.config.max_evidence_words
            if items and budget is not None and words + size > budget:
                break
            ids.add(unit.chunk_id)
            words += size
            items.append(self._evidence(unit))
        return EvidenceBundle(items=items)
