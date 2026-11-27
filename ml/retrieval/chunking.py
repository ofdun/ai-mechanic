"""Loading ParsedDocument files and splitting them into retrieval chunks."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ml.contracts import AgentError, DocumentPage, PageNumber, ParsedDocument, Text


def nfc(value: str) -> str:
    return unicodedata.normalize('NFC', value)


class Chunk(BaseModel):
    """Smallest unit the index ranks. A chunk never crosses a physical PDF page."""

    chunk_id: Text
    parent_id: Text = Field(description='Page that contains the chunk: document_id:pdf_page')
    document_id: Text
    manual: Text
    engine_models: list[Text] = Field(min_length=1)
    pdf_page: PageNumber
    printed_page: str | None = None
    section: str | None = None
    text: Text


def document_digest(document: ParsedDocument) -> str:
    canonical = json.dumps(document.model_dump(mode='json'), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def normalize_document(document: ParsedDocument) -> ParsedDocument:
    """Apply NFC to model names and check page applicability and numbering."""
    models = list(dict.fromkeys(nfc(model) for model in document.engine_models))
    pages = []
    seen = set()
    for page in document.pages:
        if page.pdf_page in seen:
            raise AgentError('RAG_CORPUS_INVALID', f'{document.document_id}: страница PDF {page.pdf_page} повторяется.')
        seen.add(page.pdf_page)
        page_models = None
        if page.engine_models is not None:
            page_models = list(dict.fromkeys(nfc(model) for model in page.engine_models))
            if not set(page_models) <= set(models):
                raise AgentError(
                    'RAG_CORPUS_INVALID',
                    f'{document.document_id}: страница {page.pdf_page} расширяет применимость документа.',
                )
        pages.append(page.model_copy(update={'engine_models': page_models}))
    return document.model_copy(update={
        'document_id': nfc(document.document_id), 'engine_models': models,
        'pages': sorted(pages, key=lambda page: page.pdf_page),
    })


class PageRecord(BaseModel):
    """One line of the ingestion export (ml/ingestion/export_pages.py): a physical PDF page."""

    document_id: Text
    models: list[Text] = Field(min_length=1)
    pdf_page: PageNumber
    printed_page: str | None = None
    section: str | None = None
    text: str = ''


class ManifestEntry(BaseModel):
    document_id: Text
    title: Text
    designation: str | None = None


def manual_names(manifest: Path) -> dict[str, str]:
    """Manual name for citations and tickets: the designation, or the title when a manual has none."""
    try:
        entries = [ManifestEntry.model_validate(item) for item in json.loads(manifest.read_bytes())['documents']]
    except (OSError, KeyError, TypeError, ValueError, ValidationError) as error:
        raise AgentError('RAG_CORPUS_INVALID', f'Не удалось прочитать каталог руководств {manifest}.') from error
    return {nfc(entry.document_id): (entry.designation or '').strip() or entry.title for entry in entries}


def documents_from_pages(pages_path: Path, manifest: Path | None = None) -> list[ParsedDocument]:
    """Group the ingestion export into ParsedDocument. The manifest defaults to manifest.json next to it."""
    manifest = manifest or pages_path.with_name('manifest.json')
    names = manual_names(manifest)
    grouped: dict[str, list[PageRecord]] = {}
    try:
        with pages_path.open(encoding='utf-8') as file:
            for number, line in enumerate(file, 1):
                if line.strip():
                    record = PageRecord.model_validate_json(line)
                    grouped.setdefault(nfc(record.document_id), []).append(record)
    except OSError as error:
        raise AgentError('RAG_CORPUS_INVALID', f'Не удалось прочитать {pages_path}.') from error
    except ValidationError as error:
        raise AgentError('RAG_CORPUS_INVALID', f'{pages_path.name}:{number} не соответствует формату страницы.') from error
    documents = []
    for document_id, records in grouped.items():
        if document_id not in names:
            raise AgentError('RAG_CORPUS_INVALID', f'{document_id} нет в каталоге {manifest.name}.')
        models = list(dict.fromkeys(model for record in records for model in record.models))
        documents.append(ParsedDocument(document_id=document_id, manual=names[document_id], engine_models=models, pages=[
            DocumentPage(
                pdf_page=record.pdf_page, printed_page=record.printed_page, section=record.section, text=record.text,
                engine_models=None if record.models == models else record.models,
            )
            for record in records
        ]))
    return documents


def load_documents(source: Path, manifest: Path | None = None) -> list[ParsedDocument]:
    """Read the corpus: an ingestion export (*.jsonl) or a directory with one ParsedDocument JSON per manual.

    Identical copies of a document are counted once, conflicting copies are rejected.
    """
    if source.is_file():
        candidates = [(document, source) for document in documents_from_pages(source, manifest)]
    else:
        candidates = []
        for path in sorted(source.glob('*.json')):
            try:
                candidates.append((ParsedDocument.model_validate_json(path.read_bytes()), path))
            except (OSError, ValidationError) as error:
                raise AgentError('RAG_CORPUS_INVALID', f'{path.name}: файл не соответствует ParsedDocument.') from error
    if not candidates:
        raise AgentError('RAG_CORPUS_INVALID', f'В {source} нет документов: нужен pages.jsonl или *.json ParsedDocument.')
    return deduplicate(candidates)


def deduplicate(candidates: list[tuple[ParsedDocument, Path]]) -> list[ParsedDocument]:
    documents: dict[str, tuple[ParsedDocument, str, Path]] = {}
    for raw, path in candidates:
        document = normalize_document(raw)
        digest = document_digest(document)
        if previous := documents.get(document.document_id):
            if previous[1] != digest:
                raise AgentError(
                    'RAG_CORPUS_INVALID',
                    f'document_id {document.document_id} встречается в {previous[2].name} и {path.name} с разным содержимым.',
                )
            continue
        documents[document.document_id] = (document, digest, path)
    return [document for document, _, _ in sorted(documents.values(), key=lambda item: item[0].document_id)]


def corpus_version(documents: list[ParsedDocument]) -> str:
    digests = sorted(f'{document.document_id}:{document_digest(document)}' for document in documents)
    return hashlib.sha256('\n'.join(digests).encode('utf-8')).hexdigest()[:16]


def _chunk(document: ParsedDocument, page: DocumentPage, chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, parent_id=f'{document.document_id}:{page.pdf_page}', document_id=document.document_id,
        manual=document.manual, engine_models=page.engine_models or document.engine_models, pdf_page=page.pdf_page,
        printed_page=page.printed_page, section=page.section, text=text,
    )


def page_chunks(document: ParsedDocument, **_) -> list[Chunk]:
    """Page-level chunking for the RET-01 baseline. Pages without text are skipped, numbering is kept."""
    return [
        _chunk(document, page, f'{document.document_id}:{page.pdf_page}', page.text)
        for page in document.pages if page.text.strip()
    ]


BLOCK_BREAK = re.compile(r'\n[ \t]*\n')


def _blocks(text: str, limit: int, overlap: int) -> list[tuple[str, int]]:
    """Paragraphs with their word counts. A paragraph longer than the limit becomes overlapping word windows."""
    blocks = []
    for paragraph in BLOCK_BREAK.split(text):
        words = paragraph.split()
        if not words:
            continue
        if len(words) <= limit:
            blocks.append((paragraph.strip('\n'), len(words)))
            continue
        step = max(1, limit - overlap)
        for start in range(0, len(words), step):
            window = words[start:start + limit]
            blocks.append((' '.join(window), len(window)))
            if start + limit >= len(words):
                break
    return blocks


def structural_chunks(document: ParsedDocument, *, chunk_words: int = 300, chunk_overlap: int = 50) -> list[Chunk]:
    """Paragraph-aware chunks inside one physical page (RET-02, RET-03).

    Paragraphs are packed up to `chunk_words` words, so a chunk never crosses a page and the citation stays exact.
    Paragraph text keeps its line breaks (table rows). Only a paragraph longer than the limit is cut into word
    windows with `chunk_overlap` words of overlap. Size is measured in words: for Russian BPE tokenizers one word
    is roughly 1.5–2 tokens.
    """
    chunks = []
    for page in document.pages:
        if not page.text.strip():
            continue
        packed: list[list[str]] = []
        current: list[str] = []
        size = 0
        for block, words in _blocks(page.text, chunk_words, chunk_overlap):
            if current and size + words > chunk_words:
                packed.append(current)
                current, size = [], 0
            current.append(block)
            size += words
        if current:
            packed.append(current)
        for number, blocks in enumerate(packed, 1):
            chunks.append(_chunk(document, page, f'{document.document_id}:{page.pdf_page}#{number}', '\n\n'.join(blocks)))
    return chunks


CHUNKERS = {'page': page_chunks, 'structural': structural_chunks}
