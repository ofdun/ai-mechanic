import argparse
import hashlib
import re
import sqlite3
import subprocess
import unicodedata
from contextlib import closing
from pathlib import Path

import httpx

from ml.config import Settings
from ml.contracts import (
    AgentError, DiagnosticResult, Evidence, EvidenceBundle, SearchPlan, TechnicalClaim, TechnicalDraft, VerifiedResponse,
)
from ml.pipeline import Pipeline


ENGINE = 'ЯМЗ-236М2'
DOCUMENT_ID = 'yamz-236-operations-2010'
MANUAL = '236-3902150-Б РЭ'
PDF_NAME = 'ЯМЗ_236_руководство по эксплуатации.pdf'
PDF_SHA256 = '4268a9519e02a668a826fe6d154de0360f6767015868ca2de7410ba570d73dd1'
FIRST_PAGE, LAST_PAGE = 52, 54
INJECTOR_ROW = re.compile(
    r'^[ \t]*(Нарушение регулировки[ \t]+или)[ \t]{2,}(Отрегулировать форсунку и, если)[ \t]*\n'
    r'[ \t]*(засорение форсунки)[ \t]{2,}(необходимо, промыть и прочистить)[ \t]*\n'
    r'[ \t]{2,}(е[её])[ \t]*$',
    re.MULTILINE | re.IGNORECASE,
)
QUERY_TERM = re.compile(r'[^\W_]{4,}', re.UNICODE)


def read_pages(data_dir: Path) -> list[str]:
    matches = [path for path in data_dir.glob('*.pdf') if unicodedata.normalize('NFC', path.name) == PDF_NAME]
    if len(matches) != 1:
        raise AgentError('RAG_INDEX_UNAVAILABLE', f'Ожидался один файл {PDF_NAME} в DATA_DIR.')
    try:
        if hashlib.sha256(matches[0].read_bytes()).hexdigest() != PDF_SHA256:
            raise AgentError('RAG_INDEX_UNAVAILABLE', 'PDF отличается от версии с проверенной применимостью страниц.')
        extraction = subprocess.run(
            ['pdftotext', '-layout', '-f', str(FIRST_PAGE), '-l', str(LAST_PAGE), str(matches[0]), '-'],
            capture_output=True, check=True, text=True, encoding='utf-8', timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise AgentError('RAG_INDEX_UNAVAILABLE', 'Не удалось прочитать PDF через pdftotext.') from error
    pages = extraction.split('\f')[:LAST_PAGE - FIRST_PAGE + 1]
    if len(pages) != LAST_PAGE - FIRST_PAGE + 1 or any(not page.strip() for page in pages):
        raise AgentError('RAG_INDEX_UNAVAILABLE', 'В PDF отсутствует ожидаемый текст физических страниц.')
    return pages


def build_index(data_dir: Path, output: Path) -> None:
    pages = read_pages(data_dir)
    if not INJECTOR_ROW.search(pages[0]):
        raise AgentError('RAG_INDEX_UNAVAILABLE', 'Извлечение PDF не сохранило проверенную строку таблицы.')
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(output)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DROP TABLE IF EXISTS manual_pages')
            db.execute('''CREATE VIRTUAL TABLE manual_pages USING fts5(
                          document_id UNINDEXED, manual UNINDEXED, engine UNINDEXED, pdf_page UNINDEXED, text)''')
            db.executemany('INSERT INTO manual_pages VALUES (?, ?, ?, ?, ?)', [
                (DOCUMENT_ID, MANUAL, ENGINE, number, page) for number, page in enumerate(pages, FIRST_PAGE)
            ])
    except (OSError, sqlite3.Error) as error:
        raise AgentError('RAG_INDEX_UNAVAILABLE', 'Не удалось собрать индекс SQLite FTS5.') from error


class PdfPageRetriever:
    def __init__(self, index: Path):
        self.index = index

    def search(self, plan: SearchPlan) -> EvidenceBundle:
        fault_terms = ['форсунки', 'впрыскивания', 'топливной'] if plan.anomaly_type == 'F08' else []
        queries = [*fault_terms, plan.rewritten_query, plan.original_query, *plan.subqueries]
        terms = dict.fromkeys(QUERY_TERM.findall(' '.join(queries).casefold()))
        if not terms:
            return EvidenceBundle()
        query = ' OR '.join(f'"{term}"' for term in list(terms)[:24])
        try:
            with closing(sqlite3.connect(self.index.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                rows = db.execute('''SELECT document_id, manual, engine, pdf_page, text FROM manual_pages
                                     WHERE manual_pages MATCH ? AND engine = ?
                                     ORDER BY bm25(manual_pages), pdf_page LIMIT 5''',
                                  (query, plan.engine_model)).fetchall()
        except sqlite3.Error as error:
            raise AgentError('RAG_INDEX_UNAVAILABLE', 'Индекс PDF недоступен. Запустите сборку индекса.') from error
        return EvidenceBundle(items=[Evidence(
            evidence_id=f'{document}:{page}', document_id=document, manual=manual, engine_models=[engine],
            pdf_page=page, text=text,
        ) for document, manual, engine, page, text in rows])


class ReviewedOperations:
    """Deterministic F08 baseline for a reviewed row of the extracted manual text."""

    @staticmethod
    def operations(evidence: EvidenceBundle) -> list[TechnicalClaim]:
        claims = []
        for item in evidence.items:
            if (item.document_id, item.manual, item.pdf_page) != (DOCUMENT_ID, MANUAL, FIRST_PAGE) or ENGINE not in item.engine_models:
                continue
            if row := INJECTOR_ROW.search(item.text):
                cause = ' '.join(f'{row[1]} {row[3]}'.split()).lower()
                repair = ' '.join((row[2], row[4], row[5])).lower()
                claims.append(TechnicalClaim(
                    kind='operation', text=f'Если подтверждено {cause}, {repair}.', evidence_ids=[item.evidence_id],
                ))
        return claims

    def generate(self, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None) -> TechnicalDraft:
        if plan.engine_model != ENGINE or plan.anomaly_type != 'F08' or not diagnostic or diagnostic.anomaly_type != 'F08':
            return TechnicalDraft()
        return TechnicalDraft(claims=self.operations(evidence))

    def verify(self, draft: TechnicalDraft, evidence: EvidenceBundle) -> VerifiedResponse:
        supported = self.operations(evidence)
        return VerifiedResponse(claims=[claim for claim in draft.claims if claim in supported])


def build_pdf_pipeline(settings: Settings, http: httpx.Client, index: Path) -> Pipeline:
    operations = ReviewedOperations()
    return Pipeline(settings, http, retriever=PdfPageRetriever(index), generator=operations, verifier=operations)


def main() -> int:
    parser = argparse.ArgumentParser(description='Сборка проверенного поискового baseline по PDF')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = Settings()
    output = args.output or settings.artifacts_dir / 'manuals.sqlite3'
    try:
        build_index(args.data_dir or settings.data_dir, output)
    except AgentError as error:
        parser.exit(1, f'{error.code}: {error}\n')
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
