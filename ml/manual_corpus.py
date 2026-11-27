"""Build page lookup and web search indexes from verified engine manuals."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

from ml.config import Settings
from ml.document_ids import document_id_variants
from ml.retrieval import RetrievalConfig, build_index as build_retrieval_index


MANIFEST = Path(__file__).with_name('ingestion') / 'manifest.json'


def document_catalog(manifest: Path | None = None) -> dict[str, dict]:
    entries = json.loads((manifest or MANIFEST).read_text(encoding='utf-8'))['documents']
    return {document_id_variants(entry['document_id'])[0]: entry for entry in entries}


def _index_pages(db: sqlite3.Connection, documents, export: TextIO | None = None) -> int:
    from ml.ingestion.export_pages import manual_name, read_document_pages

    count = 0
    for document, path in documents:
        manual = manual_name(document)
        for page in read_document_pages(document, path):
            db.executemany('INSERT INTO manual_pages VALUES (?, ?, ?, ?, ?)', (
                (document['document_id'], manual, model, page['pdf_page'], page['text'])
                for model in document['models']
            ))
            if export is not None:
                record = {'document_id': document['document_id'], 'models': document['models'], **page}
                export.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
    return count


def build_index(data_dir: Path, output: Path, manifest: Path | None = None, *,
                retrieval_dir: Path | None = None, retrieval_config: RetrievalConfig | None = None) -> int:
    from ml.ingestion.export_pages import read_manifest

    manifest = manifest or MANIFEST
    documents = list(read_manifest(manifest, data_dir))
    if output.resolve() in {source.resolve() for _, source in documents} or output.resolve() == manifest.resolve():
        raise ValueError('Путь индекса совпадает с исходным документом.')
    output.parent.mkdir(exist_ok=True, parents=True)
    with closing(sqlite3.connect(output)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DROP TABLE IF EXISTS manual_pages')
        db.execute('CREATE VIRTUAL TABLE manual_pages USING fts5('
                   'document_id UNINDEXED, manual UNINDEXED, engine UNINDEXED, pdf_page UNINDEXED, text)')
        if retrieval_dir is None:
            return _index_pages(db, documents)

        # Both lookups cite the same verified PDF extraction and physical page numbers.
        with TemporaryDirectory(dir=output.parent) as temp:
            pages = Path(temp) / 'pages.jsonl'
            with pages.open('w', encoding='utf-8') as export:
                count = _index_pages(db, documents, export)
            build_retrieval_index(pages, retrieval_dir, retrieval_config, manifest=manifest)
            return count


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=settings.data_dir)
    parser.add_argument('--output', type=Path, default=settings.artifacts_dir / 'manuals.sqlite3')
    args = parser.parse_args()
    team = settings.web_retrieval_mode == 'team'
    retrieval_dir = settings.artifacts_dir / 'retrieval' if team else None
    count = build_index(args.data_dir, args.output, settings.manual_manifest_path,
                        retrieval_dir=retrieval_dir,
                        retrieval_config=RetrievalConfig.load(settings.retrieval_config_path) if team else None)
    if team:
        print(f'{retrieval_dir}: индекс командного поиска')
    print(f'{args.output}: {count} физических страниц PDF')


if __name__ == '__main__':
    main()
