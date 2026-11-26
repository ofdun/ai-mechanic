"""Build the web application's SQLite index from the six verified hackathon manuals."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from ml.config import Settings
from ml.ingestion.export_pages import manual_name, read_document_pages, read_manifest


MANIFEST = Path(__file__).with_name('ingestion') / 'manifest.json'


def document_catalog() -> dict[str, dict]:
    entries = json.loads(MANIFEST.read_text(encoding='utf-8'))['documents']
    return {entry['document_id']: entry for entry in entries}


def build_index(data_dir: Path, output: Path) -> int:
    documents = list(read_manifest(MANIFEST, data_dir))
    if output.resolve() in {source.resolve() for _, source in documents} or output.resolve() == MANIFEST.resolve():
        raise ValueError('Путь индекса совпадает с исходным документом.')
    output.parent.mkdir(exist_ok=True, parents=True)
    count = 0
    with closing(sqlite3.connect(output)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DROP TABLE IF EXISTS manual_pages')
        db.execute('CREATE VIRTUAL TABLE manual_pages USING fts5('
                   'document_id UNINDEXED, manual UNINDEXED, engine UNINDEXED, pdf_page UNINDEXED, text)')
        for document, path in documents:
            for page in read_document_pages(document, path):
                db.executemany('INSERT INTO manual_pages VALUES (?, ?, ?, ?, ?)', (
                    (document['document_id'], manual_name(document), model, page['pdf_page'], page['text'])
                    for model in document['models']
                ))
                count += 1
    return count


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=settings.data_dir)
    parser.add_argument('--output', type=Path, default=settings.artifacts_dir / 'manuals.sqlite3')
    args = parser.parse_args()
    count = build_index(args.data_dir, args.output)
    print(f'{args.output}: {count} физических страниц PDF')


if __name__ == '__main__':
    main()
