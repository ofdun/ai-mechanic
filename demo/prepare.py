"""Prepare the isolated, synthetic fleet and its two-page search index."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from shutil import copyfile

from demo.fixtures import DOCUMENT_ID, ENGINE, MANUAL, PAGES, PDF_NAME, training_fleet


def prepare(directory: Path) -> None:
    directory.mkdir(exist_ok=True, parents=True)
    artifacts = directory / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    fleet = directory / 'vehicles.json'
    content = json.dumps({'vehicles': [v.model_dump(mode='json') for v in training_fleet()]}, ensure_ascii=False, indent=2)
    if fleet.exists() and fleet.read_text(encoding='utf-8') != content:
        raise ValueError('Каталог уже содержит другой справочник ТС. Выберите отдельный каталог для учебного стенда.')
    fleet.write_text(content, encoding='utf-8')
    copyfile(Path(__file__).with_name(PDF_NAME), directory / PDF_NAME)
    with closing(sqlite3.connect(artifacts / 'manuals.sqlite3')) as db, db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS manual_pages USING fts5('
                   'document_id UNINDEXED, manual UNINDEXED, engine UNINDEXED, pdf_page UNINDEXED, text)')
        db.execute('DELETE FROM manual_pages WHERE document_id=?', (DOCUMENT_ID,))
        db.executemany('INSERT INTO manual_pages VALUES (?, ?, ?, ?, ?)', [
            (DOCUMENT_ID, MANUAL, ENGINE, number, text) for number, text in enumerate(PAGES, 1)
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    prepare(parser.parse_args().directory)
