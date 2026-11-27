import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Event
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

pytest.importorskip('fastapi')

from ml import pdf_baseline
from ml.config import Settings
from ml.contracts import AgentError, AgentRequest
from ml.document_ids import LEGACY_YAMZ_236_OPERATIONS_ID
from ml.planner import OfflinePlanner
from demo.host import create_app
from demo.sessions import WebSessions


@pytest.fixture
def host(tmp_path):
    source = tmp_path / 'data'
    source.mkdir()
    results = []
    for vehicle_id in ('TRUCK-A', 'TRUCK-B'):
        result = {
            'vehicle_id': vehicle_id, 'session_id': 'S1', 'status': 'normal',
            'anomaly_type': None, 'anomaly_title': None, 'anomaly_score': 0.05, 'confidence': 0.9,
            'signal_quality': 'good', 'description': 'Тестовый замер', 'model_version': 'fixture',
        }
        results.append({'vehicle_id': vehicle_id, 'engine_model': 'ДВС-1', 'fuel': 'дизель',
                        'sessions': [{**result, 'checked_at': '2026-01-01T10:00:00Z'}]})
    results[0].update(make='Марка', model='Марка-1')
    (source / 'vehicles.json').write_text(json.dumps({'vehicles': results}))
    manual_bytes = (Path(__file__).resolve().parents[1] / 'demo' / 'training-stand.pdf').read_bytes()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'documents': [{
        'document_id': pdf_baseline.DOCUMENT_ID, 'file': pdf_baseline.PDF_NAME,
        'sha256': hashlib.sha256(manual_bytes).hexdigest(),
    }]}), encoding='utf-8')
    settings = Settings(_env_file=None, data_dir=source, vehicles_path=source / 'vehicles.json',
                        manual_manifest_path=manifest,
                        artifacts_dir=tmp_path, planner_mode='offline', agent_architecture='baseline')
    dist = tmp_path / 'dist'
    dist.mkdir()
    (dist / 'index.html').write_text('<title>Test UI</title>')

    def upstream(request):
        body = json.loads(request.content)
        assert sorted(body) == ['session_id', 'vehicle_id']
        return httpx.Response(200, json=next(v['sessions'][0] for v in results if v['vehicle_id'] == body['vehicle_id']))

    return create_app(settings, transport=httpx.MockTransport(upstream), frontend_dist=dist), settings


def test_web_routes_and_conversation_isolation(host, monkeypatch):
    app, settings = host
    planner_contexts = []
    plan = OfflinePlanner.plan

    def capture(self, question, model, result, *, recent_messages=()):
        planner_contexts.append((result, recent_messages))
        return plan(self, question, model, result, recent_messages=recent_messages)

    monkeypatch.setattr(OfflinePlanner, 'plan', capture)

    async def check():
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app), \
                httpx.AsyncClient(transport=transport, base_url='http://web') as first, \
                httpx.AsyncClient(transport=transport, base_url='http://web') as second:
            assert (await first.get('/health')).json() == {'status': 'ok'}
            assert 'Test UI' in (await first.get('/')).text
            listing = await first.get('/api/vehicles')
            assert 'HttpOnly' in listing.headers['set-cookie']
            catalog = listing.json()['vehicles']
            assert (catalog[0]['make'], catalog[0]['model']) == ('Марка', 'Марка-1')
            assert (catalog[1]['make'], catalog[1]['model']) == (None, None)
            assert set(catalog[0]['sessions'][0]) == {'session_id', 'checked_at', 'status', 'anomaly_score', 'anomaly_type',
                                                      'anomaly_title'}
            assert {vehicle['fleet_status']['rank'] for vehicle in catalog} == set(range(1, len(catalog) + 1))
            target = {'vehicle_id': 'TRUCK-A', 'session_id': 'S1'}
            diagnosis = await first.post('/agent/run', json={**target, 'command': 'diagnose'})
            assert diagnosis.status_code == 200
            conv = diagnosis.json()['conversation_id']
            followup = {**target, 'conversation_id': conv, 'message': 'Покажи историю.'}
            assert (await first.post('/agent/run', json=followup)).status_code == 200
            assert planner_contexts[-1][0].vehicle_id == 'TRUCK-A' and planner_contexts[-1][1]
            assert (await second.post('/agent/run', json=followup)).status_code == 200
            assert planner_contexts[-1] == (None, ())
            mismatch = await first.post('/agent/run', json={**followup, 'vehicle_id': 'TRUCK-B'})
            assert mismatch.status_code == 409 and mismatch.json()['detail']['code'] == 'CONTEXT_MISMATCH'
            forged = await first.post('/agent/run', json={**followup, 'actor_id': 'trusted'})
            assert forged.status_code == 422

    asyncio.run(check())


@pytest.mark.parametrize('indexed_id', [pdf_baseline.DOCUMENT_ID, LEGACY_YAMZ_236_OPERATIONS_ID])
def test_documents_only_indexed_known_bytes(host, indexed_id):
    app, settings = host
    pdf = settings.data_dir / pdf_baseline.PDF_NAME
    pdf.write_bytes((Path(__file__).resolve().parents[1] / 'demo' / 'training-stand.pdf').read_bytes())
    index = settings.artifacts_dir / 'manuals.sqlite3'
    with closing(sqlite3.connect(index)) as db, db:
        db.execute('CREATE TABLE manual_pages (document_id TEXT)')

    async def check():
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://web') as client:
            assert (await client.get('/api/documents/vehicles.json')).status_code == 404
            url = f'/api/documents/{pdf_baseline.DOCUMENT_ID}'
            assert (await client.get(url)).status_code == 404
            assert (await client.get(url + '/pages/2')).status_code == 404
            with closing(sqlite3.connect(index)) as db, db:
                db.execute('INSERT INTO manual_pages VALUES (?)', (indexed_id,))
            for document_id in (pdf_baseline.DOCUMENT_ID, LEGACY_YAMZ_236_OPERATIONS_ID):
                source_url = f'/api/documents/{document_id}'
                result = await client.get(source_url)
                assert result.headers['content-type'] == 'application/pdf' and result.content == pdf.read_bytes()
                page = (await client.get(source_url + '/pages/2')).json()
                assert page['page'] == page['total'] == 2
                assert 'Сверить маркировку' in page['text']
                image = await client.get(source_url + '/pages/2?view=image')
                assert image.headers['content-type'] == 'image/png' and image.content.startswith(b'\x89PNG')
            assert (await client.get(url + '/pages/0')).status_code == 404
            assert (await client.get(url + '/pages/3')).status_code == 404
            pdf.write_bytes(b'%PDF-1.4\nother')
            assert (await client.get(url)).status_code == 404
            assert (await client.get(url + '/pages/2?view=image')).status_code == 404

    asyncio.run(check())


def test_concurrent_turn_is_rejected_and_failed_turn_releases_reservation(host):
    _, settings = host
    sessions = WebSessions(settings.artifacts_dir / 'web-sessions.sqlite3')
    actor, _ = sessions.resolve(None)
    request = AgentRequest(actor_id=actor, conversation_id='chat', vehicle_id='TRUCK-A', session_id='S1', command='diagnose')
    started, release = Event(), Event()
    with httpx.Client() as client, ThreadPoolExecutor() as worker:
        pipeline = pdf_baseline.build_pdf_pipeline(settings, client, settings.artifacts_dir / 'manuals.sqlite3')

        def fail(*args, **kwargs):
            started.set()
            assert release.wait(5)
            raise AgentError('ML_UPSTREAM_ERROR', 'Тестовый сбой')

        pipeline.run = fail
        first = worker.submit(sessions.run, pipeline, request)
        assert started.wait(5)
        try:
            with pytest.raises(AgentError) as busy:
                sessions.run(pipeline, request)
            assert busy.value.code == 'CONVERSATION_BUSY'
        finally:
            release.set()
        with pytest.raises(AgentError) as failed:
            first.result()
        assert failed.value.code == 'ML_UPSTREAM_ERROR'
        with pytest.raises(AgentError) as retry:
            sessions.run(pipeline, request)
        assert retry.value.code == 'ML_UPSTREAM_ERROR'
