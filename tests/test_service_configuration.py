import hashlib
import json

import httpx
import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from demo.fixtures import DOCUMENT_ID, OPERATION, PAGES, training_fleet
from demo.prepare import prepare
from demo.host import create_app


@pytest.mark.parametrize('override', [None, 'ML_API_URL', 'MAINTENANCE_API_URL'])
def test_web_uses_independent_service_urls_from_dotenv_and_environment(tmp_path, monkeypatch, override):
    monkeypatch.chdir(tmp_path)
    prepare(tmp_path)
    source = f'{DOCUMENT_ID}:2'
    reviews = {
        'pages': {source: {'sha256': hashlib.sha256(' '.join(PAGES[1].split()).encode()).hexdigest()}},
        'topics': {}, 'entries': [{'fault_code': 'F08', 'claims': [
            {'kind': 'operation', 'text': OPERATION, 'evidence_ids': [source]},
        ]}],
    }
    (tmp_path / 'reviews.json').write_text(json.dumps(reviews))
    config = {
        'DATA_DIR': '.', 'VEHICLES_PATH': 'vehicles.json', 'ARTIFACTS_DIR': 'artifacts',
        'PLANNER_MODE': 'local', 'REVIEWED_EVIDENCE_PATH': 'reviews.json',
        'TICKET_URGENCY_BY_FAULT': '{"F08":"high"}', 'ML_API_URL': 'https://ml.test/v2/',
        'MAINTENANCE_API_URL': 'https://toir.test/workshop',
    }
    (tmp_path / '.env').write_text('\n'.join(f'{key}={value}' for key, value in config.items()))
    for key in config:
        monkeypatch.delenv(key, raising=False)
    if override:
        monkeypatch.setenv(override, 'https://replacement.test/prefix')
    diagnosis_url = config['ML_API_URL'].rstrip('/') if override != 'ML_API_URL' else 'https://replacement.test/prefix'
    ticket_url = config['MAINTENANCE_API_URL'] if override != 'MAINTENANCE_API_URL' else 'https://replacement.test/prefix'
    result = training_fleet()[0].sessions[0]
    target = {'vehicle_id': result.vehicle_id, 'session_id': result.session_id}
    posted_tickets = []

    def upstream(request):
        assert request.method == 'POST'
        payload = json.loads(request.content)
        if str(request.url) == diagnosis_url + '/diagnose':
            assert payload == target
            return httpx.Response(200, json={**result.model_dump(mode='json'), 'model_version': 'replacement'})
        assert str(request.url) == ticket_url + '/tickets'
        posted_tickets.append(payload)
        return httpx.Response(201, json={'ticket_id': 'replacement-receipt', 'status': 'created'})

    with TestClient(create_app(transport=httpx.MockTransport(upstream))) as web:
        diagnosis = web.post('/agent/run', json={**target, 'command': 'diagnose'}).json()
        assert diagnosis['diagnostic']['model_version'] == 'replacement'
        draft = web.post('/agent/run', json={**target, 'command': 'draft_ticket'}).json()['ticket_draft']
        assert draft['payload']['urgency'] == 'high' and not posted_tickets
        decision = {**target, 'draft_id': draft['draft_id'], 'draft_version': draft['draft_version'], 'confirmed': True}
        for _ in range(2):
            assert web.post('/agent/tickets/confirm', json=decision).json()['ticket_id'] == 'replacement-receipt'
        assert posted_tickets == [draft['payload']]
