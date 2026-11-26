from types import SimpleNamespace

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('python_multipart')

from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from ml.api import create_agent_router
from ml.contracts import AgentError, AgentResponse, TicketReceipt


def test_router_uses_server_actor_and_separates_ticket_confirmation():
    calls = []

    def run(request, *, context=None):
        calls.append(request)
        return AgentResponse(request_id='test', vehicle_id=request.vehicle_id, session_id=request.session_id,
                             engine_model='ЯМЗ-236М2', action='diagnose', message='ok')

    def confirm(request):
        if not request.confirmed:
            raise AgentError('CONFIRMATION_REQUIRED', 'Подтвердите заявку.')
        assert request.actor_id == 'host-authenticated'
        return TicketReceipt(ticket_id='TKT-0001', status='created')

    def actor(_key: str = Header(alias='X-Test-Actor')):
        return 'host-authenticated'

    app = FastAPI()
    app.include_router(create_agent_router(SimpleNamespace(run=run, confirm_ticket=confirm), actor))
    target = {'vehicle_id': 'V', 'session_id': 'S'}
    headers = {'X-Test-Actor': 'known-to-host'}
    with TestClient(app) as client:
        assert client.post('/agent/run', json={**target, 'command': 'diagnose'}).status_code == 422
        assert client.post('/agent/run', headers=headers, json={**target, 'actor_id': 'forged',
                                                               'command': 'diagnose'}).status_code == 422
        assert client.post('/agent/run', headers=headers, json={**target, 'command': 'diagnose'}).status_code == 200
        assert client.post('/agent/audio', headers=headers, data=target,
                           files={'audio': ('voice.wav', b'voice bytes', 'audio/wav')}).status_code == 200
        assert client.post('/agent/audio', headers=headers, data={**target, 'vehicle_id': ' '},
                           files={'audio': ('voice.wav', b'voice bytes', 'audio/wav')}).status_code == 422
        decision = {**target, 'draft_id': 'draft', 'draft_version': 1, 'confirmed': False}
        rejected = client.post('/agent/tickets/confirm', headers=headers, json=decision)
        assert rejected.status_code == 422 and rejected.json()['detail']['code'] == 'CONFIRMATION_REQUIRED'
        receipt = client.post('/agent/tickets/confirm', headers=headers, json={**decision, 'confirmed': True})
        assert receipt.json()['ticket_id'] == 'TKT-0001'
    assert [request.actor_id for request in calls] == ['host-authenticated', 'host-authenticated']
    assert calls[1].audio.content == b'voice bytes'
