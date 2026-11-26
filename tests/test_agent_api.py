import asyncio
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip('fastapi')

from fastapi import FastAPI, Header

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
    assert set(app.openapi()['paths']) == {'/agent/run', '/agent/tickets/confirm'}
    target = {'vehicle_id': 'V', 'session_id': 'S'}
    headers = {'X-Test-Actor': 'known-to-host'}

    async def check_routes():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            explicit = {**target, 'command': 'diagnose'}
            assert (await client.post('/agent/run', json=explicit)).status_code == 422
            assert (await client.post('/agent/run', headers=headers, json={**explicit, 'actor_id': 'forged'})).status_code == 422
            empty = await client.post('/agent/run', headers=headers, json=target)
            assert empty.status_code == 422 and empty.json()['detail']['code'] == 'INVALID_INPUT'
            assert (await client.post('/agent/run', headers=headers, json=explicit)).status_code == 200
            assert (await client.post('/agent/run', headers=headers, json={**target, 'message': 'Проверь двигатель.'})).status_code == 200
            decision = {**target, 'draft_id': 'draft', 'draft_version': 1, 'confirmed': False}
            rejected = await client.post('/agent/tickets/confirm', headers=headers, json=decision)
            assert rejected.status_code == 422 and rejected.json()['detail']['code'] == 'CONFIRMATION_REQUIRED'
            receipt = await client.post('/agent/tickets/confirm', headers=headers, json={**decision, 'confirmed': True})
            assert receipt.json()['ticket_id'] == 'TKT-0001'

    asyncio.run(check_routes())
    assert [request.actor_id for request in calls] == ['host-authenticated', 'host-authenticated']
    assert calls[1].message == 'Проверь двигатель.'
