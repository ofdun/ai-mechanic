import json
from pathlib import Path

import httpx
import pytest

from ml.config import Settings
from ml.contracts import AgentError, AgentRequest, EvidenceBundle, TicketConfirmation
from ml.generation import ClaimGenerator, EvidenceVerifier
from ml.pipeline import Pipeline
from ml.providers.openrouter import OpenRouterClient
from tests.test_generation_generator import ENGINE, OPERATION, PAGE, VALUE

BASE = {'vehicle_id': 'M1', 'session_id': 'S-1', 'status': 'anomaly', 'anomaly_type': 'F08',
        'anomaly_title': 'Дизельная топливоподача', 'anomaly_score': 0.9, 'confidence': 0.85,
        'signal_quality': 'good', 'description': 'Стук', 'model_version': 'test'}
FAILED = {**BASE, 'status': 'check_failed', 'anomaly_type': None, 'anomaly_title': None, 'anomaly_score': None,
          'confidence': None, 'signal_quality': 'bad'}


def make_pipeline(tmp_path: Path, record: dict, tickets: list):
    tmp_path.mkdir(parents=True, exist_ok=True)
    vehicles = tmp_path / 'vehicles.json'
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M1', 'engine_model': ENGINE, 'fuel': 'дизель',
                                                  'sessions': [{**record, 'checked_at': '2026-11-19T10:00:00+03:00'}]}]}),
                        encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'artifacts',
                        llm_api_key='test', llm_base_url='https://llm.test/api/v1', planner_mode='offline')

    def upstream(request):
        if request.url.path == '/diagnose':
            return httpx.Response(200, json=record)
        if request.url.path == '/tickets':
            tickets.append(json.loads(request.content))
            return httpx.Response(201, json={'ticket_id': 'T-1', 'status': 'created'})
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'claims': [OPERATION, VALUE]})}}]})

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=[PAGE])

    http = httpx.Client(transport=httpx.MockTransport(upstream))
    generator = ClaimGenerator(OpenRouterClient(http, settings), settings.a_generator_model)
    return Pipeline(settings, http, retriever=Retriever(), generator=generator, verifier=EvidenceVerifier(ENGINE))


def turn(command, accept=False):
    return AgentRequest(actor_id='mechanic', vehicle_id='M1', session_id='S-1', command=command,
                        accept_signal_warning=accept)


def confirm(pipeline, draft):
    return pipeline.confirm_ticket(TicketConfirmation(actor_id='mechanic', vehicle_id='M1', session_id='S-1',
                                                      draft_id=draft.draft_id, draft_version=1, confirmed=True))


@pytest.mark.parametrize(('record', 'accept', 'verdict', 'confirmable'), [
    (BASE, False, True, False),                                                     # good
    ({**BASE, 'signal_quality': 'warning'}, False, False, True),                   # warning: ask first
    ({**BASE, 'signal_quality': 'warning'}, True, True, False),                    # warning, confirmed
    ({**BASE, 'signal_quality': 'warning', 'confidence': 0.5}, True, False, False),  # below the threshold
    ({**BASE, 'signal_quality': 'bad'}, True, False, False),                       # bad: never
    (FAILED, True, False, False),                                                  # failed: never
])
def test_only_a_confident_noisy_record_can_be_confirmed(tmp_path, record, accept, verdict, confirmable):
    result = make_pipeline(tmp_path, record, []).run(turn('draft_ticket', accept))
    assert result.requires_remeasurement is not verdict
    assert (result.ticket_draft is not None) is verdict
    assert result.warning_confirmable is confirmable and result.min_confidence == 0.7
    if confirmable:
        assert 'подтвердите, что понимаете: запись с помехами' in result.message
    if verdict and record['signal_quality'] == 'warning':
        assert result.signal_warning_accepted and result.ticket_draft.signal_warning_accepted
        assert any('после подтверждения механика' in note for note in result.limitations)


def test_server_checks_the_confirmation_again_on_send(tmp_path):
    tickets = []
    noisy = {**BASE, 'signal_quality': 'warning'}
    pipeline = make_pipeline(tmp_path, noisy, tickets)
    draft = pipeline.run(turn('draft_ticket', accept=True)).ticket_draft
    assert confirm(pipeline, draft).ticket_id == 'T-1' and len(tickets) == 1

    # A draft saved without the confirmation (e.g. from a good record that is noisy now) is refused.
    clean = make_pipeline(tmp_path / 'other', BASE, tickets)
    unconfirmed = clean.run(turn('draft_ticket')).ticket_draft
    assert not unconfirmed.signal_warning_accepted
    clean.services = make_pipeline(tmp_path / 'noisy', noisy, tickets).services
    with pytest.raises(AgentError) as error:
        confirm(clean, unconfirmed)
    assert error.value.code == 'REMEASUREMENT_REQUIRED' and len(tickets) == 1
