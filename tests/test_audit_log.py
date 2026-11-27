import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from ml.config import Settings
from ml.contracts import AgentError, AgentRequest, EvidenceBundle, TicketConfirmation
from ml.generation import ClaimGenerator, EvidenceVerifier
from ml.pipeline import Pipeline
from ml.providers.openrouter import OpenRouterClient
from tests.test_generation_generator import ENGINE, OPERATION, PAGE, VALUE

RECORD = {'vehicle_id': 'M1', 'session_id': 'S-1', 'status': 'anomaly', 'anomaly_type': 'F08',
          'anomaly_title': 'Дизельная топливоподача', 'anomaly_score': 0.9, 'confidence': 0.85,
          'signal_quality': 'good', 'description': 'Стук', 'model_version': 'neuro-1'}


@pytest.fixture
def pipeline(tmp_path: Path):
    vehicles = tmp_path / 'vehicles.json'
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M1', 'engine_model': ENGINE, 'fuel': 'дизель',
                                                  'sessions': [{**RECORD, 'checked_at': '2026-11-19T10:00:00+03:00'}]}]}),
                        encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'artifacts',
                        llm_api_key='secret-key', llm_base_url='https://llm.test/api/v1', planner_mode='offline')

    def upstream(request):
        if request.url.path == '/diagnose':
            return httpx.Response(200, json=RECORD)
        if request.url.path == '/tickets':
            return httpx.Response(201, json={'ticket_id': 'T-1', 'status': 'created'})
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'claims': [OPERATION, VALUE]})}}]})

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=[PAGE])

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        generator = ClaimGenerator(OpenRouterClient(http, settings), settings.a_generator_model)
        yield Pipeline(settings, http, retriever=Retriever(), generator=generator, verifier=EvidenceVerifier(ENGINE))


def ask(pipeline, **turn):
    return pipeline.run(AgentRequest(actor_id='mechanic-7', vehicle_id='M1', session_id='S-1', **turn))


def test_chain_from_question_to_ticket(pipeline):
    ask(pipeline, command='diagnose')
    ask(pipeline, message='Какое давление впрыска?')
    draft = ask(pipeline, command='draft_ticket').ticket_draft
    pipeline.confirm_ticket(TicketConfirmation(actor_id='mechanic-7', vehicle_id='M1', session_id='S-1',
                                               draft_id=draft.draft_id, draft_version=1, confirmed=True))
    chain = pipeline.audit.chain('M1', 'S-1')
    assert [(entry['kind'], entry.get('command'), entry.get('action')) for entry in chain] == [
        ('turn', 'diagnose', 'diagnose'), ('turn', 'auto', 'search_manuals'),
        ('turn', 'draft_ticket', 'draft_ticket'), ('ticket_confirm', None, None)]
    question, drafted, sent = chain[1], chain[2], chain[3]
    assert question['question'] == 'Какое давление впрыска?' and question['answer']
    assert question['cited_pages'] == [['Fixture РЭ', 52]] and question['actor_id'] == 'mechanic-7'
    assert chain[0]['diagnostic']['model_version'] == 'neuro-1'
    assert sent['draft_id'] == drafted['draft_id'] == draft.draft_id and sent['ticket_id'] == 'T-1'
    assert 'secret-key' not in pipeline.audit.path.read_bytes().decode('utf-8', 'ignore')


def test_refusals_are_logged_too(pipeline):
    with pytest.raises(AgentError):
        pipeline.confirm_ticket(TicketConfirmation(actor_id='mechanic-7', vehicle_id='M1', session_id='S-1',
                                                   draft_id='missing', draft_version=1, confirmed=True))
    assert pipeline.audit.chain('M1', 'S-1')[-1]['error'] == 'DRAFT_NOT_FOUND'


def test_old_records_are_removed_and_a_broken_log_does_not_stop_answers(pipeline):
    ask(pipeline, command='diagnose')
    with sqlite3.connect(pipeline.audit.path) as db:
        db.execute("UPDATE audit SET at='2020-01-01T00:00:00+00:00'")
    ask(pipeline, command='diagnose')
    assert len(pipeline.audit.chain('M1', 'S-1')) == 1          # the old one is past the 90-day retention
    pipeline.audit.path.unlink()
    pipeline.audit.path.mkdir()                                  # the log file cannot be opened now
    assert ask(pipeline, command='diagnose').diagnostic.status == 'anomaly'
