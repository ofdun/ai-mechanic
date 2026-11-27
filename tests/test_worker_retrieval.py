import json

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from demo.fixtures import ENGINE, training_fleet
from ml.contracts import AgentError, DocumentPage, ParsedDocument
from ml.providers.openrouter import OpenRouterClient
from ml.retrieval import RetrievalEngine, build_index
from ml.worker import create_app


@pytest.mark.parametrize('external', [False, True])
def test_worker_searches_team_index_in_local_profile(tmp_path, monkeypatch, external):
    fleet = training_fleet()
    parsed = tmp_path / 'parsed'
    parsed.mkdir()
    document = ParsedDocument(
        document_id='training', manual='Руководство стенда', engine_models=[v.engine_model for v in fleet],
        pages=[
            DocumentPage(pdf_page=3, text='Проверить маркировку золотника при обслуживании механизма.',
                         engine_models=[ENGINE]),
            DocumentPage(pdf_page=7, text='Проверить маркировку золотника после каждой смены.',
                         engine_models=[fleet[1].engine_model]),
        ],
    )
    (parsed / 'training.json').write_text(document.model_dump_json())
    (tmp_path / 'vehicles.json').write_text(json.dumps({'vehicles': [v.model_dump(mode='json') for v in fleet]}))
    index = tmp_path / 'retrieval'
    build_index(parsed, index)
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('ARTIFACTS_DIR', str(tmp_path))
    monkeypatch.setenv('WEB_RETRIEVAL_MODE', 'team')
    monkeypatch.setenv('PLANNER_MODE', 'local')
    monkeypatch.setenv('AGENT_ARCHITECTURE', 'a')
    monkeypatch.setenv('ALLOW_EXTERNAL_EVIDENCE', str(external))
    roles = []
    sentence = document.pages[0].text

    def complete_json(self, role, model, messages, schema, **kwargs):
        roles.append(role)
        replies = {
            'a_generator': {'claims': [{'text': sentence, 'kind': 'check', 'evidence_ids': ['training:3'],
                                        'quote': sentence, 'conflict_group': ''}]},
            'a_verifier_finalizer': {'indexes': [0]},
            'plain_answer': {'lead': sentence, 'lines': [], 'assessment': 'Проверка простая.'},
        }
        return schema.model_validate(replies[role])

    monkeypatch.setattr(OpenRouterClient, 'complete_json', complete_json)
    with TestClient(create_app()) as client:
        response = client.post('/run', json={'turn': {
            'vehicle_id': fleet[0].vehicle_id, 'session_id': fleet[0].sessions[0].session_id,
            'actor_id': 'mechanic', 'message': 'Как проверить маркировку золотника?',
        }})
        assert response.status_code == 200
        answer = response.json()
        assert answer['action'] == 'search_manuals'
        assert [c['evidence_ids'] for c in answer['claims']] == [['training:3']]
        assert [e['pdf_page'] for e in answer['evidence']['items']] == [3]
        assert roles == (['a_generator', 'a_verifier_finalizer', 'plain_answer'] if external else [])
        if external:
            assert answer['message'] == answer['summary'] == sentence
    engine = RetrievalEngine.load(index)
    assert engine.page(ENGINE, 'training', 3).text == document.pages[0].text
    assert engine.page(ENGINE, 'training', 7) is None


def test_team_mode_fails_startup_if_index_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('ARTIFACTS_DIR', str(tmp_path))
    monkeypatch.setenv('WEB_RETRIEVAL_MODE', 'team')
    with pytest.raises(AgentError) as failure:
        create_app()
    assert failure.value.code == 'RAG_INDEX_UNAVAILABLE'
