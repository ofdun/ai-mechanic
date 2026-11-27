from ml.contracts import AgentResponse
from ml.generation.claims import GroundedClaim, quote_of


def test_verified_excerpt_survives_public_response_and_round_trip():
    claim = GroundedClaim(kind='check', text='Проверить учебный датчик.',
                          quote='Сначала проверить учебный датчик после остановки стенда.',
                          evidence_ids=['training:2'], conflict_group='internal-group')
    response = AgentResponse(request_id='R', vehicle_id='V', session_id='S', engine_model='Учебный Д-01',
                             action='search_manuals', message='Учебный ответ', claims=[claim])
    data = response.model_dump(mode='json')
    assert data['claims'][0]['quote'] == claim.quote
    assert 'conflict_group' not in data['claims'][0]
    restored = AgentResponse.model_validate(data)
    assert quote_of(restored.claims[0]) == claim.quote


def test_old_responses_without_excerpt_remain_readable():
    data = {'request_id': 'R', 'vehicle_id': 'V', 'session_id': 'S', 'engine_model': 'Учебный Д-01',
            'action': 'search_manuals', 'message': 'Учебный ответ', 'claims': [{
                'kind': 'check', 'text': 'Проверить учебный датчик.', 'evidence_ids': ['training:2'],
            }]}
    claim, = AgentResponse.model_validate(data).claims
    assert claim.quote == '' and quote_of(claim) == claim.text
