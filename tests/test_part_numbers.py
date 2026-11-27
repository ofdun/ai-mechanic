import pytest

from ml.contracts import AgentResponse, Evidence, EvidenceBundle, TechnicalClaim
from ml.tools.part_numbers import DESIGNATION, asks_for_parts, part_numbers

ENGINE = 'ЯМЗ-236М2'
PUMP = ('4.11 Насос водяной 236-1307010-А3\n'
        'Прокладка (поз. 12) 5340.1002246-01 [536-20-050020-530]\n'
        '406.1004045-01\n'
        'Частота вращения 1500-1800 мин-1, момент 100-125 Н·м, зазор 0,25-0,30 мм, 130.02 мм.\n')


def page(evidence_id, text, engine=ENGINE, pdf_page=238):
    return Evidence(evidence_id=evidence_id, document_id='yamz-236_rk_2014', manual='РК ЯМЗ-236', engine_models=[engine],
                    pdf_page=pdf_page, text=text)


def response(claims, *pages):
    return AgentResponse(request_id='r', vehicle_id='v', session_id='s', engine_model=ENGINE, action='search_manuals',
                         message='m', evidence=EvidenceBundle(items=list(pages)),
                         claims=[TechnicalClaim(kind=kind, text='x', evidence_ids=ids) for kind, ids in claims])


@pytest.mark.parametrize(('text', 'codes'), [
    (PUMP, ['236-1307010-А3', '5340.1002246-01', '406.1004045-01']),
    ('Шатун в сборе 236-1004045-', ['236-1004045']),
    ('датчик 1322.3771 и стартер 5234.3708000', ['1322.3771', '5234.3708000']),
    ('1500-1800 мин-1; 100-125 Н·м; 22-23 МПа; 0.0035 мм; 2026-11-19', []),
])
def test_designations_and_not_values(text, codes):
    assert [match.group(1) for match in DESIGNATION.finditer(text)] == codes


def test_only_cited_pages_of_this_engine_with_their_line():
    answer = response([('operation', ['p1'])], page('p1', PUMP), page('p2', 'Насос 238-1307010'),
                      page('p3', 'Насос 740.1307010', engine='КАМАЗ-740', pdf_page=5))
    found = part_numbers(answer)
    assert [item.designation for item in found] == ['236-1307010-А3', '5340.1002246-01', '406.1004045-01']
    assert found[0].context == '4.11 Насос водяной 236-1307010-А3' and found[0].pdf_page == 238
    # A bare code line takes its name from the line above.
    assert found[2].context.startswith('Прокладка (поз. 12)')
    other_engine = response([('operation', ['p3'])], page('p3', 'Насос 740.1307010', engine='КАМАЗ-740'))
    assert part_numbers(other_engine) == []


def test_part_questions_are_recognised():
    assert asks_for_parts(response([('operation', ['p1'])]), 'Какой номер детали у насоса?')
    assert asks_for_parts(response([('spare_part', ['p1'])]), None)
    assert not asks_for_parts(response([('operation', ['p1'])]), 'Какой момент затяжки?')


def test_pipeline_lists_designations_or_says_there_are_none(tmp_path):
    import json

    import httpx

    from ml.config import Settings
    from ml.contracts import AgentRequest
    from ml.generation import ClaimGenerator, EvidenceVerifier
    from ml.pipeline import Pipeline
    from ml.providers.openrouter import OpenRouterClient
    from tests.test_generation_generator import OPERATION, PAGE

    record = {'vehicle_id': 'M1', 'session_id': 'S-1', 'status': 'normal', 'anomaly_type': None, 'anomaly_title': None,
              'anomaly_score': 0.1, 'confidence': 0.9, 'signal_quality': 'good', 'description': '', 'model_version': 't'}
    vehicles = tmp_path / 'vehicles.json'
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M1', 'engine_model': ENGINE, 'fuel': 'дизель',
                                                  'sessions': [{**record, 'checked_at': '2026-11-19T10:00:00+03:00'}]}]}),
                        encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'a', llm_api_key='k',
                        llm_base_url='https://llm.test/api/v1', planner_mode='offline')
    with_code = PAGE.model_copy(update={'text': PAGE.text + 'Форсунка 267.1112010-02\n'})
    pages = iter([[with_code], [PAGE]])

    def upstream(request):
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({'claims': [OPERATION]})}}]})

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=next(pages))

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        pipeline = Pipeline(settings, http, retriever=Retriever(),
                            generator=ClaimGenerator(OpenRouterClient(http, settings), settings.a_generator_model),
                            verifier=EvidenceVerifier(ENGINE))
        ask = lambda text: pipeline.run(AgentRequest(actor_id='a', vehicle_id='M1', session_id='S-1', message=text))
        found = ask('Какая деталь нужна для форсунки?')
        assert [(item.designation, item.pdf_page) for item in found.part_numbers] == [('267.1112010-02', 52)]
        assert found.part_numbers_note is None
        missing = ask('Какой номер детали распылителя?')
        assert missing.part_numbers == [] and missing.part_numbers_note.startswith('На страницах руководства')
