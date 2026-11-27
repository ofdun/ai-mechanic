import json
from pathlib import Path

import httpx
import pytest

from ml.config import Settings
from ml.contracts import (
    AgentRequest,
    AgentResponse,
    EvidenceBundle,
    LocalizedText,
    TechnicalClaim,
)
from ml.generation import ClaimGenerator, EvidenceVerifier, MechanicFinalizer
from ml.generation.i18n import LABEL_KEYS, LABELS, glossary_for, labels
from ml.generation.translate import Localizer, checked, source_items
from ml.pipeline import Pipeline
from ml.providers.openrouter import OpenRouterClient
from tests.test_generation_generator import ENGINE, OPERATION, PAGE, VALUE

UZ = {
    'm': 'Dvigatel qo‘llanmasi bo‘yicha savol.',
    'c0': 'Forsunkani sozlang va kerak bo‘lsa, purkagichni yuving.',
    'c1': 'Forsunka purkashining boshlanish bosimi — 22–23 MPa.',
}


def response(**update) -> AgentResponse:
    base = AgentResponse(
        request_id='r', vehicle_id='v', session_id='s', engine_model=ENGINE, action='search_manuals',
        message='Вопрос по руководству выбранного двигателя.',
        summary='Отрегулируйте форсунку.\n⚠️ Оценка агента, не из руководства: Лучше проверить до выезда.',
        evidence=EvidenceBundle(items=[PAGE]),
        claims=[TechnicalClaim(kind='operation', text=OPERATION['text'], evidence_ids=['p52']),
                TechnicalClaim(kind='reference_value', text=VALUE['text'], evidence_ids=['p52'])],
        limitations=['Решение о ремонте принимает механик.'],
    )
    return base.model_copy(update=update)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete_json(self, role, model, messages, response_type, **options):
        self.calls.append(json.loads(messages[-1]['content']))
        items = self.reply(self.calls[-1]['items'])
        return response_type.model_validate({'items': [{'id': key, 'text': text} for key, text in items.items()]})


def test_labels_cover_every_language():
    for language in LABELS:
        assert set(labels(language)) == set(LABEL_KEYS)
        assert all(labels(language).values())
    assert labels('tg')['section.risk'] == 'Хатар' and labels('uz')['section.check'] == 'Nimani tekshirish kerak'


def test_glossary_matches_word_stems():
    found = glossary_for('uz', ['Давление масла и зазор клапанов'])
    assert found['масл'] == 'moy' and 'зазор' in found and 'форсунк' not in found
    assert 'коренн подшипник' in glossary_for('tg', ['Болты коренных подшипников'])


@pytest.mark.parametrize(('target', 'translated'), [
    ('Forsunka purkashining boshlanish bosimi — 22–23 MPa.', True),
    ('Forsunka purkashining boshlanish bosimi — 25 MPa.', False),            # value changed
    ('Forsunka purkashining boshlanish bosimi — 23–22 MPa.', False),         # order changed
    ('Давление начала впрыскивания форсунки — 22–23 МПа.', False),           # came back in Russian
    ('', False),
])
def test_translation_keeps_every_number(target, translated):
    assert checked(VALUE['text'], target).translated is translated


def test_safety_critical_lines_keep_the_russian_original():
    source = 'Не допускается работа двигателя при давлении масла ниже 100 кПа.'
    line = checked(source, 'Moy bosimi 100 kPa dan past bo‘lsa, dvigatelni ishlatish mumkin emas.')
    assert line.translated and line.original == source
    plain = checked(VALUE['text'], 'Forsunka purkashining boshlanish bosimi — 22–23 MPa.')
    assert plain.original is None


def test_localizer_translates_and_falls_back_line_by_line():
    def reply(items):
        texts = {item['id']: item['text'] for item in items}
        assert 'Оценка агента' not in json.dumps(texts, ensure_ascii=False)  # the label never reaches the model
        return {**UZ, 's0': 'Forsunkani sozlang.', 's1': 'Reysdan oldin tekshirgan ma’qul.', 'l0': '',
                'c1': 'Forsunka purkashining boshlanish bosimi — 25 MPa.'}

    client = FakeClient(reply)
    answer = Localizer(client, 'model').localize(response(), 'uz')
    assert client.calls[0]['language'] == 'узбекский (латиница)' and 'форсунк' in client.calls[0]['glossary']
    assert answer.message.text == UZ['m']
    assert [line.text for line in answer.summary] == [
        'Forsunkani sozlang.', '⚠️ Agent bahosi, qo‘llanmadan emas: Reysdan oldin tekshirgan ma’qul.'.replace('‘', "'")]
    assert answer.claims[0] == LocalizedText(text=UZ['c0'], original=OPERATION['text'])   # condition «если»
    assert answer.claims[1] == LocalizedText(text=VALUE['text'], translated=False)        # 25 instead of 22–23
    assert answer.limitations[0].translated is False
    assert answer.limitations[-1].text == labels('uz')['untranslated']
    assert answer.labels['section.operation'] == 'Nima qilish kerak'


def test_without_a_model_the_answer_stays_russian_with_local_labels():
    answer = Localizer(None, 'model').localize(response(), 'tg')
    assert not any(line.translated for line in answer.claims) and answer.labels['section.risk'] == 'Хатар'
    assert answer.limitations[-1].text == labels('tg')['untranslated']
    assert set(source_items(response())) == {'m', 's0', 's1', 'c0', 'c1', 'l0'}


@pytest.fixture
def pipeline(tmp_path: Path):
    vehicles = tmp_path / 'vehicles.json'
    record = {'vehicle_id': 'M333YK47', 'session_id': 'S-0005', 'status': 'anomaly', 'anomaly_type': 'F08',
              'anomaly_title': 'Дизельная топливоподача', 'anomaly_score': 0.9, 'confidence': 0.8,
              'signal_quality': 'good', 'description': 'Стук', 'model_version': 'test'}
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M333YK47', 'engine_model': ENGINE, 'fuel': 'дизель',
                                                  'sessions': [{**record, 'checked_at': '2026-11-19T10:00:00+03:00'}]}]}),
                        encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'artifacts',
                        llm_api_key='test', llm_base_url='https://llm.test/api/v1', planner_mode='offline',
                        allow_external_evidence=True)

    def upstream(request):
        if request.url.path == '/diagnose':
            return httpx.Response(200, json=record)
        body = json.loads(request.content)
        if body['response_format']['json_schema']['name'] == 'TranslationOutput':
            items = json.loads(body['messages'][-1]['content'])['items']
            reply = {'items': [{'id': item['id'], 'text': UZ.get(item['id'], 'Mexanik qaror qabul qiladi.')}
                               for item in items]}
        else:
            reply = {'claims': [OPERATION, VALUE]}
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(reply)}}]})

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=[PAGE])

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        generator = ClaimGenerator(OpenRouterClient(http, settings), settings.a_generator_model)
        yield Pipeline(settings, http, retriever=Retriever(), generator=generator, verifier=EvidenceVerifier(ENGINE),
                       finalizer=MechanicFinalizer())


def test_pipeline_answers_in_uzbek_and_keeps_the_ticket_russian(pipeline):
    result = pipeline.run(AgentRequest(actor_id='mechanic', vehicle_id='M333YK47', session_id='S-0005',
                                       command='draft_ticket', language='uz'))
    assert result.language == 'uz' and result.localized.language == 'uz'
    assert [claim.text for claim in result.claims] == [OPERATION['text'], VALUE['text']]
    assert result.ticket_draft.payload.operations == [OPERATION['text']]
    assert [line.text for line in result.localized.claims] == [UZ['c0'], UZ['c1']]
    assert 'Nima qilish kerak:' in result.message and "Me'yor:" in result.message
    assert f'↳ {labels("uz")["original"]}: {OPERATION["text"]}' in result.message
    assert '(Fixture РЭ, bet 52)' in result.message


def test_russian_requests_are_unchanged(pipeline):
    result = pipeline.run(AgentRequest(actor_id='mechanic', vehicle_id='M333YK47', session_id='S-0005',
                                       command='draft_ticket'))
    assert result.language == 'ru' and result.localized is None


def test_translation_check_reads_oracle_traces():
    from evaluation.generation.translation_check import answer_from_trace
    row = {'id': 'Q1', 'trace': {'summary': 'Лид.\nОтрегулируйте форсунку.',
                                 'final': [{'kind': 'operation', 'text': OPERATION['text'], 'pages': []}]}}
    answer = answer_from_trace(row)
    assert answer.message == 'Лид.' and answer.summary == 'Отрегулируйте форсунку.'
    assert answer.claims[0].text == OPERATION['text']
    assert answer_from_trace({'id': 'Q2', 'trace': {'final': []}}) is None
