import pytest

from ml.contracts import Evidence, EvidenceBundle, TechnicalClaim, TechnicalDraft
from ml.generation.claims import GroundedClaim
from ml.generation.verifier import EvidenceVerifier, is_contents_page


ENGINE = 'ЯМЗ-236М2'
# Synthetic manual text in the style of extracted PDF pages, not copied from the customer's manuals.
GUIDE = (
    'РЕГУЛИРОВКА КЛАПАННОГО МЕХАНИЗМА\n'
    'Тепловой зазор впускных и выпускных клапанов регулировать в пределах 0,25-0,30 мм.\n'
    'Зазоры регулировать только на холодном двигателе.\n'
    'Момент затяжки гаек крепления головки 196-216 Н·м (20-22 кгс·м).\n'
    'Запрещается эксплуатация двигателя при давлении масла ниже допустимого.\n'
    'Для двигателя ЯМЗ-238 угол опережения впрыскивания равен 18°.\n'
    'Нарушение регулировки форсунки    Отрегулировать форсунку и, если\n'
    'или засорение распылителя         необходимо, промыть распылитель\n'
)
CONTENTS = '\n'.join(f'{index}. Раздел {index} ........ {index * 10}' for index in range(1, 9))


def page(key='p31', text=GUIDE, models=(ENGINE,), document='re', manual='Fixture РЭ', number=31):
    return Evidence(evidence_id=key, document_id=document, manual=manual, engine_models=list(models),
                    pdf_page=number, text=text)


BUNDLE = EvidenceBundle(items=[
    page(), page('foreign', models=('ЯМЗ-536',)), page('toc', CONTENTS, number=3),
    page('p40', 'Проверить затяжку болтов.', number=40),
])


def claim(text, quote, kind='reference_value', key='p31', group=''):
    return GroundedClaim(kind=kind, text=text, quote=quote, evidence_ids=key if isinstance(key, list) else [key],
                         conflict_group=group)


def reasons(*claims, anomaly_type=None, engine=ENGINE):
    report = EvidenceVerifier(engine, anomaly_type).inspect(TechnicalDraft(claims=list(claims)), BUNDLE)
    return [check.reasons for check in report.checks]


VALUE = 'Тепловой зазор впускных и выпускных клапанов регулировать в пределах 0,25-0,30 мм.'


def test_accepts_claim_with_verbatim_excerpt():
    accepted = claim('Тепловой зазор впускных и выпускных клапанов — 0,25–0,30 мм.', VALUE)
    report = EvidenceVerifier(ENGINE).inspect(TechnicalDraft(claims=[accepted]), BUNDLE)
    assert report.accepted == [accepted]
    assert report.checks[0].excerpt.startswith('Тепловой зазор') and report.checks[0].excerpt.endswith('мм')


@pytest.mark.parametrize(('bad', 'reason'), [
    (claim('Зазор 0,25–0,30 мм.', VALUE, key='missing'), 'unknown_evidence'),
    (claim('Зазор 0,25–0,30 мм.', VALUE, key=['p31', 'p40']), 'several_sources'),
    (claim('Зазор 0,25–0,30 мм.', VALUE, key='foreign'), 'wrong_engine'),
    (claim('Раздел 2 на странице 20.', 'Раздел 2 ........ 20', key='toc'), 'contents_page'),
    (claim('Тепловой зазор клапанов 0,35 мм.', 'Тепловой зазор клапанов 0,35 мм.'), 'quote_not_found'),
    (claim('Тепловой зазор впускных и выпускных клапанов — 0,20–0,30 мм.', VALUE), 'number_not_in_quote'),
    (claim('Момент затяжки гаек крепления головки — 196–216 Н·м.', 'Момент затяжки гаек крепления головки 196-216 Н·м'),
     None),
    (claim('Момент затяжки гаек крепления головки — 200 Н·м.', 'Момент затяжки гаек крепления головки 196-216 Н·м'),
     'number_not_in_quote'),
    (claim('Регулировать зазоры на холодном двигателе.', 'Зазоры регулировать только на холодном двигателе.',
           kind='check'), 'lost_condition'),
    (claim('Эксплуатация двигателя при давлении масла ниже допустимого.',
           'Запрещается эксплуатация двигателя при давлении масла ниже допустимого.', kind='risk'), 'lost_negation'),
    (claim('Эксплуатация при низком давлении масла запрещается.',
           'эксплуатация двигателя при давлении масла ниже допустимого.', kind='risk'), None),
    (claim('Затяжка гаек крепления головки 196 Н·м.', 'крепления головки 196-216 Н·м'), 'range_truncated'),
    (claim('Момент затяжки 96–216 Н·м.', '96-216 Н·м (20-22 кгс·м)'), 'quote_splits_word'),
    (claim('Не регулировать тепловой зазор клапанов в пределах 0,25–0,30 мм.', VALUE, kind='check'),
     'invented_negation'),
    (claim('Угол опережения впрыскивания — 18°.', 'Для двигателя ЯМЗ-238 угол опережения впрыскивания равен 18°.'),
     'other_engine_in_quote'),
    (claim('Заменить головку цилиндров.', VALUE, kind='operation'), 'unrelated_to_quote'),
    (claim('Можно продолжать эксплуатацию с этим зазором 0,25–0,30 мм.', VALUE), 'unsupported_permission'),
    (claim('10 Н·м', '10 Н·м'), 'quote_too_short'),
    (claim('Для ЯМЗ-238 зазор впускных и выпускных клапанов — 0,25–0,30 мм.', VALUE), 'unknown_engine_in_claim'),
    (claim('Для ЯМЗ-236М2 зазор впускных и выпускных клапанов — 0,25–0,30 мм.', VALUE), None),
    (claim('Тепловой зазор впускных и выпускных клапанов — 0,30–0,25 мм.', VALUE), 'numbers_out_of_order'),
])
def test_rejects_unsupported_claims(bad, reason):
    found = reasons(bad)[0]
    assert (reason in found) if reason else found == ()


def test_operation_must_concern_the_faulty_unit():
    operation = claim('Отрегулировать форсунку и, если необходимо, промыть распылитель.',
                      'Отрегулировать форсунку и, если необходимо, промыть распылитель', kind='operation')
    assert reasons(operation, anomaly_type='F08') == [()]
    assert 'off_unit_operation' in reasons(operation, anomaly_type='F02')[0]
    assert reasons(operation) == [()]


def test_duplicates_and_limit():
    first = claim('Тепловой зазор впускных и выпускных клапанов — 0,25–0,30 мм.', VALUE)
    copy = claim('Тепловой  зазор впускных и выпускных клапанов — 0,25–0,30 мм.', VALUE, key='p31')
    assert reasons(first, copy) == [(), ('duplicate',)]
    verifier = EvidenceVerifier(ENGINE, max_claims=1)
    other = claim('Зазоры регулировать только на холодном двигателе.',
                  'Зазоры регулировать только на холодном двигателе.', kind='check')
    report = verifier.inspect(TechnicalDraft(claims=[first, other]), BUNDLE)
    assert report.accepted == [first] and report.checks[1].reasons == ('limit',)


@pytest.mark.parametrize(('source_text', 'operation'), [
    ('Запрещено запускать двигатель.', 'запускать двигатель.'),
    ('Только после остывания открыть крышку.', 'открыть крышку.'),
    ('Затянуть с моментом 110 Н·м.', 'с моментом 10 Н·м.'),
])
def test_claims_without_quote_are_checked_by_their_text(source_text, operation):
    evidence = EvidenceBundle(items=[page('p', source_text)])
    draft = TechnicalDraft(claims=[TechnicalClaim(kind='operation', text=operation, evidence_ids=['p'])])
    assert EvidenceVerifier(ENGINE).verify(draft, evidence).claims == []


def test_verify_keeps_claim_identity_for_the_pipeline():
    accepted = claim('Тепловой зазор впускных и выпускных клапанов — 0,25–0,30 мм.', VALUE)
    verified = EvidenceVerifier().verify(TechnicalDraft(claims=[accepted]), BUNDLE)
    assert verified.claims[0] is accepted


def test_contents_page_detection():
    assert is_contents_page(CONTENTS)
    assert not is_contents_page(GUIDE)
