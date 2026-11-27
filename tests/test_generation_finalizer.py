from datetime import datetime, timedelta, timezone

import pytest

from ml.contracts import (
    AgentResponse, DiagnosticResult, Evidence, EvidenceBundle, SearchPlan, SessionRecord, TechnicalClaim, TicketDraft,
    TicketPayload, TicketReference,
)
from ml.generation.claims import GroundedClaim
from ml.generation.finalizer import MechanicFinalizer, conflict_notes, find_conflicts, trend
from ml.generation.profiles import mentions_unit, urgency_for


ENGINE = 'ЯМЗ-236М2'
RE = Evidence(evidence_id='re31', document_id='re', manual='Fixture РЭ', engine_models=[ENGINE], pdf_page=31, text='…')
RK = Evidence(evidence_id='rk269', document_id='rk', manual='Fixture РК', engine_models=[ENGINE], pdf_page=269, text='…')
EVIDENCE = EvidenceBundle(items=[RE, RK])
FINALIZER = MechanicFinalizer()


def result(status='anomaly', *, code='F08', score=0.76, confidence=0.73, signal='good', description='Стук, дымный выхлоп'):
    return DiagnosticResult(
        vehicle_id='M333YK47', session_id='S-0005', status=status, anomaly_type=code if status == 'anomaly' else None,
        anomaly_title={'F08': 'Дизельная топливоподача', 'F02': 'Низкое давление масла'}.get(code) if status == 'anomaly' else None,
        anomaly_score=None if status == 'check_failed' else score, confidence=None if status == 'check_failed' else confidence,
        signal_quality=signal, description=description, model_version='mock-v1')


def response(action='diagnose', **fields):
    return AgentResponse(**{'request_id': 'r', 'vehicle_id': 'M333YK47', 'session_id': 'S-0005', 'engine_model': ENGINE,
                            'action': action, 'message': 'pipeline', **fields})


def claim(kind, text, key='re31', group=''):
    return GroundedClaim(kind=kind, text=text, quote=text, evidence_ids=[key], conflict_group=group)


PLAN = SearchPlan(engine_model=ENGINE, original_query='Что проверить?', rewritten_query='Что проверить?', anomaly_type='F08')
CLAIMS = [
    claim('cause', 'Возможная причина: засорение форсунки.'),
    claim('check', 'Проверить давление начала впрыскивания на стенде.'),
    claim('reference_value', 'Давление начала впрыскивания — 22 МПа.'),
    claim('operation', 'Отрегулировать форсунку.'),
]


def test_normal_verdict_is_short():
    text = FINALIZER.compose(response(diagnostic=result('normal', score=0.05, confidence=0.81)))
    assert text.splitlines()[:2] == ['✅ Норма — значимых отклонений не обнаружено.',
                                     'Можно продолжать эксплуатацию в обычном режиме.']
    assert '0,05' in text and '81%' in text


def test_failed_check_makes_no_engine_conclusion():
    text = FINALIZER.compose(response(diagnostic=result('check_failed', signal='bad', description='Перегрузка микрофона'),
                                      requires_remeasurement=True, message='Запись непригодна для заключения.'))
    assert text.startswith('🔁 Сбой проверки — заключение о двигателе не формируется.')
    assert 'Перегрузка микрофона' in text and 'повторите запись' in text
    assert 'F08' not in text and 'Приоритет' not in text and 'Риск' not in text


def test_low_confidence_anomaly_hides_the_fault_hypothesis():
    text = FINALIZER.compose(response(diagnostic=result(confidence=0.4), requires_remeasurement=True,
                                      message='Уверенность модели ниже установленного порога. Повторите замер.'))
    assert 'Дизельная топливоподача' not in text and 'порога' in text


def test_data_conflict_is_reported_without_ticket():
    text = FINALIZER.compose(response(diagnostic=result(), message='Гипотеза ML несовместима с типом топлива.'))
    assert text.startswith('⚠️ Отклонение «Дизельная топливоподача (F08)» требует проверки данных.')
    assert 'несовместима' in text and 'Заявку не формирую' in text


def test_anomaly_answer_follows_the_required_order():
    answer = FINALIZER.answer(response(diagnostic=result(), search_plan=PLAN, evidence=EVIDENCE, claims=CLAIMS,
                                       can_create_ticket=True))
    lines = answer.text.splitlines()
    assert lines[0] == '⚠️ Отклонение: Дизельная топливоподача (F08)'
    assert lines[1] == 'Приоритет: средний — дефектовка на ближайшем ТО, эксплуатация под наблюдением.'
    assert answer.urgency == 'medium' and answer.tone == 'attention'
    titles = [section.title for section in answer.sections]
    assert titles[:4] == ['Вероятная причина — надо проверить', 'Что проверить', 'Норма', 'Что сделать']
    assert '1. Проверить давление начала впрыскивания на стенде. (Fixture РЭ, стр. 31)' in lines
    assert 'Риск (оценка агента, не из руководства)' in answer.text
    assert 'Можно сформировать заявку на дефектовку.' in lines
    assert '• Fixture РЭ — стр. 31' in lines


def test_anomaly_without_verified_claims_recommends_defect_inspection():
    text = FINALIZER.compose(response(diagnostic=result(), search_plan=PLAN, evidence=EVIDENCE,
                                      limitations=['Недостаточно доказательств для технических выводов.']))
    assert 'Не буду гадать: рекомендую дефектовку узла «дизельная топливная аппаратура»' in text
    assert 'Недостаточно доказательств' not in text


def test_ticket_draft_and_missing_ticket():
    payload = TicketPayload(vehicle_id='M333YK47', session_id='S-0005', anomaly_type='F08', conclusion='…',
                            urgency='medium', operations=['Отрегулировать форсунку.'], spare_parts=[],
                            references=[TicketReference(manual='Fixture РЭ', pdf_page=31)])
    draft = TicketDraft(draft_id='d', expires_at=datetime.now(timezone.utc) + timedelta(minutes=5), payload=payload)
    text = FINALIZER.compose(response('draft_ticket', diagnostic=result(), search_plan=PLAN, evidence=EVIDENCE,
                                      claims=CLAIMS, ticket_draft=draft, can_create_ticket=True))
    assert '📝 Черновик заявки на дефектовку:' in text and '• Работа: Отрегулировать форсунку.' in text
    assert 'только после вашего подтверждения' in text and 'Можно сформировать заявку' not in text
    text = FINALIZER.compose(response('draft_ticket', diagnostic=result(), search_plan=PLAN, evidence=EVIDENCE,
                                      claims=CLAIMS[:2], limitations=['Для заявки нужна хотя бы одна подтверждённая операция.']))
    assert 'Заявка не сформирована:' in text and 'Оформите дефектовку вручную' in text
    assert 'нужна хотя бы одна' not in text


def test_manual_answer_and_honest_refusal():
    plan = SearchPlan(engine_model=ENGINE, original_query='Можно ли своим ходом перегнать машину на базу?',
                      rewritten_query='перегон')
    text = FINALIZER.compose(response('search_manuals', search_plan=plan, evidence=EVIDENCE))
    assert text.startswith('📘 Данных недостаточно.')
    assert 'нет подтверждённого ответа' in text and 'Решение о ремонте и выпуске машины' in text
    text = FINALIZER.compose(response('search_manuals', search_plan=PLAN, evidence=EVIDENCE, claims=CLAIMS[1:3]))
    assert text.startswith('📘 Ответ по руководствам для ЯМЗ-236М2') and 'Решение о ремонте' not in text


def test_disagreeing_sources_are_shown_side_by_side():
    marked = [claim('check', 'Регулировать зазоры на холодном двигателе.', 're31', 'c1'),
              claim('check', 'Регулировать зазоры не ранее чем через 15 мин после остановки.', 'rk269', 'c1')]
    detected = [claim('reference_value', 'Зазор клапанов 0,25–0,30 мм.', 're31'),
                claim('reference_value', 'Зазор клапанов 0,30–0,35 мм.', 'rk269')]
    sources = {item.evidence_id: item for item in EVIDENCE.items}
    assert find_conflicts(marked + detected, sources) == [marked, detected]
    same_manual = [claim('reference_value', 'Зазор клапанов 0,25–0,30 мм.'),
                   claim('reference_value', 'Зазор клапанов 0,30–0,35 мм.')]
    assert find_conflicts(same_manual, sources) == []
    answer = response('search_manuals', search_plan=PLAN, evidence=EVIDENCE, claims=marked)
    note, = conflict_notes(answer)
    assert note == ('Руководства расходятся: «Регулировать зазоры на холодном двигателе» (Fixture РЭ, стр. 31); '
                    '«Регулировать зазоры не ранее чем через 15 мин после остановки» (Fixture РК, стр. 269). '
                    'Уточните у мастера или специалиста, какая инструкция подходит вашему двигателю.')
    answer.limitations.append(note)
    text = FINALIZER.compose(answer)
    assert '⚖️ Руководства пишут по-разному:' in text
    assert 'Что проверить' not in text and 'Fixture РК, стр. 269' in text and 'Руководства расходятся' not in text


def test_explicit_prohibition_raises_urgency():
    prohibition = claim('risk', 'Запрещается эксплуатация двигателя при давлении масла ниже допустимого.')
    answer = FINALIZER.answer(response(diagnostic=result(code='F08', score=0.5), search_plan=PLAN, evidence=EVIDENCE,
                                       claims=[prohibition]))
    assert answer.urgency == 'high' and answer.tone == 'alert'
    assert answer.details[-1].startswith('⛔ По руководству: Запрещается эксплуатация')
    conclusion, urgency = FINALIZER.ticket_fields(result(), [prohibition])
    assert urgency == 'high' and 'запрещает эксплуатацию' in conclusion


def test_ticket_fields():
    conclusion, urgency = FINALIZER.ticket_fields(result(), CLAIMS)
    assert urgency == 'medium'
    assert conclusion.startswith('Предполагаемая неисправность: Дизельная топливоподача (F08) по акустической проверке S-0005')
    assert 'Требуется дефектовка: дизельная топливная аппаратура.' in conclusion


@pytest.mark.parametrize(('code', 'score', 'expected'), [
    ('F08', 0.76, 'medium'), ('F08', 0.9, 'high'), ('F08', 0.3, 'low'), ('F02', 0.5, 'high'), ('F02', 0.2, 'medium'),
    ('F05', 0.5, 'low'), ('F05', 0.95, 'medium'), ('F99', 0.5, 'medium'), (None, None, 'medium'),
])
def test_urgency_scheme(code, score, expected):
    assert urgency_for(code, score) == expected


def test_unit_vocabulary():
    assert mentions_unit('F08', 'Отрегулировать форсунку')
    assert not mentions_unit('F14', 'Отрегулировать сцепление')
    assert mentions_unit('F14', 'Проверить натяжение цепи')
    assert mentions_unit(None, 'что угодно')


def session(day, status, score):
    return SessionRecord(**{**result(status, score=score).model_dump(), 'session_id': f'S-{day}',
                            'checked_at': datetime(2026, 11, day, tzinfo=timezone.utc)})


@pytest.mark.parametrize(('scores', 'start'), [
    ([0.2, 0.4, 0.7], 'Оценка аномалии растёт: 0,20 → 0,40 → 0,70.'),
    ([0.7, 0.5, 0.2], 'Оценка аномалии снижается'),
    ([0.17, 0.76, 0.22], 'Оценка аномалии снизилась по сравнению с прошлой проверкой'),
    ([0.2, 0.2, 0.25], 'Оценка аномалии стабильна'),
    ([0.3], 'Для оценки динамики'),
])
def test_trend(scores, start):
    sessions = [session(day, 'anomaly', score) for day, score in enumerate(scores, 1)]
    assert trend(sessions).startswith(start)


def test_history_ignores_failed_checks_in_trend():
    sessions = [session(1, 'normal', 0.1), session(2, 'check_failed', None), session(3, 'anomaly', 0.8)]
    text = FINALIZER.compose(response('get_diagnostic_history', history=sessions))
    assert '0,10 → 0,80' in text and 'S-2 — сбой записи' in text


def test_finalizer_never_breaks_the_chat(monkeypatch):
    monkeypatch.setattr(MechanicFinalizer, 'answer', lambda self, response: 1 / 0)
    assert FINALIZER.compose(response(diagnostic=result())) == 'pipeline'


def test_plain_claims_from_older_generators_are_rendered():
    plain = TechnicalClaim(kind='operation', text='Отрегулировать форсунку.', evidence_ids=['re31'])
    text = FINALIZER.compose(response(diagnostic=result(), search_plan=PLAN, evidence=EVIDENCE, claims=[plain]))
    assert '1. Отрегулировать форсунку. (Fixture РЭ, стр. 31)' in text


def test_plain_summary_leads_the_answer_with_a_technical_line():
    marked = [claim('operation', 'Регулировать зазоры на холодном двигателе.', group='c1'),
              claim('operation', 'Регулировать зазоры не ранее чем через 15 мин после остановки.', 'rk269', 'c1')]
    summary = 'Регулируйте зазоры на холодном двигателе.\nРуководства пишут по-разному про 15 минут.'
    answer = FINALIZER.answer(response('search_manuals', search_plan=PLAN, evidence=EVIDENCE, claims=marked,
                                       summary=summary))
    assert answer.verdict == '🔧 ЯМЗ-236М2 · Fixture РЭ стр. 31 · Fixture РК стр. 269 · Руководства расходятся'
    assert answer.details == summary.splitlines()
    assert 'Источники' not in [section.title for section in answer.sections]
    assert 'Fixture РК, стр. 269' in answer.text


def test_plain_lines_may_only_restate_claims():
    from ml.generation.plain import PlainOutput, plain_text, safe_line
    claims = [claim('reference_value', 'Давление масла 400-700 кПа при 2100 об/мин.'),
              claim('operation', 'Регулировать зазоры не ранее чем через 15 мин после остановки.')]
    assert safe_line('Давление масла: 400-700 кПа на 2100 об/мин.', claims, ENGINE)
    assert not safe_line('Давление масла не ниже 350 кПа.', claims, ENGINE)
    assert not safe_line('Можно продолжать эксплуатацию.', claims[:1], ENGINE)
    assert not safe_line('Не запускайте двигатель.', claims[:1], ENGINE)
    assert not safe_line('Для ЯМЗ-536 давление 400-700 кПа.', claims, ENGINE)
    output = PlainOutput(lead='Давление масла выше 900 кПа.', lines=['Давление масла 400-700 кПа.'], assessment='')
    assert plain_text(output, claims, ENGINE) == ''


def test_agent_assessment_is_labelled_and_never_permits_driving():
    from ml.generation.plain import PlainOutput, plain_text
    claims = [claim('operation', 'При увеличенных зазорах в подшипниках двигатель направляют в ремонт.')]
    warning = PlainOutput(lead='Двигатель по руководству направляют в ремонт.', lines=[],
                          assessment='Своим ходом не ехать: может провернуть вкладыш. Лучше эвакуатор.')
    assert plain_text(warning, claims, ENGINE).splitlines()[-1] == (
        '⚠️ Оценка агента, не из руководства: Своим ходом не ехать: может провернуть вкладыш. Лучше эвакуатор.')
    permit = warning.model_copy(update={'assessment': 'Можно ехать своим ходом до базы.'})
    numbers = warning.model_copy(update={'assessment': 'Не больше 50 км на малых оборотах.'})
    assert plain_text(permit, claims, ENGINE) == plain_text(numbers, claims, ENGINE) == (
        'Двигатель по руководству направляют в ремонт.')
