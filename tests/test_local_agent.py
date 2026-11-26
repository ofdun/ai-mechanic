from types import SimpleNamespace

import pytest

from ml.contracts import DiagnosticResult, Evidence, EvidenceBundle, SearchPlan
from ml.local_evidence import LocalEvidence
from ml.planner.local import LocalPlanner


@pytest.mark.parametrize(('message', 'action'), [
    ('Проверь мотор', 'diagnose'),
    ('Оформить наряд на работы', 'draft_ticket'),
    ('Создать заявку', 'draft_ticket'),
    ('Как составить заявку на дефектовку?', 'search_manuals'),
    ('Не оформляй заявку', 'clarify'),
    ('Не оформляй заявку, покажи историю', 'get_diagnostic_history'),
    ('Какой момент затяжки крышек коренных подшипников?', 'search_manuals'),
])
def test_local_intent(message, action):
    assert LocalPlanner().plan(message, 'Стенд', None).action == action


def diagnostic():
    return DiagnosticResult(vehicle_id='V', session_id='S', status='anomaly', signal_quality='good',
                            anomaly_type='F04', anomaly_title='Ремённый привод', confidence=.85,
                            anomaly_score=.75, description='Шум на записи', model_version='synthetic')


def test_manual_candidates_cannot_authorize_a_ticket_or_become_a_norm():
    engine = 'Учебный мотор'
    page = Evidence(document_id='exercise', evidence_id='exercise:3', manual='Учебная инструкция', pdf_page=3,
                    engine_models=[engine], text='Давление испытательного контура измеряют отдельно. Температура указана ниже.')
    retriever = SimpleNamespace(search=lambda plan: EvidenceBundle(items=[page]), page=lambda *key: page)
    workflow = LocalEvidence(retriever)
    question = 'Какое давление испытательного контура?'
    plan = SearchPlan(engine_model=engine, original_query=question, rewritten_query=question)
    result = workflow.run(plan, None)
    assert not result.claims and result.evidence.items == [page]
    fault = workflow.run(plan.model_copy(update={'anomaly_type': 'F04'}), diagnostic())
    assert not fault.claims and not fault.evidence.items
    wrong_engine = workflow.run(plan.model_copy(update={'engine_model': 'Другой мотор'}), None)
    assert not wrong_engine.claims and not wrong_engine.evidence.items


def test_local_planner_grounds_an_elliptical_question_in_the_current_diagnosis():
    planner = LocalPlanner()
    message = 'Почему это опасно?'
    with_context = planner.plan(message, 'Учебный мотор', diagnostic())
    assert with_context.action == 'search_manuals' and 'Ремённый привод' in with_context.rewritten_query
    assert planner.plan(message, 'Учебный мотор', None).rewritten_query == message


def test_reviewed_norm_requires_all_pages_of_the_selected_engine():
    import hashlib

    from ml.reviewed_checks import ReviewCatalog

    texts = ['На холодном стенде.', 'Начальное давление 14,2 кПа.']
    engine = 'Atlas-24'
    pages = {
        number: Evidence(document_id='service', evidence_id=f'e{number}', manual='Регламент', engine_models=[engine],
                         pdf_page=number, text=text)
        for number, text in enumerate(texts, 1)
    }
    checks = {f'service:{number}': {'sha256': hashlib.sha256(page.text.encode()).hexdigest()}
              for number, page in pages.items()}
    checks['service:2']['captures'] = {'pressure': r'давление ([\d,]+) кПа'}
    catalog = ReviewCatalog.model_validate({
        'topics': {'pressure': ['давлен']}, 'pages': checks,
        'entries': [{'topic': 'pressure', 'claims': [{
            'kind': 'reference_value', 'text': 'На холодном стенде {pressure} кПа.',
            'evidence_ids': ['service:1', 'service:2'],
        }]}],
    })
    retriever = SimpleNamespace(search=lambda plan: EvidenceBundle(), page=lambda model, doc, n: pages.get(n))
    workflow = LocalEvidence(retriever, catalog, [engine, 'Delta-31'])
    question = 'Какое давление у Atlas-24?'
    plan = SearchPlan(engine_model=engine, original_query=question, rewritten_query=question)
    claim, = workflow.run(plan, None).claims
    assert (claim.text, claim.evidence_ids) == ('На холодном стенде 14,2 кПа.', ['e1', 'e2'])
    for other in ['Delta-31', 'Atlas-212']:
        assert not workflow.run(plan.model_copy(update={'original_query': f'Какое давление у {other}?'}), None).claims
    assert not workflow.run(plan.model_copy(update={'engine_model': 'Delta-31'}), None).claims
    pages[1].text = 'На горячем стенде.'
    assert not workflow.run(plan, None).claims
    del pages[1]
    assert not workflow.run(plan, None).claims
