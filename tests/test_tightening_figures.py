import hashlib

from ml.contracts import AgentResponse, Evidence, EvidenceBundle
from ml.generation import MechanicFinalizer
from ml.tools.figures import (
    ReviewedSequence,
    attach_sequences,
    figure_note,
    sequence_figures,
)

# Synthetic pages in the layout of the manuals; no manual text is stored in the repository.
TEXT_PAGE = ('Установить головку на блок. Затяжку болтов крепления головки цилиндров выполнить '
             'в последовательности, указанной на рис.42, в два этапа:\n– 50 Н·м;\n– доворот на 90°.')
CAPTIONS = ('Подтянуть гайки моментом 240 Н·м в порядке, указанном на рисунках 5 и 6.\n'
            'Рис. 5. Последовательность затяжки гаек\nголовок цилиндров двигателей ЯМЗ-236М2\n'
            'Рис. 6. Последовательность затяжки гаек\nголовок цилиндров двигателей ЯМЗ-238М2\n')
PAN = 'Рис. 46. Последовательность затяжки болтов масляного картера\nОтвернуть болты.'
QUESTION = 'В каком порядке затягивать болты головки цилиндров?'


def page(text, pdf_page, engine='ЯМЗ-236М2', document='rk'):
    return Evidence(evidence_id=f'{document}:{pdf_page}', document_id=document, manual='РК', engine_models=[engine],
                    pdf_page=pdf_page, text=text)


def answer(*pages, engine='ЯМЗ-236М2', claims=()):
    return AgentResponse(request_id='r', vehicle_id='v', session_id='s', engine_model=engine, action='search_manuals',
                         message='m', evidence=EvidenceBundle(items=list(pages)), claims=list(claims))


def found(response, question=QUESTION):
    return [(figure.figure, figure.pdf_page, figure.caption) for figure in sequence_figures(response, question)]


def test_the_figure_of_this_engine_and_subject_only():
    assert found(answer(page(CAPTIONS, 23))) == [
        ('5', 23, 'Последовательность затяжки гаек головок цилиндров двигателей ЯМЗ-236М2')]   # not ЯМЗ-238М2
    assert found(answer(page(TEXT_PAGE, 180), page(PAN, 185))) == [('42', 180, None)]          # not the oil pan
    assert found(answer(page(PAN, 185)), 'В каком порядке затягивать болты картера?') == [
        ('46', 185, 'Последовательность затяжки болтов масляного картера')]
    other = page(CAPTIONS, 23, engine='КАМАЗ-740')
    assert found(answer(other)) == []                                    # a page of another engine is not read
    assert found(answer(page(TEXT_PAGE, 180)), 'Какой момент затяжки болтов головки?') == []


def test_a_sequence_only_from_a_reviewed_and_unchanged_page():
    pages = [page(TEXT_PAGE, 180)]
    [figure] = sequence_figures(answer(*pages), QUESTION)
    digest = hashlib.sha256(' '.join(TEXT_PAGE.split()).encode()).hexdigest()
    reviewed = ReviewedSequence(document_id='rk', figure='42', pdf_page=180, sha256=digest,
                                steps=['болт 1 (центр)', 'болт 2', 'болт 3'], reviewed_by='Р. Р.')
    assert attach_sequences([figure], pages, [reviewed])[0].sequence == ['болт 1 (центр)', 'болт 2', 'болт 3']
    changed = reviewed.model_copy(update={'sha256': '0' * 64})          # another edition of the page
    assert attach_sequences([figure], pages, [changed])[0].sequence == []
    assert figure_note(figure).startswith('Порядок затяжки показан на рисунке 42 (РК, стр. 180) — откройте его.')

    response = answer(*pages)
    response.figures = attach_sequences([figure], pages, [reviewed])
    text = MechanicFinalizer().compose(response)
    assert 'Порядок затяжки (рис. 42):' in text and '1. болт 1 (центр)' in text


def test_pipeline_points_to_the_figure(tmp_path):
    import json

    import httpx

    from ml.config import Settings
    from ml.contracts import AgentRequest
    from ml.pipeline import Pipeline

    vehicles = tmp_path / 'vehicles.json'
    session = {'vehicle_id': 'M1', 'session_id': 'S-1', 'status': 'normal', 'anomaly_type': None, 'anomaly_title': None,
               'anomaly_score': 0.1, 'confidence': 0.9, 'signal_quality': 'good', 'description': '',
               'model_version': 't', 'checked_at': '2026-11-19T10:00:00+03:00'}
    vehicles.write_text(json.dumps({'vehicles': [{'vehicle_id': 'M1', 'engine_model': 'ЯМЗ-236М2', 'fuel': 'дизель',
                                                  'sessions': [session]}]}), encoding='utf-8')
    settings = Settings(_env_file=None, vehicles_path=vehicles, artifacts_dir=tmp_path / 'a', planner_mode='offline')

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=[page(TEXT_PAGE, 180)])

    with httpx.Client() as http:
        result = Pipeline(settings, http, retriever=Retriever()).run(
            AgentRequest(actor_id='a', vehicle_id='M1', session_id='S-1', message=QUESTION))
    assert [(figure.figure, figure.pdf_page, figure.sequence) for figure in result.figures] == [('42', 180, [])]
    assert result.limitations[-1].startswith('Порядок затяжки показан на рисунке 42')
