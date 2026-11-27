import hashlib

from ml.contracts import AgentResponse, Evidence, EvidenceBundle
from ml.generation import MechanicFinalizer
from ml.tools.figures import (
    FIGURE_PAGE,
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


def found(response, question=QUESTION, manual_pages=()):
    return [(figure.figure, figure.pdf_page, figure.caption)
            for figure in sequence_figures(response, question, manual_pages)]


def test_the_figure_of_this_engine_and_subject_only():
    assert found(answer(page(CAPTIONS, 23))) == [
        ('5', 23, 'Последовательность затяжки гаек головок цилиндров двигателей ЯМЗ-236М2')]   # not ЯМЗ-238М2
    assert found(answer(page(TEXT_PAGE, 180), page(PAN, 185))) == [('42', 180, None)]          # not the oil pan
    assert found(answer(page(PAN, 185)), 'В каком порядке затягивать болты картера?') == [
        ('46', 185, 'Последовательность затяжки болтов масляного картера')]
    other = page(CAPTIONS, 23, engine='КАМАЗ-740')
    assert found(answer(other)) == []                                    # a page of another engine is not read
    assert found(answer(page(TEXT_PAGE, 180)), 'Какой момент затяжки болтов головки?') == []


# The search ranks the torque table first; the table only says «см. рис.42», the figure is pages away (stand case).
TABLE = ('Болты крепления масляного картера1) 11,8...17,6\n'
         '1) Болты затягивать в определенной последовательности – см. рис.42')
FIGURE = 'Рис.42. Последовательность затяжки болтов крепления головки цилиндров\nОтвернуть болты крышек.'


def test_a_figure_away_from_the_answer_pages_is_found_in_the_manual():
    manual = [page(PAN, 185), page(FIGURE, 181), page(FIGURE, 181, engine='КАМАЗ-740', document='kamaz')]
    assert found(answer(page(TABLE, 231))) == []
    assert found(answer(page(TABLE, 231)), manual_pages=manual) == [
        ('42', 181, 'Последовательность затяжки болтов крепления головки цилиндров')]
    # A question that does not name what is tightened gets no figure from elsewhere in the manual.
    assert found(answer(page(TABLE, 231)), 'В каком порядке затягивать болты?', manual) == []
    # The text names figure 42, the manual has its caption: the link goes to the page with the drawing.
    assert found(answer(page(TEXT_PAGE, 180)), manual_pages=manual) == [
        ('42', 181, 'Последовательность затяжки болтов крепления головки цилиндров')]
    assert found(answer(), 'В каком порядке подтягивать гайки крепления головок цилиндров?', manual) == [
        ('42', 181, 'Последовательность затяжки болтов крепления головки цилиндров')]
    assert all(FIGURE_PAGE.search(text) for text in (FIGURE, TEXT_PAGE, PAN)) and not FIGURE_PAGE.search(TABLE)


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


def test_pipeline_finds_the_figure_the_search_did_not_rank(tmp_path):
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
    digest = hashlib.sha256(' '.join(FIGURE.split()).encode()).hexdigest()
    reviewed = tmp_path / 'sequences.json'
    reviewed.write_text(json.dumps([{'document_id': 'rk', 'figure': '42', 'pdf_page': 181, 'sha256': digest,
                                     'steps': ['1', '2'], 'reviewed_by': 'Р. Р.'}]), encoding='utf-8')

    class Retriever:
        def search(self, plan):
            return EvidenceBundle(items=[page(TABLE, 231)])

        def figure_pages(self, engine_model, pattern):
            return [item for item in (page(FIGURE, 181), page(PAN, 185)) if pattern.search(item.text)]

    with httpx.Client() as http:
        pipeline = Pipeline(settings.model_copy(update={'reviewed_sequences_path': reviewed}), http,
                            retriever=Retriever())
        result = pipeline.run(AgentRequest(actor_id='a', vehicle_id='M1', session_id='S-1', message=QUESTION))
    assert [(figure.figure, figure.pdf_page, figure.sequence) for figure in result.figures] == [('42', 181, ['1', '2'])]
