"""Check defect inspection, saved history and JSON on the running local stand."""
import json
import os
from datetime import datetime
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def exercise(page, url):
    page.goto(url)
    catalog = page.request.get(url + '/api/vehicles').json()['vehicles']

    def select(vehicle, session):
        engine = next(v['engine_model'] for v in catalog if v['vehicle_id'] == vehicle)
        page.get_by_label('Двигатель', exact=True).select_option(engine)
        page.get_by_label('Транспортное средство').select_option(vehicle)
        page.get_by_label('Проверка', exact=True).select_option(session)

    def run(button):
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
            button.click()
        assert response.value.ok, response.value.status
        return response.value.json()

    select('M333YK47', 'S-0005')
    diagnosis = run(page.get_by_role('button', name='Получить результат диагностики'))
    page.locator('#tab-ticket').click()
    draft = run(page.get_by_role('button', name='Создать заявку', exact=True))
    assert draft['ticket_draft']['draft_id']
    draft_text = page.locator('#panel-ticket .draft').inner_text()
    page.locator('#tab-result').click()
    report = page.locator('#panel-result .report').inner_text()
    form = page.locator('.inspection-form')
    comment = form.get_by_label('Комментарий')
    save = form.get_by_role('button', name='Записать результат осмотра')
    assert form.get_by_role('spinbutton').count() == 0
    expect(form.get_by_role('group', name='Для проверки по меткам')).to_have_count(0)
    checks = []
    for index, (outcome, marks) in enumerate([
        ('confirmed', None), ('not_confirmed', 'matched'), ('other_cause', 'not_matched'),
    ]):
        if index == 1:
            page.locator('#tab-ticket').click()
            assert page.locator('#panel-ticket .draft').inner_text() == draft_text
            page.locator('#tab-result').click()
            # The shipped injector case has no mark-alignment step. Exercise the other UI
            # path with a cited instruction in an otherwise real diagnosis response.
            def with_mark_instruction(route):
                reply = route.fetch()
                body = reply.json()
                source = dict(body['evidence']['items'][0], evidence_id='inspection-test-marks',
                              text='Проверьте совпадение установочных меток.')
                body['evidence']['items'].append(source)
                body['claims'].append(dict(kind='check', text=source['text'], evidence_ids=[source['evidence_id']]))
                route.fulfill(response=reply, json=body)
            page.route('**/agent/run', with_mark_instruction, times=1)
            run(page.get_by_role('button', name='Получить результат диагностики'))
            expect(form.get_by_role('group', name='Для проверки по меткам')).to_be_visible()
            page.locator('#tab-ticket').click()
            run(page.get_by_role('button', name='Создать заявку', exact=True))
            draft_text = page.locator('#panel-ticket .draft').inner_text()
            page.locator('#tab-result').click()
            report = page.locator('#panel-result .report').inner_text()
        form.locator(f'input[value="{outcome}"]').check()
        if marks:
            form.locator(f'input[value="{marks}"]').check()
        note = f'Проверка формы {index}'
        comment.fill(note)
        if index == 0:
            page.route('**/agent/run', lambda route: route.abort(), times=1)
            save.click()
            expect(page.locator('.panel-status')).to_contain_text('Нет связи')
            expect(comment).to_have_value(note)
            expect(form.locator('input[value="confirmed"]')).to_be_checked()
        result = run(save)
        expect(form.get_by_role('status')).to_have_text('Результат осмотра сохранён')
        record = result['inspections'][-1]
        assert (record['vehicle_id'], record['session_id'], record['model_version'], record['anomaly_type']) == (
            'M333YK47', 'S-0005', diagnosis['diagnostic']['model_version'], diagnosis['diagnostic']['anomaly_type'])
        assert (record['outcome'], record['marks'], record['comment']) == (outcome, marks, note)
        assert datetime.fromisoformat(record['recorded_at'].replace('Z', '+00:00')).utcoffset().total_seconds() == 0
        assert 'actor_id' not in record
        checks.append(record)
        assert page.locator('#panel-result .report').inner_text() == report
    page.locator('#tab-ticket').click()
    assert page.locator('#panel-ticket .draft').inner_text() == draft_text
    page.reload()
    page.locator('#tab-history').click()
    history = run(page.get_by_role('button', name='Показать результаты осмотра'))
    assert next(s for s in history['history'] if s['session_id'] == 'S-0005')['status'] == 'anomaly'
    expect(page.locator('#panel-history')).to_contain_text('Метки совпали')
    with page.expect_download() as download:
        page.locator('#panel-history').get_by_role('button', name='Скачать JSON').click()
    assert json.loads(Path(download.value.path()).read_text())[-3:] == checks
    assert download.value.suggested_filename == 'inspections-M333YK47.json'

    page.locator('#tab-chat').click()
    question = page.get_by_label('Вопрос по двигателю')
    if not question.is_enabled():
        page.locator('#tab-result').click()
        run(page.get_by_role('button', name='Получить результат диагностики'))
        page.locator('#tab-chat').click()
    question.fill('Покажи историю проверок')
    answer = run(page.get_by_role('button', name='Отправить', exact=True))
    assert answer['action'] == 'get_diagnostic_history'
    expect(page.locator('#panel-chat .inspection-history')).to_contain_text(checks[-1]['comment'])
    select('P456AT178', 'S-0003')
    page.locator('#tab-result').click()
    failed = run(page.get_by_role('button', name='Получить результат диагностики'))
    assert failed['requires_remeasurement']
    expect(form).to_have_count(0)
    invalid = page.request.post(url + '/agent/run', data=dict(
        vehicle_id=failed['vehicle_id'], session_id=failed['session_id'], conversation_id=failed['conversation_id'],
        command='record_inspection', inspection={'outcome': 'confirmed'}))
    assert invalid.status == 409 and invalid.json()['detail']['code'] == 'INSPECTION_UNAVAILABLE'
    page.locator('#tab-history').click()
    other = run(page.get_by_role('button', name='Показать результаты осмотра'))
    assert not {r['inspection_id'] for r in other['inspections']} & {r['inspection_id'] for r in checks}
    print('PASS: inspection outcomes, marks, retry, model version, unchanged report and ticket, vehicle history, JSON, chat and failed recording')


if __name__ == '__main__':
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(ignore_https_errors=True, http_credentials={
            'username': os.environ['STAND_USER'], 'password': os.environ['STAND_PASSWORD'],
        }, viewport={'width': 390, 'height': 844}, is_mobile=True)
        page.set_default_timeout(120000)
        exercise(page, 'https://gateway')
        browser.close()
