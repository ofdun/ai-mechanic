"""Exercise a fresh Compose slot before its shared smoke test creates the first ticket."""
import os
from urllib.request import urlopen
import json

from playwright.sync_api import expect, sync_playwright


def exercise(page, base_url='https://gateway', agent_url='http://agent:8000'):
    page.goto(base_url)
    page.get_by_label('Двигатель', exact=True).select_option('ЯМЗ-236М2')
    page.get_by_label('Транспортное средство').select_option('M333YK47')
    page.get_by_label('Проверка', exact=True).select_option('S-0005')
    page.get_by_role('button', name='Получить результат диагностики').click()
    expect(page.get_by_role('heading', name='Отклонение', exact=True)).to_be_visible()
    page.get_by_role('tab', name='Чат', exact=True).click()
    question = 'Почему это опасно?'
    editor = page.get_by_label('Вопрос по двигателю')
    editor.fill(question)
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as followup:
        editor.press('Enter')
    answer = followup.value.json()
    page.get_by_role('tab', name='Заявка', exact=True).click()
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as draft:
        page.get_by_role('button', name='Создать заявку', exact=True).click()
    draft_id = draft.value.json()['ticket_draft']['draft_id']
    with page.expect_response(lambda r: r.url.endswith('/agent/tickets/confirm')) as confirm:
        page.get_by_role('button', name='Подтвердить заявку').click()
    assert confirm.value.ok, 'Run the audit scenario first in a fresh slot'
    ticket = confirm.value.json()['ticket_id']
    expect(page.get_by_text(ticket, exact=True)).to_be_visible()

    route = '/audit?session_id=S-0005'
    assert page.request.get(base_url + route).status == 404
    with urlopen(agent_url + route, timeout=8) as result:
        chain = json.load(result)['entries']
    logged = next(e for e in chain if e.get('request_id') == answer['request_id'])
    entries = [e for e in chain if e['actor_id'] == logged['actor_id']]
    assert [e.get('action', e['kind']) for e in entries] == ['diagnose', 'search_manuals', 'draft_ticket', 'ticket_confirm']
    assert logged['question'] == question and logged['answer'] == answer['message']
    assert [c['text'] for c in logged['answer_details']['claims']] == [c['text'] for c in answer['claims']]
    assert logged['shown_pages'] and logged['cited_pages']
    assert entries[-2]['draft_id'] == entries[-1]['draft_id'] == draft_id and entries[-1]['ticket_id'] == ticket
    print('PASS: private audit matches the mechanic, answer, sources, draft and ticket in the browser.')


if __name__ == '__main__':
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(ignore_https_errors=True, http_credentials={
            'username': os.environ['STAND_USER'], 'password': os.environ['STAND_PASSWORD'],
        }, viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        context.set_default_timeout(150000)
        exercise(context.new_page())
        browser.close()
