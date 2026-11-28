"""Check a named component against the shipped manuals through the public web UI."""
import os

from playwright.sync_api import expect, sync_playwright


def exercise(page, base_url):
    page.goto(base_url)
    page.context.grant_permissions(['clipboard-read', 'clipboard-write'], origin=base_url)
    page.get_by_label('Двигатель', exact=True).select_option('ЯМЗ-236М2')
    page.get_by_label('Транспортное средство').select_option('M333YK47')
    page.get_by_label('Проверка', exact=True).select_option('S-0005')
    page.get_by_role('button', name='Получить результат диагностики').click()
    expect(page.get_by_role('heading', name='Отклонение', exact=True)).to_be_visible()
    page.get_by_role('tab', name='Чат', exact=True).click()

    def ask(text):
        field = page.get_by_label('Вопрос по двигателю')
        field.fill(text)
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as reply:
            field.press('Enter')
        assert reply.value.ok
        return reply.value.json()

    answer = ask('Какой артикул водяного насоса?')
    assert answer['part_numbers']
    parts = page.get_by_role('log').locator('.part-numbers').last
    expect(parts).to_contain_text('Наличие и совместимость проверьте у поставщика')
    for item in answer['part_numbers']:
        code = item['designation']
        source = next(e for e in answer['evidence']['items'] if e['evidence_id'] == item['evidence_id'])
        assert answer['engine_model'] in source['engine_models']
        assert any(item['evidence_id'] in c['evidence_ids'] for c in answer['claims'])
        entry = parts.get_by_role('listitem').filter(has=page.get_by_text(code, exact=True))
        entry.get_by_role('button', name='Скопировать ' + code, exact=True).click()
        assert page.evaluate('navigator.clipboard.readText()') == code
        route = f"/api/documents/{item['document_id']}/pages/{item['pdf_page']}"
        with page.expect_response(lambda r: r.url == base_url + route) as response:
            entry.locator('.source').click()
        stored = response.value.json()
        assert stored['page'] == item['pdf_page'] and code in stored['text']
        page.wait_for_function('document.querySelector(".manual img")?.naturalWidth > 0')
        page.get_by_role('button', name='К чату').click()

    for query in ('Какой артикул несуществующей муфты?', 'Какой артикул водяного насоса ЗМЗ-5234.10?'):
        missing = ask(query)
        assert not missing['part_numbers'] and missing['part_numbers_note']
        expect(page.get_by_role('log').locator('.part-numbers').last).to_contain_text(missing['part_numbers_note'])

    page.get_by_role('tab', name='Заявка', exact=True).click()
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
        page.get_by_role('button', name='Создать заявку', exact=True).click()
    draft = response.value.json()['ticket_draft']['payload']
    assert not any(item['designation'] in str(draft['spare_parts']) for item in answer['part_numbers'])
    print('PASS: named part, engine scope, copy, physical PDF, missing designation and ticket isolation.')


if __name__ == '__main__':
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(ignore_https_errors=True, viewport={'width': 390, 'height': 844},
                                http_credentials={'username': os.environ['STAND_USER'],
                                                  'password': os.environ['STAND_PASSWORD']})
        exercise(page, 'https://gateway')
        browser.close()
