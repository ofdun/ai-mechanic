"""Optional features on a running Compose stand (local-stand-testing skill: `stand e2e --test e2e/stand_features.py`).

Uses the customer's mock fleet behind the authenticated gateway: chief mechanic, score chart, vehicle memory,
inspection result, tightening-order figure, part designations and the translated interface."""
import os

from playwright.sync_api import expect, sync_playwright

BASE = os.environ.get('STAND_URL', 'https://gateway')


def exercise(page):
    page.goto(BASE)
    catalog = page.request.get(BASE + '/api/vehicles').json()['vehicles']
    assert all('fleet_status' in vehicle for vehicle in catalog), 'the agent does not send fleet statuses'

    def select(plate, session=None):
        model = next(v['engine_model'] for v in catalog if v['vehicle_id'] == plate)
        page.get_by_label('Двигатель', exact=True).select_option(model)
        page.get_by_label('Транспортное средство').select_option(plate)
        if session:
            page.get_by_label('Проверка', exact=True).select_option(session)

    def diagnose(plate, session):
        select(plate, session)
        page.get_by_role('tab', name='Результат', exact=True).click()
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as reply:
            page.get_by_role('button', name='Получить результат диагностики').click()
        assert reply.value.ok, reply.value.text()
        return reply.value.json()

    def ask(text):
        page.get_by_role('tab', name='Чат', exact=True).click()
        page.get_by_label('Вопрос по двигателю').fill(text)
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as reply:
            page.get_by_role('button', name='Отправить', exact=True).click()
        assert reply.value.ok, reply.value.text()
        return reply.value.json()

    # Chief mechanic: counters, order, a click opens the history with the score chart.
    page.get_by_role('button', name='Главный механик').click()
    counters = page.locator('.fleet-counters strong').all_inner_texts()
    states = {s: sum(v['fleet_status']['state'] == s for v in catalog) for s in ('quarantine', 'attention', 'normal')}
    assert [int(n) for n in counters] == [states['quarantine'], states['attention'], states['normal']], counters
    first = min(catalog, key=lambda v: v['fleet_status']['rank'])
    expect(page.locator('.fleet-row').first).to_contain_text(first['vehicle_id'])
    page.locator('.fleet-row button').first.click()
    expect(page.get_by_role('tab', name='История', exact=True)).to_have_attribute('aria-selected', 'true')
    expect(page.locator('#panel-history .score-chart figcaption')).to_contain_text('Оценка аномалии')
    print(f'PASS — главный механик {states}, первым {first["vehicle_id"]}, история с графиком')

    # Vehicle memory: B418HK98 has F01 on S-0008 and again on S-0009.
    repeat = diagnose('B418HK98', 'S-0009')
    assert repeat['memory'] and repeat['memory'][0].startswith('Повтор:') and '30.10.2026 (S-0008)' in repeat['memory'][0], repeat['memory']
    expect(page.locator('#panel-result')).to_contain_text('По этой машине')
    print('PASS — память по машине: ' + repeat['memory'][0])

    # Inspection result: stored, the verdict unchanged, visible in the history.
    form = page.locator('#panel-result .inspection-form')
    form.get_by_label('Гипотеза ИИ подтвердилась').check()
    form.get_by_label('Метки не проверялись').check()
    form.get_by_label('Комментарий').fill('Проверка стенда: стук клапанов подтвердился')
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as saved:
        form.get_by_role('button', name='Записать результат осмотра').click()
    record = saved.value.json()['inspections'][-1]
    assert (record['session_id'], record['model_version']) == ('S-0009', repeat['diagnostic']['model_version'])
    expect(page.locator('#panel-result .report')).to_contain_text('Отклонение')
    page.get_by_role('tab', name='История', exact=True).click()
    expect(page.locator('#panel-history')).to_contain_text('стук клапанов подтвердился')
    print('PASS — результат осмотра записан с model_version, диагноз не изменился')

    # Tightening order on ЗМЗ-40905.10: the figure of this engine, no guessed sequence.
    diagnose('A001MC178', 'S-0168')
    order = ask('В каком порядке затягивать болты крепления головки цилиндров?')
    figures = [(f['figure'], f['pdf_page']) for f in order['figures']]
    assert figures and figures[0][0] == '142' and not order['figures'][0]['sequence'], figures
    expect(page.locator('.tightening-figures').last).to_contain_text('Рисунок 142')
    print(f'PASS — порядок затяжки: рисунок 142, стр. {figures[0][1]}, без угадывания')

    # Part designations: only from this engine's pages, or an explicit «нет».
    parts = ask('Какое обозначение детали у водяного насоса?')
    codes = [p['designation'] for p in parts['part_numbers']]
    assert codes or parts['part_numbers_note'] or not parts['claims'], parts
    print(f'PASS — обозначения деталей: {codes[:4] or parts["part_numbers_note"] or "утверждений нет (baseline)"}')

    # Interface in Uzbek, including the new blocks.
    page.locator('#ui-language').select_option('uz')
    page.get_by_role('button', name='Bosh mexanik').click()
    expect(page.locator('.fleet h2')).to_contain_text('Avtopark')
    page.locator('#ui-language').select_option('ru')
    print('PASS — интерфейс на узбекском, включая новые блоки')
    print('Не покрыто данными заказчика: запись с помехами при уверенности ≥ порога (в комплекте таких нет); '
          'журнал аудита читается у агента: GET /audit, через шлюз не выводится.')


if __name__ == '__main__':
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(ignore_https_errors=True, http_credentials={
            'username': os.environ['STAND_USER'], 'password': os.environ['STAND_PASSWORD'],
        }, viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True)
        context.set_default_timeout(150000)
        exercise(context.new_page())
        browser.close()
