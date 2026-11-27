"""Run C4 against the customer's fleet via the web application's normal entry point."""
import os

from playwright.sync_api import expect, sync_playwright


def exercise(page, base_url):
    page.goto(base_url)
    catalog = page.request.get(base_url + '/api/vehicles').json()['vehicles']
    vehicles = sorted(catalog, key=lambda v: v['fleet_status']['rank'])
    assert len(vehicles) == 50
    labels = {'quarantine': 'Карантин', 'attention': 'Требует внимания', 'normal': 'Норма'}
    expected = {'quarantine': 9, 'attention': 16, 'normal': 25}
    failed = []
    order = []
    for vehicle in vehicles:
        status = vehicle['fleet_status']
        last = max(vehicle['sessions'], key=lambda s: s['checked_at'])
        assert status['session_id'] == last['session_id']
        if last['status'] == 'check_failed':
            assert status['state'] == 'attention' and status['kind'] == 'remeasure'
            failed.append(vehicle['vehicle_id'])
        order.append(({'quarantine': 0, 'attention': 1, 'normal': 3}[status['state']]
                      if status['kind'] != 'remeasure' else 2,
                      {'high': 0, 'medium': 1, 'low': 2, None: 3}[status['urgency']],
                      -(status['anomaly_score'] or 0), vehicle['vehicle_id']))
    assert sorted(order) == order and len(failed) == 2

    page.get_by_role('button', name='Главный механик').click()
    page.reload()
    fleet = page.get_by_role('region', name='Парк · 50 машин')
    expect(fleet).to_be_visible()
    rows = fleet.locator('li.fleet-row')
    expect(rows).to_have_count(50)
    assert rows.locator('.fleet-main > strong').all_text_contents() == [v['vehicle_id'] for v in vehicles]
    for state, label in labels.items():
        counter = fleet.get_by_role('button', name=f'{expected[state]} {label}')
        counter.click()
        expect(counter).to_have_attribute('aria-pressed', 'true')
        expect(rows).to_have_count(expected[state])
        expect(fleet.locator('.fleet-row.' + state)).to_have_count(expected[state])
        if state == 'attention':
            for vehicle_id in failed:
                expect(rows.filter(has=page.get_by_text(vehicle_id, exact=True))).to_contain_text('Повторите замер')
        counter.click()

    fleet.get_by_role('link', name='Как считаются статусы и порядок').click()
    expect(page.get_by_text('Стоимость ремонта и простоя не оценивается.', exact=False)).to_be_in_viewport()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')

    top = vehicles[0]
    rows.first.get_by_role('button').click()
    expect(page.get_by_role('tab', name='История')).to_have_attribute('aria-selected', 'true')
    expect(page.get_by_label('Транспортное средство')).to_have_value(top['vehicle_id'])
    expect(page.get_by_label('Проверка', exact=True)).to_have_value(top['fleet_status']['session_id'])
    expect(page.get_by_role('heading', name='История проверок · ' + top['vehicle_id'])).to_be_visible()
    expect(page.locator('.vehicle-history > li')).to_have_count(len(top['sessions']))
    expect(page.locator('.score-chart')).to_be_visible()


if __name__ == '__main__':
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for width in (390, 1280):
            page = browser.new_page(viewport={'width': width, 'height': 844}, ignore_https_errors=True,
                                    http_credentials={'username': os.environ['STAND_USER'],
                                                      'password': os.environ['STAND_PASSWORD']})
            exercise(page, 'https://gateway')
            page.close()
        browser.close()
        print('PASS: C4, 50 vehicles, counters and queue, failed measurements, latest history and chart.')
