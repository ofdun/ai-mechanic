"""Browser check of the optional features on the training stand (390×844): score chart, chief mechanic,
vehicle memory, inspection result, noisy-record confirmation, translated interface.

    PYTHONPATH=. .venv/bin/python tests/e2e/features_smoke.py [--channel chrome]
"""
import argparse
import json
import os
import re
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright

from demo import upstream
from demo.app import create_app
from tests.e2e.mobile_smoke import prepare, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        data = Path(directory)
        prepare(data)
        vehicles = json.loads((data / 'vehicles.json').read_text(encoding='utf-8'))
        for vehicle in vehicles['vehicles']:
            for session in vehicle['sessions']:
                if session['session_id'] == 'F08-DEMO':
                    session['signal_quality'] = 'warning'
        (data / 'vehicles.json').write_text(json.dumps(vehicles, ensure_ascii=False), encoding='utf-8')
        key = ('ДЕМО-01', 'F08-DEMO')
        stack.enter_context(patch.dict(upstream.measurements, {
            key: upstream.measurements[key].model_copy(update={'signal_quality': 'warning'})}))
        url = stack.enter_context(serve(upstream.app))
        stack.enter_context(patch.dict(os.environ, {
            'DATA_DIR': directory, 'ARTIFACTS_DIR': str(data / 'artifacts'), 'VEHICLES_PATH': str(data / 'vehicles.json'),
            'ML_API_URL': url, 'MAINTENANCE_API_URL': url, 'PLANNER_MODE': 'offline'}))
        base = stack.enter_context(serve(create_app()))
        playwright = stack.enter_context(sync_playwright())
        browser = playwright.chromium.launch(channel=args.channel)
        stack.callback(browser.close)
        page = browser.new_context(viewport={'width': 390, 'height': 844}, is_mobile=True, has_touch=True,
                                   accept_downloads=True).new_page()
        page.goto(base)

        page.get_by_role('button', name='Главный механик').click()
        expect(page.locator('.fleet-row')).to_have_count(2)
        expect(page.locator('.fleet-counters')).to_contain_text('Требует внимания')
        page.locator('.fleet-row button').first.click()
        expect(page.get_by_role('tab', name='История')).to_have_attribute('aria-selected', 'true')
        print('PASS — главный механик: статусы, очередь, переход к истории')

        page.get_by_label('Двигатель', exact=True).select_option('Учебный Д-02')
        page.get_by_label('Транспортное средство').select_option('ДЕМО-02')
        page.get_by_role('tab', name='История').click()
        expect(page.locator('.score-chart')).to_contain_text('сбой')
        print('PASS — график оценки: сбой показан разрывом')

        page.get_by_label('Двигатель', exact=True).select_option('Учебный Д-01')
        page.get_by_label('Транспортное средство').select_option('ДЕМО-01')
        page.get_by_role('tab', name='Результат').click()
        page.get_by_role('button', name='Получить результат диагностики').click()
        form = page.locator('#panel-result .warning-confirm')
        expect(form).to_be_visible()
        expect(page.locator('#panel-result .inspection-form')).to_have_count(0)
        form.get_by_role('button', name='Продолжить с этой записью').click()          # unchecked: refused
        expect(form).to_be_visible()
        form.get_by_label(re.compile('Понимаю, что запись с помехами')).check()
        form.get_by_role('button', name='Продолжить с этой записью').click()
        expect(page.locator('.report-reading')).to_contain_text('Порог уверенности 70%')
        expect(page.locator('#panel-result')).to_contain_text('после подтверждения механика')
        expect(page.locator('#panel-result')).to_contain_text('По этой машине')
        print('PASS — запись с помехами: вывод только после подтверждения, порог показан, память по машине')

        page.get_by_role('tab', name='Заявка').click()
        page.get_by_role('button', name='Создать заявку', exact=True).click()
        with page.expect_response(lambda r: r.url.endswith('/agent/tickets/confirm')) as confirmed:
            page.get_by_role('button', name='Подтвердить заявку').click()
        assert confirmed.value.ok, confirmed.value.text()
        print('PASS — заявка по подтверждённой записи с помехами создана')

        page.get_by_role('tab', name='Результат').click()
        inspection = page.locator('.inspection-form')
        inspection.get_by_label('Причина другая').check()
        inspection.get_by_label('Метки совпали').check()
        inspection.get_by_label('Комментарий').fill('Шум от генератора')
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as saved:
            inspection.get_by_role('button', name='Записать результат осмотра').click()
        assert saved.value.json()['inspections'][0]['marks'] == 'matched'
        expect(page.locator('.report')).to_contain_text('Отклонение')
        page.get_by_role('tab', name='История').click()
        expect(page.locator('#panel-history')).to_contain_text('Шум от генератора')
        with page.expect_download() as download:
            page.get_by_role('button', name='Скачать JSON').click()
        assert json.loads(Path(download.value.path()).read_text())[0]['comment'] == 'Шум от генератора'
        print('PASS — результат осмотра: запись, история, JSON; диагноз не изменился')

        page.locator('#ui-language').select_option('uz')
        expect(page.get_by_role('tab', name='Tarix')).to_be_visible()
        expect(page.locator('#panel-history')).to_contain_text("Ko'rik natijalari")
        print('PASS — интерфейс на узбекском, включая новые блоки')


if __name__ == '__main__':
    main()
