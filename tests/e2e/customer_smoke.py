"""Exercise the release UI with the supplied fleet, actual mocks and a private ticket store."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from playwright.sync_api import expect, sync_playwright

from ml.config import Settings
from web.app import create_app
from mobile_smoke import serve


def exercise(page, base_url):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(base_url)
    catalog = page.request.get(base_url + '/api/vehicles').json()['vehicles']
    assert len(catalog) == 50 and sum(len(v['sessions']) for v in catalog) == 195
    assert len({v['engine_model'] for v in catalog}) == 5
    vehicle = page.get_by_label('Транспортное средство')
    session = page.get_by_label('Проверка', exact=True)
    diagnose = page.get_by_role('button', name='Получить результат диагностики')
    chat = page.get_by_role('tab', name='Чат', exact=True)
    ticket = page.get_by_role('tab', name='Заявка', exact=True)
    open_ticket = page.get_by_role('button', name='К заявке', exact=True)
    create_ticket = page.get_by_role('button', name='Создать заявку', exact=True)
    thread = page.get_by_role('log')
    question = page.get_by_label('Вопрос по двигателю')

    open_ticket.click()
    expect(ticket).to_be_focused()
    expect(page.get_by_role('tabpanel', name='Заявка')).to_contain_text('Сначала получите результат диагностики')
    expect(create_ticket).to_be_disabled()
    vehicle.select_option('M333YK47')
    session.select_option('S-0005')
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
        diagnose.click()
    diagnosis = response.value.json()
    assert diagnosis['diagnostic']['anomaly_type'] == 'F08'
    assert diagnosis['engine_model'] == 'ЯМЗ-236М2' and diagnosis['can_create_ticket']
    assert {'risk', 'operation', 'reference_value'} <= {c['kind'] for c in diagnosis['claims']}
    expect(page.get_by_role('heading', name='Отклонение', exact=True)).to_be_visible()
    chat.click()
    expect(thread).to_contain_text('Отклонение')
    question.fill('Можно ли своим ходом перегнать машину на базу?')
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
        question.press('Enter')
    followup = response.value.json()
    assert followup['action'] == 'search_manuals'
    if followup['claims']:
        assert all(claim['evidence_ids'] for claim in followup['claims'])
    else:
        expect(thread).to_contain_text('Рекомендуется дефектовка')
    page.get_by_role('tab', name='Результат', exact=True).click()
    open_ticket.click()
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
        create_ticket.click()
    draft = response.value.json()['ticket_draft']['payload']
    assert draft['operations'] and draft['references'] and draft['anomaly_type'] == 'F08'
    assert (draft['vehicle_id'], draft['session_id']) == ('M333YK47', 'S-0005')
    with page.expect_response(lambda r: r.url.endswith('/agent/tickets/confirm')) as response:
        page.get_by_role('button', name='Подтвердить заявку').click()
    confirmed = response.value
    assert confirmed.ok
    receipt = confirmed.json()['ticket_id']
    expect(page.get_by_text(receipt, exact=True)).to_be_visible()
    replay = page.request.post(confirmed.url, data=confirmed.request.post_data_json)
    assert replay.ok and replay.json()['ticket_id'] == receipt

    vehicle.select_option('P456AT178')
    for measurement, verdict, reason in [('S-0001', 'Норма', 'Заявка на дефектовку не требуется'),
                                         ('S-0003', 'Сбой проверки', 'Повторите замер')]:
        session.select_option(measurement)
        diagnose.click()
        expect(page.get_by_role('heading', name=verdict, exact=True)).to_be_visible()
        open_ticket.click()
        expect(page.get_by_role('tabpanel', name='Заявка')).to_contain_text(reason)
        expect(create_ticket).to_be_disabled()

    target = next(v for v in catalog if v['engine_model'] == 'ЗМЗ-5234.10')
    engines = page.get_by_label('Двигатель', exact=True)
    engines.select_option(label=target['engine_model'])
    expect(vehicle.locator('option')).to_have_count(sum(v['engine_model'] == target['engine_model'] for v in catalog))
    vehicle.select_option(target['vehicle_id'])
    chat.click()
    question.fill('Какой момент затяжки крышек коренных подшипников?')
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
        question.press('Enter')
    answer = response.value.json()
    assert answer['engine_model'] == target['engine_model']
    evidence = {e['evidence_id']: e for e in answer['evidence']['items']}
    norms = [claim for claim in answer['claims'] if claim['kind'] == 'reference_value']
    assert any(evidence[key]['pdf_page'] == 137 for claim in norms for key in claim['evidence_ids'])
    assert all(answer['engine_model'] in evidence[key]['engine_models'] for claim in norms for key in claim['evidence_ids'])
    thread.get_by_role('button', name=re.compile(r'стр\. 137\b')).first.click()
    expect(page.get_by_text('Страница 137 из 158')).to_be_visible()
    page.wait_for_function('document.querySelector(".manual img")?.naturalWidth > 0')
    page.get_by_role('button', name='К чату').click()
    expect(question).to_be_visible()
    engines.select_option(label='Все двигатели')
    expect(vehicle.locator('option')).to_have_count(50)
    expect(vehicle).to_have_value(target['vehicle_id'])
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    print('PASS: 50 vehicles / 195 sessions, C1 ticket and replay, C2, C3, normal, engine-specific source on PDF page 137.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--mock-dir', type=Path, required=True, help='Unpacked directory containing mock_ml and mock_toir')
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--channel')
    args = parser.parse_args()
    sys.path.insert(0, str(args.mock_dir.resolve(strict=True)))
    from mock_ml.app import app as mock_ml
    from mock_toir.app import app as mock_toir

    with ExitStack() as stack:
        state = Path(stack.enter_context(TemporaryDirectory()))
        (state / 'manuals.sqlite3').symlink_to(args.index.resolve(strict=True))
        settings = Settings(
            _env_file=None, data_dir=args.data_dir, vehicles_path=args.mock_dir / 'data' / 'vehicles.json',
            artifacts_dir=state, planner_mode='local', agent_architecture='baseline',
            allow_external_evidence=False, ml_api_url=stack.enter_context(serve(mock_ml)),
            maintenance_api_url=stack.enter_context(serve(mock_toir)),
        )
        base_url = stack.enter_context(serve(create_app(settings)))
        pw = stack.enter_context(sync_playwright())
        browser = pw.chromium.launch(channel=args.channel)
        stack.callback(browser.close)
        page = browser.new_page(viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
        exercise(page, base_url)


if __name__ == '__main__':
    main()
