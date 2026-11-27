"""Exercise the release UI with the supplied fleet, actual mocks and a private ticket store."""
import argparse
import os
from contextlib import ExitStack
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from playwright.sync_api import expect, sync_playwright

from ml.retrieval import RetrievalEngine
from ml.worker import create_app
import socket
import subprocess
import time
from urllib.request import urlopen
from urllib.error import URLError
from mobile_smoke import serve


def exercise(page, base_url, ranked_pages):
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
    canonical = 'yamz-236-operations-2010'
    legacy = 'yamz-236_re_2010'
    route = f'/api/documents/{canonical}/pages/52'
    current = page.request.get(base_url + route)
    old = page.request.get(base_url + route.replace(canonical, legacy))
    assert current.ok and old.ok and current.json() == old.json() and old.json()['page'] == 52
    question.fill('Почему это опасно?')
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as followup_response:
        question.press('Enter')
    cited = followup_response.value.json()
    assert any(s['document_id'] == canonical and s['pdf_page'] == 52 for s in cited['evidence']['items'])
    with page.expect_response(lambda r: r.url == base_url + route) as cited_page:
        thread.get_by_role('button', name=re.compile(r'стр\. 52\b')).first.click()
    assert cited_page.value.ok
    expect(page.get_by_text('Страница 52 из 172')).to_be_visible()
    page.wait_for_function('document.querySelector(".manual img")?.naturalWidth > 0')
    page.get_by_role('button', name='К чату').click()
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
    with page.expect_response(lambda r: r.url == base_url + route) as draft_page:
        page.get_by_role('tabpanel', name='Заявка').get_by_role('button', name=re.compile(r'стр\. 52\b')).first.click()
    assert draft_page.value.ok
    expect(page.get_by_text('Страница 52 из 172')).to_be_visible()
    page.wait_for_function('document.querySelector(".manual img")?.naturalWidth > 0')
    page.get_by_role('button', name='К заявке').click()
    page.get_by_label('Приоритет', exact=True).select_option('high')
    with page.expect_response(lambda r: r.url.endswith('/agent/tickets/confirm')) as response:
        page.get_by_role('button', name='Подтвердить заявку').click()
    confirmed = response.value
    assert confirmed.ok
    assert confirmed.request.post_data_json['urgency'] == 'high'
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

    vehicle.select_option('Y652CK98')
    session.select_option('S-0132')
    diagnose.click()
    expect(page.get_by_role('heading', name='Ошибка распознавания')).to_be_visible()
    chat.click()
    expect(thread).to_contain_text('Ошибка распознавания')
    expect(thread).not_to_contain_text('Уверенность ИИ')
    ticket.click()
    expect(create_ticket).to_be_disabled()

    vehicle.select_option('P854BT78')
    session.select_option('S-0021')
    diagnose.click()
    report = page.get_by_role('tabpanel', name='Результат')
    expect(report).to_contain_text('66%')
    expect(report).to_contain_text('порог для этой проверки 70%')
    expect(report).to_contain_text('помехи')

    vehicle.select_option('T801PO178')
    session.select_option('S-0046')
    diagnose.click()
    expect(report).to_contain_text('F11')
    open_ticket.click()
    handoff = page.get_by_role('region', name='Рекомендация на дефектовку')
    for field in ('T801PO178', 'ЯМЗ-236М2', 'S-0046', 'F11', 'порядку автопарка'):
        expect(handoff).to_contain_text(field)
    expect(create_ticket).to_be_disabled()

    target = next(v for v in catalog if v['engine_model'] == 'ЗМЗ-5234.10')
    engines = page.get_by_label('Двигатель', exact=True)
    engines.select_option(label=target['engine_model'])
    expect(vehicle.locator('option')).to_have_count(sum(v['engine_model'] == target['engine_model'] for v in catalog))
    vehicle.select_option(target['vehicle_id'])
    chat.click()
    query = 'Как проверить термостат?'
    question.fill(query)
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as response:
        question.press('Enter')
    search = response.value.json()
    assert search['action'] == 'search_manuals' and search['engine_model'] == target['engine_model']
    candidates = ranked_pages.get((target['engine_model'], query), set())
    assert candidates, 'The browser question did not reach the team retrieval engine'
    sources = search['evidence']['items']
    assert sources and all(s['evidence_id'] in candidates and target['engine_model'] in s['engine_models']
                           for s in sources)
    source = sources[0]
    route = f"/api/documents/{source['document_id']}/pages/{source['pdf_page']}"
    with page.expect_response(lambda r: r.url == base_url + route) as physical_page:
        thread.get_by_role('button', name=re.compile(rf"стр\. {source['pdf_page']}$")).first.click()
    assert physical_page.value.ok and physical_page.value.json()['page'] == source['pdf_page']
    page.wait_for_function('document.querySelector(".manual img")?.naturalWidth > 0')
    page.get_by_role('button', name='К чату').click()

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
    print('PASS: 50 vehicles / 195 sessions, C1–C3, confirmed ticket and replay, team search and cited PDF pages.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--mock-dir', type=Path, required=True, help='Unpacked directory containing mock_ml and mock_toir')
    parser.add_argument('--index', type=Path, help='Reuse manuals.sqlite3 with its sibling retrieval directory')
    parser.add_argument('--channel')
    parser.add_argument('--backend', type=Path, required=True, help='Compiled Go web server')
    args = parser.parse_args()
    sys.path.insert(0, str(args.mock_dir.resolve(strict=True)))
    from mock_ml.app import app as mock_ml
    from mock_toir.app import app as mock_toir

    with ExitStack() as stack:
        state = Path(stack.enter_context(TemporaryDirectory()))
        stack.enter_context(patch.dict(os.environ, {
            'DATA_DIR': str(args.data_dir), 'VEHICLES_PATH': str(args.mock_dir / 'data' / 'vehicles.json'),
            'ARTIFACTS_DIR': str(state), 'PLANNER_MODE': 'local', 'AGENT_ARCHITECTURE': 'baseline',
            'WEB_RETRIEVAL_MODE': 'team', 'REPEAT_ON_WARNING': 'true',
            'ALLOW_EXTERNAL_EVIDENCE': 'false', 'MIN_DIAGNOSTIC_CONFIDENCE': '0.70',
            'ML_API_URL': stack.enter_context(serve(mock_ml)),
            'MAINTENANCE_API_URL': stack.enter_context(serve(mock_toir)),
        }))
        if args.index is not None:
            (state / 'manuals.sqlite3').symlink_to(args.index.resolve(strict=True))
            (state / 'retrieval').symlink_to((args.index.parent / 'retrieval').resolve(strict=True))
        else:
            subprocess.run([sys.executable, '-m', 'ml.manual_corpus'], check=True)

        ranked_pages = {}
        search = RetrievalEngine.search

        def record_search(engine, plan):
            evidence = search(engine, plan)
            ranked_pages[(plan.engine_model, plan.original_query)] = {s.evidence_id for s in evidence.items}
            return evidence

        stack.enter_context(patch.object(RetrievalEngine, 'search', record_search))
        agent_url = stack.enter_context(serve(create_app()))
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            address = f'127.0.0.1:{listener.getsockname()[1]}'
        base_url = f'http://{address}'
        process = subprocess.Popen([args.backend.resolve(strict=True)], env={
            **os.environ, 'LISTEN_ADDR': address, 'AGENT_URL': agent_url,
            'FRONTEND_DIST': str(Path('app/frontend/dist').resolve()),
        })
        stack.callback(process.wait, 10)
        stack.callback(process.terminate)
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(base_url + '/health', timeout=1):
                    break
            except URLError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        pw = stack.enter_context(sync_playwright())
        browser = pw.chromium.launch(channel=args.channel)
        stack.callback(browser.close)
        page = browser.new_page(viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
        exercise(page, base_url, ranked_pages)


if __name__ == '__main__':
    main()
