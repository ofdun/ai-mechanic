"""Standalone browser acceptance run against the local synthetic training stand."""
import argparse
from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import re
import socket
import tempfile
from threading import Thread
import time
from unittest.mock import patch

import uvicorn
from playwright.sync_api import expect, sync_playwright

from demo import upstream
from demo.app import create_app
from demo.fixtures import DOCUMENT_ID, MANUAL, OPERATION, PDF_NAME, RISK
from demo.prepare import prepare


@contextmanager
def serve(app):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
        worker = Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not worker.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError('The local test server did not start.')
                time.sleep(0.05)
            yield f'http://127.0.0.1:{port}'
        finally:
            server.should_exit = True
            worker.join(timeout=10)


def exercise(context, base_url, pdf_path):
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(base_url)
    vehicle = page.get_by_label('Транспортное средство')
    measurement = page.get_by_label('Проверка', exact=True)
    diagnose = page.get_by_role('button', name='Получить результат диагностики')
    draft = page.get_by_role('button', name='Создать заявку', exact=True)
    confirm = page.get_by_role('button', name='Подтвердить заявку', exact=True)
    vehicle.select_option('ДЕМО-01')
    measurement.select_option('F08-DEMO')
    diagnose.tap()
    expect(page.get_by_role('heading', name='Отклонение', exact=True)).to_be_visible()
    edit_selection = page.get_by_role('button', name=re.compile('Изменить'))
    expect(edit_selection).to_have_attribute('aria-expanded', 'false')
    expect(vehicle).to_be_hidden()
    expect(page.get_by_text(RISK, exact=False).first).to_be_visible()
    expect(page.get_by_text(OPERATION, exact=False).first).to_be_visible()

    link = page.get_by_role('link', name=f'{MANUAL}, стр. 2').first
    href = f'/api/documents/{DOCUMENT_ID}#page=2'
    expect(link).to_have_attribute('href', href)
    with context.expect_page() as opened:
        link.tap()
    source = opened.value
    source.wait_for_url(base_url + href)
    document = context.request.get(source.url)
    assert document.ok and document.headers['content-type'] == 'application/pdf'
    assert document.body() == pdf_path.read_bytes()
    source.close()
    print('PASS — отклонение, риск, работа, ссылка и PDF на физической странице 2')

    question = page.get_by_label('Вопрос по двигателю')
    send = page.get_by_role('button', name='Отправить', exact=True)
    question.fill('Почему это опасно?')
    send.tap()
    expect(page.get_by_role('article')).to_have_count(2)
    answer = page.get_by_role('article').nth(1)
    expect(answer).to_contain_text(RISK)
    expect(answer.get_by_role('link')).to_have_attribute('href', href)
    print('PASS — F4, уточнение по предыдущей диагностике с подтверждённым риском и ссылкой')

    question.fill('Можно ли перегнать машину на базу?')
    page.get_by_role('button', name='Отправить', exact=True).tap()
    expect(page.get_by_text(re.compile('Рекомендуется дефектовка'))).to_be_visible()
    print('PASS — C2, уточнение без применимого источника')

    thread = page.get_by_role('log', name='Сообщения')
    previous_messages = thread.inner_text()
    draft.tap()
    preview = page.get_by_role('region', name='Черновик заявки')
    expect(preview).to_contain_text(OPERATION)
    assert thread.inner_text() == previous_messages, 'Создание черновика изменило историю диагностики'
    expect(preview).to_have_count(1)
    expect(preview).to_contain_text('F08-DEMO')
    expect(preview.get_by_role('link')).to_have_attribute('href', href)
    with page.expect_response(lambda res: res.url.endswith('/agent/tickets/confirm')) as completed:
        confirm.tap()
    receipt_response = completed.value
    assert receipt_response.ok
    ticket_id = receipt_response.json()['ticket_id']
    expect(page.get_by_text(re.compile(re.escape(ticket_id)))).to_be_visible()
    expect(confirm).to_have_count(0)
    replay = context.request.post(receipt_response.url, data=receipt_response.request.post_data_json)
    assert replay.ok and replay.json()['ticket_id'] == ticket_id
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    print('PASS — C1, черновик, явное подтверждение и повтор с тем же ticket_id')

    edit_selection.tap()
    expect(vehicle).to_be_focused()
    vehicle.select_option('ДЕМО-02')
    measurement.select_option('NORMAL-DEMO')
    expect(page.get_by_text(re.compile(re.escape(ticket_id)))).to_have_count(0)
    expect(page.get_by_text(OPERATION, exact=False)).to_have_count(0)
    diagnose.tap()
    expect(page.get_by_role('heading', name='Норма', exact=True)).to_be_visible()
    expect(page.get_by_text(re.compile('можно продолжить эксплуатацию'))).to_be_visible()
    expect(draft).to_have_count(0)
    question.fill('Почему это опасно?')
    send.tap()
    expect(page.get_by_text(re.compile('Рекомендуется дефектовка'))).to_be_visible()
    expect(page.get_by_text(RISK, exact=False)).to_have_count(0)
    print('PASS — норма, смена ТС очищает предыдущую заявку, ответ и источник')

    edit_selection.tap()
    measurement.select_option('REPEAT-DEMO')
    expect(page.get_by_role('heading', name='Норма', exact=True)).to_have_count(0)
    diagnose.tap()
    expect(page.get_by_role('heading', name='Сбой проверки', exact=True)).to_be_visible()
    expect(page.get_by_text(re.compile('Повторите замер'))).to_be_visible()
    expect(draft).to_have_count(0)
    expect(page.get_by_role('link', name=re.compile('стр\\.'))).to_have_count(0)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    print('PASS — C3, смена проверки, просьба повторить замер, заявки и рекомендации отсутствуют')
    return ticket_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', help='Например, chrome для установленного Google Chrome')
    args = parser.parse_args()
    if not Path('frontend/dist/index.html').is_file():
        parser.exit(1, 'Сначала соберите интерфейс командой npm --prefix frontend run build.\n')
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        data_dir = Path(directory)
        prepare(data_dir)
        ticket_posts = []

        async def observe_upstream(scope, receive, send):
            if scope['type'] == 'http' and scope['method'] == 'POST' and scope['path'] == '/tickets':
                ticket_posts.append(scope['path'])
            await upstream.app(scope, receive, send)

        upstream_url = stack.enter_context(serve(observe_upstream))
        stack.enter_context(patch.dict(os.environ, {
            'DATA_DIR': directory, 'ARTIFACTS_DIR': str(data_dir / 'artifacts'),
            'VEHICLES_PATH': str(data_dir / 'vehicles.json'),
            'ML_API_URL': upstream_url, 'MAINTENANCE_API_URL': upstream_url, 'PLANNER_MODE': 'offline',
        }))
        base_url = stack.enter_context(serve(create_app()))
        playwright = stack.enter_context(sync_playwright())
        browser = playwright.chromium.launch(channel=args.channel)
        stack.callback(browser.close)
        context = browser.new_context(viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
        stack.callback(context.close)
        print(f'Chromium {browser.version} · 390×844 · учебные HTTP-сервисы')
        first_ticket = exercise(context, base_url, data_dir / PDF_NAME)
        assert len(ticket_posts) == 1, f'Ожидалась одна запись в ТОиР, отправлено {len(ticket_posts)}'

        # The public training stand gives each visitor their own copy of the exercise.
        guest = browser.new_context()
        stack.callback(guest.close)
        guest_page = guest.new_page()
        guest_page.goto(base_url)
        guest_page.get_by_label('Транспортное средство').select_option('ДЕМО-01')
        guest_page.get_by_role('button', name='Получить результат диагностики').click()
        guest_page.get_by_role('button', name='Создать заявку', exact=True).click()
        with guest_page.expect_response(lambda r: r.url.endswith('/agent/tickets/confirm')) as confirmation:
            guest_page.get_by_role('button', name='Подтвердить заявку').click()
        assert confirmation.value.ok
        second_ticket = confirmation.value.json()['ticket_id']
        assert first_ticket != second_ticket and len(ticket_posts) == 2
        expect(guest_page.get_by_text(re.compile(re.escape(second_ticket)))).to_be_visible()
        print('PASS — второй посетитель может пройти учебную диагностику со своей заявкой')


if __name__ == '__main__':
    main()
