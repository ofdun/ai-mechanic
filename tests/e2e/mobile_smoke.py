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
from demo.fixtures import DOCUMENT_ID, MANUAL, OPERATION, RISK
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


def exercise(context, base_url):
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(base_url)
    vehicle = page.get_by_label('Транспортное средство')
    engine = page.get_by_label('Двигатель', exact=True)
    measurement = page.get_by_label('Проверка', exact=True)
    diagnose = page.get_by_role('button', name='Получить результат диагностики')
    draft = page.get_by_role('button', name='Создать заявку', exact=True)
    confirm = page.get_by_role('button', name='Подтвердить заявку', exact=True)
    vehicle.select_option('ДЕМО-01')
    expect(engine).to_have_value('')
    engine.select_option('Учебный Д-01')
    expect(vehicle.locator('option')).to_have_count(1)
    expect(vehicle.locator('optgroup[label="Учебный Д-01"] option')).to_have_text('ДЕМО-01 · Учебный Д-01')
    measurement.select_option('F08-DEMO')
    chat_view = page.get_by_role('tab', name='Чат', exact=True)
    ticket_view = page.get_by_role('tab', name='Заявка', exact=True)
    footer = page.locator('.panel-footer')
    def footer_offset():
        return footer.evaluate('(node) => node.getBoundingClientRect().top + window.scrollY')
    initial_offset = footer_offset()
    diagnose.tap()
    expect(page.get_by_role('heading', name='Отклонение', exact=True)).to_be_visible()
    expect(vehicle).to_be_visible()
    assert abs(footer_offset() - initial_offset) < 2
    expect(page.get_by_text(RISK, exact=False).first).to_be_visible()
    expect(page.get_by_text(OPERATION, exact=False).first).to_be_visible()

    report = page.get_by_role('tabpanel', name='Результат', exact=True)
    expect(report.get_by_role('heading', name='Что делать дальше')).to_be_visible()
    source_name = f'{MANUAL}, стр. 2'
    source = report.get_by_role('button', name=source_name, exact=True)
    doc_requests, downloads, popups = [], [], []
    page.on('request', lambda req: doc_requests.append(req.url) if '/api/documents/' in req.url else None)
    page.on('download', lambda download: downloads.append(download))
    page.on('popup', lambda popup: popups.append(popup))
    expect(report).to_contain_text('Искусственный интеллект')
    expect(report).to_contain_text('91%')
    expect(report).not_to_contain_text('ML')
    source.scroll_into_view_if_needed()
    old_scroll = report.evaluate('el => el.scrollTop')
    source.tap()
    expect(page.get_by_text('Страница 2 из 2')).to_be_visible()
    page.wait_for_function('document.querySelector(".manual img")?.naturalWidth > 0')
    page.get_by_role('button', name='Текст', exact=True).tap()
    expect(page.locator('.manual-text')).to_contain_text('Сверить маркировку')
    page.get_by_role('button', name='Предыдущая страница').tap()
    expect(page.locator('.manual-text')).to_contain_text('Демонстрационные данные')
    expect(page.get_by_role('button', name='Предыдущая страница')).to_be_disabled()
    page.go_back()
    expect(source).to_be_focused()
    assert abs(report.evaluate('el => el.scrollTop') - old_scroll) < 2
    report.get_by_role('button', name=re.compile('^Источник 1')).first.tap()
    expect(page.get_by_text('Страница 2 из 2')).to_be_visible()
    page.get_by_role('button', name='К результату').tap()
    print('PASS — физическая страница 2 в приложении, оригинал и текст, перелистывание и возврат к той же строке')

    page.get_by_role('button', name='Задать вопрос').tap()
    question = page.get_by_label('Вопрос по двигателю')
    expect(question).to_be_focused()
    thread = page.get_by_role('log', name='Сообщения')
    verdict = thread.get_by_role('heading', name='Отклонение', exact=True)
    expect(verdict).to_have_count(1)
    expect(thread).to_contain_text('91%')
    expect(thread).not_to_contain_text(OPERATION)
    with page.expect_response(lambda res: res.url.endswith('/agent/run')) as repeated:
        diagnose.tap()
    assert repeated.value.ok
    expect(chat_view).to_have_attribute('aria-selected', 'true')
    expect(verdict).to_have_count(1)
    print('PASS — F2, один вердикт в чате после двух запросов диагностики')
    send = page.get_by_role('button', name='Отправить', exact=True)
    question.fill('Почему это опасно?')
    send.tap()
    expect(page.get_by_role('article')).to_have_count(1)
    answer = page.get_by_role('article').first
    expect(answer).to_contain_text(RISK)
    answer.get_by_role('button', name=source_name).tap()
    expect(page.get_by_text('Страница 2 из 2')).to_be_visible()
    page.get_by_role('button', name='К чату').tap()
    expect(answer.get_by_role('button', name=source_name)).to_be_focused()
    print('PASS — F4, уточнение по предыдущей диагностике с подтверждённым риском и ссылкой')

    # The offline backend supplies genuine evidence; stand in only for the optional writer.
    summary = 'Сначала проверьте настройку стенда.\nПосле этого повторите учебный замер.'

    def with_summary(route):
        upstream = route.fetch()
        body = upstream.json()
        assert body['claims'] and body['action'] == 'search_manuals'
        route.fulfill(response=upstream, json={**body, 'message': summary, 'summary': summary})

    page.route('**/agent/run', with_summary, times=1)
    question.fill('Почему это опасно?')
    send.tap()
    generated = thread.get_by_role('article').last
    expect(generated.locator('.manual-answer')).to_have_text(summary)
    expect(generated.get_by_text(summary, exact=True)).to_have_count(1)
    expect(generated.get_by_text(RISK, exact=False).first).to_be_visible()
    generated.get_by_role('button', name=source_name).tap()
    expect(page.get_by_text('Страница 2 из 2')).to_be_visible()
    page.get_by_role('button', name='К чату').tap()
    expect(generated.locator('details')).to_have_count(0)
    expect(generated.get_by_text(RISK, exact=False).first).to_be_visible()
    print('PASS — один краткий ответ, его источник и видимая формулировка из руководства')

    delayed = []
    page.route('**/agent/run', lambda route: delayed.append(route), times=1)
    text = 'Почему это опасно при следующем замере?'
    question.fill(text)
    question.press('Enter')
    turn = thread.locator('.turn').last
    expect(turn.get_by_text(text, exact=True)).to_be_visible()
    expect(turn.get_by_text('Ищем ответ…')).to_be_visible()
    expect(question).to_have_value('')
    question.fill('Следующий вопрос')
    delayed[0].abort()
    expect(turn.get_by_role('button', name='Повторить')).to_be_visible()
    turn.get_by_role('button', name='Повторить').tap()
    expect(turn.get_by_role('article')).to_be_visible()
    expect(thread.get_by_text(text, exact=True)).to_have_count(1)
    expect(question).to_have_value('Следующий вопрос')

    question.fill('Можно ли перегнать машину на базу?')
    page.get_by_role('button', name='Отправить', exact=True).tap()
    expect(page.get_by_text(re.compile('Рекомендуется дефектовка'))).to_be_visible()
    print('PASS — C2, уточнение без применимого источника')

    question.fill('Сколько стоит маркировка учебного узла?')
    send.tap()
    reply = thread.locator('.turn').last.get_by_role('article')
    candidates = reply.get_by_role('heading', name='Страницы по теме для самостоятельной проверки').locator('..')
    expect(reply.locator('.search-fallback')).to_have_count(1)
    expect(reply.get_by_role('heading', name='Ограничения')).to_have_count(0)
    candidate_page = candidates.get_by_role('button', name=source_name)
    candidate_page.tap()
    expect(page.get_by_text('Страница 2 из 2')).to_be_visible()
    page.get_by_role('button', name='К чату').tap()
    expect(candidate_page).to_be_focused()

    previous_messages = thread.inner_text()
    ticket_view.tap()
    draft.tap()
    preview = page.get_by_role('region', name='Черновик заявки')
    expect(preview).to_contain_text(OPERATION)
    work_value = preview.locator('dt', has_text='Работы').locator('+ dd')
    expect(work_value).to_have_text(OPERATION)
    expect(work_value.get_by_role('list')).to_have_count(0)
    assert abs(footer_offset() - initial_offset) < 2
    chat_view.tap()
    assert thread.inner_text() == previous_messages, 'Создание черновика изменило историю'
    ticket_view.tap()
    expect(preview).to_have_count(1)
    expect(preview).to_contain_text('F08-DEMO')
    preview.get_by_role('button', name=source_name).tap()
    expect(page.get_by_text('Страница 2 из 2')).to_be_visible()
    page.get_by_role('button', name='К заявке').tap()
    expect(preview.get_by_role('button', name=source_name)).to_be_focused()
    assert not popups and not downloads
    assert doc_requests and all(f'/api/documents/{DOCUMENT_ID}/pages/' in url for url in doc_requests)
    priority = preview.get_by_label('Приоритет', exact=True)
    priority.select_option('high')
    expect(preview).to_contain_text('до выпуска в рейс')
    held = []
    page.route('**/agent/tickets/confirm', lambda route: held.append(route))
    with page.expect_response(lambda res: res.url.endswith('/agent/tickets/confirm')) as completed:
        confirm.tap()
        expect(engine).to_be_disabled()
        expect(vehicle).to_be_disabled()
        expect(measurement).to_be_disabled()
        expect(priority).to_be_disabled()
        held[0].continue_()
    receipt_response = completed.value
    assert receipt_response.ok
    assert receipt_response.request.post_data_json['urgency'] == 'high'
    ticket_id = receipt_response.json()['ticket_id']
    expect(page.get_by_text(re.compile(re.escape(ticket_id)))).to_be_visible()
    expect(confirm).to_have_count(0)
    replay = context.request.post(receipt_response.url, data=receipt_response.request.post_data_json)
    assert replay.ok and replay.json()['ticket_id'] == ticket_id
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    print('PASS — C1, черновик, явное подтверждение и повтор с тем же ticket_id')

    engine.select_option('Учебный Д-02')
    expect(engine).to_be_focused()
    expect(vehicle).to_have_value('ДЕМО-02')
    expect(vehicle.locator('option')).to_have_count(1)
    expect(measurement).to_have_value('REPEAT-DEMO')
    agent_calls = []
    page.on('request', lambda req: agent_calls.append(req.url) if '/agent/' in req.url else None)
    saved_turns = thread.locator('.turn').count()
    page.get_by_role('tab', name='История', exact=True).tap()
    panel = page.get_by_role('tabpanel', name='История', exact=True)
    history = panel.locator('.vehicle-history')
    expect(panel.get_by_role('heading', name='История проверок · ДЕМО-02')).to_be_visible()
    expect(thread.locator('.turn')).to_have_count(saved_turns)
    expect(history.locator('li')).to_have_count(2)
    expect(history.locator('li').first).to_contain_text('REPEAT-DEMO')
    expect(history.locator('li').first).to_contain_text('Сбой проверки')
    expect(history.locator('li').last).to_contain_text('Норма')
    assert not agent_calls
    measurement.select_option('NORMAL-DEMO')
    expect(history).to_have_count(0)
    engine.select_option('')
    expect(vehicle.locator('option')).to_have_count(2)
    expect(vehicle).to_have_value('ДЕМО-02')
    expect(measurement).to_have_value('NORMAL-DEMO')
    expect(page.get_by_text(re.compile(re.escape(ticket_id)))).to_have_count(0)
    expect(page.get_by_text(OPERATION, exact=False)).to_have_count(0)
    diagnose.tap()
    expect(page.get_by_role('heading', name='Норма', exact=True)).to_be_visible()
    expect(report.get_by_text(re.compile('можно продолжить эксплуатацию'))).to_be_visible()
    expect(draft).to_have_count(0)
    chat_view.tap()
    expect(thread).to_contain_text('можно продолжить эксплуатацию')
    expect(verdict).to_have_count(0)
    question.fill('Почему это опасно?')
    send.tap()
    expect(page.get_by_text(re.compile('Рекомендуется дефектовка'))).to_be_visible()
    expect(page.get_by_text(RISK, exact=False)).to_have_count(0)
    print('PASS — норма, смена ТС очищает предыдущую заявку, ответ и источник')

    measurement.select_option('REPEAT-DEMO')
    expect(page.get_by_role('heading', name='Норма', exact=True)).to_have_count(0)
    diagnose.tap()
    expect(page.get_by_role('heading', name='Сбой проверки', exact=True)).to_be_visible()
    expect(report.get_by_text(re.compile('Повторите замер'))).to_be_visible()
    expect(draft).to_have_count(0)
    expect(page.get_by_role('button', name=re.compile('стр\\.'))).to_have_count(0)
    chat_view.tap()
    expect(thread).to_contain_text('Повторите замер')
    expect(thread).not_to_contain_text('можно продолжить эксплуатацию')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    print('PASS — C3, смена проверки, просьба повторить замер, заявки и рекомендации отсутствуют')
    return ticket_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', help='Например, chrome для установленного Google Chrome')
    args = parser.parse_args()
    if not Path('app/frontend/dist/index.html').is_file():
        parser.exit(1, 'Сначала соберите интерфейс командой npm --prefix app/frontend run build.\n')
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
        first_ticket = exercise(context, base_url)
        assert len(ticket_posts) == 1, f'Ожидалась одна запись в ТОиР, отправлено {len(ticket_posts)}'

        # The public training stand gives each visitor their own copy of the exercise.
        guest = browser.new_context()
        stack.callback(guest.close)
        guest_page = guest.new_page()
        guest_page.goto(base_url)
        guest_page.get_by_label('Транспортное средство').select_option('ДЕМО-01')
        guest_page.get_by_role('button', name='Получить результат диагностики').click()
        result_tab = guest_page.get_by_role('tab', name='Результат')
        result_tab.focus()
        result_tab.press('End')
        ticket_tab = guest_page.get_by_role('tab', name='Заявка')
        expect(ticket_tab).to_be_focused()
        expect(ticket_tab).to_have_attribute('aria-selected', 'true')
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
