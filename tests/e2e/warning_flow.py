"""Check consent when the local ML fixture changes quality during an inspection."""
from unittest.mock import patch

from playwright.sync_api import expect


def exercise(page, base_url, records):
    key = ('M333YK47', 'S-0005')
    clean = records[key]
    page.goto(base_url)
    page.get_by_label('Двигатель', exact=True).select_option('ЯМЗ-236М2')
    page.get_by_label('Транспортное средство').select_option(key[0])
    page.get_by_label('Проверка', exact=True).select_option(key[1])
    diagnosis = page.get_by_role('button', name='Получить результат диагностики')
    with page.expect_response(lambda r: r.url.endswith('/agent/run')) as first:
        diagnosis.click()
    body = first.value.request.post_data_json
    drafts = {**body, 'command': 'draft_ticket'}
    old = page.request.post(base_url + '/agent/run', data=drafts).json()['ticket_draft']

    with patch.dict(records, {key: {**clean, 'signal_quality': 'warning', 'confidence': 0.70}}):
        confirm = page.request.post(base_url + '/agent/tickets/confirm', data={
            'vehicle_id': key[0], 'session_id': key[1], 'draft_id': old['draft_id'],
            'draft_version': old['draft_version'], 'confirmed': True,
        })
        assert confirm.status == 409 and confirm.json()['detail']['code'] == 'REMEASUREMENT_REQUIRED'
        denied = page.request.post(base_url + '/agent/run', data=drafts).json()
        assert denied['warning_confirmable'] and not denied['ticket_draft']
        diagnosis.click()
        report = page.locator('#panel-result')
        checkbox = report.get_by_role('checkbox')
        proceed = report.get_by_role('button', name='Продолжить с этой записью')
        expect(checkbox).not_to_be_checked()
        proceed.click()
        expect(checkbox).to_be_focused()
        checkbox.check()
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as accepted:
            proceed.click()
        assert accepted.value.json()['signal_warning_accepted']
        expect(report).to_contain_text('после подтверждения механика')
        expect(report).to_contain_text('Порог уверенности 70%')
        page.get_by_role('tab', name='Чат', exact=True).click()
        expect(page.get_by_role('log').locator('.turn').last.get_by_role('heading', name='Отклонение')).to_be_visible()
        lang = page.locator('#ui-language')
        with page.expect_response(lambda r: r.url.endswith('/agent/run')):
            lang.select_option('uz')
        page.locator('#tab-result').click()
        expect(report).to_contain_text('Ishonch chegarasi 70%')
        with page.expect_response(lambda r: r.url.endswith('/agent/run')):
            lang.select_option('ru')
        page.get_by_role('tab', name='Заявка', exact=True).click()
        with page.expect_response(lambda r: r.url.endswith('/agent/run')) as draft:
            page.get_by_role('button', name='Создать заявку', exact=True).click()
        assert draft.value.json()['ticket_draft']['signal_warning_accepted']
        with page.expect_response(lambda r: r.url.endswith('/agent/tickets/confirm')) as sent:
            page.get_by_role('button', name='Подтвердить заявку').click()
        expect(page.get_by_text(sent.value.json()['ticket_id'], exact=True)).to_be_visible()
    print('PASS: rejected stale draft and missing consent, confirmed warning at threshold, chat verdict and receipt.')
