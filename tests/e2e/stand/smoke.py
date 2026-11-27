"""Exercise the running Compose stack through its authenticated gateway."""
import os
from playwright.sync_api import expect, sync_playwright


def exercise(page, base_url="https://gateway"):
    page.goto(base_url)
    catalog = page.request.get(base_url + "/api/vehicles").json()["vehicles"]
    assert len(catalog) == 50 and sum(len(v["sessions"]) for v in catalog) == 195

    def diagnose(plate, session):
        model = next(v["engine_model"] for v in catalog if v["vehicle_id"] == plate)
        page.get_by_label("Двигатель", exact=True).select_option(model)
        page.get_by_label("Транспортное средство").select_option(plate)
        page.get_by_label("Проверка", exact=True).select_option(session)
        page.get_by_role("tab", name="Результат", exact=True).click()
        with page.expect_response(lambda r: r.url.endswith("/agent/run")) as reply:
            page.get_by_role("button", name="Получить результат диагностики").click()
        assert reply.value.ok
        return reply.value.json()

    diagnosis = diagnose("M333YK47", "S-0005")
    assert diagnosis["diagnostic"]["status"] == "anomaly" and diagnosis["can_create_ticket"]
    page.get_by_role("button", name="К заявке", exact=True).click()
    page.get_by_role("button", name="Создать заявку", exact=True).click()
    expect(page.get_by_role("button", name="Печать")).to_be_visible()
    with page.expect_response(lambda r: r.url.endswith("/agent/tickets/confirm")) as confirmation:
        page.get_by_role("button", name="Подтвердить заявку").click()
    response = confirmation.value
    if response.status == 409 and response.json().get("detail", {}).get("code") == "DUPLICATE_TICKET":
        expect(page.get_by_text("Для этой проверки уже есть заявка", exact=False)).to_be_visible()
        print("C1: существующая попытка, проверена защита от дубликата. Для новой квитанции используйте чистый слот.")
    else:
        assert response.ok, response.status
        ticket = response.json()["ticket_id"]
        expect(page.get_by_text(ticket, exact=True)).to_be_visible()
        replay = page.request.post(response.url, data=response.request.post_data_json)
        assert replay.ok and replay.json()["ticket_id"] == ticket
        print("C1: получена квитанция и проверено повторное подтверждение.")
    page.get_by_role("tab", name="Чат", exact=True).click()
    page.get_by_label("Вопрос по двигателю").fill("Можно ли своим ходом доехать до базы?")
    with page.expect_response(lambda r: r.url.endswith("/agent/run")) as followup:
        page.get_by_role("button", name="Отправить", exact=True).click()
    answer = followup.value.json()
    assert followup.value.ok and answer["action"] == "search_manuals" and answer["limitations"] and not answer["claims"]
    assert diagnose("P456AT178", "S-0001")["diagnostic"]["status"] == "normal"
    failed = diagnose("P456AT178", "S-0003")
    assert failed["requires_remeasurement"] and not failed["can_create_ticket"]
    page.get_by_role("button", name="К заявке", exact=True).click()
    expect(page.get_by_role("button", name="Создать заявку")).to_be_disabled()
    print("PASS: каталог, отклонение, кнопка печати, заявка, ограничение на перегон, норма, плохая запись")


if __name__ == "__main__":
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(ignore_https_errors=True, http_credentials={
            "username": os.environ["STAND_USER"], "password": os.environ["STAND_PASSWORD"],
        }, viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        context.set_default_timeout(150000)
        exercise(context.new_page())
        browser.close()
