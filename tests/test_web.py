"""The manager-facing page.

These cover the promises the page makes to someone who will never read the
code: that it does not call a stale number "Tomorrow", that it says so loudly
when the data has fallen behind, that it frames the number against a normal
week, and that a name from a third-party feed cannot inject markup.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.routers import web


def day(date="2026-01-02", covers=45.0, **extra):
    base = {
        "date": date,
        "day_of_week": dt.date.fromisoformat(date).strftime("%A"),
        "forecast_covers": covers,
        "covers_rollavg_4wk_weekday": 45.0,
        "is_holiday": 0,
        "holiday_name": None,
        "is_local_event": 0.0,
        "event_name": None,
        "is_rain": 0,
        "is_rain_heavy": 0,
        "is_promo": 0,
    }
    base.update(extra)
    return base


def payload(days, last_history="2026-01-01", mae=9.2):
    return {
        "source": "csv",
        "last_history_date": last_history,
        "trained_days": 361,
        "accuracy": {"holdout_mae": mae, "holdout_mape": 13.1, "holdout_days": 49},
        "forecast": days,
    }


# --- the "Tomorrow" label --------------------------------------------------


def test_says_tomorrow_only_when_the_day_really_is_tomorrow():
    today = dt.date(2026, 1, 1)
    assert web.lead_label("2026-01-02", today) == "Tomorrow · "
    assert web.lead_label("2026-01-01", today) == "Today · "
    # The forecast starts after the last day of history, which may be weeks
    # old. Calling that "Tomorrow" would be a lie.
    assert web.lead_label("2025-09-14", today) == ""


def test_stale_first_day_is_not_labelled_tomorrow():
    """Regression: the page called a month-old first day 'Tomorrow'."""
    stale = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    html = web.render_forecast(payload([day(date=stale)], last_history=stale))
    assert "Tomorrow" not in html


# --- staleness -------------------------------------------------------------


def test_staleness_banner_appears_once_history_falls_behind():
    note = web.staleness_note("2026-01-01", dt.date(2026, 2, 1))
    assert "31 days behind" in note


@pytest.mark.parametrize("last_history", ["2026-01-31", "2026-01-30"])
def test_no_staleness_banner_within_the_grace_window(last_history):
    # A single missed overnight run should not cry wolf.
    assert web.staleness_note(last_history, dt.date(2026, 2, 1)) == ""


def test_current_data_renders_no_warning_and_says_tomorrow():
    today = dt.date.today()
    tomorrow = (today + dt.timedelta(days=1)).isoformat()
    html = web.render_forecast(payload([day(date=tomorrow)], last_history=today.isoformat()))
    assert "days behind" not in html
    assert "Tomorrow" in html


# --- framing the number ----------------------------------------------------


@pytest.mark.parametrize(
    "forecast,typical,expected",
    [
        (45.0, 45.0, "About the same as a normal"),
        (60.0, 45.0, "Busier than a normal"),
        (30.0, 45.0, "Quieter than a normal"),
        # Inside the tolerance band, so not worth calling out either way.
        (47.0, 45.0, "About the same as a normal"),
    ],
)
def test_comparison_wording(forecast, typical, expected):
    assert expected in web.comparison(forecast, typical, "Friday")


def test_comparison_handles_a_missing_weekday_average():
    assert "No Friday average" in web.comparison(45.0, None, "Friday")


def test_number_is_rounded_and_given_a_range_not_a_decimal():
    html = web.render_forecast(payload([day(covers=44.6)], mae=9.2))
    assert ">45</p>" in html
    assert "44.6" not in html
    assert "36–54" in html  # 45 +/- 9


def test_range_is_omitted_when_the_model_reports_no_accuracy():
    body = payload([day()])
    body["accuracy"] = {}
    assert "Usually within" not in web.render_forecast(body)


def test_empty_forecast_explains_itself_instead_of_crashing():
    html = web.render_forecast(payload([]))
    assert "No forecast available" in html


# --- drivers ---------------------------------------------------------------


def test_only_the_drivers_in_play_are_shown():
    # The chip list is omitted entirely on an unremarkable day; `chip` alone
    # would match the stylesheet, so look for the rendered element.
    assert '<ul class="chips">' not in web.render_forecast(payload([day()]))
    assert "Heavy rain expected" in web.render_forecast(payload([day(is_rain_heavy=1)]))


def test_a_trace_of_rain_is_not_worth_a_chip():
    """`is_rain` is true on four Miami days in five, so a chip driven by it is
    noise. Only rain heavy enough to change the forecast gets shown."""
    html = web.render_forecast(payload([day(is_rain=1, is_rain_heavy=0)]))
    assert '<ul class="chips">' not in html


def test_feed_supplied_names_cannot_inject_markup():
    """Holiday and event names come from third-party feeds, so they are data."""
    nasty = '<script>alert("x")</script>'
    html = web.render_forecast(
        payload([day(is_holiday=1, holiday_name=nasty, is_local_event=1.0, event_name=nasty)])
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_page_makes_no_external_requests():
    """No CDN, no fonts, no analytics -- it must render on bad wifi."""
    html = web.render_forecast(payload([day()]))
    for marker in ("http://", "https://", "//cdn", "<script"):
        assert marker not in html


# --- the route -------------------------------------------------------------


def test_today_route_renders_a_page(client):
    resp = client.get("/today")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "covers expected" in resp.text


def test_root_index_points_at_the_page(client):
    assert client.get("/").json()["today"] == "/today"


# --- access control --------------------------------------------------------


def test_page_is_open_when_no_password_is_set(client):
    """Local development must stay a single command with no credentials."""
    assert client.get("/today").status_code == 200


def test_page_demands_credentials_once_a_password_is_set(settings, monkeypatch):
    """Hosting this without a password would publish a real business's trade."""
    from fastapi.testclient import TestClient

    secured = settings.model_copy(update={"web_password": "s3cret"})
    monkeypatch.setattr("app.main.get_settings", lambda: secured)
    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        assert c.get("/today").status_code == 401
        assert c.get("/today", auth=("manager", "wrong")).status_code == 401
        assert c.get("/today", auth=("wrong", "s3cret")).status_code == 401
        assert c.get("/today", auth=("manager", "s3cret")).status_code == 200


def test_health_stays_reachable_without_credentials(settings, monkeypatch):
    """Uptime checks must not need the page password."""
    from fastapi.testclient import TestClient

    secured = settings.model_copy(update={"web_password": "s3cret"})
    monkeypatch.setattr("app.main.get_settings", lambda: secured)
    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        assert c.get("/health").status_code == 200


def test_staleness_threshold_matches_the_refresh_cadence():
    """A weekly export must not trip a warning designed for a daily ingest."""
    today = dt.date(2026, 2, 10)
    five_days_old = "2026-02-05"

    # Daily cadence: five days behind is genuinely wrong.
    assert web.staleness_note(five_days_old, today, threshold=2) != ""
    # Weekly cadence: five days behind is simply Friday.
    assert web.staleness_note(five_days_old, today, threshold=8) == ""
    # Even a generous threshold still catches data that is really stale.
    assert "30 days behind" in web.staleness_note("2026-01-11", today, threshold=8)
