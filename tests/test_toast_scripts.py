"""The Toast preflight and backfill scripts.

All HTTP is faked, so these run with no credentials and no network. What is
worth testing here is not the client -- test_toast.py covers that -- but the
operational promises the scripts make: that preflight tells you *which* stage
failed, that it can tell real guest counts from Toast's default of one per
order, and that backfill resumes rather than refetching.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.sources.toast import ToastClient, ToastCoversSource
from app.store import Store

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str):
    """Import a file from scripts/, which is not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load_script("toast_preflight")
backfill = load_script("toast_backfill")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}
        self.text = text or (json.dumps(payload) if payload is not None else "")

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Serves login, restaurant lookup and per-business-date order pages.

    `orders_by_date` maps a yyyyMMdd string to the orders for that date; any
    date not present returns an empty list, which is how Toast reports a day
    with no orders.
    """

    def __init__(self, orders_by_date=None, restaurant=None, orders_status=200):
        self.orders_by_date = orders_by_date or {}
        self.restaurant = restaurant or {"general": {"name": "Test Kitchen"}}
        self.orders_status = orders_status
        self.order_requests = []

    def post(self, url, json=None, timeout=None):
        return FakeResponse(payload={"token": {"accessToken": "tok", "expiresIn": 3600}})

    def get(self, url, params=None, headers=None, timeout=None):
        if "/restaurants/" in url:
            return FakeResponse(payload=self.restaurant)
        if self.orders_status != 200:
            return FakeResponse(status_code=self.orders_status, text="forbidden")
        business_date = (params or {}).get("businessDate")
        self.order_requests.append(business_date)
        page = (params or {}).get("page", 1)
        # Everything fits on page one in these fixtures.
        orders = self.orders_by_date.get(business_date, []) if page == 1 else []
        return FakeResponse(payload=orders)


def make_source(session, db_path=None):
    settings = Settings(
        source="toast",
        toast_hostname="ws-api.example.com",
        toast_client_id="cid",
        toast_client_secret="secret",
        toast_restaurant_guid="guid-1",
        db_path=db_path or Path("/tmp/unused.sqlite3"),
    )
    client = ToastClient(
        hostname=settings.toast_hostname,
        client_id=settings.toast_client_id,
        client_secret=settings.toast_client_secret,
        restaurant_guid=settings.toast_restaurant_guid,
        session=session,
    )
    return ToastCoversSource(settings, client=client), settings


def orders(*guest_counts):
    return [{"numberOfGuests": n, "voided": False, "deleted": False} for n in guest_counts]


def recent_dates(n):
    """The business dates the scripts will ask for, newest first."""
    yesterday = dt.date.today() - dt.timedelta(days=1)
    return [(yesterday - dt.timedelta(days=i)).strftime("%Y%m%d") for i in range(n)]


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(preflight, "SLEEP_BETWEEN_DAYS", 0)
    monkeypatch.setattr(preflight.time, "sleep", lambda *_: None)
    monkeypatch.setattr(backfill.time, "sleep", lambda *_: None)


def patch_source(monkeypatch, module, source, settings):
    monkeypatch.setattr(module, "Settings", lambda: settings)
    monkeypatch.setattr(module, "ToastCoversSource", lambda *a, **k: source)


# --- preflight -------------------------------------------------------------


def test_preflight_passes_and_spots_real_guest_counts(monkeypatch, capsys):
    # Three guests on one order, so covers and order counts diverge.
    by_date = {d: orders(3, 2) for d in recent_dates(3)}
    source, settings = make_source(FakeSession(orders_by_date=by_date))
    patch_source(monkeypatch, preflight, source, settings)

    assert preflight.main(["--days", "3"]) == 0

    out = capsys.readouterr().out
    assert "Test Kitchen" in out
    assert "Guest counts look real" in out
    # 5 guests over 2 orders.
    assert "2.50 guests per order" in out


def test_preflight_flags_defaulted_guest_counts(monkeypatch, capsys):
    # Every order carries exactly one guest -- Toast's un-disableable default.
    by_date = {d: orders(1, 1, 1) for d in recent_dates(3)}
    source, settings = make_source(FakeSession(orders_by_date=by_date))
    patch_source(monkeypatch, preflight, source, settings)

    assert preflight.main(["--days", "3"]) == 0

    out = capsys.readouterr().out
    assert "Covers equal order counts on every sampled day" in out
    assert "forecasting transactions, not covers" in out


def test_preflight_blames_the_orders_scope_on_403(monkeypatch, capsys):
    source, settings = make_source(FakeSession(orders_status=403))
    patch_source(monkeypatch, preflight, source, settings)

    assert preflight.main(["--days", "2"]) == 1

    out = capsys.readouterr().out
    # The restaurant lookup succeeded, so the failure must be pinned on orders.
    assert "GUID resolves to" in out
    assert "orders:read" in out


def test_preflight_reports_missing_settings_without_calling_toast(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "Settings", lambda: Settings(source="toast"))

    assert preflight.main([]) == 1

    out = capsys.readouterr().out
    assert "COVERS_TOAST_CLIENT_ID" in out
    assert "Authentication" not in out


# --- backfill --------------------------------------------------------------


def test_backfill_writes_days_to_the_store(monkeypatch, tmp_path):
    db = tmp_path / "covers.sqlite3"
    by_date = {"20250101": orders(2, 2), "20250102": orders(4)}
    source, settings = make_source(FakeSession(orders_by_date=by_date), db_path=db)
    patch_source(monkeypatch, backfill, source, settings)

    code = backfill.main(["--start", "2025-01-01", "--end", "2025-01-02", "--sleep", "0"])
    assert code == 0

    stored = Store(db).load_daily_covers()
    assert len(stored) == 2
    assert stored.set_index(stored["date"].dt.date)["covers"].to_dict() == {
        dt.date(2025, 1, 1): 4.0,
        dt.date(2025, 1, 2): 4.0,
    }
    assert stored.set_index(stored["date"].dt.date)["order_count"].to_dict() == {
        dt.date(2025, 1, 1): 2,
        dt.date(2025, 1, 2): 1,
    }


def test_backfill_resumes_by_skipping_stored_dates(monkeypatch, tmp_path, capsys):
    db = tmp_path / "covers.sqlite3"
    Store(db).upsert_daily_covers(
        pd.DataFrame([{"date": pd.Timestamp("2025-01-01"), "covers": 4.0, "order_count": 2}]),
        source="toast",
    )

    session = FakeSession(orders_by_date={"20250102": orders(4)})
    source, settings = make_source(session, db_path=db)
    patch_source(monkeypatch, backfill, source, settings)

    assert backfill.main(["--start", "2025-01-01", "--end", "2025-01-02", "--sleep", "0"]) == 0

    # The already-stored day must not have cost a request.
    assert session.order_requests == ["20250102"]
    assert "1 already stored, skipping" in capsys.readouterr().out


def test_backfill_overwrite_refetches_stored_dates(monkeypatch, tmp_path):
    db = tmp_path / "covers.sqlite3"
    Store(db).upsert_daily_covers(
        pd.DataFrame([{"date": pd.Timestamp("2025-01-01"), "covers": 1.0, "order_count": 1}]),
        source="toast",
    )

    session = FakeSession(orders_by_date={"20250101": orders(5, 5)})
    source, settings = make_source(session, db_path=db)
    patch_source(monkeypatch, backfill, source, settings)

    assert (
        backfill.main(
            ["--start", "2025-01-01", "--end", "2025-01-01", "--sleep", "0", "--overwrite"]
        )
        == 0
    )

    assert session.order_requests == ["20250101"]
    assert Store(db).load_daily_covers()["covers"].tolist() == [10.0]


def test_backfill_leaves_a_failed_day_unstored_rather_than_zero(monkeypatch, tmp_path, capsys):
    """A day we could not fetch must read as unknown, never as 'closed'."""
    db = tmp_path / "covers.sqlite3"

    class FlakySession(FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):
            if "/restaurants/" in url:
                return FakeResponse(payload=self.restaurant)
            if (params or {}).get("businessDate") == "20250101":
                return FakeResponse(status_code=500, text="boom")
            return super().get(url, params=params, headers=headers, timeout=timeout)

    session = FlakySession(orders_by_date={"20250102": orders(3)})
    source, settings = make_source(session, db_path=db)
    patch_source(monkeypatch, backfill, source, settings)

    assert backfill.main(["--start", "2025-01-01", "--end", "2025-01-02", "--sleep", "0"]) == 0

    stored = Store(db).load_daily_covers()
    assert stored["date"].dt.date.tolist() == [dt.date(2025, 1, 2)]
    assert "1 date(s) failed" in capsys.readouterr().out


def test_backfill_rejects_a_backwards_range(monkeypatch, tmp_path, capsys):
    source, settings = make_source(FakeSession(), db_path=tmp_path / "c.sqlite3")
    patch_source(monkeypatch, backfill, source, settings)

    assert backfill.main(["--start", "2025-02-01", "--end", "2025-01-01"]) == 1
    assert "is before" in capsys.readouterr().out
