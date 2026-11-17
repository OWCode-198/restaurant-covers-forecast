"""Toast client and covers aggregation.

All HTTP is faked, so these run with no credentials and no network.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from app.config import Settings
from app.sources.base import SourceError, SourceNotConfigured
from app.sources.toast import (
    MAX_PAGE_SIZE,
    ToastAuthError,
    ToastClient,
    ToastCoversSource,
    guests_in_order,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text or (json.dumps(payload) if payload is not None else "")

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Records calls and replays queued responses."""

    def __init__(self, login=None, get_responses=None):
        self.login = login or FakeResponse(
            payload={"token": {"accessToken": "tok-1", "expiresIn": 3600}}
        )
        self.get_responses = list(get_responses or [])
        self.post_calls = []
        self.get_calls = []

    def post(self, url, json=None, timeout=None):
        self.post_calls.append({"url": url, "json": json})
        return self.login

    def get(self, url, params=None, headers=None, timeout=None):
        self.get_calls.append({"url": url, "params": params, "headers": headers})
        if not self.get_responses:
            return FakeResponse(payload=[])
        return self.get_responses.pop(0)


def make_client(session) -> ToastClient:
    return ToastClient(
        hostname="toast-api.example.com",
        client_id="cid",
        client_secret="secret",
        restaurant_guid="guid-123",
        session=session,
    )


# --- guest counting --------------------------------------------------------


@pytest.mark.parametrize(
    "order, expected",
    [
        ({"numberOfGuests": 4}, 4.0),
        ({"numberOfGuests": 0, "calculatedGuestCount": 2.5}, 2.5),
        ({"numberOfGuests": None, "calculatedGuestCount": 3}, 3.0),
        ({"numberOfGuests": 6, "voided": True}, 0.0),
        ({"numberOfGuests": 6, "deleted": True}, 0.0),
        ({}, 0.0),
        # A boolean must never be read as a guest count of 1.
        ({"numberOfGuests": True}, 0.0),
    ],
)
def test_guests_in_order(order, expected):
    assert guests_in_order(order) == expected


# --- auth ------------------------------------------------------------------


def test_authenticate_caches_token():
    session = FakeSession()
    client = make_client(session)

    assert client.authenticate() == "tok-1"
    assert client.authenticate() == "tok-1"
    # The second call must reuse the cached token rather than logging in again.
    assert len(session.post_calls) == 1

    body = session.post_calls[0]["json"]
    assert body["clientId"] == "cid"
    assert body["userAccessType"] == "TOAST_MACHINE_CLIENT"


def test_authenticate_accepts_flat_token_shape():
    session = FakeSession(login=FakeResponse(payload={"accessToken": "flat", "expiresIn": 100}))
    assert make_client(session).authenticate() == "flat"


def test_authenticate_rejects_bad_credentials():
    session = FakeSession(login=FakeResponse(status_code=401, text="nope"))
    with pytest.raises(ToastAuthError, match="rejected the credentials"):
        make_client(session).authenticate()


def test_hostname_is_normalized():
    assert make_client(FakeSession()).base_url == "https://toast-api.example.com"

    client = ToastClient(
        hostname="https://host.example.com/",
        client_id="a",
        client_secret="b",
        restaurant_guid="c",
        session=FakeSession(),
    )
    assert client.base_url == "https://host.example.com"


# --- orders / pagination ---------------------------------------------------


def test_sends_required_headers():
    session = FakeSession(get_responses=[FakeResponse(payload=[])])
    make_client(session).fetch_orders_for_business_date(dt.date(2026, 2, 14))

    headers = session.get_calls[0]["headers"]
    assert headers["Authorization"] == "Bearer tok-1"
    assert headers["Toast-Restaurant-External-ID"] == "guid-123"
    assert session.get_calls[0]["params"]["businessDate"] == "20260214"


def test_pagination_follows_until_short_page():
    full_page = [{"numberOfGuests": 2}] * MAX_PAGE_SIZE
    session = FakeSession(
        get_responses=[
            FakeResponse(payload=full_page),
            FakeResponse(payload=[{"numberOfGuests": 2}] * 3),
        ]
    )
    orders = make_client(session).fetch_orders_for_business_date(dt.date(2026, 2, 14))

    assert len(orders) == MAX_PAGE_SIZE + 3
    assert [c["params"]["page"] for c in session.get_calls] == [1, 2]


def test_404_means_no_orders_not_an_error():
    session = FakeSession(get_responses=[FakeResponse(status_code=404)])
    assert make_client(session).fetch_orders_for_business_date(dt.date(2026, 2, 14)) == []


def test_non_array_response_is_an_error():
    session = FakeSession(get_responses=[FakeResponse(payload={"unexpected": "shape"})])
    with pytest.raises(SourceError, match="JSON array"):
        make_client(session).fetch_orders_for_business_date(dt.date(2026, 2, 14))


# --- daily aggregation -----------------------------------------------------


def _toast_settings(tmp_path) -> Settings:
    return Settings(
        source="toast",
        db_path=tmp_path / "t.sqlite3",
        toast_hostname="toast-api.example.com",
        toast_client_id="cid",
        toast_client_secret="secret",
        toast_restaurant_guid="guid-123",
        enrichment_enabled=False,
    )


def test_daily_covers_sums_guests_per_business_date(tmp_path):
    session = FakeSession(
        get_responses=[
            # 2026-03-01: 4 + 2 guests, plus a voided order that must not count.
            FakeResponse(
                payload=[
                    {"numberOfGuests": 4},
                    {"numberOfGuests": 2},
                    {"numberOfGuests": 9, "voided": True},
                ]
            ),
            # 2026-03-02: no orders at all -> a closed day, recorded as 0.
            FakeResponse(payload=[]),
        ]
    )
    source = ToastCoversSource(_toast_settings(tmp_path), client=make_client(session))
    df = source.fetch_daily_covers(dt.date(2026, 3, 1), dt.date(2026, 3, 2))

    assert list(df["covers"]) == [6.0, 0.0]
    assert list(df["order_count"]) == [2, 0]


def test_a_failing_date_is_skipped_not_zeroed(tmp_path):
    session = FakeSession(
        get_responses=[
            FakeResponse(payload=[{"numberOfGuests": 5}]),
            FakeResponse(status_code=400, text="bad"),
            FakeResponse(payload=[{"numberOfGuests": 3}]),
        ]
    )
    source = ToastCoversSource(_toast_settings(tmp_path), client=make_client(session))
    df = source.fetch_daily_covers(dt.date(2026, 3, 1), dt.date(2026, 3, 3))

    # The middle date is absent entirely -- "unknown", never a fabricated 0.
    assert [d.date().isoformat() for d in df["date"]] == ["2026-03-01", "2026-03-03"]


def test_requires_an_explicit_date_range(tmp_path):
    source = ToastCoversSource(_toast_settings(tmp_path), client=make_client(FakeSession()))
    with pytest.raises(SourceError, match="explicit start and end"):
        source.fetch_daily_covers()


def test_rejects_inverted_range(tmp_path):
    source = ToastCoversSource(_toast_settings(tmp_path), client=make_client(FakeSession()))
    with pytest.raises(SourceError, match="is before start"):
        source.fetch_daily_covers(dt.date(2026, 3, 5), dt.date(2026, 3, 1))


def test_missing_credentials_is_a_clear_error(tmp_path):
    settings = Settings(source="toast", db_path=tmp_path / "t.sqlite3", toast_hostname=None)
    with pytest.raises(SourceNotConfigured, match="COVERS_TOAST_HOSTNAME"):
        ToastCoversSource(settings)
