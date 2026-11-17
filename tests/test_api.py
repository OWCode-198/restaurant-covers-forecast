"""The HTTP surface, driven through FastAPI's TestClient.

No network: the `settings` fixture points at a temporary CSV and disables
enrichment.
"""
from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.config import Settings


def test_health_reports_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["source"] == "csv"
    assert body["csv_exists"] is True
    # Nothing ingested yet.
    assert body["covers"]["days"] == 0


def test_health_with_source_check(client):
    body = client.get("/health?check_source=true").json()
    assert body["source_check"]["reachable"] is True
    assert body["source_check"]["rows"] == 400


def test_ingest_then_forecast(client):
    ingested = client.post("/ingest", json={}).json()
    assert ingested["days_ingested"] == 400
    assert ingested["first_date"] == "2024-01-01"

    body = client.get("/forecast?days=7").json()
    assert len(body["forecast"]) == 7
    assert body["trained_days"] > 300
    assert body["uses_reservations"] is True
    assert body["accuracy"]["holdout_mape"] < 15

    day = body["forecast"][0]
    # Every forecast must come with the inputs behind it.
    for field in ("date", "day_of_week", "forecast_covers", "covers_lag_7", "is_holiday"):
        assert field in day


def test_ingest_is_idempotent(client):
    first = client.post("/ingest", json={}).json()
    second = client.post("/ingest", json={}).json()
    assert first["days_ingested"] == second["days_ingested"]

    assert client.get("/health").json()["covers"]["days"] == 400


def test_ingest_respects_a_date_range(client):
    body = client.post("/ingest", json={"start": "2024-01-01", "end": "2024-01-31"}).json()
    assert body["days_ingested"] == 31
    assert body["last_date"] == "2024-01-31"


def test_ingest_rejects_inverted_range(client):
    resp = client.post("/ingest", json={"start": "2024-03-01", "end": "2024-01-01"})
    assert resp.status_code == 422  # caught by the schema validator


def test_forecast_with_overrides(client):
    client.post("/ingest", json={})

    plain = client.get("/forecast?days=1&log=false").json()["forecast"][0]
    date = plain["date"]

    body = client.post(
        "/forecast",
        json={
            "days": 1,
            "log": False,
            "overrides": [{"date": date, "is_promo": 1, "is_local_event": 1}],
        },
    ).json()
    day = body["forecast"][0]

    # The override must reach the feature row and be reported back.
    assert day["is_promo"] == 1
    assert day["is_local_event"] == 1
    assert plain["is_promo"] == 0
    # The forecast value itself is not asserted here: the synthetic history has
    # no promo days, so the fitted promo coefficient is legitimately ~0 and the
    # number should barely move. `test_promo_coefficient_moves_forecast` covers
    # the effect on history that does vary.


def test_forecast_days_is_bounded(client):
    client.post("/ingest", json={})
    assert client.get("/forecast?days=0").status_code == 422
    assert client.get("/forecast?days=999").status_code == 422


def test_forecast_start_date_is_honoured(client):
    client.post("/ingest", json={})
    body = client.get("/forecast?days=3&start=2026-06-01&log=false").json()
    assert [d["date"] for d in body["forecast"]] == ["2026-06-01", "2026-06-02", "2026-06-03"]


def test_logged_forecasts_and_accuracy_flow(client):
    client.post("/ingest", json={})
    forecast = client.get("/forecast?days=3").json()
    first = forecast["forecast"][0]

    logged = client.get("/forecasts").json()
    assert len(logged) == 3
    assert logged[0]["actual_covers"] is None

    recorded = client.post("/actuals", json={"date": first["date"], "actual_covers": 150.0}).json()
    assert recorded["forecast_rows_updated"] == 1

    accuracy = client.get("/accuracy").json()
    assert accuracy["days_compared"] == 1
    assert accuracy["days"][0]["actual_covers"] == 150.0


def test_actuals_rejects_negative_covers(client):
    resp = client.post("/actuals", json={"date": "2026-01-01", "actual_covers": -5})
    assert resp.status_code == 422


def test_history_endpoint(client):
    client.post("/ingest", json={})
    body = client.get("/history?limit=10").json()
    assert body["days"] == 10
    # The fixture ends 400 days after 2024-01-01; the last 10 start here.
    assert body["history"][0]["date"] == "2025-01-25"


def test_forecast_without_enough_history_returns_409(tmp_path, monkeypatch):
    """A near-empty CSV is a state problem (409), not a bad request."""
    csv = tmp_path / "tiny.csv"
    pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5), "covers": [10, 20, 30, 40, 50]}
    ).to_csv(csv, index=False)

    settings = Settings(
        source="csv",
        csv_path=csv,
        db_path=tmp_path / "tiny.sqlite3",
        enrichment_enabled=False,
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        resp = c.get("/forecast?days=7")
        assert resp.status_code == 409
        assert "complete days of history" in resp.json()["detail"]


def test_toast_source_requires_a_range(tmp_path, monkeypatch):
    settings = Settings(
        source="toast",
        db_path=tmp_path / "t.sqlite3",
        toast_hostname="toast-api.example.com",
        toast_client_id="cid",
        toast_client_secret="secret",
        toast_restaurant_guid="guid",
        enrichment_enabled=False,
    )
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        resp = c.post("/ingest", json={})
        assert resp.status_code == 400
        assert "start" in resp.json()["detail"]

        health = c.get("/health").json()
        assert health["source"] == "toast"
        assert health["toast_configured"] is True


def test_unconfigured_toast_source_is_a_400(tmp_path, monkeypatch):
    settings = Settings(source="toast", db_path=tmp_path / "t.sqlite3", enrichment_enabled=False)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        assert c.get("/health").json()["toast_configured"] is False
        resp = c.post("/ingest", json={"start": "2026-01-01", "end": "2026-01-02"})
        assert resp.status_code == 400
        assert "COVERS_TOAST_HOSTNAME" in resp.json()["detail"]


def test_model_cache_is_invalidated_by_new_data(service):
    first = service.fitted_model()
    assert service.fitted_model() is first  # cached

    service.ingest()
    assert service.fitted_model() is not first  # refit after ingest


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Predictive Covers API"
    assert "/forecast" in schema["paths"]
    assert "/ingest" in schema["paths"]
