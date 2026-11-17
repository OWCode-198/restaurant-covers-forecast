"""Shared fixtures.

Everything here is synthetic and local: no network calls, no Toast
credentials, and nothing touches the project's real CSVs or database.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.config import Location, Settings
from app.service import CoversService
from app.store import Store


@pytest.fixture(autouse=True)
def isolate_local_config(monkeypatch):
    """Keep tests away from this machine's real configuration.

    `Settings` reads `.env`, and `get_location` falls back to
    `.covers_forecast_config.json`. Without this, a developer's own coordinates
    and API keys leak into every test, so results differ between machines and
    depend on whether `.env` happens to exist.
    """
    import app.config as config_module

    monkeypatch.setitem(config_module.Settings.model_config, "env_file", None)
    monkeypatch.setattr(config_module, "_load_cli_config", lambda: {})
    # get_settings() is lru_cached, so a value built before this fixture ran
    # would otherwise survive into the test.
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


@pytest.fixture
def location() -> Location:
    return Location(
        latitude=25.77,
        longitude=-80.19,
        timezone="America/New_York",
        country_code="US",
        ticketmaster_api_key=None,
        name="Test City",
    )


def make_covers_frame(days: int = 400, seed: int = 7) -> pd.DataFrame:
    """A plausible covers history with a weekday effect.

    Deterministic, so a test asserting on a forecast is not flaky.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(dt.date(2024, 1, 1), periods=days, freq="D")

    # Busier on Fri/Sat/Sun, with mild noise and a slow upward trend.
    dow_effect = {0: 90, 1: 100, 2: 115, 3: 130, 4: 180, 5: 205, 6: 150}
    covers = [dow_effect[d.dayofweek] + i * 0.05 + rng.normal(0, 6) for i, d in enumerate(dates)]

    return pd.DataFrame(
        {
            "date": dates,
            "covers": np.round(covers, 0),
            "reservations_covers": np.round(np.array(covers) * 0.55, 0),
            "temp_high_f": 78 + 8 * np.sin(np.arange(days) / 58.0),
            "precip_in": rng.choice([0.0, 0.0, 0.0, 0.2], size=days),
        }
    )


@pytest.fixture
def covers_csv(tmp_path):
    """A synthetic covers CSV on disk."""
    path = tmp_path / "covers.csv"
    make_covers_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def settings(tmp_path, covers_csv) -> Settings:
    """Settings pointed entirely at temporary files, enrichment off.

    Enrichment is disabled so no test reaches out to Open-Meteo / Nager.Date;
    the synthetic CSV already carries weather.
    """
    return Settings(
        source="csv",
        csv_path=covers_csv,
        db_path=tmp_path / "test.sqlite3",
        enrichment_enabled=False,
        latitude=25.77,
        longitude=-80.19,
        timezone="America/New_York",
        country_code="US",
    )


@pytest.fixture
def store(settings) -> Store:
    return Store(settings.db_path)


@pytest.fixture
def service(settings, store) -> CoversService:
    return CoversService(settings, store=store, location=None)


@pytest.fixture
def client(settings, monkeypatch):
    """A TestClient whose app uses the temporary settings."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as c:
        yield c
