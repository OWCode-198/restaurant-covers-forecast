"""Store round-tripping, CSV column sniffing, and the forecasting model."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.core import model as model_core
from app.sources.base import SourceError
from app.sources.csv_source import CsvCoversSource, detect_column
from tests.conftest import make_covers_frame


# --- CSV source ------------------------------------------------------------


def test_detect_column_ignores_case_and_separators():
    cols = ["Date", "Total Guests", "Net sales"]
    assert detect_column(cols, ("date",)) == "Date"
    assert detect_column(cols, ("covers", "total_guests")) == "Total Guests"
    assert detect_column(cols, ("nothing",)) is None


def test_reads_toast_style_export(tmp_path):
    """The raw Toast sales export uses 'Total guests', not 'covers'."""
    path = tmp_path / "export.csv"
    pd.DataFrame(
        {
            "date": ["2025-08-25", "2025-08-26"],
            "Net sales": [202.0, 146.6],
            "Total orders": [22, 17],
            "Total guests": [22, 17],
        }
    ).to_csv(path, index=False)

    df = CsvCoversSource(path).fetch_daily_covers()
    assert list(df["covers"]) == [22.0, 17.0]


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        CsvCoversSource(tmp_path / "nope.csv").fetch_daily_covers()


def test_csv_without_a_covers_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2025-01-01"], "revenue": [100]}).to_csv(path, index=False)
    with pytest.raises(SourceError, match="needs a date column and a covers column"):
        CsvCoversSource(path).fetch_daily_covers()


def test_date_range_filter(covers_csv):
    df = CsvCoversSource(covers_csv).fetch_daily_covers(
        start=dt.date(2024, 2, 1), end=dt.date(2024, 2, 10)
    )
    assert len(df) == 10
    assert df["date"].min().date() == dt.date(2024, 2, 1)


# --- store -----------------------------------------------------------------


def test_upsert_round_trips_feature_columns(store):
    store.upsert_daily_covers(make_covers_frame(days=5), source="csv")

    out = store.load_daily_covers()
    assert len(out) == 5
    # Weather and reservations from the source must survive the round trip,
    # otherwise the model silently loses features present in the input.
    assert "temp_high_f" in out.columns
    assert "reservations_covers" in out.columns
    assert out["temp_high_f"].notna().all()


def test_upsert_updates_rather_than_duplicating(store):
    frame = make_covers_frame(days=3)
    store.upsert_daily_covers(frame, source="csv")

    revised = frame.copy()
    revised.loc[0, "covers"] = 999.0
    store.upsert_daily_covers(revised, source="toast")

    out = store.load_daily_covers()
    assert len(out) == 3
    assert out.iloc[0]["covers"] == 999.0
    assert out.iloc[0]["source"] == "toast"


def test_a_source_without_weather_does_not_blank_existing_weather(store):
    """A Toast ingest knows covers but not temperature. It must not erase the
    weather an earlier CSV ingest recorded."""
    store.upsert_daily_covers(make_covers_frame(days=3), source="csv")

    bare = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "covers": [10.0, 20.0, 30.0]}
    )
    store.upsert_daily_covers(bare, source="toast")

    out = store.load_daily_covers()
    assert list(out["covers"]) == [10.0, 20.0, 30.0]
    assert out["temp_high_f"].notna().all()


def test_empty_store_returns_empty_frame(store):
    assert store.load_daily_covers().empty


def test_coverage_counts_only_known_days(store):
    store.upsert_daily_covers(make_covers_frame(days=4), source="csv")
    coverage = store.covers_coverage()
    assert coverage["days"] == 4
    assert coverage["first_date"] == "2024-01-01"


def test_forecast_log_and_actuals(store):
    store.log_forecasts(
        [{"date": "2026-01-01", "forecast_covers": 120.0, "is_holiday": 1}], source="csv"
    )
    logged = store.list_forecasts()
    assert len(logged) == 1
    assert logged[0]["features"]["is_holiday"] == 1
    assert logged[0]["actual_covers"] is None

    assert store.record_actual("2026-01-01", 131.0) == 1

    report = store.accuracy_report()
    assert report["days_compared"] == 1
    assert report["mae"] == 11.0


def test_latest_forecast_per_date_wins(store):
    store.log_forecasts([{"date": "2026-01-01", "forecast_covers": 100.0}], source="csv")
    store.log_forecasts([{"date": "2026-01-01", "forecast_covers": 140.0}], source="csv")

    latest = store.list_forecasts(latest_per_date=True)
    assert len(latest) == 1
    assert latest[0]["forecast_covers"] == 140.0

    assert len(store.list_forecasts(latest_per_date=False)) == 2


# --- model -----------------------------------------------------------------


def test_prepare_history_fills_calendar_gaps_and_flags():
    frame = pd.DataFrame(
        {
            # 2024-01-03 is missing entirely; 2024-01-04 is a recorded zero.
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            "covers": [100.0, 110.0, 0.0],
        }
    )
    out = model_core.prepare_history(frame)

    assert len(out) == 4  # the gap is filled in
    missing = out[out["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    closed = out[out["date"] == pd.Timestamp("2024-01-04")].iloc[0]

    # A gap is "unknown"; a recorded zero is "closed". They differ.
    assert missing["covers_missing"] == 1 and missing["is_closed"] == 0
    assert closed["covers_missing"] == 0 and closed["is_closed"] == 1
    assert out["day_of_week"].iloc[0] == "Monday"


def test_prepare_history_always_has_temp_column():
    frame = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "covers": [10.0, 20.0, 30.0]}
    )
    out = model_core.prepare_history(frame)
    # `fit` selects temp_high_f by name, so it must exist even with no weather.
    assert "temp_high_f" in out.columns
    assert out["temp_high_f"].notna().all()


def test_fit_and_forecast_are_sane():
    df = model_core.prepare_history(make_covers_frame(days=400))
    model = model_core.fit(df)

    assert model.trained_days > 300
    assert model.has_reservations is True
    assert model.accuracy is not None
    # The synthetic series is nearly deterministic, so error should be small.
    assert model.accuracy.mape < 15

    results = model_core.forecast_days(model, location=None, days=7, enrichment_enabled=False)
    assert len(results) == 7
    expected_first = (model.last_history_date + pd.Timedelta(days=1)).date().isoformat()
    assert results[0]["date"] == expected_first

    for row in results:
        # Weekday covers run 90-210 in the fixture; a forecast far outside that
        # band would mean the features were wired up wrong.
        assert 40 < row["forecast_covers"] < 400
        assert row["weather_estimated"] is True  # no location supplied


def test_weekend_forecast_exceeds_midweek():
    df = model_core.prepare_history(make_covers_frame(days=400))
    model = model_core.fit(df)
    results = model_core.forecast_days(model, location=None, days=14, enrichment_enabled=False)

    by_dow = {r["day_of_week"]: r["forecast_covers"] for r in results}
    # The fixture makes Saturday the busiest day and Monday the quietest.
    assert by_dow["Saturday"] > by_dow["Monday"]


def test_fit_rejects_too_little_history():
    df = model_core.prepare_history(make_covers_frame(days=20))
    with pytest.raises(model_core.NotEnoughHistory) as exc:
        model_core.fit(df, min_training_days=30)
    assert exc.value.required == 30


def test_predict_one_tolerates_missing_features():
    df = model_core.prepare_history(make_covers_frame(days=200))
    model = model_core.fit(df)

    row = {"day_of_week": "Friday"}  # every numeric feature absent
    # Should fall back to 0 for each rather than raising.
    assert isinstance(model_core.predict_one(row, model), float)


def test_forecast_never_returns_negative_covers():
    frame = make_covers_frame(days=200)
    frame["covers"] = frame["covers"] * 0.01  # push the fit toward zero
    model = model_core.fit(model_core.prepare_history(frame))
    results = model_core.forecast_days(model, location=None, days=7, enrichment_enabled=False)
    assert all(r["forecast_covers"] >= 0 for r in results)


def test_overrides_beat_fetched_values():
    df = model_core.prepare_history(make_covers_frame(days=300))
    model = model_core.fit(df)

    base = model_core.forecast_days(model, None, days=1, enrichment_enabled=False)[0]
    promo = model_core.forecast_days(
        model,
        None,
        days=1,
        overrides_by_date={base["date"]: {"is_promo": 1, "temp_high_f": 95.0}},
        enrichment_enabled=False,
    )[0]

    assert promo["is_promo"] == 1
    assert promo["temp_high_f"] == 95.0
    assert base["is_promo"] == 0


def test_promo_coefficient_moves_forecast():
    """When history actually contains promo days that ran hot, flagging a future
    day as a promo must raise its forecast."""
    frame = make_covers_frame(days=400)
    frame["is_promo"] = 0
    # Every 10th day was a promo, and promos reliably added ~40 covers.
    promo_rows = frame.index % 10 == 0
    frame.loc[promo_rows, "is_promo"] = 1
    frame.loc[promo_rows, "covers"] = frame.loc[promo_rows, "covers"] + 40

    model = model_core.fit(model_core.prepare_history(frame))

    base = model_core.forecast_days(model, None, days=1, enrichment_enabled=False)[0]
    promo = model_core.forecast_days(
        model,
        None,
        days=1,
        overrides_by_date={base["date"]: {"is_promo": 1}},
        enrichment_enabled=False,
    )[0]

    assert promo["forecast_covers"] > base["forecast_covers"] + 20


def test_reservations_estimate_is_on_the_reservations_scale():
    """Regression: the forecast must not feed a covers-scale number into the
    reservations feature. The fixture books 55% of covers, so the estimate for a
    future day should sit near 0.55x that weekday's covers average, not at 1.0x.
    """
    model = model_core.fit(model_core.prepare_history(make_covers_frame(days=400)))
    assert model.has_reservations
    assert 0.5 < model.reservations_ratio < 0.6

    row = model_core.forecast_days(model, None, days=1, enrichment_enabled=False)[0]
    ratio = row["reservations_covers"] / row["covers_rollavg_4wk_weekday"]
    assert 0.5 < ratio < 0.6


def test_graded_event_pressure_moves_the_forecast(monkeypatch):
    """The point of the whole feature: a big event next door must raise the
    forecast more than a small one at the edge of the radius.

    The event lookup is stubbed, so no network and no Ticketmaster key.
    """
    from app.config import Location
    from app.core import enrichment, events

    # History where event days ran ~35 covers hotter, so the coefficient is real.
    frame = make_covers_frame(days=400)
    frame["is_local_event"] = 0
    event_rows = frame.index % 12 == 0
    frame.loc[event_rows, "is_local_event"] = 1
    frame.loc[event_rows, "covers"] = frame.loc[event_rows, "covers"] + 35

    model = model_core.fit(model_core.prepare_history(frame))

    location = Location(
        latitude=25.7617,
        longitude=-80.1918,
        timezone="America/New_York",
        country_code="US",
        ticketmaster_api_key="stub-key",
        name="Test",
        event_radius_miles=15.0,
    )

    monkeypatch.setattr(enrichment, "holiday_for_date", lambda d, loc: "")
    monkeypatch.setattr(enrichment, "weather_for_date", lambda d, loc: None)

    def set_event(raw_events):
        def _fake(day, loc, radius_miles=15.0, pressure_cap=events.PRESSURE_CAP, settings=None):
            return events.score_events(
                [events.CandidateEvent.from_ticketmaster(e) for e in raw_events],
                loc.latitude,
                loc.longitude,
                radius_miles=radius_miles,
                pressure_cap=pressure_cap,
            )

        monkeypatch.setattr(enrichment, "event_pressure_for_date", _fake)

    def one_event(distance, segment, genre=None):
        classification = {"segment": {"name": segment}}
        if genre:
            classification["genre"] = {"name": genre}
        return [
            {
                "id": "e1",
                "name": "Stub Event",
                "dates": {"start": {"localDate": "2099-01-01"}},
                "classifications": [classification],
                "_embedded": {
                    "venues": [{"name": "Stub Venue", "distance": distance, "units": "miles"}]
                },
            }
        ]

    set_event([])
    baseline = model_core.forecast_days(model, location, days=1)[0]

    set_event(one_event(14.0, "Film"))
    far_small = model_core.forecast_days(model, location, days=1)[0]

    set_event(one_event(0.5, "Sports", "Football"))
    near_big = model_core.forecast_days(model, location, days=1)[0]

    assert baseline["event_pressure"] == 0.0
    assert far_small["event_pressure"] < near_big["event_pressure"]

    # That ordering has to survive into the actual covers number.
    assert baseline["forecast_covers"] < near_big["forecast_covers"]
    assert far_small["forecast_covers"] < near_big["forecast_covers"]

    # The graded value is what reaches the model, not a flattened flag.
    assert near_big["is_local_event"] > 1.0
    assert 0.0 <= far_small["is_local_event"] < 1.0


def test_prepare_history_keeps_event_pressure_continuous():
    """Regression: casting is_local_event to int would truncate 1.7 to 1 and
    0.6 to 0, discarding the distance/size grading entirely."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "covers": [100.0, 110.0, 120.0],
            "is_local_event": [0.0, 0.6, 1.7],
        }
    )
    out = model_core.prepare_history(frame)
    assert list(out["is_local_event"]) == [0.0, 0.6, 1.7]


def test_has_reservations_is_a_python_bool():
    """numpy.bool_ leaks into the JSON response and trips Pydantic."""
    model = model_core.fit(model_core.prepare_history(make_covers_frame(days=200)))
    assert type(model.has_reservations) is bool


def test_nan_features_serialize_as_null():
    assert model_core._as_optional_float(np.nan) is None
    assert model_core._as_optional_float(None) is None
    assert model_core._as_optional_float(3.14159) == 3.14


def test_enrichment_runs_before_prepare_history(tmp_path, monkeypatch, location):
    """Regression: enrichment was a no-op for any source without its own weather.

    `prepare_history` backfills temp_high_f with a neutral constant and
    is_holiday with 0 so that `fit` always has the columns it selects by name.
    `enrich_history` decides what to fetch by looking for NaNs in exactly those
    columns, so running it *after* prepare meant it always saw a full frame and
    fetched nothing -- weather and holidays were silently dead.
    """
    from app.config import Settings
    from app.core import enrichment
    from app.service import CoversService

    csv = tmp_path / "bare.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=120),
            "covers": [40 + (i % 7) * 5 for i in range(120)],
        }
    ).to_csv(csv, index=False)

    def fake_weather(lat, lon, start, end, tz="auto"):
        days = pd.date_range(start, end)
        # Deliberately varying, so a neutral constant cannot pass this test.
        return pd.DataFrame(
            {
                "date": days,
                "temp_high_f": [60.0 + (i % 30) for i in range(len(days))],
                "precip_in": [0.4 if i % 3 == 0 else 0.0 for i in range(len(days))],
            }
        )

    monkeypatch.setattr(enrichment, "fetch_weather_historical", fake_weather)
    monkeypatch.setattr(
        enrichment,
        "fetch_holidays",
        lambda country, years: pd.DataFrame(
            [{"date": "2025-01-01", "holiday_name": "New Year's Day"}]
        ),
    )

    settings = Settings(
        source="csv",
        csv_path=csv,
        db_path=tmp_path / "e.sqlite3",
        enrichment_enabled=True,
    )
    history = CoversService(settings, location=location).history()

    assert history["temp_high_f"].nunique() > 1, "weather never reached the frame"
    assert history["precip_in"].notna().any()
    assert history["is_rain"].sum() > 0
    assert history["is_holiday"].sum() == 1


def test_store_refuses_to_mix_two_restaurants(tmp_path):
    """Regression: the real store ended up holding 805 days from two venues.

    `daily_covers` is keyed by date alone, so re-pointing the service at a
    different export merged into the old history instead of replacing it. The
    model then trained across the seam -- 146 covers/day on one side, 61 on
    the other -- and every number computed from it was meaningless.
    """
    from app.config import Settings
    from app.service import CoversService, ServiceError
    from app.store import Store

    def csv_for(name, covers):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(
            {"date": pd.date_range("2025-01-01", periods=60), "covers": covers}
        ).to_csv(path, index=False)
        return path

    db = tmp_path / "shared.sqlite3"
    first = csv_for("venue_a", [50] * 60)
    second = csv_for("venue_b", [150] * 60)

    def service(path):
        return CoversService(
            Settings(source="csv", csv_path=path, db_path=db, enrichment_enabled=False)
        )

    service(first).ingest()

    with pytest.raises(ServiceError, match="two restaurants"):
        service(second).ingest()

    # The first venue's history must be untouched by the refused ingest.
    stored = Store(db).load_daily_covers()
    assert len(stored) == 60
    assert stored["covers"].max() == 50


def test_reingesting_the_same_source_is_fine(tmp_path):
    """The guard must not block the normal case of a refreshed export."""
    from app.config import Settings
    from app.service import CoversService
    from app.store import Store

    path = tmp_path / "venue.csv"
    pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=60), "covers": [50] * 60}
    ).to_csv(path, index=False)
    db = tmp_path / "s.sqlite3"
    settings = Settings(source="csv", csv_path=path, db_path=db, enrichment_enabled=False)

    CoversService(settings).ingest()
    CoversService(settings).ingest()  # must not raise
    assert len(Store(db).load_daily_covers()) == 60


# --- the features the model actually uses ----------------------------------


def test_rain_features_split_a_drizzle_from_a_downpour():
    """`is_rain` fired on 80% of this restaurant's days, so the model reads a
    threshold and the amount instead. Both are derived in `prepare_history`."""
    out = model_core.prepare_history(
        pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=6, freq="D"),
                "covers": [50, 52, 48, 55, 51, 49],
                "precip_in": [0.0, 0.01, 0.2, 0.26, 1.0, 9.0],
            }
        )
    )
    assert out["is_rain"].tolist() == [0, 1, 1, 1, 1, 1]
    assert out["is_rain_heavy"].tolist() == [0, 0, 0, 1, 1, 1]
    # Capped: a 9-inch day must not read as 9x as discouraging as a 1-inch day.
    assert out["precip_in_capped"].tolist() == [
        0.0,
        0.01,
        0.2,
        0.26,
        1.0,
        model_core.PRECIP_CAP_IN,
    ]


def test_a_supplied_precip_moves_the_rain_features_with_it():
    """A caller overriding precip_in -- the backtest does exactly this, with the
    weather that actually happened -- must not leave the rain features behind on
    the fetched forecast's values."""
    df = model_core.prepare_history(make_covers_frame(days=200))
    model = model_core.fit(df)
    target = model.last_history_date + pd.Timedelta(days=1)

    wet = model_core.build_feature_row(
        target, model, location=None, overrides={"precip_in": 0.9},
        enrichment_enabled=False,
    )
    assert wet["is_rain_heavy"] == 1
    assert wet["precip_in_capped"] == 0.9

    dry = model_core.build_feature_row(
        target, model, location=None, overrides={"precip_in": 0.0},
        enrichment_enabled=False,
    )
    assert dry["is_rain_heavy"] == 0
    assert dry["precip_in_capped"] == 0.0


def test_the_model_reads_the_28_day_level_and_not_the_dropped_features():
    """The slower level sits alongside the 7-day trend so the fit can tell a
    busy week from a busier restaurant. lag_14 and temperature were measured
    out of the model; keep them out without a backtest that says otherwise."""
    df = model_core.prepare_history(make_covers_frame(days=200))
    model = model_core.fit(df)
    assert "covers_rollavg_28d_level" in model.feature_cols
    assert "covers_lag_14" not in model.feature_cols
    assert "temp_high_f" not in model.feature_cols

    target = model.last_history_date + pd.Timedelta(days=1)
    row = model_core.build_feature_row(
        target, model, location=None, enrichment_enabled=False
    )
    assert row["covers_rollavg_28d_level"] > 0
