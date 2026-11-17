"""The covers forecasting model.

An interpretable least-squares regression. It began as a direct port of the
notebooks and of `put in terminal/covers_forecast_cli.py` (steps 5 and 6), with
the prints, prompts and `sys.exit` calls replaced by return values and
exceptions so it can be driven by an HTTP request.

This module is now the source of truth for the feature set, and the CLI and
notebooks have NOT been brought along: on 2026-10-19 the features here were
re-chosen against the walk-forward backtest (see `NUMERIC_FEATURES` and
`scripts/evaluate.py`), which dropped `covers_lag_14` and `temp_high_f`, split
the rain flag at a threshold that means something, and added the 28-day level.
The CLI still fits the older set, so the two will not agree on a number. Change
features here, measure with `scripts/evaluate.py`, and treat its MAE as the
figure to quote.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import Location
from app.core import enrichment, events

logger = logging.getLogger(__name__)

DOW_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
WEEKEND_DAYS = {"Friday", "Saturday", "Sunday"}

# Every feature here earns its place in the walk-forward backtest
# (scripts/evaluate.py). Measured on 65 origins x 7 days, excluding the known
# outlier days, each of these was tried and dropped:
#
#   covers_lag_14   fitted a negative coefficient on noise; removing it is free.
#   temp_high_f     Miami temperature barely moves within a season (coef -0.02).
#   is_rain         true on 80% of days, because any trace of precipitation
#                   counted. Replaced by `is_rain_heavy` + the amount, which is
#                   the single biggest accuracy win available here
#                   (clean MAE 9.17 -> 8.83).
#   8-week weekday average, annual sin/cos seasonality, a log target, recency
#                   weighting, ridge, Huber/L1 losses and per-weekday models:
#                   all neutral or worse. Do not re-add without a backtest.
#
# `is_local_event` and `is_promo` are constant 0 across the current history, so
# their coefficients fit to exactly 0 and they cannot move a forecast. They stay
# in the list as the plumbing for the day that history exists -- see the note on
# event backfill in scripts/evaluate.py.
NUMERIC_FEATURES = [
    "covers_lag_7",
    "covers_rollavg_4wk_weekday",
    "covers_rollavg_7d_trend",
    "covers_rollavg_28d_level",
    "is_holiday",
    "is_local_event",
    "is_promo",
    "is_rain_heavy",
    "precip_in_capped",
]

# Rain only moves covers once it is enough to change whether someone walks over.
# At `> 0` the flag fired on 80% of Miami days and carried no signal.
RAIN_HEAVY_THRESHOLD_IN = 0.25
# A hurricane's rainfall is not 40x as discouraging as a thunderstorm's; past
# this the extra inches say nothing new, and uncapped they dominate the fit.
PRECIP_CAP_IN = 1.5

# Reservations are only used as a feature when populated for most of the
# history -- a mostly-empty column would hurt more than it helps.
RESERVATIONS_COVERAGE_THRESHOLD = 0.8


class NotEnoughHistory(Exception):
    """Raised when there are too few complete days to fit a usable model."""

    def __init__(self, available: int, required: int) -> None:
        self.available = available
        self.required = required
        super().__init__(
            f"Only {available} complete days of history; at least {required} are needed "
            "to fit a reliable model (a few months is better)."
        )


@dataclass
class Accuracy:
    """Hold-out self-check on the most recent weeks of history."""

    mae: float
    mape: float
    tested_days: int


@dataclass
class FittedModel:
    coefs: np.ndarray
    feature_cols: List[str]
    has_reservations: bool
    # Mean reservations / covers across history. Used to estimate reservations
    # for a future date from that weekday's covers average; without it the
    # forecast would be fed a covers-scale number in a reservations-scale slot.
    reservations_ratio: float
    trained_days: int
    hist: pd.DataFrame
    last_history_date: pd.Timestamp
    accuracy: Optional[Accuracy] = None
    get_lag: Optional[Callable[[pd.Timestamp, int], float]] = field(repr=False, default=None)
    get_rollavg_weekday: Optional[Callable[[pd.Timestamp, str], float]] = field(
        repr=False, default=None
    )
    get_rollavg_7d_trend: Optional[Callable[[pd.Timestamp], float]] = field(
        repr=False, default=None
    )
    get_rollavg_28d_level: Optional[Callable[[pd.Timestamp], float]] = field(
        repr=False, default=None
    )


def make_feature_functions(covers_by_date: pd.Series, hist_df: pd.DataFrame):
    """Lag and rolling-average lookups. Identical to the CLI / notebook 06."""

    def get_lag(target_date: pd.Timestamp, days_back: int) -> float:
        return covers_by_date.get(target_date - pd.Timedelta(days=days_back), np.nan)

    def get_rollavg_weekday(target_date: pd.Timestamp, dow: str, n: int = 4) -> float:
        past = (
            hist_df[(hist_df["date"] < target_date) & (hist_df["day_of_week"] == dow)]
            .sort_values("date")
            .tail(n)
        )
        return past["covers"].mean() if len(past) else np.nan

    def _trailing_mean(
        target_date: pd.Timestamp, lag: int, window: int, min_periods: int
    ) -> float:
        window_dates = [target_date - pd.Timedelta(days=i) for i in range(lag, lag + window)]
        vals = [covers_by_date.get(d) for d in window_dates]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.mean(vals)) if len(vals) >= min_periods else np.nan

    def get_rollavg_7d_trend(
        target_date: pd.Timestamp, lag: int = 7, window: int = 7, min_periods: int = 3
    ) -> float:
        return _trailing_mean(target_date, lag, window, min_periods)

    def get_rollavg_28d_level(
        target_date: pd.Timestamp, lag: int = 7, window: int = 28, min_periods: int = 14
    ) -> float:
        """The slower-moving level the 7-day trend swings around.

        With both in the model the fit can separate "this week was busy" from
        "the restaurant is busier than it was", which a single window conflates.
        Worth ~0.3 covers of MAE in the backtest.
        """
        return _trailing_mean(target_date, lag, window, min_periods)

    return get_lag, get_rollavg_weekday, get_rollavg_7d_trend, get_rollavg_28d_level


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a daily covers frame: one row per calendar date, with the
    derived calendar / closure / rain flags the model expects.

    Mirrors the non-network half of the CLI's `build_clean_dataset`.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")

    # Reindex onto a gapless daily range so lag lookups line up with the calendar.
    full_range = pd.DataFrame(
        {"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")}
    )
    df = full_range.merge(df, on="date", how="left")

    df["covers_missing"] = df["covers"].isna().astype(int)
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND_DAYS).astype(int)
    # A recorded zero means the restaurant was closed; a blank means we simply
    # have no data for that day. They must not be treated the same.
    df["is_closed"] = ((df["covers"] == 0) & (df["covers_missing"] == 0)).astype(int)

    for col in ("is_promo", "is_holiday"):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Continuous, not a flag: event scoring stores graded pressure here, where
    # 1.0 is a typical event day. Casting to int would truncate 1.7 to 1 and
    # 0.6 to 0, throwing away exactly the distance/size grading. Historical
    # rows are 0/1, which is the same scale.
    if "is_local_event" not in df.columns:
        df["is_local_event"] = 0.0
    df["is_local_event"] = (
        pd.to_numeric(df["is_local_event"], errors="coerce").fillna(0.0).astype(float)
    )

    for col in ("event_name", "promo_name", "holiday_name"):
        if col not in df.columns:
            df[col] = np.nan

    # These must exist whether or not enrichment ran -- `fit` selects them by
    # name, and a missing column is a crash rather than a missing feature.
    for col in ("temp_high_f", "precip_in"):
        if col not in df.columns:
            df[col] = np.nan
    # `is_rain` stays "any measurable precipitation" because the page and the API
    # have always reported it that way. The model uses the two columns below.
    if "is_rain" not in df.columns:
        df["is_rain"] = (df["precip_in"].fillna(0) > 0).astype(int)
    df["is_rain"] = df["is_rain"].fillna(0).astype(int)
    df["is_rain_heavy"] = (
        df["precip_in"].fillna(0) > RAIN_HEAVY_THRESHOLD_IN
    ).astype(int)
    df["precip_in_capped"] = df["precip_in"].fillna(0).clip(0, PRECIP_CAP_IN)

    # With enrichment off and no weather in the source, temperature carries no
    # signal. A neutral constant keeps it from reading as 0F, which would
    # dominate the fit; the coefficient then simply goes to zero.
    if df["temp_high_f"].isna().all():
        df["temp_high_f"] = enrichment.NEUTRAL_TEMP_F

    return df.reset_index(drop=True)


def fit(df: pd.DataFrame, min_training_days: int = 30, holdout_days: int = 55) -> FittedModel:
    """Fit the regression. Raises NotEnoughHistory if there is too little data."""
    hist = df[df["is_closed"] == 0].reset_index(drop=True).copy()
    hist["day_of_week"] = pd.Categorical(hist["day_of_week"], categories=DOW_ORDER)
    covers_by_date = df.set_index("date")["covers"]

    (
        get_lag,
        get_rollavg_weekday,
        get_rollavg_7d_trend,
        get_rollavg_28d_level,
    ) = make_feature_functions(covers_by_date, hist)

    hist["covers_lag_7"] = hist["date"].apply(lambda d: get_lag(d, 7))
    hist["covers_rollavg_4wk_weekday"] = [
        get_rollavg_weekday(r.date, r.day_of_week) for r in hist.itertuples()
    ]
    hist["covers_rollavg_7d_trend"] = hist["date"].apply(get_rollavg_7d_trend)
    hist["covers_rollavg_28d_level"] = hist["date"].apply(get_rollavg_28d_level)

    # bool() matters: the pandas comparison yields numpy.bool_, which leaks all
    # the way into the JSON response and trips Pydantic's numpy deprecation path.
    has_reservations = bool(
        "reservations_covers" in hist.columns
        and hist["reservations_covers"].notna().mean() >= RESERVATIONS_COVERAGE_THRESHOLD
    )

    numeric_feats = list(NUMERIC_FEATURES)
    reservations_ratio = 1.0
    if has_reservations:
        numeric_feats.append("reservations_covers")
        both = hist[["reservations_covers", "covers"]].dropna()
        both = both[both["covers"] > 0]
        if len(both):
            reservations_ratio = float((both["reservations_covers"] / both["covers"]).mean())

    dow_dummies = pd.get_dummies(hist["day_of_week"], prefix="dow", drop_first=True).astype(float)
    X = pd.concat([dow_dummies, hist[numeric_feats]], axis=1)
    y = hist["covers"]

    complete = X.notna().all(axis=1) & y.notna()
    X_train, y_train = X[complete], y[complete]
    if len(X_train) < min_training_days:
        raise NotEnoughHistory(len(X_train), min_training_days)

    X_design = np.column_stack([np.ones(len(X_train)), X_train.values])
    coefs, _, _, _ = np.linalg.lstsq(X_design, y_train.values, rcond=None)

    accuracy = _holdout_accuracy(hist, X, y, complete, holdout_days, min_training_days)

    return FittedModel(
        coefs=coefs,
        feature_cols=list(X.columns),
        has_reservations=has_reservations,
        reservations_ratio=reservations_ratio,
        trained_days=int(len(X_train)),
        hist=hist,
        last_history_date=hist["date"].max(),
        accuracy=accuracy,
        get_lag=get_lag,
        get_rollavg_weekday=get_rollavg_weekday,
        get_rollavg_7d_trend=get_rollavg_7d_trend,
        get_rollavg_28d_level=get_rollavg_28d_level,
    )


def _holdout_accuracy(
    hist: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    complete: pd.Series,
    holdout_days: int,
    min_training_days: int,
) -> Optional[Accuracy]:
    """Refit on everything before the hold-out window and score on it. Returns
    None when there is not enough history to spare."""
    test_start = hist["date"].max() - pd.Timedelta(days=holdout_days)
    test_mask = complete & (hist["date"] >= test_start)
    train_mask = complete & (hist["date"] < test_start)
    if train_mask.sum() < min_training_days or test_mask.sum() < 7:
        return None

    X_tr = np.column_stack([np.ones(int(train_mask.sum())), X[train_mask].values])
    coefs, _, _, _ = np.linalg.lstsq(X_tr, y[train_mask].values, rcond=None)
    X_te = np.column_stack([np.ones(int(test_mask.sum())), X[test_mask].values])
    pred = X_te @ coefs
    actual = y[test_mask].values

    # Guard against a zero actual producing an infinite MAPE.
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(pred[nonzero] - actual[nonzero]) / actual[nonzero]) * 100)
        if nonzero.any()
        else float("nan")
    )
    return Accuracy(
        mae=float(np.mean(np.abs(pred - actual))),
        mape=mape,
        tested_days=int(test_mask.sum()),
    )


def predict_one(row: Dict, model: FittedModel) -> float:
    """Score a single feature row against the fitted coefficients."""
    dow_keys = {f"dow_{d}" for d in DOW_ORDER[1:]}
    x = [1.0]
    for col in model.feature_cols:
        if col in dow_keys:
            x.append(1.0 if col == f"dow_{row['day_of_week']}" else 0.0)
        else:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0  # graceful fallback rather than crashing on a missing input
            x.append(float(val))
    return float(np.array(x) @ model.coefs)


def build_feature_row(
    target: pd.Timestamp,
    model: FittedModel,
    location: Optional[Location],
    overrides: Optional[Dict] = None,
    enrichment_enabled: bool = True,
) -> Dict:
    """Assemble a full feature row for `target`, auto-filling holiday, event and
    weather values from the public APIs.

    `overrides` wins over anything fetched -- that is how a caller says "we are
    running a promo that day" or supplies its own weather estimate.
    """
    overrides = overrides or {}
    dow = target.day_name()
    row: Dict = {
        "date": target,
        "day_of_week": dow,
        "covers_lag_7": model.get_lag(target, 7),
        "covers_rollavg_4wk_weekday": model.get_rollavg_weekday(target, dow),
        "covers_rollavg_7d_trend": model.get_rollavg_7d_trend(target),
        "covers_rollavg_28d_level": model.get_rollavg_28d_level(target),
        "is_promo": 0,
        "is_holiday": 0,
        "is_local_event": 0,
        "holiday_name": "",
        "event_name": "",
        "weather_estimated": False,
    }

    use_apis = enrichment_enabled and location is not None

    if use_apis and "is_holiday" not in overrides:
        hname = enrichment.holiday_for_date(target, location)
        if hname is not None:
            row["is_holiday"] = 1 if hname else 0
            row["holiday_name"] = hname

    if use_apis and "is_local_event" not in overrides:
        pressure = enrichment.event_pressure_for_date(
            target,
            location,
            radius_miles=getattr(location, "event_radius_miles", 15.0),
            pressure_cap=getattr(location, "event_pressure_cap", events.PRESSURE_CAP),
        )
        if pressure is not None:
            row["event_detail"] = pressure.to_dict()
            row["event_pressure"] = pressure.pressure
            row["event_count"] = pressure.event_count
            row["nearest_event_miles"] = pressure.nearest_miles
            row["event_name"] = pressure.summary() or ""

            # The model's coefficient was fitted on a binary 0/1 column, so
            # `is_local_event` is where the pressure has to go -- and pressure
            # is calibrated so 1.0 == a typical event day, which is exactly
            # what that coefficient means. A nearer or larger event therefore
            # scales the effect linearly instead of being flattened to 1.
            if getattr(location, "event_scoring_enabled", True):
                row["is_local_event"] = pressure.pressure
            else:
                row["is_local_event"] = pressure.is_local_event

    weather = enrichment.weather_for_date(target, location) if use_apis else None
    if weather is not None:
        row["temp_high_f"] = float(weather["temp_high_f"])
        row["precip_in"] = float(weather["precip_in"])
        row["is_rain"] = 1 if weather["precip_in"] > 0 else 0
    else:
        # Live forecast unreachable -- estimate from this month's recent history
        # rather than leaving temp blank (a blank reads as 0F to the model).
        temp, rainy = enrichment.estimate_seasonal_weather(target, model.hist)
        row["temp_high_f"] = temp
        row["precip_in"] = np.nan
        row["is_rain"] = 1 if rainy else 0
        row["weather_estimated"] = True

    # Derived from precip, so they are set after the override-free weather above
    # and recomputed after `overrides` lands -- a caller that supplies its own
    # precip_in must move the rain features with it.
    _apply_precip_features(row)

    if model.has_reservations:
        # No bookings feed for a future date, so estimate it: this weekday's
        # recent covers average, scaled by the historical reservations share.
        # Using the covers average directly would feed a covers-scale value
        # into a reservations-scale feature and inflate every forecast.
        weekday_covers = row["covers_rollavg_4wk_weekday"]
        row["reservations_covers"] = (
            np.nan
            if weekday_covers is None or pd.isna(weekday_covers)
            else weekday_covers * model.reservations_ratio
        )

    row.update(overrides)
    if "precip_in" in overrides and "is_rain_heavy" not in overrides:
        _apply_precip_features(row)
    return row


def _apply_precip_features(row: Dict) -> None:
    """Fill the two precipitation features the model actually uses.

    A missing precip reading means "we do not know", and the honest stand-in for
    an unknown is a dry day -- the common case -- rather than a NaN that
    `predict_one` would silently turn into 0.0 anyway.
    """
    precip = row.get("precip_in")
    amount = 0.0 if precip is None or pd.isna(precip) else float(precip)
    row["is_rain_heavy"] = 1 if amount > RAIN_HEAVY_THRESHOLD_IN else 0
    row["precip_in_capped"] = min(max(amount, 0.0), PRECIP_CAP_IN)


def forecast_days(
    model: FittedModel,
    location: Optional[Location],
    days: int = 7,
    start: Optional[pd.Timestamp] = None,
    overrides_by_date: Optional[Dict[str, Dict]] = None,
    enrichment_enabled: bool = True,
) -> List[Dict]:
    """Forecast `days` consecutive days starting the day after the last day of
    history (or from `start`, if given).

    Returns one dict per day with the prediction and the inputs behind it, so a
    caller can show *why* a number came out the way it did.
    """
    overrides_by_date = overrides_by_date or {}
    first = start if start is not None else model.last_history_date + pd.Timedelta(days=1)
    targets = [pd.Timestamp(first) + pd.Timedelta(days=i) for i in range(days)]

    results = []
    for target in targets:
        iso = target.date().isoformat()
        row = build_feature_row(
            target,
            model,
            location,
            overrides=overrides_by_date.get(iso),
            enrichment_enabled=enrichment_enabled,
        )
        # Covers cannot be negative; a sparse-history extrapolation can go there.
        forecast = max(0.0, predict_one(row, model))
        results.append(
            {
                "date": iso,
                "day_of_week": row["day_of_week"],
                "forecast_covers": round(forecast, 1),
                "is_holiday": int(row["is_holiday"]),
                "holiday_name": row.get("holiday_name") or None,
                # Now continuous: the value actually fed to the model, where
                # 1.0 is a typical event day. Reported as given rather than
                # rounded to a flag, so the grading is visible.
                "is_local_event": _as_optional_float(row["is_local_event"]) or 0.0,
                "event_name": row.get("event_name") or None,
                "event_pressure": _as_optional_float(row.get("event_pressure")),
                "event_count": row.get("event_count"),
                "nearest_event_miles": _as_optional_float(row.get("nearest_event_miles")),
                "event_detail": row.get("event_detail"),
                "is_promo": int(row["is_promo"]),
                "temp_high_f": _as_optional_float(row.get("temp_high_f")),
                "precip_in": _as_optional_float(row.get("precip_in")),
                "is_rain": int(row["is_rain"]),
                # What the model actually reacts to. `is_rain` fires on any
                # trace -- four Miami days in five -- so it is kept for the API
                # contract but is not what to show a manager.
                "is_rain_heavy": int(row.get("is_rain_heavy") or 0),
                "weather_estimated": bool(row.get("weather_estimated")),
                "covers_lag_7": _as_optional_float(row.get("covers_lag_7")),
                "covers_rollavg_4wk_weekday": _as_optional_float(
                    row.get("covers_rollavg_4wk_weekday")
                ),
                "reservations_covers": _as_optional_float(row.get("reservations_covers")),
            }
        )
    return results


def _as_optional_float(value) -> Optional[float]:
    """JSON has no NaN -- surface missing numbers as null instead."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 2)
