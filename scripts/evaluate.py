#!/usr/bin/env python3
"""Walk-forward evaluation: how good is the forecast, really?

Why this exists
---------------
`fit()` reports a single hold-out score, and that number is optimistic for two
separate reasons:

1. **One window.** A single 55-day hold-out is one sample. Comparing two
   feature sets on it reliably "discovers" differences that vanish the moment
   you look at another window. Several candidate changes to this project
   looked like 5-8% wins on one window and were worth nothing across five.

2. **Leaked features.** `_holdout_accuracy` refits on data before the window
   but scores using the feature matrix built from the *whole* history. For a
   test day deep inside the hold-out, `covers_rollavg_4wk_weekday` averages
   actual covers from inside the hold-out -- days you would not have when
   forecasting. The further in, the more it knows that it should not.

This harness avoids both. It walks an origin through the history and, at each
one, refits on data up to that origin only, then forecasts the next `--horizon`
days through the same `forecast_days` path the web page uses. Nothing after the
origin is visible. Historical weather and holidays for the target days are
passed as overrides, so the backtest uses the conditions actually recorded
rather than calling a live API.

It also scores two baselines, because an accuracy number means nothing alone:

* **last week**   -- same weekday, seven days ago; the forecast a manager
  already makes in their head.
* **weekday avg** -- that weekday's mean over the previous four weeks.

A model that cannot beat both is not earning its complexity.

    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --horizon 7 --step 7
    python3 scripts/evaluate.py --ablate
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Running this as a script puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core import model as model_core  # noqa: E402
from app.service import CoversService  # noqa: E402

# Features whose true historical values are handed to the forecaster as
# overrides -- a backtest should use the weather that happened, not today's.
# `precip_in` is the one that matters: `build_feature_row` re-derives
# is_rain_heavy and precip_in_capped from it, so overriding the amount moves the
# features the model actually reads.
OVERRIDE_COLS = ("temp_high_f", "precip_in", "is_rain", "is_holiday", "is_local_event")
METHODS = ("model", "last_week", "weekday_avg")
# Where `--save` publishes the honest score for the web page to read.
ACCURACY_FILENAME = "walk_forward_accuracy.json"


def accuracy_path(settings) -> Path:
    """Beside the database, so it travels with the deployment it describes."""
    return Path(settings.db_path).parent / ACCURACY_FILENAME


def load_history(settings: Settings) -> pd.DataFrame:
    service = CoversService(settings)
    service.ingest()
    return service.history()


def origins_for(
    hist: pd.DataFrame, min_train: int, horizon: int, step: int
) -> List[pd.Timestamp]:
    """Evenly spaced forecast origins with enough history behind and ahead."""
    usable = hist[hist["covers"].notna()].reset_index(drop=True)
    if len(usable) < min_train + horizon:
        return []
    first = usable.loc[min_train, "date"]
    last = usable["date"].max() - pd.Timedelta(days=horizon)
    out, cursor = [], first
    while cursor <= last:
        out.append(cursor)
        cursor += pd.Timedelta(days=step)
    return out


def evaluate(
    hist: pd.DataFrame, origins: List[pd.Timestamp], horizon: int, min_train: int
) -> pd.DataFrame:
    """One row per (origin, horizon) with the model's and the baselines' errors."""
    actual_by_date = dict(zip(hist["date"], hist["covers"]))
    rows: List[Dict] = []

    for origin in origins:
        past = hist[hist["date"] <= origin]
        if past["covers"].notna().sum() < min_train:
            continue
        try:
            model = model_core.fit(past, min_training_days=min_train, holdout_days=0)
        except model_core.NotEnoughHistory:
            continue

        # True recorded conditions for the target days, so this does not also
        # end up testing the weather forecast.
        overrides = {}
        for h in range(1, horizon + 1):
            target = origin + pd.Timedelta(days=h)
            match = hist[hist["date"] == target]
            if match.empty:
                continue
            overrides[target.date().isoformat()] = {
                col: match.iloc[0][col] for col in OVERRIDE_COLS if col in match.columns
            }

        preds = model_core.forecast_days(
            model,
            location=None,
            days=horizon,
            start=origin + pd.Timedelta(days=1),
            overrides_by_date=overrides,
            enrichment_enabled=False,
        )

        for h, pred in enumerate(preds, start=1):
            target = pd.Timestamp(pred["date"])
            actual = actual_by_date.get(target)
            if actual is None or pd.isna(actual):
                continue  # a gap day is unknown, not zero

            last_week = actual_by_date.get(target - pd.Timedelta(days=7))
            same_dow = past[
                (past["day_of_week"] == target.day_name()) & past["covers"].notna()
            ].tail(4)["covers"]

            rows.append(
                {
                    "origin": origin,
                    "horizon": h,
                    "date": target,
                    "actual": float(actual),
                    "model": float(pred["forecast_covers"]),
                    "last_week": (
                        float(last_week)
                        if last_week is not None and not pd.isna(last_week)
                        else np.nan
                    ),
                    "weekday_avg": float(same_dow.mean()) if len(same_dow) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def mae(frame: pd.DataFrame, col: str) -> float:
    d = frame[[col, "actual"]].dropna()
    return float(np.mean(np.abs(d[col] - d["actual"]))) if len(d) else float("nan")


def mape(frame: pd.DataFrame, col: str) -> float:
    d = frame[[col, "actual"]].dropna()
    d = d[d["actual"] != 0]
    if not len(d):
        return float("nan")
    return float(np.mean(np.abs(d[col] - d["actual"]) / d["actual"]) * 100)


def report(results: pd.DataFrame, horizon: int) -> None:
    print(
        f"\n{len(results)} forecasts from {results['origin'].nunique()} origins "
        f"({results['date'].min().date()} -> {results['date'].max().date()})"
    )

    print("\nOVERALL")
    print(f"  {'method':14s} {'MAE':>7s} {'MAPE':>8s}")
    for m in METHODS:
        print(f"  {m:14s} {mae(results, m):7.2f} {mape(results, m):7.1f}%")

    print("\nBY HORIZON (MAE)")
    print(f"  {'days ahead':12s} " + " ".join(f"{m:>12s}" for m in METHODS))
    for h in range(1, horizon + 1):
        sub = results[results["horizon"] == h]
        if sub.empty:
            continue
        print(f"  {'+' + str(h):12s} " + " ".join(f"{mae(sub, m):12.2f}" for m in METHODS))

    # Spread across origins is the honest yardstick: a difference smaller than
    # this is noise, however pleasing it looks.
    per_origin = pd.DataFrame(
        [{m: mae(g, m) for m in METHODS} for _, g in results.groupby("origin")]
    )
    print("\nPER-ORIGIN SPREAD (MAE)")
    print(f"  {'method':14s} {'mean':>7s} {'sd':>7s} {'min':>7s} {'max':>7s}")
    for m in METHODS:
        col = per_origin[m].dropna()
        print(
            f"  {m:14s} {col.mean():7.2f} {col.std():7.2f} "
            f"{col.min():7.2f} {col.max():7.2f}"
        )

    # Paired, not a comparison of spreads. Origins differ wildly in how hard
    # they are (MAE 4.5 to 27 here), but that difficulty hits every method
    # equally, so comparing each method's sd would hide a real effect behind
    # variance both methods share. What matters is the per-forecast difference.
    from scipy import stats

    print("\nIS THE MODEL ACTUALLY BETTER? (paired, per forecast)")
    errors = {m: (results[m] - results["actual"]).abs() for m in METHODS}
    for baseline in ("weekday_avg", "last_week"):
        pair = pd.DataFrame(
            {"model": errors["model"], "base": errors[baseline]}
        ).dropna()
        if len(pair) < 10:
            continue
        diff = pair["base"] - pair["model"]  # positive = model better
        _, p = stats.ttest_rel(pair["base"], pair["model"])
        lo = diff.mean() - 1.96 * diff.sem()
        hi = diff.mean() + 1.96 * diff.sem()
        wins = int((diff > 0).sum())
        verdict = "better" if p < 0.05 and diff.mean() > 0 else "NOT clearly better"
        print(
            f"  vs {baseline:12s} {diff.mean():+5.2f} covers  "
            f"95% CI [{lo:+5.2f},{hi:+5.2f}]  p={p:.3f}  "
            f"wins {100 * wins / len(pair):.0f}%  -> {verdict}"
        )


def ablate(hist: pd.DataFrame, origins, horizon: int, min_train: int) -> None:
    """The same harness with feature groups switched off."""

    def strip(df, weather=False, holidays=False):
        df = df.copy()
        if weather:
            df["temp_high_f"] = 75.0
            df["precip_in"] = 0.0
        if holidays:
            df["is_holiday"] = 0
        # Re-derive rather than edit: is_rain, is_rain_heavy and
        # precip_in_capped all come off precip_in, and setting the amount to
        # zero while leaving "it rained" behind would ablate nothing.
        return model_core.prepare_history(df.drop(columns=["is_rain"]))

    variants = {
        "full": hist,
        "no weather": strip(hist, weather=True),
        "no holidays": strip(hist, holidays=True),
        "neither": strip(hist, weather=True, holidays=True),
    }
    print("\nABLATION (walk-forward, same origins)")
    print(f"  {'variant':14s} {'MAE':>7s} {'MAPE':>8s}")
    for name, frame in variants.items():
        res = evaluate(frame, origins, horizon, min_train)
        print(f"  {name:14s} {mae(res, 'model'):7.2f} {mape(res, 'model'):7.1f}%")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=7, help="Days ahead (default 7).")
    # Coprime with 7 on purpose. A step that is a multiple of 7 puts every
    # origin on the same weekday, so horizon +4 is always (say) Sunday and the
    # by-horizon table measures which weekday is hardest rather than how
    # accuracy decays with distance.
    parser.add_argument(
        "--step", type=int, default=5, help="Days between origins (default 5; avoid multiples of 7)."
    )
    parser.add_argument(
        "--min-train", type=int, default=90, help="Minimum training days (default 90)."
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write per-forecast rows here.")
    parser.add_argument("--ablate", action="store_true", help="Also run the feature ablation.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Publish this score for the web page to quote, instead of the "
        "optimistic in-process hold-out figure.",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    print(
        f"source={settings.source}  path={settings.csv_path}  "
        f"enrichment={settings.enrichment_enabled}"
    )
    hist = load_history(settings)

    origins = origins_for(hist, args.min_train, args.horizon, args.step)
    if not origins:
        print("Not enough history for even one origin.")
        return 1

    results = evaluate(hist, origins, args.horizon, args.min_train)
    if results.empty:
        print("No scoreable forecasts produced.")
        return 1

    report(results, args.horizon)
    if args.ablate:
        ablate(hist, origins, args.horizon, args.min_train)
    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"\nWrote {len(results)} rows to {args.csv}")

    if args.save:
        out = accuracy_path(settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "mae": round(mae(results, "model"), 2),
                    "mape": round(mape(results, "model"), 2),
                    "baseline_weekday_avg_mae": round(mae(results, "weekday_avg"), 2),
                    "forecasts": int(len(results)),
                    "origins": int(results["origin"].nunique()),
                    "horizon": args.horizon,
                    "through": results["date"].max().date().isoformat(),
                    "generated": dt.date.today().isoformat(),
                },
                indent=2,
            )
        )
        print(f"\nPublished walk-forward accuracy to {out}")
        print("The web page will now quote this instead of the hold-out figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
