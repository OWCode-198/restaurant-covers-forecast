#!/usr/bin/env python3
"""Annotate the covers CSV with what the model would have predicted.

Adds two columns so actual and predicted sit side by side in a spreadsheet:

    predicted   what the forecast would have said the evening before
    error       predicted - actual (positive = we over-forecast)

The predictions are **honest**, not hindsight. For every date the model is
refitted on history up to the previous day only and asked for a one-day-ahead
forecast, exactly as the page does. Nothing after the forecast origin is ever
visible, so these are the numbers a manager would really have been shown.

That distinction matters. Fitting once on all the data and reading predictions
back off the fitted line would flatter the model badly, because every day
would have helped choose the coefficients used to predict it.

Early dates come out blank: they sit before there is enough history to fit
anything, and a guess there would be worse than an honest gap.

    python3 scripts/add_predictions.py
    python3 scripts/add_predictions.py --out annotated.csv

Re-run after each import; it is safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.core import model as model_core  # noqa: E402

OVERRIDE_COLS = ("temp_high_f", "precip_in", "is_rain", "is_holiday", "is_local_event")


def load_evaluate():
    """Reuse the backtest's history loading rather than duplicating it."""
    spec = importlib.util.spec_from_file_location(
        "evaluate", Path(__file__).resolve().parent / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def next_day_predictions(hist: pd.DataFrame, min_train: int) -> pd.DataFrame:
    """One honest one-day-ahead forecast per date."""
    rows: List[dict] = []
    dates = hist.loc[hist["covers"].notna(), "date"].sort_values().tolist()
    total = len(dates)

    for index, target in enumerate(dates, start=1):
        past = hist[hist["date"] < target]
        if past["covers"].notna().sum() < min_train:
            continue
        try:
            model = model_core.fit(past, min_training_days=min_train, holdout_days=0)
        except model_core.NotEnoughHistory:
            continue

        # Hand it the conditions actually recorded for that day, so this does
        # not also end up being a test of the weather forecast.
        match = hist[hist["date"] == target].iloc[0]
        overrides = {c: match[c] for c in OVERRIDE_COLS if c in hist.columns}
        preds = model_core.forecast_days(
            model,
            location=None,
            days=1,
            start=target,
            overrides_by_date={target.date().isoformat(): overrides},
            enrichment_enabled=False,
        )
        rows.append({"date": target, "predicted": float(preds[0]["forecast_covers"])})

        if index % 50 == 0:
            print(f"  {index}/{total} days...")

    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="Write here instead of in place.")
    parser.add_argument("--min-train", type=int, default=90)
    args = parser.parse_args(argv)

    settings = Settings()
    source_csv = Path(settings.csv_path)
    if not source_csv.exists():
        raise SystemExit(f"{source_csv} not found. Check COVERS_CSV_PATH in .env.")

    ev = load_evaluate()
    print(f"Reading {source_csv}")
    hist = ev.load_history(settings)

    print("Refitting once per day (this takes a minute)...")
    preds = next_day_predictions(hist, args.min_train)
    if preds.empty:
        raise SystemExit("Not enough history to predict anything.")

    original = pd.read_csv(source_csv)
    original["date"] = pd.to_datetime(original["date"])
    # Drop any previous run's columns so re-running does not stack them up.
    original = original.drop(columns=[c for c in ("predicted", "error") if c in original.columns])

    merged = original.merge(preds, on="date", how="left")
    merged["error"] = (merged["predicted"] - merged["covers"]).round(0)
    merged["predicted"] = merged["predicted"].round(0)

    out = args.out or source_csv
    merged.to_csv(out, index=False, date_format="%Y-%m-%d")

    scored = merged.dropna(subset=["predicted"])
    blank = len(merged) - len(scored)
    print(f"\nWrote {len(merged)} rows to {out}")
    print(f"  {len(scored)} carry a prediction; {blank} sit before there is enough history.")
    print(f"  mean absolute error: {scored['error'].abs().mean():.2f} covers")
    print(
        f"  over-forecast on {int((scored['error'] > 0).sum())} days, "
        f"under on {int((scored['error'] < 0).sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
