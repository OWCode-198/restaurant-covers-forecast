#!/usr/bin/env python3
"""The days the forecast got most wrong -- as a sheet to hand to the restaurant.

The model's error is concentrated in a handful of days: on this project's data
the worst 10 days out of 306 carry 13% of all error, and forecasting just
those perfectly would take MAE from 10.7 to 9.2. Nearly all of them are
weekends or holidays that went far busier than any feature in the model could
anticipate.

No public data source explains them. Event listings turned up the King Mango
Strut (+67 covers) but nothing at all for the two largest spikes, while the
Coconut Grove Arts Festival -- 100,000 visitors, a mile away -- moved covers
by zero. Size and proximity do not predict impact.

The people who work there do know. So this prints the days worth asking about,
with a blank column for the answer. Anything they can name becomes an entry in
`recurring_events.json` (see scripts/setup_recurring_events.py) and the model
stops being blind to it.

    python3 scripts/worst_days.py
    python3 scripts/worst_days.py --top 20 --out sheet.csv

Errors come from the walk-forward backtest in scripts/evaluate.py, so
"expected" is what the model would genuinely have said at the time, using only
data available before that day -- not a number fitted with hindsight.
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


def load_evaluate():
    """Reuse the backtest rather than reimplementing it."""
    spec = importlib.util.spec_from_file_location(
        "evaluate", Path(__file__).resolve().parent / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_sheet(results: pd.DataFrame, top: int) -> pd.DataFrame:
    """One row per date, worst misses first, with a column for their answer."""
    # A date is forecast from several origins; average them so each day
    # appears once rather than up to seven times.
    per_day = (
        results.groupby("date")
        .agg(actual=("actual", "first"), expected=("model", "mean"))
        .reset_index()
    )
    per_day["difference"] = (per_day["actual"] - per_day["expected"]).round(0).astype(int)
    per_day["abs_diff"] = per_day["difference"].abs()

    worst = per_day.nlargest(top, "abs_diff").sort_values("date")
    return pd.DataFrame(
        {
            "date": worst["date"].dt.date,
            "day": worst["date"].dt.day_name(),
            "actual_covers": worst["actual"].round(0).astype(int),
            "expected": worst["expected"].round(0).astype(int),
            "difference": worst["difference"].map(lambda v: f"{v:+d}"),
            "busier_or_quieter": worst["difference"].map(
                lambda v: "BUSIER" if v > 0 else "QUIETER"
            ),
            "what_happened": "",
        }
    )


def _rows(frame: pd.DataFrame) -> None:
    print(f"  {'date':12s} {'day':10s} {'actual':>7s} {'usual':>7s} {'diff':>7s}")
    for _, r in frame.iterrows():
        print(
            f"  {str(r['date']):12s} {r['day']:10s} "
            f"{r['actual_covers']:7d} {r['expected']:7d} {r['difference']:>7s}"
        )


def print_sheet(sheet: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DAYS WE CANNOT EXPLAIN  --  do any of these ring a bell?")
    print("=" * 70)

    busier = sheet[sheet["busier_or_quieter"] == "BUSIER"]
    quieter = sheet[sheet["busier_or_quieter"] == "QUIETER"]
    if len(busier):
        print("\nMUCH BUSIER than normal for that day of the week:\n")
        _rows(busier)
    if len(quieter):
        print("\nMUCH QUIETER than normal for that day of the week:\n")
        _rows(quieter)

    print("\n" + "-" * 70)
    print("Worth asking about each one:")
    print("  - an event nearby, or something on in the bookstore?")
    print("  - a post of yours, or someone else posting about you?")
    print("  - closed, short-staffed, or opened late?")
    print("  - street closure, construction, nowhere to park?")
    print("  - a promotion, on a delivery app or your own?")
    print("  - weather bad enough to be memorable (storm, warning)?")
    print("-" * 70)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=15, help="How many days (default 15).")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "worst_days_to_ask_about.csv",
        help="Where to write the sheet.",
    )
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=90)
    args = parser.parse_args(argv)

    ev = load_evaluate()
    settings = Settings()
    print(f"source={settings.source}  path={settings.csv_path}")

    hist = ev.load_history(settings)
    origins = ev.origins_for(hist, args.min_train, args.horizon, args.step)
    if not origins:
        print("Not enough history to backtest.")
        return 1

    results = ev.evaluate(hist, origins, args.horizon, args.min_train)
    if results.empty:
        print("No scoreable forecasts produced.")
        return 1

    sheet = build_sheet(results, args.top)
    print_sheet(sheet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False)
    try:
        shown = args.out.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        shown = args.out
    print(f"\nWrote {len(sheet)} rows to {shown}")
    print("Open it in Excel or Google Sheets and fill in 'what_happened'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
