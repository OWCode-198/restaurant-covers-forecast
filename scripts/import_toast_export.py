#!/usr/bin/env python3
"""Turn a raw Toast Sales summary export into a covers CSV this project reads.

This is the documented, repeatable version of "clean up the export". Run it on
every new export rather than reshaping anything by hand.

What it does, and why each step exists
--------------------------------------
1. **Reads `Sales by day.csv`** from the export folder. That is the only sheet
   with a per-day series; the rest are period totals.

2. **Parses `yyyyMMdd` explicitly as %Y%m%d.** Left to infer, pandas reads a
   bare 20250825 as nanoseconds since the epoch and returns 1970-01-01 -- a
   wrong answer rather than an error, which would shift the whole file off the
   calendar.

3. **Maps `Total guests` to `covers`.** Note what that means: Toast defaults
   every order to one guest and the default cannot be switched off, so where
   staff do not enter guest counts `Total guests` is simply the order count.
   The report below says which case this export is.

4. **Drops any row dated today or later.** The current day is still trading
   when you export, so its total is partial. Keeping it teaches the model that
   the most recent day was unusually quiet, and the forecast page would treat
   a half-finished day as history.

5. **Leaves missing days missing.** A date absent from the export is reported
   as a gap, never written as 0. Downstream an absent date reads as "unknown"
   and a 0 reads as "closed"; conflating them teaches the model that a day
   nobody recorded was a day nobody came.

    python3 scripts/import_toast_export.py "SalesSummary_2025-08-15_2026-10-15"
    python3 scripts/import_toast_export.py <folder> --out data/processed/x.csv

Then point the service at the result:

    COVERS_SOURCE=csv
    COVERS_CSV_PATH=data/processed/covers_daily_toast.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Running this as a script puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_SHEET = "Sales by day.csv"
DINING_SHEET = "Dining options summary.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "covers_daily_toast.csv"

# Columns as Toast writes them.
DATE_COL = "yyyyMMdd"
GUESTS_COL = "Total guests"
ORDERS_COL = "Total orders"
SALES_COL = "Net sales"


def read_daily(export_dir: Path) -> pd.DataFrame:
    path = export_dir / DAILY_SHEET
    if not path.exists():
        raise SystemExit(
            f"No '{DAILY_SHEET}' in {export_dir}.\n"
            "Export Reports -> Sales summary from Toast Web and pass the folder "
            "it unzips to."
        )
    raw = pd.read_csv(path)
    missing = [c for c in (DATE_COL, GUESTS_COL, ORDERS_COL) if c not in raw.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing {', '.join(missing)}. Found: {', '.join(raw.columns)}."
        )

    out = pd.DataFrame(
        {
            # Explicit format: see step 2 in the module docstring.
            "date": pd.to_datetime(raw[DATE_COL], format="%Y%m%d", errors="coerce"),
            "covers": pd.to_numeric(raw[GUESTS_COL], errors="coerce"),
            "order_count": pd.to_numeric(raw[ORDERS_COL], errors="coerce"),
        }
    )
    if SALES_COL in raw.columns:
        out["net_sales"] = pd.to_numeric(raw[SALES_COL], errors="coerce")

    unparsed = int(out["date"].isna().sum())
    if unparsed:
        print(f"  ! dropped {unparsed} row(s) with an unreadable date")
    return out[out["date"].notna()].sort_values("date").reset_index(drop=True)


def gaps(df: pd.DataFrame) -> List[dt.date]:
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return [d.date() for d in full.difference(df["date"])]


def report(df: pd.DataFrame, dropped_partial: Optional[dt.date], export_dir: Path) -> None:
    print(f"\n  {len(df)} day(s): {df['date'].min().date()} -> {df['date'].max().date()}")

    if dropped_partial:
        print(f"  dropped {dropped_partial} (still trading when exported)")

    missing = gaps(df)
    if missing:
        shown = ", ".join(str(d) for d in missing[:8])
        print(
            f"  {len(missing)} gap(s), left as unknown rather than zero: {shown}"
            + (" ..." if len(missing) > 8 else "")
        )
    else:
        print("  no gaps")

    # Covers or transactions? This decides what the model is forecasting.
    differing = int((df["covers"] != df["order_count"]).sum())
    if differing:
        ratio = df["covers"].sum() / max(df["order_count"].sum(), 1)
        print(
            f"  guest counts look real ({differing} day(s) differ from order "
            f"count, {ratio:.2f} guests/order)"
        )
    else:
        print("  ! covers == order count on every day: nobody enters guest")
        print("    counts, so this series is TRANSACTIONS, not covers.")

    dining = export_dir / DINING_SHEET
    if dining.exists():
        mix = pd.read_csv(dining)
        if {"Dining option", "Orders"} <= set(mix.columns):
            mix = mix[mix["Dining option"] != "Total"]
            total = max(mix["Orders"].sum(), 1)
            share = ", ".join(
                f"{r['Dining option']} {r['Orders'] / total * 100:.0f}%"
                for _, r in mix.nlargest(3, "Orders").iterrows()
            )
            print(f"  dining mix: {share}")

    monthly = df.set_index("date")["covers"].resample("MS").mean()
    if len(monthly) >= 3 and monthly.iloc[0] > 0 and monthly.iloc[-1] / monthly.iloc[0] >= 2:
        print(
            f"  ! volume grew {monthly.iloc[-1] / monthly.iloc[0]:.1f}x since "
            f"{monthly.index[0].date():%b %Y} "
            f"({monthly.iloc[0]:.0f} -> {monthly.iloc[-1]:.0f}/day). Early months"
        )
        print("    are a different business; check whether trimming them helps.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path, help="The unzipped Sales summary folder.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write the clean CSV (default: data/processed/covers_daily_toast.csv).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite --out if it already exists."
    )
    args = parser.parse_args(argv)

    if not args.export_dir.is_dir():
        raise SystemExit(f"{args.export_dir} is not a directory.")

    print(f"Reading {args.export_dir / DAILY_SHEET}")
    df = read_daily(args.export_dir)
    if df.empty:
        raise SystemExit("No usable rows in the export.")

    # Step 4: today is partial, and so is anything dated ahead of it.
    today = dt.date.today()
    partial = df[df["date"] >= pd.Timestamp(today)]
    dropped = partial["date"].max().date() if not partial.empty else None
    df = df[df["date"] < pd.Timestamp(today)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("Every row is today or later -- nothing complete to import.")

    report(df, dropped, args.export_dir)

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"\n{args.out} already exists. Pass --force to overwrite it, or --out "
            "to write elsewhere.\n(Refusing by default: "
            "data/processed/covers_daily_clean.csv is a different dataset.)"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, date_format="%Y-%m-%d")

    try:
        rel = args.out.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = args.out
    print(f"\nWrote {len(df)} row(s) to {rel}")
    print("\nPoint the service at it:")
    print("  COVERS_SOURCE=csv")
    print(f"  COVERS_CSV_PATH={rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
