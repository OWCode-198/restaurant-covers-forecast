#!/usr/bin/env python3
"""Load Toast order history into the store, slowly and resumably.

`POST /ingest` already pulls a date range, but it does the whole range in one
in-process pass and writes only at the end. Toast's orders API is fetched one
business date per request, so two years is ~730 sequential requests: long
enough that a dropped connection, a rate limit or a closed laptop will happen,
and losing the lot at request 700 is not acceptable.

So this script:

  * writes to the store every `--chunk-days` rather than once at the end, so
    progress survives whatever interrupts it
  * skips business dates already stored, which makes a re-run a resume (pass
    `--overwrite` to refetch them anyway)
  * sleeps between requests, because Toast asks for historical pulls to be
    spaced out and a backfill is the least urgent thing you will ever run
  * flushes what it has on Ctrl-C and tells you where to resume from

    python3 scripts/toast_backfill.py --start 2024-10-01
    python3 scripts/toast_backfill.py --start 2024-10-01 --end 2025-01-01
    python3 scripts/toast_backfill.py --start 2024-10-01 --sleep 2.0

Run `scripts/toast_preflight.py` first: it fails in one second on a bad
credential, where this would fail 730 times.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# Running this as a script puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.sources.base import SourceError, SourceNotConfigured  # noqa: E402
from app.sources.toast import ToastCoversSource, guests_in_order  # noqa: E402
from app.store import Store  # noqa: E402

# Toast's guidance for historical pulls is to space calls several seconds
# apart. One business date per request is lighter than a month-wide range
# query, so default to something polite but not glacial; --sleep raises it.
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_CHUNK_DAYS = 14
# Two years is the useful horizon for a daily model with yearly seasonality.
DEFAULT_LOOKBACK_DAYS = 730


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a date (use YYYY-MM-DD).")


def stored_dates(store: Store) -> Set[dt.date]:
    """Business dates already in the store, so a re-run resumes."""
    existing = store.load_daily_covers()
    if existing.empty:
        return set()
    return {d.date() for d in pd.to_datetime(existing["date"])}


def flush(store: Store, rows: List[Dict]) -> int:
    """Write a chunk of fetched days and return how many landed."""
    if not rows:
        return 0
    written = store.upsert_daily_covers(pd.DataFrame(rows), source="toast")
    rows.clear()
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start",
        type=parse_date,
        default=None,
        help=f"First business date (default: {DEFAULT_LOOKBACK_DAYS} days ago).",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=None,
        help="Last business date (default: yesterday; today is still open).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Seconds between requests (default: {DEFAULT_SLEEP_SECONDS}).",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help=f"Write to the store every N days (default: {DEFAULT_CHUNK_DAYS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refetch business dates already in the store instead of skipping.",
    )
    args = parser.parse_args(argv)

    end = args.end or dt.date.today() - dt.timedelta(days=1)
    start = args.start or end - dt.timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if end < start:
        print(f"--end ({end}) is before --start ({start}).")
        return 1
    if args.chunk_days < 1:
        print("--chunk-days must be at least 1.")
        return 1
    if args.sleep < 0:
        print("--sleep cannot be negative.")
        return 1

    settings = Settings()
    try:
        source = ToastCoversSource(settings)
    except SourceNotConfigured as exc:
        print(f"Toast is not configured: {exc}")
        print("Run scripts/toast_preflight.py for a step-by-step check.")
        return 1

    store = Store(settings.db_path)
    already = set() if args.overwrite else stored_dates(store)

    wanted = [start + dt.timedelta(days=offset) for offset in range((end - start).days + 1)]
    todo = [d for d in wanted if d not in already]

    print(f"Backfilling Toast {start} -> {end}")
    print(f"  {len(wanted)} business date(s) in range")
    if already:
        print(
            f"  {len(wanted) - len(todo)} already stored, skipping "
            "(--overwrite to refetch)"
        )
    print(f"  {len(todo)} to fetch, ~{args.sleep:.1f}s apart")
    if not todo:
        print("\nNothing to do.")
        return 0
    print(f"  rough estimate: {len(todo) * max(args.sleep, 0.1) / 60:.0f} min\n")

    pending: List[Dict] = []
    written = 0
    failures: List[str] = []
    last_done: Optional[dt.date] = None

    try:
        for index, day in enumerate(todo, start=1):
            try:
                orders = source.client.fetch_orders_for_business_date(
                    day, page_size=settings.toast_page_size
                )
            except SourceError as exc:
                # One bad day must not void the backfill. An omitted date reads
                # downstream as "unknown", which is correct -- unlike a zero,
                # which would read as "closed".
                print(f"  {day}  SKIPPED: {exc}")
                failures.append(day.isoformat())
                continue

            live = [o for o in orders if not (o.get("voided") or o.get("deleted"))]
            pending.append(
                {
                    "date": pd.Timestamp(day),
                    "covers": float(sum(guests_in_order(o) for o in live)),
                    "order_count": len(live),
                }
            )
            last_done = day

            if index % 10 == 0 or index == len(todo):
                print(f"  {day}  ({index}/{len(todo)})")
            if len(pending) >= args.chunk_days:
                written += flush(store, pending)

            if index < len(todo) and args.sleep:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        written += flush(store, pending)
        print("\n\nInterrupted.")
        print(f"  {written} day(s) saved.")
        if last_done is not None:
            print(f"  Resume with: --start {last_done + dt.timedelta(days=1)}")
            print("  (or just re-run the same command -- stored dates are skipped)")
        return 130

    written += flush(store, pending)

    coverage = store.covers_coverage()
    print(f"\nDone. {written} day(s) written.")
    print(
        f"  store now holds {coverage['days']} day(s): "
        f"{coverage['first_date']} -> {coverage['last_date']}"
    )
    if failures:
        print(f"  {len(failures)} date(s) failed and were left unstored:")
        print("    " + ", ".join(failures[:10]) + ("..." if len(failures) > 10 else ""))
        print("  Re-run the same command to retry just those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
