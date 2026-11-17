#!/usr/bin/env python3
"""Check Toast credentials the moment they arrive, before backfilling anything.

Run this first. It is read-only and writes nothing -- it walks the things that
can be wrong, in the order they fail, so a bad setup produces one clear
sentence instead of a stack trace half way through a two-year backfill:

    1. the four settings are present
    2. the client id / secret authenticate
    3. the restaurant GUID resolves (proves the credential set covers this
       location)
    4. recent business dates return orders (proves the orders:read scope)

`GET /health?check_source=true` covers roughly steps 1-3, but it needs the
server running, reports any failure as a single opaque string, and never
fetches an order -- so it cannot tell a missing orders:read scope from a
working integration. This can, and it runs from a cold checkout.

It then answers the question that decides what the model is actually
forecasting: are guest counts real, or is every order Toast's un-disableable
default of one guest? If covers equal order counts on every sampled day the
target is transactions, not covers -- fine in itself, but worth knowing before
you build on it rather than after.

    python3 scripts/toast_preflight.py
    python3 scripts/toast_preflight.py --days 14

Exits 0 when Toast is ready to backfill, 1 otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Running this as a script puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.sources.base import SourceError, SourceNotConfigured  # noqa: E402
from app.sources.toast import (  # noqa: E402
    ToastAuthError,
    ToastCoversSource,
    guests_in_order,
)

# A preflight makes only a handful of requests, but there is no reason to burst
# them at an API whose rate limits we are guests of.
SLEEP_BETWEEN_DAYS = 0.5


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def note(msg: str) -> None:
    print(f"        {msg}")


def restaurant_label(info: Dict) -> str:
    """A human name for the restaurant, however this Toast account nests it.

    The restaurants API has moved these fields between the top level and
    `general` over time, so try the known shapes and fall back to reporting
    what did come back rather than raising on a missing key.
    """
    general = info.get("general") if isinstance(info.get("general"), dict) else {}
    for candidate in (
        general.get("name"),
        general.get("locationName"),
        info.get("name"),
        info.get("locationName"),
    ):
        if candidate:
            return str(candidate)
    return f"(no name field; keys returned: {', '.join(sorted(info)) or 'none'})"


def sample_days(source: ToastCoversSource, days: int) -> Tuple[List[Dict], List[str]]:
    """Fetch the last `days` complete business dates, one request each.

    Today is skipped deliberately: it is still open, so a low count would read
    as a problem with the credentials rather than a day in progress.
    """
    rows: List[Dict] = []
    errors: List[str] = []
    yesterday = dt.date.today() - dt.timedelta(days=1)

    for offset in range(days):
        day = yesterday - dt.timedelta(days=offset)
        try:
            orders = source.client.fetch_orders_for_business_date(
                day, page_size=source.settings.toast_page_size
            )
        except SourceError as exc:
            errors.append(f"{day}: {exc}")
            continue

        live = [o for o in orders if not (o.get("voided") or o.get("deleted"))]
        rows.append(
            {
                "date": day,
                "covers": sum(guests_in_order(o) for o in live),
                "orders": len(live),
            }
        )
        if offset < days - 1:
            time.sleep(SLEEP_BETWEEN_DAYS)

    rows.sort(key=lambda r: r["date"])
    return rows, errors


def report_guest_counts(rows: List[Dict]) -> None:
    """Say whether guest counts are real or Toast's default of one per order."""
    scored = [r for r in rows if r["orders"] > 0]
    if not scored:
        note("No orders in the sampled window, so guest counts cannot be judged yet.")
        return

    differing = [r for r in scored if r["covers"] != r["orders"]]
    if differing:
        ratio = sum(r["covers"] for r in scored) / sum(r["orders"] for r in scored)
        ok(
            f"Guest counts look real: they differ from order counts on "
            f"{len(differing)}/{len(scored)} day(s), averaging {ratio:.2f} "
            "guests per order."
        )
        return

    print()
    print("  NOTE  Covers equal order counts on every sampled day.")
    note("Toast defaults every order to 1 guest and that default cannot be")
    note("switched off, so this almost certainly means nobody enters guest")
    note("counts -- you are forecasting transactions, not covers.")
    note("That is a perfectly good target. Just name it honestly, and do not")
    note("pool it with a dine-in restaurant's real covers later on.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="How many recent business dates to sample (default: 7).",
    )
    args = parser.parse_args(argv)

    if args.days < 1:
        print("--days must be at least 1.")
        return 1

    settings = Settings()
    print("Toast preflight")
    print("=" * 62)

    # --- 1. configuration --------------------------------------------------
    print("\n1. Settings")
    try:
        source = ToastCoversSource(settings)
    except SourceNotConfigured as exc:
        fail(str(exc))
        note("Fill these into .env, then re-run. See .env.example.")
        return 1
    ok(f"All four Toast settings present (host: {source.client.base_url}).")
    if settings.source != "toast":
        note(
            f"COVERS_SOURCE is '{settings.source}'. Credentials are checked "
            "anyway, but set it to 'toast' before ingesting."
        )

    # --- 2. authentication -------------------------------------------------
    print("\n2. Authentication")
    try:
        source.client.authenticate()
    except ToastAuthError as exc:
        fail(str(exc))
        note("A 401/403 here means the client id or secret is wrong, or the")
        note("credential set was never enabled for this location.")
        return 1
    ok("Client credentials accepted; bearer token issued.")

    # --- 3. restaurant lookup ----------------------------------------------
    print("\n3. Restaurant")
    try:
        info = source.client.fetch_restaurant()
    except SourceError as exc:
        fail(str(exc))
        note("The credentials work but this GUID does not resolve. Check")
        note("COVERS_TOAST_RESTAURANT_GUID against Toast's confirmation email,")
        note("and that it is a location GUID, not a restaurant group GUID.")
        return 1
    ok(f"GUID resolves to: {restaurant_label(info)}")

    # --- 4. orders ---------------------------------------------------------
    print(f"\n4. Orders (last {args.days} complete business date(s))")
    rows, errors = sample_days(source, args.days)

    for err in errors:
        fail(err)
    if not rows:
        note("No business date could be fetched. If the errors above are 403s,")
        note("the credential set is missing the orders:read scope -- the")
        note("restaurant can add it to the same credential set in Toast Web.")
        return 1

    for row in rows:
        print(
            f"        {row['date']}  covers={row['covers']:>8.0f}  "
            f"orders={row['orders']:>6d}"
        )
    total_orders = sum(r["orders"] for r in rows)
    ok(f"Fetched {len(rows)} business date(s), {total_orders} live order(s) total.")
    if errors:
        note(f"{len(errors)} date(s) failed -- see above. Backfill will skip those.")

    # --- 5. what the data means --------------------------------------------
    print("\n5. Guest counts")
    report_guest_counts(rows)

    print("\n" + "=" * 62)
    print("Ready. Next:")
    print("  python3 scripts/toast_backfill.py --start YYYY-MM-DD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
