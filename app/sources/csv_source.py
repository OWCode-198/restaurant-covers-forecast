"""Covers history from a CSV export.

This is the default source: it works today, with no Toast credentials, against
the exports already in this project (`data/processed/covers_daily_clean.csv`,
`data/raw/bhan_covers.csv`, ...). The column sniffing mirrors the CLI's
`detect_column` so the same files load without being reshaped by hand.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from app.sources.base import SourceError, validate_frame

logger = logging.getLogger(__name__)

# "yyyyMMdd" is the literal header Toast writes on the "Sales by day" sheet of
# a Sales summary export, so a raw export loads without being reshaped by hand.
DATE_CANDIDATES = ("date", "business_date", "day", "service_date", "yyyyMMdd")
COVERS_CANDIDATES = (
    "covers",
    "total_covers",
    "total guests",
    "total_guests",
    "guests",
    "guest_count",
    "covers_actual",
)
RESERVATION_CANDIDATES = ("reservations_covers", "reservations", "reserved_covers")

# Columns worth carrying through when the export already has them -- the model
# uses these as features instead of re-fetching them.
PASSTHROUGH_COLUMNS = (
    "reservations_covers",
    "temp_high_f",
    "precip_in",
    "is_rain",
    "is_holiday",
    "holiday_name",
    "is_local_event",
    "event_name",
    "is_promo",
    "promo_name",
    "is_closed",
)


def detect_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Find the first column matching a candidate, case-insensitively and
    ignoring spaces / underscores."""

    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "")

    lookup = {norm(c): c for c in columns}
    for cand in candidates:
        hit = lookup.get(norm(cand))
        if hit is not None:
            return hit
    return None


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse a date column, including Toast's numeric yyyyMMdd business dates.

    Left to its own inference, `pd.to_datetime` reads a bare 20250825 as
    nanoseconds since the epoch and returns 1970-01-01 -- a wrong answer rather
    than an error, which would quietly shift an entire export off the calendar.
    Eight-digit values are therefore parsed explicitly.
    """
    as_text = values.astype("string").str.strip()
    known = as_text.dropna()
    if len(known) and known.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


class CsvCoversSource:
    """Reads daily covers from a CSV file on disk."""

    name = "csv"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def fetch_daily_covers(
        self,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise SourceError(
                f"Covers CSV not found at {self.path}. Set COVERS_CSV_PATH to a file "
                "with a date column and a covers column."
            )
        try:
            raw = pd.read_csv(self.path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise SourceError(f"Could not read {self.path}: {exc}") from exc

        date_col = detect_column(raw.columns, DATE_CANDIDATES)
        covers_col = detect_column(raw.columns, COVERS_CANDIDATES)
        if date_col is None or covers_col is None:
            raise SourceError(
                f"{self.path.name} needs a date column and a covers column. "
                f"Found: {', '.join(raw.columns)}."
            )

        out = pd.DataFrame(
            {
                "date": parse_dates(raw[date_col]),
                "covers": pd.to_numeric(raw[covers_col], errors="coerce"),
            }
        )

        # A reservations column may be named differently from our canonical one.
        res_col = detect_column(raw.columns, RESERVATION_CANDIDATES)
        if res_col is not None:
            out["reservations_covers"] = pd.to_numeric(raw[res_col], errors="coerce")

        for col in PASSTHROUGH_COLUMNS:
            if col in out.columns:
                continue
            match = detect_column(raw.columns, (col,))
            if match is not None:
                out[col] = raw[match]

        dropped = int(out["date"].isna().sum())
        if dropped:
            logger.warning(
                "Dropped %d row(s) from %s with unparseable dates.", dropped, self.path.name
            )
        out = out[out["date"].notna()]

        if start is not None:
            out = out[out["date"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["date"] <= pd.Timestamp(end)]

        # Later rows win on a duplicated date -- a re-export of the same day is
        # the corrected one.
        out = out.drop_duplicates(subset="date", keep="last")
        return validate_frame(out, self.name)
