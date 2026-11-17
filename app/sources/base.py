"""The covers data-source contract.

Everything downstream (feature building, model fitting, the HTTP layer) works
against `CoversSource`, so swapping a CSV export for the live Toast API is a
configuration change rather than a code change.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional, Protocol, runtime_checkable

import pandas as pd

# The canonical daily shape every source must return. Extra columns are allowed
# and passed through (the project's own CSVs carry weather, for instance);
# these two are required.
REQUIRED_COLUMNS = ("date", "covers")


class SourceError(Exception):
    """A data source could not produce history (bad path, auth failure, ...)."""


class SourceNotConfigured(SourceError):
    """The selected source is missing required configuration."""


@runtime_checkable
class CoversSource(Protocol):
    """A provider of daily covers history.

    `name` identifies the source in /health and in responses so it is always
    clear whether a forecast came from a CSV export or from live Toast data.
    """

    name: str

    def fetch_daily_covers(
        self,
        start: Optional[dt.date] = None,
        end: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        """Return one row per calendar date with at least `date` and `covers`.

        `date` must be datetime64 and `covers` numeric -- NaN where unknown, 0
        only where the restaurant was genuinely closed, since the model treats
        those two cases differently.
        """
        ...


def validate_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Check a source's output against the contract and normalise dtypes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SourceError(
            f"Source '{source_name}' returned no {' and '.join(missing)} column "
            f"(got: {', '.join(df.columns) or 'nothing'})."
        )
    if df.empty:
        raise SourceError(f"Source '{source_name}' returned no rows.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["covers"] = pd.to_numeric(df["covers"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)
