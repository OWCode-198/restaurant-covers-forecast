"""Persistence for ingested covers and logged forecasts.

SQLite, via the standard library -- no ORM and no server to run. It replaces
the CLI's `forecast_log.csv` append-and-hope approach with something that can
be upserted and queried, which matters once ingestion happens on a schedule
rather than by hand.

Dates are stored as ISO `YYYY-MM-DD` text, which sorts and compares correctly
in SQLite and keeps the rows readable with any sqlite3 client.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Optional per-day feature columns. A source that already knows these (the
# project's own CSV exports carry weather and reservations) must be able to
# round-trip them through the store, or the model would lose features that were
# present in the input. NULL means "unknown", and enrichment fills it in later.
FEATURE_COLUMNS = (
    "reservations_covers",
    "temp_high_f",
    "precip_in",
    "is_holiday",
    "holiday_name",
    "is_local_event",
    "event_name",
    "is_promo",
)

# Columns that are genuinely 0/1 flags. `is_local_event` is deliberately NOT
# here: event scoring made it continuous (1.0 = a typical event day), and
# coercing it to an int would throw the grading away.
INTEGER_FLAG_COLUMNS = ("is_holiday", "is_promo")
TEXT_COLUMNS = ("holiday_name", "event_name")

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_covers (
    date                TEXT PRIMARY KEY,   -- ISO YYYY-MM-DD
    covers              REAL,               -- NULL = unknown, 0 = closed
    order_count         INTEGER,
    reservations_covers REAL,
    temp_high_f         REAL,
    precip_in           REAL,
    is_holiday          INTEGER,
    holiday_name        TEXT,
    is_local_event      REAL,               -- continuous event pressure
    event_name          TEXT,
    is_promo            INTEGER,
    source              TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

-- Which dataset this store holds. `daily_covers` is keyed by date alone, so
-- pointing the service at a different restaurant's export does not replace
-- the old history -- it merges into it, and the model silently trains on two
-- businesses at once. This lets `ingest` notice and refuse.
CREATE TABLE IF NOT EXISTS dataset_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forecast_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at    TEXT NOT NULL,
    target_date     TEXT NOT NULL,  -- ISO YYYY-MM-DD
    forecast_covers REAL NOT NULL,
    features        TEXT,           -- JSON blob of the inputs behind the number
    actual_covers   REAL,           -- filled in later, once the day happens
    source          TEXT
);

CREATE INDEX IF NOT EXISTS idx_forecast_log_target
    ON forecast_log (target_date);
"""


def _as_float(value) -> Optional[float]:
    """None for anything missing -- SQLite stores that as NULL ("unknown")."""
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Store:
    """All database access for the service."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            # WAL keeps a scheduled ingest from blocking a concurrent forecast read.
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # --- dataset identity --------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM dataset_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO dataset_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, value, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    # --- daily covers ------------------------------------------------------

    def upsert_daily_covers(self, df: pd.DataFrame, source: str) -> int:
        """Insert or update one row per date. Returns the number of rows written.

        A later ingest wins for a date it covers -- Toast can revise a business
        date after the fact (late voids, reopened checks), and the newest read
        is the truthful one.
        """
        if df.empty:
            return 0

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        present = [c for c in FEATURE_COLUMNS if c in df.columns]
        columns = ["date", "covers", "order_count"] + present + ["source", "updated_at"]

        rows = []
        for record in df.to_dict("records"):
            row = [
                pd.Timestamp(record["date"]).date().isoformat(),
                _as_float(record.get("covers")),
                _as_int(record.get("order_count")),
            ]
            for col in present:
                value = record.get(col)
                if col in TEXT_COLUMNS:
                    row.append(None if value is None or pd.isna(value) else str(value))
                elif col in INTEGER_FLAG_COLUMNS:
                    row.append(_as_int(value))
                else:
                    # Includes is_local_event, which is continuous.
                    row.append(_as_float(value))
            row.extend([source, now])
            rows.append(tuple(row))

        placeholders = ", ".join("?" for _ in columns)
        # COALESCE on the feature columns: a source that does not know a value
        # must not blank out one an earlier ingest or enrichment already found.
        updates = ",\n                    ".join(
            [
                "covers      = excluded.covers",
                "order_count = COALESCE(excluded.order_count, daily_covers.order_count)",
            ]
            + [f"{c} = COALESCE(excluded.{c}, daily_covers.{c})" for c in present]
            + ["source      = excluded.source", "updated_at  = excluded.updated_at"]
        )

        with self._connect() as conn:
            conn.executemany(
                f"""
                INSERT INTO daily_covers ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(date) DO UPDATE SET
                    {updates}
                """,
                rows,
            )
        return len(rows)

    def load_daily_covers(self) -> pd.DataFrame:
        """All stored covers as a frame with `date` (datetime64), `covers` and
        whichever feature columns have been populated."""
        columns = ["date", "covers", "order_count"] + list(FEATURE_COLUMNS) + ["source"]
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT {', '.join(columns)} FROM daily_covers ORDER BY date"
            )
            rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        # Drop feature columns that are entirely empty so downstream code can
        # tell "never populated" from "populated with nulls".
        for col in FEATURE_COLUMNS:
            if df[col].isna().all():
                df = df.drop(columns=[col])
        return df

    def covers_coverage(self) -> Dict:
        """First date, last date and day count -- for /health."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS days "
                "FROM daily_covers WHERE covers IS NOT NULL"
            ).fetchone()
        return {
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "days": row["days"] or 0,
        }

    # --- forecast log ------------------------------------------------------

    def log_forecasts(self, forecasts: Iterable[Dict], source: str) -> int:
        """Append forecasts to the log, keeping the feature inputs alongside."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for f in forecasts:
            features = {k: v for k, v in f.items() if k not in ("date", "forecast_covers")}
            rows.append(
                (now, f["date"], float(f["forecast_covers"]), json.dumps(features), source)
            )
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO forecast_log "
                "(predicted_at, target_date, forecast_covers, features, source) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def list_forecasts(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        latest_per_date: bool = True,
        limit: int = 500,
    ) -> List[Dict]:
        """Logged forecasts, most recent target dates first.

        With `latest_per_date`, only the newest forecast for each target date is
        returned -- re-forecasting a day should not clutter the history.
        """
        clauses, params = [], []
        if start:
            clauses.append("target_date >= ?")
            params.append(start)
        if end:
            clauses.append("target_date <= ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        if latest_per_date:
            sql = f"""
                SELECT f.* FROM forecast_log f
                JOIN (
                    SELECT target_date, MAX(id) AS id
                    FROM forecast_log {where}
                    GROUP BY target_date
                ) latest ON latest.id = f.id
                ORDER BY f.target_date DESC
                LIMIT ?
            """
        else:
            sql = f"SELECT * FROM forecast_log {where} ORDER BY target_date DESC, id DESC LIMIT ?"

        params.append(int(limit))
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

        for row in rows:
            try:
                row["features"] = json.loads(row["features"]) if row["features"] else {}
            except json.JSONDecodeError:
                row["features"] = {}
        return rows

    def record_actual(self, target_date: str, actual_covers: float) -> int:
        """Attach the real covers count to every forecast for that date, and to
        the covers history, so forecast-vs-actual can be compared later.

        Returns the number of forecast rows updated.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE forecast_log SET actual_covers = ? WHERE target_date = ?",
                (float(actual_covers), target_date),
            )
            updated = cur.rowcount
            # Only `covers` is asserted here -- any weather / holiday values
            # already recorded for that date stay as they are.
            conn.execute(
                """
                INSERT INTO daily_covers (date, covers, source, updated_at)
                VALUES (?, ?, 'actual', ?)
                ON CONFLICT(date) DO UPDATE SET
                    covers     = excluded.covers,
                    source     = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (target_date, float(actual_covers), now),
            )
        return updated

    def accuracy_report(self, limit: int = 200) -> Dict:
        """Forecast-vs-actual error over the logged days that have both."""
        with self._connect() as conn:
            rows = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT f.target_date, f.forecast_covers, f.actual_covers
                    FROM forecast_log f
                    JOIN (
                        SELECT target_date, MAX(id) AS id FROM forecast_log
                        WHERE actual_covers IS NOT NULL GROUP BY target_date
                    ) latest ON latest.id = f.id
                    ORDER BY f.target_date DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            ]

        if not rows:
            return {"days_compared": 0, "mae": None, "mape": None, "days": []}

        errors = [abs(r["forecast_covers"] - r["actual_covers"]) for r in rows]
        pcts = [
            abs(r["forecast_covers"] - r["actual_covers"]) / r["actual_covers"] * 100
            for r in rows
            if r["actual_covers"]
        ]
        return {
            "days_compared": len(rows),
            "mae": round(sum(errors) / len(errors), 1),
            "mape": round(sum(pcts) / len(pcts), 1) if pcts else None,
            "days": [
                {
                    "date": r["target_date"],
                    "forecast_covers": r["forecast_covers"],
                    "actual_covers": r["actual_covers"],
                    "error": round(r["forecast_covers"] - r["actual_covers"], 1),
                }
                for r in rows
            ],
        }
