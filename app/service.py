"""Application service layer.

Holds the wiring between data sources, the store and the model so the HTTP
routers stay thin. One instance is created at startup and shared.

The fitted model is cached: fitting is a few hundred milliseconds of pandas
work, and refitting on every request would make a dashboard that polls
/forecast needlessly slow. The cache is invalidated whenever covers data
changes (an ingest, or a recorded actual).
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from app.config import Location, Settings, get_location
from app.core import model as model_core
from app.sources.base import CoversSource, SourceError, SourceNotConfigured
from app.sources.csv_source import CsvCoversSource
from app.sources.toast import ToastCoversSource
from app.store import Store

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """A request could not be satisfied (surfaced as a 4xx/5xx by the routers)."""


def build_source(settings: Settings) -> CoversSource:
    """Instantiate the configured covers source."""
    if settings.source == "toast":
        return ToastCoversSource(settings)
    return CsvCoversSource(settings.csv_path)


class CoversService:
    def __init__(
        self,
        settings: Settings,
        store: Optional[Store] = None,
        source: Optional[CoversSource] = None,
        location: Optional[Location] = None,
    ) -> None:
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self._source = source
        self._location = location if location is not None else get_location(settings)
        self._model: Optional[model_core.FittedModel] = None
        # Read-only forecasts are memoised. Every call re-runs enrichment for
        # each target day -- weather, holidays, events -- which dominates the
        # request: profiling /today showed ~1.5s per view, almost all of it
        # repeated lookups for days whose answer had not changed. The page is
        # read-only and the forecast only moves when new covers arrive, so
        # serving repeat views from here is both correct and ~1000x faster.
        self._forecast_cache: Dict = {}
        # Fitting mutates the cache; serialise it so two concurrent requests
        # cannot fit at once and hand back a half-built model.
        self._lock = threading.Lock()

    # --- source ------------------------------------------------------------

    @property
    def source(self) -> CoversSource:
        """Built lazily so a misconfigured Toast source fails on the request
        that needs it, not at import time."""
        if self._source is None:
            self._source = build_source(self.settings)
        return self._source

    @property
    def location(self) -> Optional[Location]:
        return self._location

    def invalidate_model(self) -> None:
        with self._lock:
            self._model = None
            self._forecast_cache = {}

    # --- ingest ------------------------------------------------------------

    def _source_fingerprint(self) -> str:
        """Identifies which restaurant's data this store is meant to hold."""
        if self.settings.source == "toast":
            return f"toast:{self.settings.toast_restaurant_guid}"
        return f"csv:{Path(self.settings.csv_path).resolve()}"

    def _guard_dataset_identity(self) -> None:
        """Refuse to merge one restaurant's history into another's.

        `daily_covers` is keyed by date alone, so re-pointing the service at a
        different export does not replace the old rows -- it merges into them.
        This actually happened here: the store ended up holding 805 days from
        two restaurants, averaging 146 covers before one date and 61 after, and
        every model fitted on it was training across the seam.
        """
        seen = self.store.get_meta("source_fingerprint")
        current = self._source_fingerprint()
        if seen is not None and seen != current:
            raise ServiceError(
                f"This store already holds data from {seen}, and you are ingesting "
                f"from {current}. Mixing two restaurants in one database would "
                "train the model across both. Point COVERS_DB_PATH at a new file "
                "for this restaurant, or delete the existing one to start over."
            )
        self.store.set_meta("source_fingerprint", current)

    def ingest(self, start: Optional[dt.date] = None, end: Optional[dt.date] = None) -> Dict:
        """Pull covers from the configured source into the store."""
        self._guard_dataset_identity()
        try:
            df = self.source.fetch_daily_covers(start=start, end=end)
        except SourceNotConfigured:
            # Missing configuration is the caller's problem (400), not an
            # upstream failure (502), so it must not be wrapped as a ServiceError.
            raise
        except SourceError as exc:
            raise ServiceError(str(exc)) from exc

        written = self.store.upsert_daily_covers(df, source=self.source.name)
        self.invalidate_model()

        return {
            "source": self.source.name,
            "days_ingested": written,
            "first_date": df["date"].min().date().isoformat(),
            "last_date": df["date"].max().date().isoformat(),
            "covers_total": float(df["covers"].fillna(0).sum()),
        }

    # --- history / model ---------------------------------------------------

    def history(self) -> pd.DataFrame:
        """Stored covers, enriched and normalised, ready for fitting.

        Falls back to reading the source directly when the store is empty, so a
        fresh install can forecast without an explicit ingest step first.
        """
        df = self.store.load_daily_covers()
        if df.empty:
            logger.info("Store is empty; reading history directly from %s.", self.source.name)
            try:
                df = self.source.fetch_daily_covers()
            except SourceNotConfigured:
                raise  # a 400, not a 409 -- see `ingest`
            except SourceError as exc:
                raise ServiceError(
                    f"No stored covers and the {self.source.name} source could not "
                    f"provide any: {exc}"
                ) from exc
            self.store.upsert_daily_covers(df, source=self.source.name)

        # Enrich BEFORE preparing. `prepare_history` backfills the columns that
        # must exist whether or not enrichment ran -- temp_high_f becomes a
        # neutral constant, is_holiday becomes 0 -- and `enrich_history` decides
        # what to fetch by looking for NaNs in exactly those columns. Running it
        # second meant it always saw a full frame and fetched nothing, so for
        # any source without its own weather these features were silently dead.
        if self.settings.enrichment_enabled:
            from app.core import enrichment  # local import keeps startup cheap

            df = enrichment.enrich_history(df, self._location)
        prepared = model_core.prepare_history(df)
        return prepared

    def fitted_model(self, refresh: bool = False) -> model_core.FittedModel:
        with self._lock:
            if self._model is not None and not refresh:
                return self._model
            df = self.history()
            try:
                self._model = model_core.fit(
                    df,
                    min_training_days=self.settings.min_training_days,
                    holdout_days=self.settings.holdout_days,
                )
            except model_core.NotEnoughHistory as exc:
                raise ServiceError(str(exc)) from exc
            return self._model

    # --- forecast ----------------------------------------------------------

    def forecast(
        self,
        days: int = 7,
        start: Optional[dt.date] = None,
        overrides_by_date: Optional[Dict[str, Dict]] = None,
        log: bool = True,
        refresh: bool = False,
    ) -> Dict:
        model = self.fitted_model(refresh=refresh)

        # Only read-only calls are cacheable: `log=True` writes to the forecast
        # log, and serving that from a cache would silently drop the write.
        # Keyed on the model's own last history date so fresh data always
        # misses, and on today so a page left open overnight re-forecasts.
        cacheable = log is False and overrides_by_date is None and not refresh
        cache_key = (
            days,
            start.isoformat() if start else None,
            model.last_history_date.date().isoformat(),
            dt.date.today().isoformat(),
        )
        if cacheable:
            hit = self._forecast_cache.get(cache_key)
            if hit is not None:
                return hit

        results = model_core.forecast_days(
            model,
            self._location,
            days=days,
            start=pd.Timestamp(start) if start else None,
            overrides_by_date=overrides_by_date,
            enrichment_enabled=self.settings.enrichment_enabled,
        )
        if log:
            self.store.log_forecasts(results, source=self.source.name)
            # A fresh forecast supersedes anything memoised for this model.
            self._forecast_cache = {}

        accuracy = None
        if model.accuracy is not None:
            accuracy = {
                "holdout_mae": round(model.accuracy.mae, 1),
                "holdout_mape": round(model.accuracy.mape, 1),
                "holdout_days": model.accuracy.tested_days,
            }

        payload = {
            "source": self.source.name,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "last_history_date": model.last_history_date.date().isoformat(),
            "trained_days": model.trained_days,
            "uses_reservations": model.has_reservations,
            "location": self._location.name if self._location else None,
            "enrichment_enabled": self.settings.enrichment_enabled and self._location is not None,
            "accuracy": accuracy,
            "forecast": results,
        }
        if cacheable:
            self._forecast_cache[cache_key] = payload
        return payload

    # --- actuals -----------------------------------------------------------

    def record_actual(self, target_date: dt.date, actual_covers: float) -> Dict:
        iso = target_date.isoformat()
        updated = self.store.record_actual(iso, actual_covers)
        self.invalidate_model()
        return {
            "date": iso,
            "actual_covers": actual_covers,
            "forecast_rows_updated": updated,
        }

    def accuracy(self, limit: int = 200) -> Dict:
        return self.store.accuracy_report(limit=limit)

    # --- health ------------------------------------------------------------

    def health(self, check_source: bool = False) -> Dict:
        coverage = self.store.covers_coverage()
        out: Dict = {
            "status": "ok",
            "source": self.settings.source,
            "database": str(self.settings.db_path),
            "covers": coverage,
            "location": self._location.name if self._location else None,
            "enrichment_enabled": self.settings.enrichment_enabled and self._location is not None,
            "restaurant_address": self._location.address if self._location else None,
            "location_source": self._location.source if self._location else None,
            "location_precise": self._location.is_precise if self._location else False,
            "warnings": (
                self._location.warnings()
                if self._location is not None
                else [
                    "No restaurant location configured. Weather, holidays and event "
                    "distances are all unavailable until COVERS_LATITUDE / "
                    "COVERS_LONGITUDE are set (resolve them with GET /geocode)."
                ]
            ),
        }

        if self.settings.source == "toast":
            out["toast_configured"] = all(
                [
                    self.settings.toast_hostname,
                    self.settings.toast_client_id,
                    self.settings.toast_client_secret,
                    self.settings.toast_restaurant_guid,
                ]
            )
        else:
            out["csv_path"] = str(self.settings.csv_path)
            out["csv_exists"] = self.settings.csv_path.exists()

        if check_source:
            # Deliberately opt-in: this makes a live network call to Toast.
            try:
                if self.settings.source == "toast":
                    restaurant = self.source.client.fetch_restaurant()
                    out["source_check"] = {
                        "reachable": True,
                        "restaurant_name": restaurant.get("general", {}).get("name")
                        or restaurant.get("name"),
                    }
                else:
                    df = self.source.fetch_daily_covers()
                    out["source_check"] = {"reachable": True, "rows": int(len(df))}
            except (SourceError, ServiceError, AttributeError) as exc:
                out["status"] = "degraded"
                out["source_check"] = {"reachable": False, "error": str(exc)}

        return out
