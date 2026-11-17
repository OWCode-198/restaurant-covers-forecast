"""Weather, public-holiday and local-event lookups.

Ported from `put in terminal/covers_forecast_cli.py` with the interactive
prompts removed, so the same feature values the CLI produces are available to
the service. Every function degrades to None / empty rather than raising, so a
third-party outage never takes the forecast endpoint down.

Sources (all free; only Ticketmaster needs a key):
  - Open-Meteo archive  -- historical daily weather
  - Open-Meteo forecast -- today + ~16 days ahead
  - Nager.Date          -- public holidays by country
  - Ticketmaster        -- upcoming ticketed events nearby (no historical archive)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from app.config import Location
from app.core import events as events_core

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOLIDAYS_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
TICKETMASTER_URL = "https://app.ticketmaster.com/discovery/v2/events.json"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's policy requires a User-Agent identifying the application.
NOMINATIM_USER_AGENT = "predictive-covers-api/0.2 (restaurant covers forecasting)"

# Used when no weather data at all is available for a date range, matching the
# CLI's behaviour. A blank temperature reads as 0F to the model, which would
# skew the forecast badly almost anywhere.
NEUTRAL_TEMP_F = 75.0
NEUTRAL_PRECIP_IN = 0.0

_LOOKUP_ERRORS = (requests.RequestException, ValueError, KeyError)

# A failed weather lookup does not raise -- it falls back to NEUTRAL_TEMP_F, so
# one upstream blip silently replaces real weather with a constant and the
# feature stops working without anyone noticing. Open-Meteo returns the
# occasional 502, so retry the transient statuses before accepting that.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_LOOKUP_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.5


# Forecasting seven days calls weather_for_date and holiday_for_date once per
# day, and each of those re-fetches a payload covering *all* of the days -- the
# same 16-day forecast, and the same year of holidays, fetched seven times
# over. That is 14 redundant round trips on every page view. These collapse
# them to two.
_TTL_CACHE: Dict = {}
WEATHER_FORECAST_TTL_SECONDS = 3600.0  # forecasts are revised hourly at most
HOLIDAY_TTL_SECONDS = 86400.0  # a year's public holidays do not move


def _cached(key, ttl: float, producer):
    """Memoise `producer()` under `key` for `ttl` seconds.

    Deliberately tiny and process-local: this exists to stop one request making
    the same call seven times, not to be a durable cache.
    """
    now = time.time()
    hit = _TTL_CACHE.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = producer()
    _TTL_CACHE[key] = (now, value)
    return value


def clear_caches() -> None:
    """Drop memoised lookups (for tests, and for a long-lived idle process)."""
    _TTL_CACHE.clear()


def _get_with_retry(url: str, params: Optional[Dict] = None, timeout: int = 30):
    """GET, retrying the statuses that mean "try again" rather than "no"."""
    for attempt in range(1, MAX_LOOKUP_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == MAX_LOOKUP_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code in RETRY_STATUSES and attempt < MAX_LOOKUP_ATTEMPTS:
            logger.warning(
                "%s returned %s; retrying (%d/%d)",
                url,
                resp.status_code,
                attempt,
                MAX_LOOKUP_ATTEMPTS,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
        return resp

    raise RuntimeError(f"No response from {url}")  # pragma: no cover - defensive


def geocode_location(query: str, count: int = 5) -> List[dict]:
    """Look up candidate lat/lon for a place name, most populous first."""
    resp = requests.get(
        GEOCODE_URL,
        params={"name": query, "count": count, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return sorted(results, key=lambda r: r.get("population", 0), reverse=True)


def fetch_weather_historical(
    lat: float, lon: float, start_date: str, end_date: str, timezone: str = "auto"
) -> pd.DataFrame:
    resp = _get_with_retry(
        WEATHER_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": timezone,
        },
        timeout=30,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temp_high_f": daily["temperature_2m_max"],
            "precip_in": daily["precipitation_sum"],
        }
    )


def fetch_weather_forecast(
    lat: float, lon: float, timezone: str = "auto", days: int = 16
) -> pd.DataFrame:
    """Live forecast (today + up to `days` ahead). Memoised: one 7-day forecast
    asks for this same payload once per target day."""
    return _cached(
        ("weather_forecast", lat, lon, timezone, days),
        WEATHER_FORECAST_TTL_SECONDS,
        lambda: _fetch_weather_forecast_uncached(lat, lon, timezone, days),
    )


def _fetch_weather_forecast_uncached(
    lat: float, lon: float, timezone: str = "auto", days: int = 16
) -> pd.DataFrame:
    resp = _get_with_retry(
        WEATHER_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": timezone,
            "forecast_days": days,
        },
        timeout=30,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temp_high_f": daily["temperature_2m_max"],
            "precip_in": daily["precipitation_sum"],
        }
    )


def fetch_holidays(country_code: str, years: Sequence[int]) -> pd.DataFrame:
    """Public holidays for whole years. Memoised: a 7-day forecast asks for the
    same year seven times, and a year's holidays do not change."""
    return _cached(
        ("holidays", country_code, tuple(sorted(years))),
        HOLIDAY_TTL_SECONDS,
        lambda: _fetch_holidays_uncached(country_code, years),
    )


def _fetch_holidays_uncached(country_code: str, years: Sequence[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        resp = _get_with_retry(
            HOLIDAYS_URL.format(year=year, country=country_code), timeout=15
        )
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        for h in resp.json():
            rows.append({"date": h["date"], "holiday_name": h["localName"]})
    return pd.DataFrame(rows, columns=["date", "holiday_name"])


def fetch_events(
    lat: float,
    lon: float,
    api_key: str,
    start_date: str,
    end_date: str,
    radius_miles: float = 15.0,
) -> List[Dict]:
    """Raw Ticketmaster event objects for a date range near a point.

    Returns the event dicts untouched rather than a name/date summary, because
    the scoring in `app.core.events` needs the venue coordinates, the
    classification segment and the venue metadata. An earlier version of this
    function kept only name and date, which is why distance could not be
    graded at all.

    Ticketmaster Discovery only surfaces current/upcoming events -- there is no
    historical archive -- so this is useful at forecast time, never for
    backfilling training history.
    """
    resp = requests.get(
        TICKETMASTER_URL,
        params={
            "apikey": api_key,
            "latlong": f"{lat},{lon}",
            "radius": int(round(radius_miles)),
            "unit": "miles",
            "startDateTime": f"{start_date}T00:00:00Z",
            "endDateTime": f"{end_date}T23:59:59Z",
            "size": 50,
            "sort": "date,asc",
        },
        timeout=20,
    )
    resp.raise_for_status()
    events = resp.json().get("_embedded", {}).get("events", [])
    # Drop anything with no local date -- it cannot be attributed to a day.
    return [e for e in events if ((e.get("dates") or {}).get("start") or {}).get("localDate")]


# Nominatim matches street names literally -- it does not expand US postal
# abbreviations, so "3409 Main Hwy" finds nothing while "3409 Main Highway"
# resolves exactly. Expanding these is the difference between the address
# working and failing silently.
STREET_ABBREVIATIONS = {
    "hwy": "Highway",
    "st": "Street",
    "ave": "Avenue",
    "av": "Avenue",
    "blvd": "Boulevard",
    "rd": "Road",
    "dr": "Drive",
    "ln": "Lane",
    "ct": "Court",
    "pl": "Place",
    "pkwy": "Parkway",
    "cir": "Circle",
    "ter": "Terrace",
    "trl": "Trail",
    "sq": "Square",
}


def expand_street_abbreviations(address: str) -> str:
    """Spell out common US street-type abbreviations for the geocoder.

    Only whole words are replaced, and only the street type -- directional
    prefixes like NE or SW are left alone, since Nominatim handles those.
    """
    out = []
    for token in address.split():
        stripped = token.strip(".,").lower()
        replacement = STREET_ABBREVIATIONS.get(stripped)
        if replacement is None:
            out.append(token)
            continue
        # Preserve a trailing comma, which separates address components.
        out.append(replacement + ("," if token.rstrip(".").endswith(",") else ""))
    return " ".join(out)


class GeocodeUnavailable(Exception):
    """The geocoder could not be reached, or refused the request.

    Distinct from "no such address": a transient failure must never be cached
    as a permanent miss, or one rate-limited run poisons the cache for good.
    """


# Nominatim's policy is at most one request per second. Anything that geocodes
# in a loop must respect this or it gets 429s, as happened when every stadium
# in the country was being looked up.
NOMINATIM_MIN_INTERVAL_SECONDS = 1.1
_last_nominatim_call = 0.0


def _throttle_nominatim() -> None:
    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < NOMINATIM_MIN_INTERVAL_SECONDS:
        time.sleep(NOMINATIM_MIN_INTERVAL_SECONDS - elapsed)
    _last_nominatim_call = time.time()


def _geocode_once(query: str) -> Optional[Dict]:
    """One lookup. Returns None if the address is unknown; raises
    GeocodeUnavailable if the geocoder itself could not answer."""
    _throttle_nominatim()
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise GeocodeUnavailable(f"could not reach the geocoder: {exc}") from exc

    if resp.status_code == 429 or resp.status_code >= 500:
        raise GeocodeUnavailable(f"geocoder returned HTTP {resp.status_code}")

    try:
        resp.raise_for_status()
        results = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise GeocodeUnavailable(f"unusable geocoder response: {exc}") from exc

    if not results:
        return None
    hit = results[0]
    address = hit.get("address") or {}
    try:
        return {
            "latitude": float(hit["lat"]),
            "longitude": float(hit["lon"]),
            "display_name": hit.get("display_name"),
            "matched_query": query,
            # Captured so event sources can cheaply pre-filter venues by region
            # instead of geocoding every one of them.
            "city": address.get("city") or address.get("town") or address.get("village"),
            "state": address.get("state"),
            "state_code": (address.get("ISO3166-2-lvl4") or "").split("-")[-1] or None,
            "country_code": (address.get("country_code") or "").upper() or None,
        }
    except (KeyError, TypeError, ValueError):
        return None


def geocode_address(address: str) -> Optional[Dict]:
    """Street-level geocoding via OpenStreetMap Nominatim.

    Open-Meteo's geocoder resolves place names, not street addresses, so it
    returns a city centroid -- useless when the whole point is grading an event
    two miles away versus eight. Nominatim handles a full street address.

    Several progressively looser forms are tried, because Nominatim is strict:
    it wants street types spelled out, and a neighbourhood in the city slot
    ("Coconut Grove" rather than "Miami") makes it return nothing at all.

    Nominatim's usage policy requires an identifying User-Agent and at most one
    request per second. This is meant to be called rarely -- once, to find your
    coordinates, which you then put in `.env`.
    """
    address = address.strip()
    if not address:
        return None

    candidates = [address]

    expanded = expand_street_abbreviations(address)
    if expanded != address:
        candidates.append(expanded)

    # Drop a middle component: a neighbourhood sitting where the city should be
    # ("3409 Main Highway, Coconut Grove, FL 33133") defeats an exact match.
    parts = [p.strip() for p in expanded.split(",") if p.strip()]
    if len(parts) >= 3:
        candidates.append(", ".join([parts[0]] + parts[2:]))

    for query in candidates:
        hit = _geocode_once(query)  # throttled internally; may raise
        if hit is not None:
            if query != address:
                logger.info("Geocoded %r via the reformulated query %r", address, query)
            return hit

    logger.info("No geocoding match for %r after %d attempt(s).", address, len(candidates))
    return None


# ---------------------------------------------------------------------------
# Per-date helpers used at forecast time
# ---------------------------------------------------------------------------


def event_pressure_for_date(
    day: pd.Timestamp,
    location: Location,
    radius_miles: float = 15.0,
    pressure_cap: float = events_core.PRESSURE_CAP,
    settings=None,
) -> Optional[events_core.EventPressure]:
    """Distance- and size-weighted event pressure for a date.

    Aggregates every enabled source -- Ticketmaster, ESPN, and the operator's
    recurring-event calendar -- then scores them together.

    Returns an `EventPressure` (possibly with pressure 0.0 and no events), or
    None when no source is available at all, so the caller can tell "no events
    nearby" from "could not check".
    """
    from app.config import get_settings
    from app.core import event_sources

    settings = settings if settings is not None else get_settings()

    sources = event_sources.build_sources(settings, location)
    if not sources:
        # Nothing configured: no Ticketmaster key, ESPN disabled, no recurring
        # rules. "Unknown", not "no events".
        return None

    candidates = event_sources.gather_candidates(day.date(), location, settings)
    return events_core.score_events(
        candidates,
        origin_lat=location.latitude,
        origin_lon=location.longitude,
        radius_miles=radius_miles,
        pressure_cap=pressure_cap,
    )


def events_for_date(day: pd.Timestamp, location: Location) -> Optional[List[str]]:
    """Event names for a date, [] if none found, or None if unavailable.

    Kept as a thin convenience wrapper; the forecast path uses
    `event_pressure_for_date` so it gets the distance and size weighting too.
    """
    pressure = event_pressure_for_date(day, location)
    if pressure is None:
        return None
    return [e.name for e in pressure.events]


def weather_for_date(day: pd.Timestamp, location: Location) -> Optional[pd.Series]:
    """Try the live forecast first (today + ~16 days), then the historical
    archive (any past date). Returns None if neither is available."""
    try:
        fc = fetch_weather_forecast(location.latitude, location.longitude, location.timezone)
        match = fc[fc["date"] == day]
        if len(match):
            return match.iloc[0]
    except _LOOKUP_ERRORS as exc:
        logger.warning("Weather forecast lookup failed for %s: %s", day.date(), exc)

    iso = day.date().isoformat()
    try:
        hist = fetch_weather_historical(
            location.latitude, location.longitude, iso, iso, location.timezone
        )
        return hist.iloc[0] if len(hist) else None
    except _LOOKUP_ERRORS as exc:
        logger.warning("Weather archive lookup failed for %s: %s", iso, exc)
        return None


def holiday_for_date(day: pd.Timestamp, location: Location) -> Optional[str]:
    """Holiday name for a date, "" if it is not a holiday, or None if the
    lookup failed (caller should skip rather than guess)."""
    try:
        holidays = fetch_holidays(location.country_code, [day.year])
    except _LOOKUP_ERRORS as exc:
        logger.warning("Holiday lookup failed for %s: %s", day.date(), exc)
        return None
    if not len(holidays):
        return ""
    holidays = holidays.copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    match = holidays[holidays["date"] == day]
    return str(match.iloc[0]["holiday_name"]) if len(match) else ""


def estimate_seasonal_weather(target: pd.Timestamp, hist: pd.DataFrame) -> Tuple[float, bool]:
    """Fallback when the live forecast is unreachable: use this calendar month's
    history, then overall history, then a neutral default."""
    if "temp_high_f" not in hist.columns or not len(hist):
        return NEUTRAL_TEMP_F, False

    same_month = hist[hist["date"].dt.month == target.month]
    pool = same_month if same_month["temp_high_f"].notna().any() else hist

    temps = pool["temp_high_f"].dropna()
    temp = float(temps.mean()) if len(temps) else NEUTRAL_TEMP_F

    if "is_rain" in pool.columns and pool["is_rain"].notna().any():
        rainy = bool(pool["is_rain"].dropna().mean() > 0.5)
    else:
        rainy = False
    return temp, rainy


def enrich_history(df: pd.DataFrame, location: Optional[Location]) -> pd.DataFrame:
    """Fill weather and holiday columns across a daily history frame.

    Mirrors the CLI's `build_clean_dataset` enrichment step. Values already
    present in `df` win, so a CSV that already carries weather (as this
    project's own exports do) is not re-fetched.
    """
    df = df.copy()
    for col in ("temp_high_f", "precip_in"):
        if col not in df.columns:
            df[col] = np.nan

    if df["temp_high_f"].isna().any() and location is not None:
        try:
            weather = fetch_weather_historical(
                location.latitude,
                location.longitude,
                df["date"].min().date().isoformat(),
                df["date"].max().date().isoformat(),
                location.timezone,
            )
            df = df.merge(weather, on="date", how="left", suffixes=("", "_fetched"))
            for col in ("temp_high_f", "precip_in"):
                fetched = f"{col}_fetched"
                if fetched in df.columns:
                    df[col] = df[col].fillna(df[fetched])
                    df = df.drop(columns=[fetched])
        except _LOOKUP_ERRORS as exc:
            logger.warning("Historical weather enrichment failed: %s", exc)

    if df["temp_high_f"].isna().all():
        logger.info("No weather data for this range; using neutral defaults.")
        df["temp_high_f"] = NEUTRAL_TEMP_F
        df["precip_in"] = NEUTRAL_PRECIP_IN
    elif df["temp_high_f"].isna().any():
        df[["temp_high_f", "precip_in"]] = df[["temp_high_f", "precip_in"]].interpolate(
            limit_direction="both"
        )

    if "is_holiday" not in df.columns:
        df["is_holiday"] = np.nan
    if "holiday_name" not in df.columns:
        df["holiday_name"] = np.nan

    # `prepare_history` runs before this and fills is_holiday with 0, so an
    # isna() check alone never fires for CSV-sourced history and holidays were
    # silently never fetched at all. An entirely empty holiday_name column is
    # the reliable signal that this range has not been enriched yet.
    never_enriched = df["holiday_name"].isna().all()
    if (never_enriched or df["is_holiday"].isna().any()) and location is not None:
        try:
            years = range(int(df["date"].min().year), int(df["date"].max().year) + 1)
            holidays = fetch_holidays(location.country_code, list(years))
        except _LOOKUP_ERRORS as exc:
            logger.warning("Holiday enrichment failed: %s", exc)
            holidays = pd.DataFrame(columns=["date", "holiday_name"])

        if len(holidays):
            holidays["date"] = pd.to_datetime(holidays["date"])
            # Some countries have more than one named holiday on the same date
            # (e.g. a federal plus a regional one). Collapse to one row per date
            # so the merge cannot duplicate a date -- duplicates would break the
            # lag / rolling-average lookups, which expect one value per date.
            holidays = holidays.groupby("date", as_index=False)["holiday_name"].agg(
                lambda names: " / ".join(sorted(set(names)))
            )
            df = df.merge(holidays, on="date", how="left", suffixes=("", "_fetched"))
            if "holiday_name_fetched" in df.columns:
                df["holiday_name"] = df["holiday_name"].fillna(df["holiday_name_fetched"])
                df = df.drop(columns=["holiday_name_fetched"])
            df["is_holiday"] = df["holiday_name"].notna().astype(int)

    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # Safety net: every merge above should preserve exactly one row per date.
    if df["date"].duplicated().any():
        logger.warning("Duplicate dates after enrichment; keeping the first of each.")
        df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)

    return df
