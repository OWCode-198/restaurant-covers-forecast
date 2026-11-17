"""Service configuration.

Values come from environment variables or a local `.env` file. The defaults
are deliberately the "works right now, no credentials" ones: the CSV data
source and the existing project config file.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The CLI's config file -- reused so the service and the CLI agree on location,
# timezone and the Ticketmaster key instead of duplicating that setup.
CLI_CONFIG_PATH = PROJECT_ROOT / ".covers_forecast_config.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="COVERS_",
        extra="ignore",
    )

    # --- data source -------------------------------------------------------
    # "csv"   -- read covers history from a CSV export (no credentials needed)
    # "toast" -- pull orders live from the Toast API (needs credentials)
    source: Literal["csv", "toast"] = "csv"

    csv_path: Path = PROJECT_ROOT / "data" / "processed" / "covers_daily_toast.csv"

    # --- the page ----------------------------------------------------------
    # /today carries no auth of its own: it is meant to be opened on a phone
    # with no login. Left unset it is open to anyone with the URL, which is
    # right for local work and wrong the moment it is hosted, because it shows
    # a real business's daily trade. Set a password for any deployment
    # reachable from the internet.
    web_username: str = "manager"
    web_password: Optional[str] = None

    # How old the newest day of history may be before the page warns that the
    # forecast is built on stale data. Match it to how often you actually
    # refresh: 2 suits a daily automated ingest, 8 suits a weekly manual
    # export. Set too low it cries wolf every day and people learn to ignore
    # it, which is worse than not warning at all.
    stale_after_days: int = 2

    # --- Toast API ---------------------------------------------------------
    # The hostname is issued with your credentials. Toast does not publish a
    # single public sandbox host, so this must be set explicitly.
    toast_hostname: Optional[str] = None
    toast_client_id: Optional[str] = None
    toast_client_secret: Optional[str] = None
    toast_restaurant_guid: Optional[str] = None
    toast_user_access_type: str = "TOAST_MACHINE_CLIENT"
    # Toast's orders endpoint is paginated and rate limited; keep pages modest.
    toast_page_size: int = Field(default=100, ge=1, le=100)
    toast_timeout_seconds: int = 30

    # --- storage -----------------------------------------------------------
    db_path: Path = PROJECT_ROOT / "data" / "covers_service.sqlite3"

    # --- location / enrichment --------------------------------------------
    # Left as None so they fall back to the CLI config file (see `get_location`).
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None
    country_code: Optional[str] = None
    ticketmaster_api_key: Optional[str] = None

    # The restaurant's street address. Informational, plus the input to
    # GET /geocode -- resolve it once and then set latitude/longitude
    # explicitly, because event distance grading is only as precise as this
    # origin point. A city-level centroid makes "2 miles" vs "8 miles"
    # meaningless.
    restaurant_address: Optional[str] = None

    # State / province, used to pre-filter nationwide event feeds before
    # spending a geocode on each venue. Match the spelling the feed uses
    # ("Florida" for ESPN). Blank disables the filter.
    restaurant_region: Optional[str] = None

    # --- event scoring -----------------------------------------------------
    # How far out to look for events. This doubles as the distance at which an
    # event's weight reaches zero, by construction: weight = 1 - distance/radius.
    event_radius_miles: float = Field(default=15.0, gt=0, le=100)

    # Ceiling on total event pressure. The model's event coefficient was fitted
    # on a binary 0/1 column, so values far above 1 extrapolate beyond the
    # fitted range; this bounds that extrapolation.
    event_pressure_cap: float = Field(default=3.0, ge=1.0, le=10.0)

    # false reverts to the old binary behaviour: any event in radius -> 1.
    event_scoring_enabled: bool = True

    # --- event sources -----------------------------------------------------
    # ESPN's undocumented scoreboard API: no key required, covers sports
    # fixtures including college games. Unofficial, so it can change without
    # notice -- one input among several rather than a hard dependency.
    espn_enabled: bool = True
    # Comma-separated ESPN league paths. Each costs one request per forecast
    # day, so trim this to the leagues that actually play near you.
    espn_leagues: Optional[str] = None

    # Recurring events the operator stated once (a Saturday market, school
    # terms). Per-restaurant; written by scripts/setup_recurring_events.py.
    recurring_events_path: Path = PROJECT_ROOT / "recurring_events.json"

    # ESPN venues have no coordinates, so they are geocoded once and cached
    # here. Safe to delete; it will be rebuilt.
    venue_cache_path: Path = PROJECT_ROOT / "data" / "venue_geocode_cache.json"

    # --- model -------------------------------------------------------------
    min_training_days: int = 30
    # Days of recent history held back for the accuracy self-check.
    holdout_days: int = 55

    enrichment_enabled: bool = True


class Location:
    """Resolved restaurant location used for weather / holiday / event lookups.

    Carries the event-scoring knobs too, so the forecast path does not have to
    thread the whole `Settings` object down into feature building.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone: str,
        country_code: str,
        ticketmaster_api_key: Optional[str] = None,
        name: Optional[str] = None,
        address: Optional[str] = None,
        event_radius_miles: float = 15.0,
        event_pressure_cap: float = 3.0,
        event_scoring_enabled: bool = True,
        source: str = "explicit",
        region: Optional[str] = None,
    ) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.country_code = country_code
        self.ticketmaster_api_key = ticketmaster_api_key
        self.name = name
        self.address = address
        self.event_radius_miles = event_radius_miles
        self.event_pressure_cap = event_pressure_cap
        self.event_scoring_enabled = event_scoring_enabled
        # Where the coordinates came from: "explicit" (this deployment set
        # COVERS_LATITUDE/LONGITUDE) or "inherited_config" (fell back to
        # .covers_forecast_config.json). Event distances are only trustworthy
        # when a deployment set its own origin, so /health says which it is.
        self.source = source
        self.region = region

    @property
    def is_precise(self) -> bool:
        return self.source == "explicit"

    def warnings(self) -> list:
        """Configuration problems worth surfacing on /health."""
        out = []
        if not self.is_precise:
            out.append(
                "Coordinates were inherited from .covers_forecast_config.json rather than "
                "set for this deployment. Event distances are measured from this point, so "
                "set COVERS_LATITUDE / COVERS_LONGITUDE for this restaurant "
                "(resolve them with GET /geocode)."
            )
        if not self.address:
            out.append(
                "No COVERS_RESTAURANT_ADDRESS is set, so there is no record of which "
                "address these coordinates belong to."
            )
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Location({self.name or 'unnamed'} @ {self.latitude},{self.longitude})"


def _load_cli_config() -> dict:
    """Read the CLI's config file, if present. Never raises."""
    if not CLI_CONFIG_PATH.exists():
        return {}
    try:
        with open(CLI_CONFIG_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_location(settings: Optional[Settings] = None) -> Optional[Location]:
    """Resolve the restaurant location from env vars, falling back to the CLI
    config file.

    Returns None when no usable location is configured, so callers can degrade
    to forecasting without weather/holiday features rather than failing.
    """
    settings = settings or get_settings()
    cli = _load_cli_config()

    explicit = settings.latitude is not None and settings.longitude is not None
    lat = settings.latitude if settings.latitude is not None else cli.get("latitude")
    lon = settings.longitude if settings.longitude is not None else cli.get("longitude")
    if lat is None or lon is None:
        return None

    return Location(
        latitude=float(lat),
        longitude=float(lon),
        timezone=settings.timezone or cli.get("timezone") or "auto",
        country_code=settings.country_code or cli.get("country_code") or "US",
        ticketmaster_api_key=(
            settings.ticketmaster_api_key
            or cli.get("ticketmaster_api_key")
            or os.environ.get("TICKETMASTER_API_KEY")
        ),
        name=cli.get("location_name"),
        address=settings.restaurant_address,
        event_radius_miles=settings.event_radius_miles,
        event_pressure_cap=settings.event_pressure_cap,
        event_scoring_enabled=settings.event_scoring_enabled,
        source="explicit" if explicit else "inherited_config",
        region=settings.restaurant_region,
    )
