"""Request and response models for the HTTP API.

These are the contract: they drive validation and the generated OpenAPI docs
at /docs, so a client can be written against them without reading the service
code.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

MAX_FORECAST_DAYS = 60


class IngestRequest(BaseModel):
    """Pull covers history from the configured source into the store.

    The CSV source reads the whole file when no range is given. The Toast
    source requires a range -- it fetches one business date per request, so an
    unbounded backfill could mean thousands of API calls.
    """

    start: Optional[dt.date] = Field(
        default=None, description="First business date to ingest (inclusive)."
    )
    end: Optional[dt.date] = Field(
        default=None, description="Last business date to ingest (inclusive)."
    )

    @model_validator(mode="after")
    def check_range(self) -> "IngestRequest":
        if self.start and self.end and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class IngestResponse(BaseModel):
    source: str
    days_ingested: int
    first_date: str
    last_date: str
    covers_total: float


class DayOverride(BaseModel):
    """Caller-supplied facts about one future date.

    Anything set here beats what the weather / holiday / events lookups return.
    This is how you say "we are running a prix fixe on Valentine's Day" or
    supply a weather estimate when the forecast APIs are unreachable.
    """

    date: dt.date
    is_promo: Optional[int] = Field(default=None, ge=0, le=1)
    is_holiday: Optional[int] = Field(default=None, ge=0, le=1)
    # Graded, not a flag: 0 = nothing nearby, 1 = a typical event day, 2 = about
    # twice that effect. Setting it skips the Ticketmaster lookup for that date,
    # which is how you account for an event Ticketmaster does not list (a street
    # festival, a convention, a nearby wedding).
    is_local_event: Optional[float] = Field(default=None, ge=0, le=10)
    temp_high_f: Optional[float] = Field(default=None, ge=-80, le=140)
    is_rain: Optional[int] = Field(default=None, ge=0, le=1)
    reservations_covers: Optional[float] = Field(default=None, ge=0)

    def to_feature_overrides(self) -> Dict:
        """Only the fields actually supplied, so an unset field does not clobber
        a fetched value with None."""
        return {k: v for k, v in self.model_dump(exclude={"date"}).items() if v is not None}


class ForecastRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=MAX_FORECAST_DAYS)
    start: Optional[dt.date] = Field(
        default=None,
        description="First day to forecast. Defaults to the day after the last day of history.",
    )
    overrides: List[DayOverride] = Field(default_factory=list)
    log: bool = Field(default=True, description="Record these forecasts in the forecast log.")
    refresh: bool = Field(default=False, description="Refit the model before forecasting.")


class ScoredEvent(BaseModel):
    """One nearby event and why it scored the way it did."""

    name: str
    date: str
    venue_name: Optional[str] = None
    distance_miles: Optional[float] = None
    distance_weight: float = Field(
        description="Linear falloff: 1.0 at the restaurant, 0.0 at the search radius."
    )
    size_weight: float = Field(description="Relative event size, where 1.0 is a typical event.")
    size_basis: str = Field(
        description="Which proxy produced size_weight: capacity, segment, segment+genre or "
        "default, optionally with +venue_activity."
    )
    segment: Optional[str] = None
    genre: Optional[str] = None
    capacity: Optional[int] = None
    pressure: float = Field(description="distance_weight x size_weight.")


class EventDetail(BaseModel):
    """The event picture behind one day's forecast."""

    pressure: float
    event_count: int
    nearest_miles: Optional[float] = None
    capped: bool = False
    events: List[ScoredEvent] = Field(default_factory=list)


class DayForecast(BaseModel):
    date: str
    day_of_week: str
    forecast_covers: float
    is_holiday: int
    holiday_name: Optional[str] = None
    # Continuous since event scoring was added: this is the value fed to the
    # model, where 1.0 means a typical event day, 0.0 means nothing nearby,
    # and 2.0 means roughly twice the effect of a typical event.
    is_local_event: float
    event_name: Optional[str] = None
    event_pressure: Optional[float] = None
    event_count: Optional[int] = None
    nearest_event_miles: Optional[float] = None
    event_detail: Optional[EventDetail] = None
    is_promo: int
    temp_high_f: Optional[float] = None
    precip_in: Optional[float] = None
    is_rain: int
    is_rain_heavy: int = 0
    weather_estimated: bool = False
    covers_lag_7: Optional[float] = None
    covers_rollavg_4wk_weekday: Optional[float] = None
    reservations_covers: Optional[float] = None


class ModelAccuracy(BaseModel):
    holdout_mae: float
    holdout_mape: float
    holdout_days: int


class ForecastResponse(BaseModel):
    source: str
    generated_at: str
    last_history_date: str
    trained_days: int
    uses_reservations: bool
    location: Optional[str] = None
    enrichment_enabled: bool
    accuracy: Optional[ModelAccuracy] = None
    forecast: List[DayForecast]


class ActualRequest(BaseModel):
    """Record what actually happened on a day already forecast."""

    date: dt.date
    actual_covers: float = Field(ge=0)


class ActualResponse(BaseModel):
    date: str
    actual_covers: float
    forecast_rows_updated: int


class AccuracyDay(BaseModel):
    date: str
    forecast_covers: float
    actual_covers: float
    error: float


class AccuracyResponse(BaseModel):
    days_compared: int
    mae: Optional[float] = None
    mape: Optional[float] = None
    days: List[AccuracyDay] = Field(default_factory=list)


class CoversCoverage(BaseModel):
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    days: int = 0


class SourceCheck(BaseModel):
    reachable: bool
    restaurant_name: Optional[str] = None
    rows: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    source: str
    database: str
    covers: CoversCoverage
    location: Optional[str] = None
    enrichment_enabled: bool
    restaurant_address: Optional[str] = None
    location_source: Optional[str] = Field(
        default=None,
        description='"explicit" when this deployment set its own coordinates, '
        '"inherited_config" when it fell back to .covers_forecast_config.json.',
    )
    location_precise: bool = Field(
        default=False,
        description="False means event distances are being measured from an origin this "
        "deployment did not set, so they should not be trusted.",
    )
    warnings: List[str] = Field(default_factory=list)
    toast_configured: Optional[bool] = None
    csv_path: Optional[str] = None
    csv_exists: Optional[bool] = None
    source_check: Optional[SourceCheck] = None


class LoggedForecast(BaseModel):
    id: int
    predicted_at: str
    target_date: str
    forecast_covers: float
    actual_covers: Optional[float] = None
    source: Optional[str] = None
    features: Dict = Field(default_factory=dict)


class GeocodeResponse(BaseModel):
    """Street-address lookup, for pinning down the restaurant's coordinates."""

    query: str
    latitude: float
    longitude: float
    display_name: Optional[str] = None
    note: str = (
        "Put these in .env as COVERS_LATITUDE / COVERS_LONGITUDE. Event distance "
        "grading is only as precise as this origin point."
    )


class EventPreviewResponse(BaseModel):
    """What the event scorer sees for one date, without running a forecast."""

    date: str
    radius_miles: float
    origin_latitude: float
    origin_longitude: float
    available: bool = Field(
        description="false when no Ticketmaster key is configured or the lookup failed."
    )
    detail: Optional[EventDetail] = None


class ErrorResponse(BaseModel):
    detail: str
