"""HTTP endpoints.

Thin wrappers over `CoversService`: validate, call, translate errors into
status codes. No business logic lives here.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app import schemas
from app.service import CoversService, ServiceError
from app.sources.base import SourceNotConfigured

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service(request: Request) -> CoversService:
    """The shared service instance, created at startup in `app.main`."""
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - startup always sets this
        raise HTTPException(status_code=503, detail="Service is not initialised yet.")
    return service


@router.get("/health", response_model=schemas.HealthResponse, tags=["ops"])
def health(
    check_source: bool = Query(
        False,
        description="Also make a live call to the data source to confirm credentials work.",
    ),
    service: CoversService = Depends(get_service),
) -> Dict:
    """Configuration, stored-data coverage, and optionally a live source check."""
    return service.health(check_source=check_source)


@router.post(
    "/ingest",
    response_model=schemas.IngestResponse,
    tags=["data"],
    responses={400: {"model": schemas.ErrorResponse}, 502: {"model": schemas.ErrorResponse}},
)
def ingest(
    body: schemas.IngestRequest,
    service: CoversService = Depends(get_service),
) -> Dict:
    """Pull covers history from the configured source into the store.

    Safe to re-run: a date already stored is updated, not duplicated. Run it on
    a schedule (cron, a worker) to keep the forecast current.
    """
    if service.settings.source == "toast" and not (body.start and body.end):
        raise HTTPException(
            status_code=400,
            detail=(
                "The Toast source needs both `start` and `end` -- it fetches one "
                "business date per request, so an unbounded backfill is not safe."
            ),
        )
    try:
        return service.ingest(start=body.start, end=body.end)
    except SourceNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ServiceError as exc:
        # The source was configured but could not deliver -- upstream's problem.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/forecast",
    response_model=schemas.ForecastResponse,
    tags=["forecast"],
    responses={409: {"model": schemas.ErrorResponse}},
)
def forecast(
    body: schemas.ForecastRequest,
    service: CoversService = Depends(get_service),
) -> Dict:
    """Forecast covers, with optional per-day overrides.

    Use this form when you know something the APIs do not -- a promo, a private
    buyout, a street closure.
    """
    overrides = {o.date.isoformat(): o.to_feature_overrides() for o in body.overrides}
    try:
        return service.forecast(
            days=body.days,
            start=body.start,
            overrides_by_date=overrides,
            log=body.log,
            refresh=body.refresh,
        )
    except SourceNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ServiceError as exc:
        # Not enough history is a state problem, not a bad request.
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/forecast",
    response_model=schemas.ForecastResponse,
    tags=["forecast"],
    responses={409: {"model": schemas.ErrorResponse}},
)
def forecast_get(
    days: int = Query(7, ge=1, le=schemas.MAX_FORECAST_DAYS),
    start: Optional[dt.date] = Query(
        None,
        description="First day to forecast; defaults to the day after the last history day.",
    ),
    log: bool = Query(True, description="Record these forecasts in the forecast log."),
    refresh: bool = Query(False, description="Refit the model before forecasting."),
    service: CoversService = Depends(get_service),
) -> Dict:
    """The next N days, no body required -- the convenient read for a dashboard."""
    try:
        return service.forecast(days=days, start=start, log=log, refresh=refresh)
    except SourceNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/forecasts", response_model=List[schemas.LoggedForecast], tags=["forecast"])
def list_forecasts(
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    all_revisions: bool = Query(
        False, description="Include superseded forecasts, not just the latest per date."
    ),
    limit: int = Query(500, ge=1, le=5000),
    service: CoversService = Depends(get_service),
) -> List[Dict]:
    """Previously logged forecasts, newest target date first."""
    return service.store.list_forecasts(
        start=start.isoformat() if start else None,
        end=end.isoformat() if end else None,
        latest_per_date=not all_revisions,
        limit=limit,
    )


@router.post("/actuals", response_model=schemas.ActualResponse, tags=["data"])
def record_actual(
    body: schemas.ActualRequest,
    service: CoversService = Depends(get_service),
) -> Dict:
    """Record the real covers count for a day, so accuracy can be tracked.

    With the Toast source this is usually unnecessary -- the next ingest picks
    the real number up on its own.
    """
    return service.record_actual(body.date, body.actual_covers)


@router.get("/accuracy", response_model=schemas.AccuracyResponse, tags=["forecast"])
def accuracy(
    limit: int = Query(200, ge=1, le=2000),
    service: CoversService = Depends(get_service),
) -> Dict:
    """Forecast-vs-actual error across every logged day that has both."""
    return service.accuracy(limit=limit)


@router.get(
    "/geocode",
    response_model=schemas.GeocodeResponse,
    tags=["ops"],
    responses={404: {"model": schemas.ErrorResponse}},
)
def geocode(
    address: str = Query(
        ..., min_length=3, description="A full street address, e.g. '123 Main St, Miami, FL'."
    ),
) -> Dict:
    """Resolve a street address to coordinates, for pinning down the origin
    used by event distance grading.

    Run this once, then put the result in `.env`. Event weighting measures from
    this point, so a city-level centroid makes "2 miles away" versus "8 miles
    away" meaningless.
    """
    from app.core import enrichment

    hit = enrichment.geocode_address(address)
    if hit is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not geocode {address!r}. Try adding the city, state and postal code.",
        )
    return {
        "query": address,
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
        "display_name": hit.get("display_name"),
    }


@router.get(
    "/events",
    response_model=schemas.EventPreviewResponse,
    tags=["forecast"],
    responses={409: {"model": schemas.ErrorResponse}},
)
def events_preview(
    date: dt.date = Query(..., description="The date to score events for."),
    radius_miles: Optional[float] = Query(
        None, gt=0, le=100, description="Overrides the configured search radius."
    ),
    service: CoversService = Depends(get_service),
) -> Dict:
    """Show what the event scorer sees for a date, and why.

    Returns each nearby event with its distance, its distance weight, its size
    weight and the proxy that produced it -- so an event-driven forecast can be
    checked without rerunning the model.
    """
    location = service.location
    if location is None:
        raise HTTPException(
            status_code=409,
            detail="No restaurant location configured, so event distance cannot be measured. "
            "Set COVERS_LATITUDE / COVERS_LONGITUDE (see GET /geocode).",
        )

    from app.core import enrichment

    radius = radius_miles or location.event_radius_miles
    pressure = enrichment.event_pressure_for_date(
        pd.Timestamp(date),
        location,
        radius_miles=radius,
        pressure_cap=location.event_pressure_cap,
    )
    return {
        "date": date.isoformat(),
        "radius_miles": radius,
        "origin_latitude": location.latitude,
        "origin_longitude": location.longitude,
        "available": pressure is not None,
        "detail": None if pressure is None else pressure.to_dict(),
    }


@router.get("/recurring-events", tags=["forecast"])
def recurring_events(service: CoversService = Depends(get_service)) -> Dict:
    """The recurring events configured for this restaurant.

    These are the calendar facts no ticketing feed knows -- a weekly market,
    school terms, an annual festival. Add or change them by running:

        python3 scripts/setup_recurring_events.py

    The file is read per request, so changes apply to the next forecast with
    no restart.
    """
    from app.core import recurring

    path = service.settings.recurring_events_path
    event_set = recurring.load(path)
    today = dt.date.today()

    # A short preview makes a misconfigured rule obvious immediately.
    upcoming = []
    for offset in range(28):
        day = today + dt.timedelta(days=offset)
        hits = event_set.occurrences_on(day)
        if hits:
            upcoming.append(
                {
                    "date": day.isoformat(),
                    "day_of_week": day.strftime("%A"),
                    "events": [
                        {"name": h.name, "size": h.size, "size_weight": h.size_weight}
                        for h in hits
                    ],
                }
            )

    return {
        "config_path": str(path),
        "configured": len(event_set.events),
        "events": [
            {**e.to_dict(), "description": e.describe(), "size_weight": e.size_weight}
            for e in event_set.events
        ],
        "next_28_days": upcoming,
        "hint": (
            "Run scripts/setup_recurring_events.py to add or change these."
            if event_set.events
            else "None configured. Run scripts/setup_recurring_events.py to add some -- "
            "a weekly market or school terms are often the cheapest accuracy win, "
            "since no ticketing API reports them."
        ),
    }


@router.get("/history", tags=["data"])
def history(
    limit: int = Query(400, ge=1, le=5000),
    service: CoversService = Depends(get_service),
) -> Dict:
    """The stored daily covers the model trains on, most recent last."""
    try:
        df = service.history()
    except ServiceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    cols = [
        c
        for c in (
            "date",
            "covers",
            "day_of_week",
            "is_weekend",
            "is_closed",
            "covers_missing",
            "is_holiday",
            "holiday_name",
            "temp_high_f",
            "precip_in",
            "is_rain",
            "reservations_covers",
        )
        if c in df.columns
    ]
    tail = df[cols].tail(limit).copy()
    tail["date"] = tail["date"].dt.strftime("%Y-%m-%d")
    # JSON has no NaN; `where(notna)` turns them into nulls.
    records = tail.astype(object).where(tail.notna(), None).to_dict("records")
    return {"days": len(records), "history": records}
