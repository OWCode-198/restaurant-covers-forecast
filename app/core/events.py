"""Scoring nearby events by distance and size.

The original feature was binary: any ticketed event within 15 miles set
`is_local_event = 1`. That treats a stadium show two blocks away and a club
gig fifteen miles out as the same thing, which they are not.

Events now arrive from several sources (Ticketmaster, ESPN, a recurring-event
calendar) and are normalised into `CandidateEvent` before scoring, so
source-specific JSON shapes stay in the source modules. Each day gets one
continuous `event_pressure`:

    pressure = sum over events of  distance_weight x size_weight

Calibration (the part that matters)
-----------------------------------
The model's `is_local_event` coefficient was fitted on a strictly binary
column: in this project's history, 706 zeros and 26 ones. The coefficient
therefore means "the effect of a typical event day". So `event_pressure` is
deliberately calibrated so that **1.0 means a typical event day** -- one
reasonably notable event, reasonably close. Then:

    pressure 0.0  ->  nothing nearby, no effect
    pressure 1.0  ->  exactly the effect the model was trained on
    pressure 2.0  ->  twice that effect

Feeding an uncalibrated score into a coefficient trained at 0/1 would repeat
the reservations-scaling bug this project already had, where a covers-scale
number in a reservations-scale slot inflated every forecast by ~35%.

Because the coefficient was only ever observed at 0 and 1, anything far above
1 is linear extrapolation beyond the fitted range. `PRESSURE_CAP` bounds that
honestly rather than pretending a festival weekend is 12x a normal event.

Distance
--------
Linear, as specified: full weight at the restaurant's door, decaying straight
to zero at the search radius.

    weight = max(0, 1 - distance / radius)

Size
----
Resolved from the best signal a source can offer, most trustworthy first:

  1. an explicit weight (a recurring event the operator sized themselves)
  2. a real attendance figure (ESPN reports these, but only after a game)
  3. a venue capacity, where a feed carries one
  4. the classification segment / genre (a stadium sport draws more than a
     studio theatre show)
  5. the venue's count of upcoming events, a mild hint at venue scale

Each response reports which basis was used, so a genre guess is never
mistaken for a measured crowd.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EARTH_RADIUS_MILES = 3958.7613
KM_TO_MILES = 0.621371

# A typical event's size weight is 1.0 by definition -- see Calibration above.
BASELINE_SIZE_WEIGHT = 1.0

# The crowd treated as "typical". Impact is assumed to grow with the square
# root, not linearly: a 60,000-seat stadium does not send 12x as many diners as
# a 5,000-seat hall, because only a small spillover fraction eats nearby.
REFERENCE_CAPACITY = 5000.0
MAX_CAPACITY_WEIGHT = 4.0

# Segment weights, used when no crowd figure is available. Relative to a
# typical event (1.0), not absolute crowd sizes.
SEGMENT_WEIGHTS: Dict[str, float] = {
    "sports": 1.6,
    "music": 1.3,
    "arts & theatre": 1.0,
    "arts and theatre": 1.0,
    "community": 1.0,
    "miscellaneous": 0.8,
    "film": 0.5,
}

# Genres that reliably mean a large venue, applied on top of the segment.
GENRE_BOOSTS: Dict[str, float] = {
    "football": 1.25,
    "basketball": 1.15,
    "baseball": 1.15,
    "hockey": 1.1,
    "soccer": 1.1,
    "motorsports": 1.2,
    "motorsports/racing": 1.2,
    "rock": 1.1,
    "pop": 1.1,
    "hip-hop/rap": 1.1,
    "festival": 1.3,
    "comedy": 0.9,
    "classical": 0.85,
    "theatre": 0.85,
}

# A venue hosting hundreds of events a year is usually a major one. Weak
# signal, so it is bounded tightly.
UPCOMING_EVENTS_REFERENCE = 50.0
MAX_UPCOMING_EVENTS_WEIGHT = 1.3
MIN_UPCOMING_EVENTS_WEIGHT = 0.85

# Hard ceiling on total pressure. See Calibration.
PRESSURE_CAP = 3.0

# Pressure below this is treated as no event at all, so a single tiny listing
# at the edge of the radius does not nudge every forecast.
PRESSURE_FLOOR = 0.05


def _as_positive_number(value) -> Optional[float]:
    """Positive float, or None. Booleans are rejected: `True` must never read
    as a crowd of 1."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


@dataclass
class CandidateEvent:
    """One event from any source, normalised for scoring.

    A source supplies whichever location and size signals it has. Everything
    is optional except a name and a date, because sources differ widely in
    what they expose.
    """

    name: str
    date: str
    source: str = "unknown"
    id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # Some feeds state the distance directly (Ticketmaster does on a geo
    # search), which is preferred over recomputing it.
    distance_miles: Optional[float] = None
    attendance: Optional[float] = None
    capacity: Optional[float] = None
    segment: Optional[str] = None
    genre: Optional[str] = None
    venue_name: Optional[str] = None
    upcoming_events: Optional[float] = None
    # Set when the operator sized the event themselves; beats every heuristic.
    size_weight_override: Optional[float] = None

    @classmethod
    def from_ticketmaster(cls, event: Dict) -> "CandidateEvent":
        """Normalise a Ticketmaster Discovery event object."""
        venues = (event.get("_embedded") or {}).get("venues") or []
        venue = venues[0] if venues else {}
        location = venue.get("location") or {}
        classifications = event.get("classifications") or []
        classification = classifications[0] if classifications else {}

        distance = _as_positive_number(venue.get("distance"))
        if distance is not None and str(venue.get("units") or "miles").lower().startswith("km"):
            distance *= KM_TO_MILES

        def _coord(key):
            try:
                return float(location[key])
            except (KeyError, TypeError, ValueError):
                return None

        return cls(
            name=event.get("name") or "Event",
            date=((event.get("dates") or {}).get("start") or {}).get("localDate") or "",
            source="ticketmaster",
            id=event.get("id"),
            latitude=_coord("latitude"),
            longitude=_coord("longitude"),
            distance_miles=distance,
            capacity=_as_positive_number(venue.get("capacity")),
            segment=((classification.get("segment") or {}).get("name")) or None,
            genre=((classification.get("genre") or {}).get("name")) or None,
            venue_name=venue.get("name"),
            upcoming_events=_as_positive_number(
                (venue.get("upcomingEvents") or {}).get("_total")
            ),
        )


@dataclass
class NearbyEvent:
    """One scored event, with the reasoning kept attached."""

    name: str
    date: str
    distance_miles: Optional[float]
    distance_weight: float
    size_weight: float
    pressure: float
    source: str = "unknown"
    venue_name: Optional[str] = None
    segment: Optional[str] = None
    genre: Optional[str] = None
    capacity: Optional[int] = None
    attendance: Optional[int] = None
    size_basis: str = "default"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "date": self.date,
            "source": self.source,
            "venue_name": self.venue_name,
            "distance_miles": (
                None if self.distance_miles is None else round(self.distance_miles, 2)
            ),
            "distance_weight": round(self.distance_weight, 3),
            "size_weight": round(self.size_weight, 3),
            "size_basis": self.size_basis,
            "segment": self.segment,
            "genre": self.genre,
            "capacity": self.capacity,
            "attendance": self.attendance,
            "pressure": round(self.pressure, 3),
        }


@dataclass
class EventPressure:
    """The aggregate score for one date."""

    pressure: float
    event_count: int
    nearest_miles: Optional[float]
    events: List[NearbyEvent] = field(default_factory=list)
    capped: bool = False
    sources: List[str] = field(default_factory=list)

    @property
    def is_local_event(self) -> int:
        """Binary view, for reporting and for callers that want the old flag."""
        return 1 if self.pressure >= PRESSURE_FLOOR else 0

    def summary(self, limit: int = 3) -> str:
        """Human-readable list of the biggest contributors."""
        top = sorted(self.events, key=lambda e: e.pressure, reverse=True)[:limit]
        parts = []
        for e in top:
            if e.distance_miles is None:
                parts.append(e.name)
            else:
                parts.append(f"{e.name} ({e.distance_miles:.1f}mi)")
        extra = len(self.events) - len(top)
        if extra > 0:
            parts.append(f"+{extra} more")
        return ", ".join(parts)

    def to_dict(self, limit: int = 5) -> Dict:
        top = sorted(self.events, key=lambda e: e.pressure, reverse=True)[:limit]
        return {
            "pressure": round(self.pressure, 3),
            "event_count": self.event_count,
            "nearest_miles": None if self.nearest_miles is None else round(self.nearest_miles, 2),
            "capped": self.capped,
            "sources": self.sources,
            "events": [e.to_dict() for e in top],
        }


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


def distance_weight(distance_miles: Optional[float], radius_miles: float) -> float:
    """Linear falloff: 1.0 at the restaurant, 0.0 at the radius edge.

    An unknown distance gets the weight of an event at half the radius -- it is
    somewhere in the search area, so neither full credit nor none.
    """
    if radius_miles <= 0:
        return 0.0
    if distance_miles is None:
        return 0.5
    if distance_miles <= 0:
        return 1.0
    return max(0.0, 1.0 - (distance_miles / radius_miles))


def event_distance_miles(
    candidate: CandidateEvent, origin_lat: float, origin_lon: float
) -> Optional[float]:
    """Distance from the restaurant to the event, or None if unknowable."""
    if candidate.distance_miles is not None:
        return candidate.distance_miles
    if candidate.latitude is None or candidate.longitude is None:
        return None
    return haversine_miles(origin_lat, origin_lon, candidate.latitude, candidate.longitude)


def _crowd_weight(crowd: float) -> float:
    return min(MAX_CAPACITY_WEIGHT, math.sqrt(crowd / REFERENCE_CAPACITY))


def size_weight(candidate: CandidateEvent) -> Tuple[float, str]:
    """Relative size of an event, and which signal produced it.

    1.0 is a typical event. Returns (weight, basis) so a response can show
    whether the number came from a measured crowd or from a genre guess.
    """
    # 1. The operator sized it themselves.
    override = _as_positive_number(candidate.size_weight_override)
    if override is not None:
        return min(MAX_CAPACITY_WEIGHT, override), "operator"

    # 2. A real attendance figure. ESPN reports these, but only once a game has
    #    been played -- a future fixture reports 0, which _as_positive_number
    #    rejects, so forecasting falls through to the proxies below.
    attendance = _as_positive_number(candidate.attendance)
    if attendance is not None:
        return _crowd_weight(attendance), "attendance"

    # 3. A stated venue capacity.
    capacity = _as_positive_number(candidate.capacity)
    if capacity is not None:
        return _crowd_weight(capacity), "capacity"

    # 4. Segment, refined by genre.
    segment = candidate.segment
    weight = SEGMENT_WEIGHTS.get((segment or "").strip().lower(), BASELINE_SIZE_WEIGHT)
    basis = "segment" if segment else "default"

    if candidate.genre:
        boost = GENRE_BOOSTS.get(candidate.genre.strip().lower())
        if boost is not None:
            weight *= boost
            basis = "segment+genre"

    # 5. Venue activity, as a bounded nudge.
    total = _as_positive_number(candidate.upcoming_events)
    if total is not None:
        activity = math.sqrt(total / UPCOMING_EVENTS_REFERENCE)
        activity = max(MIN_UPCOMING_EVENTS_WEIGHT, min(MAX_UPCOMING_EVENTS_WEIGHT, activity))
        weight *= activity
        basis = f"{basis}+venue_activity"

    return weight, basis


def score_events(
    candidates: Sequence[CandidateEvent],
    origin_lat: float,
    origin_lon: float,
    radius_miles: float = 15.0,
    pressure_cap: float = PRESSURE_CAP,
) -> EventPressure:
    """Aggregate a day's events into one pressure value.

    Events are deduplicated on (source, id) and then on (name, date), since the
    same game can appear in more than one feed and a feed can list an event
    twice across pages.
    """
    seen_ids = set()
    seen_names: Dict[tuple, str] = {}
    scored: List[NearbyEvent] = []
    distances: List[float] = []
    sources = []

    for candidate in candidates:
        key = (candidate.source, candidate.id)
        if candidate.id is not None:
            if key in seen_ids:
                continue
            seen_ids.add(key)

        # Cross-source duplicate: an NFL game listed by both ESPN and
        # Ticketmaster must not count twice. Only applied ACROSS sources --
        # within one feed, two entries with distinct ids are distinct events,
        # even when they share a name (festival acts, a double-header).
        name_key = (candidate.name.strip().lower(), candidate.date)
        previous_source = seen_names.get(name_key)
        if previous_source is not None and previous_source != candidate.source:
            logger.debug(
                "Skipping %r from %s: already counted from %s.",
                candidate.name,
                candidate.source,
                previous_source,
            )
            continue
        seen_names.setdefault(name_key, candidate.source)

        distance = event_distance_miles(candidate, origin_lat, origin_lon)
        dweight = distance_weight(distance, radius_miles)
        sweight, basis = size_weight(candidate)
        pressure = dweight * sweight

        if distance is not None:
            distances.append(distance)

        # An event at or beyond the radius contributes nothing; keep it out of
        # the list rather than reporting a zero-weight row.
        if pressure <= 0:
            continue

        if candidate.source not in sources:
            sources.append(candidate.source)

        scored.append(
            NearbyEvent(
                name=candidate.name,
                date=candidate.date,
                source=candidate.source,
                distance_miles=distance,
                distance_weight=dweight,
                size_weight=sweight,
                pressure=pressure,
                venue_name=candidate.venue_name,
                segment=candidate.segment,
                genre=candidate.genre,
                capacity=None if candidate.capacity is None else int(candidate.capacity),
                attendance=None if candidate.attendance is None else int(candidate.attendance),
                size_basis=basis,
            )
        )

    total = sum(e.pressure for e in scored)
    capped = total > pressure_cap
    if capped:
        logger.info(
            "Event pressure %.2f exceeded the cap of %.2f (%d events); clamping.",
            total,
            pressure_cap,
            len(scored),
        )
        total = pressure_cap

    if total < PRESSURE_FLOOR:
        total = 0.0

    return EventPressure(
        pressure=total,
        event_count=len(scored),
        nearest_miles=min(distances) if distances else None,
        events=scored,
        capped=capped,
        sources=sources,
    )
