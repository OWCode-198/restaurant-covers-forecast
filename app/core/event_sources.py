"""Event sources, merged into one list of scoring candidates.

Each source answers the same question -- "what is happening near here on this
date?" -- and returns `CandidateEvent`s, so adding a feed never touches the
scoring logic.

  ticketmaster  ticketed events (concerts, shows, pro sports)
  espn          sports fixtures, including college games
  recurring     calendar facts the operator stated once

Why ESPN is worth having
------------------------
Ticketmaster only lists events it sells tickets for, which misses a great deal
-- in this project's own history the flagged event days are mostly a city
marathon, a street festival, a convention and a college game day. ESPN's
undocumented scoreboard API needs no key and covers fixtures across the major
leagues.

Two limits found by testing it, both handled here:

  * Venues carry a city and state but **no coordinates**, so each venue is
    geocoded once and cached to disk. Without the cache this would be one
    Nominatim request per game per forecast.
  * `attendance` is a post-game figure: a future fixture reports 0. So size
    for an upcoming game comes from a documented per-league typical crowd,
    and a real attendance figure is used only where the feed has one.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

from app.config import Location, Settings
from app.core.events import CandidateEvent

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{league}/scoreboard"

# Leagues worth polling, as ESPN path segments. Each costs one request per
# forecast day, so the default set is deliberately small.
DEFAULT_ESPN_LEAGUES = (
    "football/nfl",
    "football/college-football",
    "basketball/nba",
    "baseball/mlb",
    "hockey/nhl",
)

# Typical attendance per league, used for a future fixture where ESPN reports
# attendance 0. Rounded regular-season figures; the scorer takes the square
# root of these, so the right order of magnitude matters more than precision.
LEAGUE_TYPICAL_ATTENDANCE: Dict[str, int] = {
    "football/nfl": 68000,
    "football/college-football": 40000,
    "basketball/nba": 18000,
    "basketball/mens-college-basketball": 9000,
    "baseball/mlb": 28000,
    "hockey/nhl": 17000,
    "soccer/usa.1": 21000,
}
FALLBACK_TYPICAL_ATTENDANCE = 15000

LEAGUE_GENRES: Dict[str, str] = {
    "football/nfl": "Football",
    "football/college-football": "Football",
    "basketball/nba": "Basketball",
    "basketball/mens-college-basketball": "Basketball",
    "baseball/mlb": "Baseball",
    "hockey/nhl": "Hockey",
    "soccer/usa.1": "Soccer",
}

_REQUEST_ERRORS = (requests.RequestException, ValueError, KeyError)

# Full state name -> two-letter code, for matching a configured region against
# whatever spelling a feed uses. Codes map to themselves so normalising is
# idempotent.
US_STATE_CODES: Dict[str, str] = {}
for _name, _code in {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}.items():
    US_STATE_CODES[_name] = _code
    US_STATE_CODES[_code] = _code


class VenueGeocodeCache:
    """Venue name/city -> coordinates, persisted to JSON.

    ESPN gives no coordinates, and Nominatim asks for at most one request per
    second, so caching is what makes this source usable at all. Venues do not
    move, and a miss is remembered too, so a stadium Nominatim cannot find is
    not looked up again on every forecast.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._cache: Dict[str, Optional[Dict]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._cache = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read the venue cache at %s: %s", self.path, exc)

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self._cache, f, indent=2, sort_keys=True)
            self._dirty = False
        except OSError as exc:
            logger.warning("Could not write the venue cache to %s: %s", self.path, exc)

    def get(self, key: str, resolver) -> Optional[Dict]:
        """Cached coordinates for `key`, calling `resolver(key)` on a miss.

        A definitive "no such place" is cached so it is not retried forever. A
        transient failure -- a rate limit, a timeout -- is deliberately NOT
        cached: caching it would poison the entry permanently after one bad run.
        """
        if key in self._cache:
            return self._cache[key]

        from app.core.enrichment import GeocodeUnavailable

        try:
            result = resolver(key)
        except GeocodeUnavailable as exc:
            logger.warning("Geocoder unavailable for %r (not cached): %s", key, exc)
            return None

        coords = (
            {"latitude": result["latitude"], "longitude": result["longitude"]} if result else None
        )
        self._cache[key] = coords
        self._dirty = True
        return coords


class TicketmasterSource:
    """Ticketed events near the restaurant."""

    name = "ticketmaster"

    def __init__(self, radius_miles: float = 15.0) -> None:
        self.radius_miles = radius_miles

    def candidates_for(self, day: dt.date, location: Location) -> List[CandidateEvent]:
        if not location.ticketmaster_api_key:
            return []

        from app.core import enrichment  # imported here to avoid an import cycle

        iso = day.isoformat()
        try:
            raw = enrichment.fetch_events(
                location.latitude,
                location.longitude,
                location.ticketmaster_api_key,
                iso,
                iso,
                radius_miles=self.radius_miles,
            )
        except _REQUEST_ERRORS as exc:
            logger.warning("Ticketmaster lookup failed for %s: %s", iso, exc)
            return []

        out = []
        for event in raw:
            # The range query can return neighbouring days.
            if ((event.get("dates") or {}).get("start") or {}).get("localDate") != iso:
                continue
            out.append(CandidateEvent.from_ticketmaster(event))
        return out


class EspnSource:
    """Sports fixtures from ESPN's undocumented scoreboard API.

    No API key. Venues are geocoded once and cached; attendance is used when
    the feed reports a real figure, and a per-league typical crowd otherwise.
    """

    name = "espn"

    def __init__(
        self,
        leagues: Sequence[str] = DEFAULT_ESPN_LEAGUES,
        cache: Optional[VenueGeocodeCache] = None,
        timeout: int = 15,
    ) -> None:
        self.leagues = list(leagues)
        self.cache = cache
        self.timeout = timeout

    def _fetch_league(self, league: str, day: dt.date) -> List[Dict]:
        try:
            resp = requests.get(
                ESPN_SCOREBOARD_URL.format(league=league),
                params={"dates": day.strftime("%Y%m%d")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("events", []) or []
        except _REQUEST_ERRORS as exc:
            logger.warning("ESPN %s lookup failed for %s: %s", league, day, exc)
            return []

    @staticmethod
    def _normalize_region(value: str) -> str:
        """Fold a US state to its two-letter code, so "Florida" and "FL" match.

        ESPN reports abbreviations ("GA", "OH"); a human configuring the
        service is as likely to write the full name. Comparing raw strings
        silently filters out every venue, which is exactly what happened.
        """
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return US_STATE_CODES.get(text, text)

    @classmethod
    def is_plausibly_nearby(cls, venue: Dict, location: Location) -> bool:
        """Cheap string pre-filter, before spending a geocode on the venue.

        This matters a lot. ESPN's scoreboard is nationwide, so a single
        forecast day can list 40+ venues across the country. Geocoding each one
        to discover that Lumen Field is 2,700 miles from Miami is pure waste,
        and in practice it got the geocoder to rate-limit us.

        A 15-mile radius means the venue is in the same metro, which in the US
        means the same state. So a venue is only worth geocoding when its
        region matches the restaurant's. With no configured region we cannot
        filter, and fall back to allowing it.
        """
        region = cls._normalize_region(getattr(location, "region", None))
        if not region:
            return True

        address = venue.get("address") or {}
        venue_region = cls._normalize_region(address.get("state"))
        if not venue_region:
            # No state at all usually means an international venue.
            country = str(address.get("country") or "").strip().lower()
            return not country or country in ("usa", "united states")
        return venue_region == region

    def _venue_coords(self, venue: Dict) -> Optional[Dict]:
        """Geocode a venue from its name and city, via the shared cache."""
        address = venue.get("address") or {}
        parts = [
            venue.get("fullName"),
            address.get("city"),
            address.get("state"),
            address.get("country"),
        ]
        key = ", ".join(p for p in parts if p)
        if not key or self.cache is None:
            return None

        from app.core import enrichment

        return self.cache.get(key, enrichment.geocode_address)

    def candidates_for(self, day: dt.date, location: Location) -> List[CandidateEvent]:
        out = []
        for league in self.leagues:
            for event in self._fetch_league(league, day):
                competitions = event.get("competitions") or []
                competition = competitions[0] if competitions else {}
                venue = competition.get("venue") or {}

                # Filter on the cheap signal first; geocoding is the expensive,
                # rate-limited step.
                if not self.is_plausibly_nearby(venue, location):
                    continue

                coords = self._venue_coords(venue)
                if coords is None:
                    # Without a location the event cannot be distance-graded,
                    # and counting it at an assumed distance would be guesswork.
                    continue

                # Real attendance only exists after a game is played; a future
                # fixture reports 0, so fall back to the league's typical crowd.
                attendance = competition.get("attendance") or 0
                if not attendance:
                    attendance = LEAGUE_TYPICAL_ATTENDANCE.get(
                        league, FALLBACK_TYPICAL_ATTENDANCE
                    )

                out.append(
                    CandidateEvent(
                        name=event.get("name") or event.get("shortName") or "Game",
                        date=day.isoformat(),
                        source=self.name,
                        id=str(event.get("id")) if event.get("id") else None,
                        latitude=coords["latitude"],
                        longitude=coords["longitude"],
                        attendance=attendance,
                        segment="Sports",
                        genre=LEAGUE_GENRES.get(league),
                        venue_name=venue.get("fullName"),
                    )
                )

        if self.cache is not None:
            self.cache.flush()
        return out


class RecurringSource:
    """Calendar facts the operator stated once."""

    name = "recurring"

    def __init__(self, event_set) -> None:
        self.event_set = event_set

    def candidates_for(self, day: dt.date, location: Location) -> List[CandidateEvent]:
        out = []
        for rule in self.event_set.occurrences_on(day):
            out.append(
                CandidateEvent(
                    name=rule.name,
                    date=day.isoformat(),
                    source=self.name,
                    id=f"{rule.name}:{day.isoformat()}",
                    distance_miles=rule.distance_miles,
                    size_weight_override=rule.size_weight,
                    segment="Community",
                    venue_name=rule.note,
                )
            )
        return out


def build_sources(settings: Settings, location: Location) -> List:
    """Assemble the enabled event sources."""
    sources: List = []

    if location.ticketmaster_api_key:
        sources.append(TicketmasterSource(radius_miles=location.event_radius_miles))

    if settings.espn_enabled:
        leagues = (
            [lg.strip() for lg in settings.espn_leagues.split(",") if lg.strip()]
            if settings.espn_leagues
            else list(DEFAULT_ESPN_LEAGUES)
        )
        sources.append(
            EspnSource(leagues=leagues, cache=VenueGeocodeCache(settings.venue_cache_path))
        )

    from app.core import recurring

    event_set = recurring.load(settings.recurring_events_path)
    if event_set.events:
        sources.append(RecurringSource(event_set))

    return sources


def gather_candidates(
    day: dt.date, location: Location, settings: Settings
) -> List[CandidateEvent]:
    """Every candidate event for a date, across all enabled sources.

    A failing source contributes nothing and is logged; it never prevents the
    others from reporting.
    """
    candidates: List[CandidateEvent] = []
    for source in build_sources(settings, location):
        try:
            candidates.extend(source.candidates_for(day, location))
        except Exception as exc:  # a third-party feed must not break a forecast
            logger.warning("Event source %r failed for %s: %s", source.name, day, exc)
    return candidates
