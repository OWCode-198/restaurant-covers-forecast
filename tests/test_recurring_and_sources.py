"""Recurring-event rules, the ESPN source, and merging sources together.

No network: ESPN responses and geocoding are both monkeypatched.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from app.config import Location, Settings
from app.core import event_sources, recurring
from app.core.events import CandidateEvent, score_events

ORIGIN_LAT, ORIGIN_LON = 25.7278151, -80.2423594


def make_location(**kw) -> Location:
    defaults = dict(
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LON,
        timezone="America/New_York",
        country_code="US",
        name="Test",
        event_radius_miles=15.0,
    )
    defaults.update(kw)
    return Location(**defaults)


# --- recurrence rules ------------------------------------------------------


def test_weekly_rule_matches_only_that_weekday():
    rule = recurring.RecurringEvent(name="Saturday Market", kind="weekly", weekday=5)
    # 2026-10-03 is a Saturday.
    assert rule.occurs_on(dt.date(2026, 10, 3)) is True
    assert rule.occurs_on(dt.date(2026, 10, 4)) is False
    assert rule.occurs_on(dt.date(2026, 10, 10)) is True


def test_monthly_date_rule():
    rule = recurring.RecurringEvent(name="Pay Day", kind="monthly_date", day_of_month=15)
    assert rule.occurs_on(dt.date(2026, 1, 15)) is True
    assert rule.occurs_on(dt.date(2026, 2, 15)) is True
    assert rule.occurs_on(dt.date(2026, 2, 16)) is False


@pytest.mark.parametrize(
    "day, expected",
    [
        (dt.date(2026, 10, 3), True),    # first Saturday
        (dt.date(2026, 10, 10), False),  # second
        (dt.date(2026, 10, 4), False),   # a Sunday
    ],
)
def test_first_weekday_of_month(day, expected):
    rule = recurring.RecurringEvent(name="Art Walk", kind="monthly_weekday", weekday=5, nth=1)
    assert rule.occurs_on(day) is expected


def test_last_weekday_of_month():
    rule = recurring.RecurringEvent(
        name="Last Friday Social", kind="monthly_weekday", weekday=4, nth=-1
    )
    # October 2026: Fridays fall on 2, 9, 16, 23, 30.
    assert rule.occurs_on(dt.date(2026, 10, 30)) is True
    assert rule.occurs_on(dt.date(2026, 10, 23)) is False


def test_annual_rule():
    rule = recurring.RecurringEvent(
        name="Independence Day Parade", kind="annual", month=7, day_of_month=4
    )
    assert rule.occurs_on(dt.date(2026, 7, 4)) is True
    assert rule.occurs_on(dt.date(2027, 7, 4)) is True
    assert rule.occurs_on(dt.date(2026, 7, 5)) is False


def test_date_range_rule():
    rule = recurring.RecurringEvent(
        name="School Term", kind="date_range", start_date="2026-08-17", end_date="2026-12-18"
    )
    assert rule.occurs_on(dt.date(2026, 9, 1)) is True
    assert rule.occurs_on(dt.date(2026, 8, 16)) is False
    assert rule.occurs_on(dt.date(2026, 12, 19)) is False


def test_date_bounds_keep_a_rule_out_of_older_history():
    """A market that started in 2026 must not be applied to 2024 covers."""
    rule = recurring.RecurringEvent(
        name="New Market", kind="weekly", weekday=5, start_date="2026-01-01"
    )
    assert rule.occurs_on(dt.date(2024, 10, 5)) is False  # a Saturday, but too early
    assert rule.occurs_on(dt.date(2026, 10, 3)) is True


def test_size_presets_map_to_weights():
    small = recurring.RecurringEvent(name="A", kind="weekly", weekday=0, size="small")
    huge = recurring.RecurringEvent(name="B", kind="weekly", weekday=0, size="huge")
    assert small.size_weight < 1.0 < huge.size_weight


def test_invalid_kind_is_rejected():
    with pytest.raises(recurring.RecurringConfigError, match="unknown kind"):
        recurring.RecurringEvent(name="X", kind="fortnightly")


def test_invalid_size_is_rejected():
    with pytest.raises(recurring.RecurringConfigError, match="unknown size"):
        recurring.RecurringEvent(name="X", kind="weekly", weekday=1, size="enormous")


def test_weekly_rule_requires_a_weekday():
    with pytest.raises(recurring.RecurringConfigError, match="needs a weekday"):
        recurring.RecurringEvent(name="X", kind="weekly")


def test_date_range_requires_both_bounds():
    with pytest.raises(recurring.RecurringConfigError, match="needs both start_date"):
        recurring.RecurringEvent(name="X", kind="date_range", start_date="2026-01-01")


def test_describe_is_readable():
    rule = recurring.RecurringEvent(
        name="Market", kind="weekly", weekday=5, size="large", distance_miles=0.2
    )
    text = rule.describe()
    assert "Saturday" in text and "large" in text


# --- persistence -----------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "recurring_events.json"
    original = recurring.RecurringEventSet(
        events=[
            recurring.RecurringEvent(name="Market", kind="weekly", weekday=5, size="large"),
            recurring.RecurringEvent(
                name="Term", kind="date_range", start_date="2026-08-17", end_date="2026-12-18"
            ),
        ]
    )
    recurring.save(path, original)

    loaded = recurring.load(path)
    assert [e.name for e in loaded.events] == ["Market", "Term"]
    assert loaded.events[0].size == "large"


def test_missing_file_is_an_empty_set(tmp_path):
    assert recurring.load(tmp_path / "nope.json").events == []


def test_malformed_json_does_not_raise(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json")
    assert recurring.load(path).events == []


def test_one_bad_rule_does_not_discard_the_good_ones(tmp_path):
    """A typo in an optional config file must not stop forecasting."""
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {"name": "Good", "kind": "weekly", "weekday": 5},
                    {"name": "Bad", "kind": "every-other-tuesday"},
                    {"name": "AlsoGood", "kind": "monthly_date", "day_of_month": 1},
                ]
            }
        )
    )
    loaded = recurring.load(path)
    assert [e.name for e in loaded.events] == ["Good", "AlsoGood"]


def test_comment_keys_are_ignored(tmp_path):
    path = tmp_path / "commented.json"
    path.write_text(
        json.dumps(
            {"events": [{"_comment": "hi", "name": "Market", "kind": "weekly", "weekday": 5}]}
        )
    )
    assert len(recurring.load(path).events) == 1


# --- the recurring source --------------------------------------------------


def test_recurring_source_emits_candidates():
    event_set = recurring.RecurringEventSet(
        events=[
            recurring.RecurringEvent(
                name="Saturday Market",
                kind="weekly",
                weekday=5,
                size="large",
                distance_miles=0.1,
            )
        ]
    )
    source = event_sources.RecurringSource(event_set)

    saturday = source.candidates_for(dt.date(2026, 10, 3), make_location())
    sunday = source.candidates_for(dt.date(2026, 10, 4), make_location())

    assert len(saturday) == 1
    assert saturday[0].name == "Saturday Market"
    # The operator's own sizing must win over any heuristic.
    assert saturday[0].size_weight_override == pytest.approx(1.75)
    assert sunday == []


def test_operator_sizing_beats_the_heuristics():
    from app.core.events import size_weight

    candidate = CandidateEvent(
        name="Market",
        date="2026-10-03",
        source="recurring",
        size_weight_override=1.75,
        segment="Film",  # would otherwise score 0.5
    )
    weight, basis = size_weight(candidate)
    assert basis == "operator"
    assert weight == pytest.approx(1.75)


# --- the ESPN source -------------------------------------------------------


def espn_payload(name="Dolphins at Jets", attendance=0, venue="Hard Rock Stadium", city="Miami"):
    return {
        "events": [
            {
                "id": "401",
                "name": name,
                "shortName": "MIA @ NYJ",
                "competitions": [
                    {
                        "attendance": attendance,
                        "venue": {
                            "fullName": venue,
                            "address": {"city": city, "state": "FL"},
                        },
                    }
                ],
            }
        ]
    }


class FakeEspnResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def espn_source(tmp_path, monkeypatch):
    """An ESPN source with HTTP and geocoding stubbed, and a real cache file."""
    monkeypatch.setattr(
        event_sources.requests,
        "get",
        lambda url, params=None, timeout=None: FakeEspnResponse(espn_payload()),
    )

    from app.core import enrichment

    monkeypatch.setattr(
        enrichment,
        "geocode_address",
        lambda q: {"latitude": 25.958, "longitude": -80.2389, "display_name": q},
    )
    cache = event_sources.VenueGeocodeCache(tmp_path / "venues.json")
    return event_sources.EspnSource(leagues=["football/nfl"], cache=cache)


def test_espn_source_produces_a_located_candidate(espn_source):
    out = espn_source.candidates_for(dt.date(2026, 10, 18), make_location())
    assert len(out) == 1

    game = out[0]
    assert game.source == "espn"
    assert game.segment == "Sports"
    assert game.genre == "Football"
    assert game.latitude == pytest.approx(25.958)


def test_future_fixture_falls_back_to_a_typical_league_crowd(espn_source):
    """ESPN reports attendance 0 before kickoff, so a future game must still
    be sized rather than scoring as nothing."""
    out = espn_source.candidates_for(dt.date(2026, 10, 18), make_location())
    assert out[0].attendance == event_sources.LEAGUE_TYPICAL_ATTENDANCE["football/nfl"]


def test_real_attendance_is_preferred_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(
        event_sources.requests,
        "get",
        lambda url, params=None, timeout=None: FakeEspnResponse(espn_payload(attendance=104168)),
    )
    from app.core import enrichment

    monkeypatch.setattr(
        enrichment, "geocode_address", lambda q: {"latitude": 25.95, "longitude": -80.23}
    )
    source = event_sources.EspnSource(
        leagues=["football/nfl"], cache=event_sources.VenueGeocodeCache(tmp_path / "v.json")
    )
    out = source.candidates_for(dt.date(2025, 11, 15), make_location())
    assert out[0].attendance == 104168


def test_ungeocodable_venue_is_skipped_not_guessed(tmp_path, monkeypatch):
    """Without coordinates an event cannot be distance-graded, and assuming a
    distance would be guesswork."""
    monkeypatch.setattr(
        event_sources.requests,
        "get",
        lambda url, params=None, timeout=None: FakeEspnResponse(espn_payload()),
    )
    from app.core import enrichment

    monkeypatch.setattr(enrichment, "geocode_address", lambda q: None)
    source = event_sources.EspnSource(
        leagues=["football/nfl"], cache=event_sources.VenueGeocodeCache(tmp_path / "v.json")
    )
    assert source.candidates_for(dt.date(2026, 10, 18), make_location()) == []


def test_espn_network_failure_yields_nothing(tmp_path, monkeypatch):
    def boom(url, params=None, timeout=None):
        raise event_sources.requests.RequestException("down")

    monkeypatch.setattr(event_sources.requests, "get", boom)
    source = event_sources.EspnSource(
        leagues=["football/nfl"], cache=event_sources.VenueGeocodeCache(tmp_path / "v.json")
    )
    assert source.candidates_for(dt.date(2026, 10, 18), make_location()) == []


# --- venue geocode cache ---------------------------------------------------


def test_venue_cache_resolves_once_and_persists(tmp_path):
    calls = []

    def resolver(key):
        calls.append(key)
        return {"latitude": 1.0, "longitude": 2.0}

    path = tmp_path / "venues.json"
    cache = event_sources.VenueGeocodeCache(path)
    assert cache.get("Hard Rock Stadium, Miami", resolver)["latitude"] == 1.0
    assert cache.get("Hard Rock Stadium, Miami", resolver)["latitude"] == 1.0
    assert len(calls) == 1  # the second read came from the cache
    cache.flush()

    # A fresh instance reads it back from disk rather than geocoding again.
    reloaded = event_sources.VenueGeocodeCache(path)
    assert reloaded.get("Hard Rock Stadium, Miami", resolver)["latitude"] == 1.0
    assert len(calls) == 1


def test_venue_cache_remembers_a_miss(tmp_path):
    """A stadium Nominatim cannot find must not be retried every forecast."""
    calls = []

    def resolver(key):
        calls.append(key)
        return None

    cache = event_sources.VenueGeocodeCache(tmp_path / "v.json")
    assert cache.get("Nowhere Arena", resolver) is None
    assert cache.get("Nowhere Arena", resolver) is None
    assert len(calls) == 1


# --- merging sources -------------------------------------------------------


def test_cross_source_duplicate_is_counted_once():
    """The same game from ESPN and Ticketmaster must not double-count."""
    shared = dict(name="Dolphins at Jets", date="2026-10-18", distance_miles=1.0)
    result = score_events(
        [
            CandidateEvent(source="espn", attendance=68000, **shared),
            CandidateEvent(source="ticketmaster", segment="Sports", **shared),
        ],
        ORIGIN_LAT,
        ORIGIN_LON,
    )
    assert result.event_count == 1
    assert result.sources == ["espn"]


def test_same_source_same_name_still_counts_twice():
    """Two festival acts sharing a name within one feed are distinct events."""
    result = score_events(
        [
            CandidateEvent(
                name="Fest Act",
                date="2026-10-18",
                source="ticketmaster",
                id="a",
                distance_miles=1.0,
                segment="Music",
            ),
            CandidateEvent(
                name="Fest Act",
                date="2026-10-18",
                source="ticketmaster",
                id="b",
                distance_miles=1.0,
                segment="Music",
            ),
        ],
        ORIGIN_LAT,
        ORIGIN_LON,
    )
    assert result.event_count == 2


def test_sources_are_reported():
    result = score_events(
        [
            CandidateEvent(
                name="Game",
                date="2026-10-18",
                source="espn",
                attendance=68000,
                distance_miles=2.0,
            ),
            CandidateEvent(
                name="Market",
                date="2026-10-18",
                source="recurring",
                size_weight_override=1.0,
                distance_miles=0.1,
            ),
        ],
        ORIGIN_LAT,
        ORIGIN_LON,
    )
    assert set(result.sources) == {"espn", "recurring"}
    assert result.event_count == 2


def test_build_sources_respects_configuration(tmp_path):
    settings = Settings(
        db_path=tmp_path / "db.sqlite3",
        espn_enabled=False,
        recurring_events_path=tmp_path / "none.json",
        venue_cache_path=tmp_path / "v.json",
    )
    # No Ticketmaster key, ESPN off, no recurring file -> nothing to ask.
    assert event_sources.build_sources(settings, make_location()) == []


def test_build_sources_includes_enabled_ones(tmp_path):
    recurring.save(
        tmp_path / "rec.json",
        recurring.RecurringEventSet(
            events=[recurring.RecurringEvent(name="Market", kind="weekly", weekday=5)]
        ),
    )
    settings = Settings(
        db_path=tmp_path / "db.sqlite3",
        espn_enabled=True,
        espn_leagues="football/nfl",
        recurring_events_path=tmp_path / "rec.json",
        venue_cache_path=tmp_path / "v.json",
    )
    names = [
        s.name
        for s in event_sources.build_sources(settings, make_location(ticketmaster_api_key="key"))
    ]
    assert names == ["ticketmaster", "espn", "recurring"]


def test_a_failing_source_does_not_stop_the_others(tmp_path, monkeypatch):
    class Boom:
        name = "boom"

        def candidates_for(self, day, location):
            raise RuntimeError("feed is down")

    class Fine:
        name = "fine"

        def candidates_for(self, day, location):
            return [
                CandidateEvent(
                    name="Market", date=day.isoformat(), source="fine", distance_miles=0.1
                )
            ]

    monkeypatch.setattr(event_sources, "build_sources", lambda s, loc: [Boom(), Fine()])
    settings = Settings(db_path=tmp_path / "db.sqlite3")
    out = event_sources.gather_candidates(dt.date(2026, 10, 3), make_location(), settings)
    assert [c.source for c in out] == ["fine"]
