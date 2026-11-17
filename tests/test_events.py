"""Distance- and size-weighted event scoring.

Pure functions over synthetic Ticketmaster-shaped dicts: no network, no key.
"""
from __future__ import annotations

import pytest

from app.core.events import (
    MAX_CAPACITY_WEIGHT,
    CandidateEvent,
    PRESSURE_CAP,
    PRESSURE_FLOOR,
    REFERENCE_CAPACITY,
    distance_weight,
    event_distance_miles,
    haversine_miles,
    score_events,
    size_weight,
)

# Downtown Miami, roughly -- the origin for every test below.
ORIGIN_LAT, ORIGIN_LON = 25.7617, -80.1918


def make_event(
    name="Test Event",
    date="2026-03-01",
    lat=None,
    lon=None,
    distance=None,
    units="miles",
    segment=None,
    genre=None,
    capacity=None,
    upcoming=None,
    event_id=None,
    venue_name="Test Venue",
):
    """Build a minimal event in the Discovery API's shape."""
    venue = {"name": venue_name}
    if lat is not None and lon is not None:
        venue["location"] = {"latitude": str(lat), "longitude": str(lon)}
    if distance is not None:
        venue["distance"] = distance
        venue["units"] = units
    if capacity is not None:
        venue["capacity"] = capacity
    if upcoming is not None:
        venue["upcomingEvents"] = {"_total": upcoming}

    event = {
        "name": name,
        "dates": {"start": {"localDate": date}},
        "_embedded": {"venues": [venue]},
    }
    if event_id is not None:
        event["id"] = event_id
    if segment or genre:
        classification = {}
        if segment:
            classification["segment"] = {"name": segment}
        if genre:
            classification["genre"] = {"name": genre}
        event["classifications"] = [classification]
    return event


def cand(event):
    """Normalise a Ticketmaster-shaped dict into a scoring candidate."""
    return CandidateEvent.from_ticketmaster(event)


# --- distance --------------------------------------------------------------


def test_haversine_known_distance():
    """Miami to Fort Lauderdale is about 25 miles."""
    d = haversine_miles(ORIGIN_LAT, ORIGIN_LON, 26.1224, -80.1373)
    assert 22 < d < 27


def test_haversine_is_zero_for_same_point():
    assert haversine_miles(ORIGIN_LAT, ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON) == pytest.approx(0)


def test_haversine_is_symmetric():
    a = haversine_miles(ORIGIN_LAT, ORIGIN_LON, 26.1224, -80.1373)
    b = haversine_miles(26.1224, -80.1373, ORIGIN_LAT, ORIGIN_LON)
    assert a == pytest.approx(b)


@pytest.mark.parametrize(
    "distance, expected",
    [
        (0.0, 1.0),    # at the door
        (3.75, 0.75),  # a quarter of the way out
        (7.5, 0.5),    # halfway
        (15.0, 0.0),   # at the radius edge
        (20.0, 0.0),   # beyond it -- never negative
    ],
)
def test_distance_weight_is_linear(distance, expected):
    assert distance_weight(distance, 15.0) == pytest.approx(expected)


def test_unknown_distance_gets_mid_weight():
    """It is somewhere in the radius, so neither full credit nor none."""
    assert distance_weight(None, 15.0) == 0.5


def test_zero_radius_is_safe():
    assert distance_weight(5.0, 0.0) == 0.0


def test_closer_event_always_outweighs_farther():
    assert distance_weight(1.0, 15.0) > distance_weight(14.0, 15.0)


def test_prefers_ticketmasters_own_distance():
    event = make_event(distance=4.0, units="miles", lat=26.1224, lon=-80.1373)
    # The reported 4 miles wins over the ~25 miles the coordinates imply.
    assert event_distance_miles(cand(event), ORIGIN_LAT, ORIGIN_LON) == pytest.approx(4.0)


def test_converts_reported_kilometres():
    event = make_event(distance=10.0, units="km")
    assert event_distance_miles(cand(event), ORIGIN_LAT, ORIGIN_LON) == pytest.approx(6.21371, abs=1e-4)


def test_falls_back_to_venue_coordinates():
    event = make_event(lat=25.77, lon=-80.19)
    d = event_distance_miles(cand(event), ORIGIN_LAT, ORIGIN_LON)
    assert d is not None and d < 1.0


def test_distance_is_none_without_any_location():
    assert event_distance_miles(cand(make_event()), ORIGIN_LAT, ORIGIN_LON) is None


# --- size ------------------------------------------------------------------


def test_capacity_is_preferred_and_sublinear():
    """A venue at 4x the reference capacity gets only 2x the weight: impact
    grows with the square root, since only a spillover fraction dines out."""
    weight, basis = size_weight(cand(make_event(capacity=REFERENCE_CAPACITY * 4)))
    assert basis == "capacity"
    assert weight == pytest.approx(2.0)


def test_reference_capacity_is_a_typical_event():
    weight, _ = size_weight(cand(make_event(capacity=REFERENCE_CAPACITY)))
    assert weight == pytest.approx(1.0)


def test_capacity_weight_is_capped():
    weight, _ = size_weight(cand(make_event(capacity=500_000)))
    assert weight == MAX_CAPACITY_WEIGHT


def test_sports_outweighs_theatre():
    sports, _ = size_weight(cand(make_event(segment="Sports")))
    theatre, _ = size_weight(cand(make_event(segment="Arts & Theatre")))
    assert sports > theatre


def test_segment_matching_ignores_case():
    lower, _ = size_weight(cand(make_event(segment="sports")))
    upper, _ = size_weight(cand(make_event(segment="SPORTS")))
    assert lower == upper


def test_genre_boosts_on_top_of_segment():
    plain, _ = size_weight(cand(make_event(segment="Sports")))
    football, basis = size_weight(cand(make_event(segment="Sports", genre="Football")))
    assert football > plain
    assert basis == "segment+genre"


def test_unknown_segment_falls_back_to_typical():
    weight, basis = size_weight(cand(make_event(segment="Something New")))
    assert weight == pytest.approx(1.0)
    assert basis == "segment"


def test_no_classification_is_the_default():
    weight, basis = size_weight(cand(make_event()))
    assert weight == pytest.approx(1.0)
    assert basis == "default"


def test_venue_activity_nudges_and_is_bounded():
    quiet, basis = size_weight(cand(make_event(segment="Music", upcoming=1)))
    busy, _ = size_weight(cand(make_event(segment="Music", upcoming=100_000)))
    base, _ = size_weight(cand(make_event(segment="Music")))

    assert "venue_activity" in basis
    assert busy > quiet
    # Bounded, so a mega-venue cannot run away with the score.
    assert busy <= base * 1.3 + 1e-9
    assert quiet >= base * 0.85 - 1e-9


def test_capacity_rejects_booleans():
    """`capacity: true` must not read as a capacity of 1."""
    event = make_event(segment="Music")
    event["_embedded"]["venues"][0]["capacity"] = True
    _, basis = size_weight(cand(event))
    assert basis != "capacity"


# --- aggregate -------------------------------------------------------------


def _score(events, radius=15.0, cap=PRESSURE_CAP):
    """Score Ticketmaster-shaped dicts, normalising them first."""
    candidates = [cand(e) for e in events]
    return score_events(candidates, ORIGIN_LAT, ORIGIN_LON, radius_miles=radius, pressure_cap=cap)


def test_no_events_is_zero_pressure():
    result = _score([])
    assert result.pressure == 0.0
    assert result.event_count == 0
    assert result.is_local_event == 0
    assert result.nearest_miles is None


def test_a_typical_nearby_event_is_about_one():
    """The calibration the model depends on: one ordinary event close by should
    land near 1.0, which is what the binary coefficient was fitted on."""
    result = _score([make_event(distance=0.5, segment="Arts & Theatre")])
    assert 0.9 < result.pressure <= 1.0


def test_closer_event_produces_more_pressure():
    near = _score([make_event(distance=1.0, segment="Music")]).pressure
    far = _score([make_event(distance=13.0, segment="Music")]).pressure
    assert near > far * 3


def test_bigger_event_produces_more_pressure():
    """Same distance, different size -- the point of the size weighting."""
    stadium = _score([make_event(distance=2.0, segment="Sports", genre="Football")]).pressure
    recital = _score(
        [make_event(distance=2.0, segment="Arts & Theatre", genre="Classical")]
    ).pressure
    assert stadium > recital


def test_big_event_far_away_can_lose_to_small_event_next_door():
    """The trade-off the grading exists to express."""
    far_stadium = _score([make_event(distance=14.0, segment="Sports", genre="Football")]).pressure
    near_club = _score([make_event(distance=0.2, segment="Music")]).pressure
    assert near_club > far_stadium


def test_event_beyond_radius_is_excluded():
    result = _score([make_event(distance=20.0, segment="Music")], radius=15.0)
    assert result.pressure == 0.0
    assert result.event_count == 0


def test_multiple_events_accumulate():
    one = _score([make_event(distance=2.0, segment="Music", event_id="a")]).pressure
    two = _score(
        [
            make_event(distance=2.0, segment="Music", event_id="a"),
            make_event(distance=2.0, segment="Music", event_id="b"),
        ]
    ).pressure
    assert two > one


def test_duplicate_ids_are_counted_once():
    result = _score(
        [
            make_event(distance=2.0, segment="Music", event_id="same"),
            make_event(distance=2.0, segment="Music", event_id="same"),
        ]
    )
    assert result.event_count == 1


def test_pressure_is_capped():
    """A festival weekend must not extrapolate the coefficient to absurdity."""
    many = [
        make_event(distance=0.1, segment="Sports", genre="Football", event_id=str(i))
        for i in range(40)
    ]
    result = _score(many)
    assert result.pressure == PRESSURE_CAP
    assert result.capped is True


def test_tiny_pressure_is_floored_to_zero():
    """A single distant listing should not nudge every forecast."""
    result = _score([make_event(distance=14.99, segment="Film")])
    assert result.pressure == 0.0


def test_nearest_miles_reports_the_closest():
    result = _score(
        [
            make_event(distance=8.0, event_id="a"),
            make_event(distance=1.5, event_id="b"),
            make_event(distance=5.0, event_id="c"),
        ]
    )
    assert result.nearest_miles == pytest.approx(1.5)


def test_summary_lists_biggest_contributors_first():
    result = _score(
        [
            make_event(name="Tiny Gig", distance=12.0, segment="Film", event_id="a"),
            make_event(
                name="Big Game",
                distance=0.5,
                segment="Sports",
                genre="Football",
                event_id="b",
            ),
        ]
    )
    assert result.summary().startswith("Big Game")


def test_to_dict_exposes_the_reasoning():
    result = _score([make_event(distance=2.0, segment="Sports", genre="Football", capacity=40000)])
    payload = result.to_dict()

    assert payload["event_count"] == 1
    event = payload["events"][0]
    # Every component of the score is reported, so a number is explainable.
    for key in ("distance_miles", "distance_weight", "size_weight", "size_basis", "pressure"):
        assert key in event
    assert event["size_basis"] == "capacity"
    assert event["capacity"] == 40000


def test_radius_changes_the_grading():
    event = [make_event(distance=10.0, segment="Music")]
    tight = _score(event, radius=12.0).pressure
    wide = _score(event, radius=30.0).pressure
    # The same event is relatively closer within a wider radius.
    assert wide > tight


def test_floor_constant_is_sane():
    assert 0 < PRESSURE_FLOOR < 1


# --- origin provenance -----------------------------------------------------
#
# Event distances are measured from the configured coordinates, so a
# deployment that never set its own must not silently inherit another
# restaurant's origin.


def test_explicit_coordinates_are_marked_precise(tmp_path):
    from app.config import Settings, get_location

    settings = Settings(
        db_path=tmp_path / "x.sqlite3",
        latitude=25.7278151,
        longitude=-80.2423594,
        restaurant_address="3409 Main Hwy, Coconut Grove, FL 33133",
    )
    location = get_location(settings)

    assert location is not None
    assert location.source == "explicit"
    assert location.is_precise is True
    assert location.warnings() == []


def test_inherited_coordinates_are_flagged(tmp_path, monkeypatch):
    """Falling back to the shared config file must produce a warning."""
    import app.config as config_module
    from app.config import Settings, get_location

    monkeypatch.setattr(
        config_module,
        "_load_cli_config",
        lambda: {"latitude": 25.7617, "longitude": -80.1918, "location_name": "Somewhere"},
    )
    location = get_location(Settings(db_path=tmp_path / "x.sqlite3"))

    assert location is not None
    assert location.source == "inherited_config"
    assert location.is_precise is False
    assert any("inherited" in w for w in location.warnings())


def test_missing_address_is_flagged(tmp_path):
    from app.config import Settings, get_location

    location = get_location(
        Settings(db_path=tmp_path / "x.sqlite3", latitude=1.0, longitude=2.0)
    )
    assert location is not None
    assert any("COVERS_RESTAURANT_ADDRESS" in w for w in location.warnings())


def test_no_coordinates_at_all_returns_none(tmp_path, monkeypatch):
    import app.config as config_module
    from app.config import Settings, get_location

    monkeypatch.setattr(config_module, "_load_cli_config", lambda: {})
    assert get_location(Settings(db_path=tmp_path / "x.sqlite3")) is None


# --- address normalisation -------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The case that actually failed: Nominatim does not expand "Hwy".
        ("3409 Main Hwy, Coconut Grove, FL 33133", "3409 Main Highway, Coconut Grove, FL 33133"),
        ("100 NE 2nd Ave, Miami, FL", "100 NE 2nd Avenue, Miami, FL"),
        ("1 Ocean Blvd", "1 Ocean Boulevard"),
        ("5 Elm St", "5 Elm Street"),
        # Already spelled out, so unchanged.
        ("3409 Main Highway, Miami, FL", "3409 Main Highway, Miami, FL"),
    ],
)
def test_expand_street_abbreviations(raw, expected):
    from app.core.enrichment import expand_street_abbreviations

    assert expand_street_abbreviations(raw) == expected


def test_expansion_preserves_component_commas():
    """The comma separating address components must survive, or the looser
    retry queries are built wrong."""
    from app.core.enrichment import expand_street_abbreviations

    out = expand_street_abbreviations("3409 Main Hwy, Coconut Grove, FL 33133")
    assert out.count(",") == 2


def test_geocode_tries_looser_queries(monkeypatch):
    """The exact address fails, the expanded one fails, and dropping the
    neighbourhood succeeds -- which is what happened for a real address."""
    from app.core import enrichment

    attempts = []

    def fake_once(query):
        attempts.append(query)
        if "Coconut Grove" in query or "Hwy" in query:
            return None
        return {"latitude": 25.7278151, "longitude": -80.2423594, "display_name": "ok"}

    monkeypatch.setattr(enrichment, "_geocode_once", fake_once)
    monkeypatch.setattr(enrichment.time, "sleep", lambda s: None)

    hit = enrichment.geocode_address("3409 Main Hwy, Coconut Grove, FL 33133")

    assert hit is not None
    assert hit["latitude"] == pytest.approx(25.7278151)
    assert len(attempts) == 3  # original, expanded, neighbourhood dropped


def test_geocode_returns_none_when_every_query_fails(monkeypatch):
    from app.core import enrichment

    monkeypatch.setattr(enrichment, "_geocode_once", lambda q: None)
    monkeypatch.setattr(enrichment.time, "sleep", lambda s: None)
    assert enrichment.geocode_address("nowhere at all, XX") is None


def test_geocode_rejects_empty_input():
    from app.core.enrichment import geocode_address

    assert geocode_address("   ") is None
