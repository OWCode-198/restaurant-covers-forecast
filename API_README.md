# Predictive Covers API

The covers forecaster from this project, as a service. Daily covers come from
**Toast** or from a **CSV export**, get enriched with weather, public holidays
and nearby ticketed events, and are fed to the same interpretable regression
used in the notebooks and `put in terminal/covers_forecast_cli.py`.

Nothing about the model changed in the move to HTTP, with one deliberate
exception — see [Reservations scaling](#reservations-scaling-a-fix).

## Quick start

No Toast credentials needed: the default configuration reads the CSV exports
already in this repo.

```bash
python3 -m venv .venv-api
```

```bash
.venv-api/bin/pip install -r requirements-api.txt
```

```bash
.venv-api/bin/uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for interactive docs, or:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```bash
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
```

```bash
curl -s 'localhost:8000/forecast?days=7' | python3 -m json.tool
```

> The project's existing `.venv` is broken — its scripts still point at the old
> `~/Downloads/Predictive Covers project` path from before the directory moved,
> so `pip` fails there. `.venv-api` is separate and leaves it untouched.

## Configuration

Everything is environment variables, prefixed `COVERS_`, read from a `.env`
file if present. Copy the template and edit:

```bash
cp .env.example .env
```

Location, timezone and the Ticketmaster key fall back to
`.covers_forecast_config.json`, so the service and the CLI stay in agreement
rather than each holding their own copy.

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/health` | Configuration and stored-data coverage. `?check_source=true` also makes a live call to Toast to prove the credentials work. |
| `POST` | `/ingest` | Pull covers history into the store. Body: `{"start": "...", "end": "..."}`, both optional for CSV, **required** for Toast. Safe to re-run. |
| `GET` | `/forecast?days=7` | The next N days. Convenient read for a dashboard. |
| `POST` | `/forecast` | Same, with per-day overrides for things the APIs cannot know. |
| `GET` | `/forecasts` | Previously logged forecasts, newest target date first. |
| `POST` | `/actuals` | Record the real covers for a day, so accuracy can be tracked. |
| `GET` | `/accuracy` | Forecast-vs-actual error across every logged day that has both. |
| `GET` | `/history` | The daily covers the model trains on. |
| `GET` | `/events?date=` | What the event scorer sees for a date, and why — per-event distance, weights, size basis and source. |
| `GET` | `/recurring-events` | The recurring rules configured for this restaurant, plus a 28-day preview. |
| `GET` | `/geocode?address=` | Resolve your street address to coordinates, for pinning down the event-distance origin. |

Every forecast carries the inputs behind it — the lag, the weekday average,
the weather, whether a holiday or event was detected — so a number is always
explainable rather than arriving bare.

### Overrides

Tell the model something it cannot look up:

```bash
curl -s -X POST localhost:8000/forecast -H 'content-type: application/json' -d '{"days": 7, "overrides": [{"date": "2026-02-14", "is_promo": 1}]}'
```

Supported per day: `is_promo`, `is_holiday`, `is_local_event`, `temp_high_f`,
`precip_in`, `is_rain`, `reservations_covers`. Anything you set beats what gets
fetched; anything you leave out is still filled in automatically.

For weather, set `precip_in` (inches) rather than `is_rain`. The model reads the
amount and a heavy-rain threshold, both derived from `precip_in`, so overriding
the amount moves the forecast; `is_rain` is reported for the record but is true
on four days in five here and is not what the forecast reacts to.

## Switching to Toast

Set `COVERS_SOURCE=toast` and the credentials in `.env`, then confirm the
connection before trusting it:

```bash
curl -s 'localhost:8000/health?check_source=true' | python3 -m json.tool
```

Backfill a range, then forecast:

```bash
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"start": "2025-01-01", "end": "2026-01-01"}'
```

### Getting Toast credentials

You cannot get Toast API credentials as an unaffiliated individual. There are
two routes, both tied to a restaurant or a company:

1. **Standard API Access** — read-only and self-service, which is all this
   service needs. Purchased through the Toast Shop with a Toast Web login,
   after which you create credential sets yourself under **Integrations →
   Toast API access → Manage credentials**, each scoped to specific locations.
   Included in Restaurant Management Essentials for US customers, otherwise a
   standalone subscription. Requires access to the restaurant's Toast account.
2. **Toast Partner Program** — a company applies, is vetted for business fit
   and technical readiness, signs a partner agreement, builds against a sandbox
   the integrations team issues, and passes a certification review before
   production credentials are granted. There is no self-serve developer sandbox.

### How covers are derived from Toast

- One request per **business date** (`GET /orders/v2/ordersBulk?businessDate=`),
  not a timestamp range. Toast's business date already accounts for the
  restaurant's own day-rollover cutoff, so a day of covers needs no timezone
  arithmetic here.
- Covers for a day are the sum of each order's `numberOfGuests`, falling back
  to `calculatedGuestCount` where the POS recorded no party size.
- Voided and deleted orders are excluded.
- A date with no orders is stored as `0` ("closed"). A date that **failed** to
  fetch is omitted entirely ("unknown"). The model treats these differently, so
  a network failure is never quietly recorded as a day with no customers.

## Event scoring: distance and size

The original event feature was binary — any ticketed event within 15 miles set
`is_local_event = 1`. That treated a stadium game two blocks away and a club
gig fifteen miles out as identical. Now each day gets a continuous
**event pressure**:

```
pressure = sum over nearby events of  distance_weight x size_weight
```

**Distance** falls off linearly, as asked: full weight at your door, straight
down to zero at the search radius.

```
distance_weight = max(0, 1 - distance / radius)
```

Distance is measured from your configured coordinates to each event's venue,
using the `distance` Ticketmaster returns on a geo search when present and
falling back to a haversine calculation from the venue's latitude/longitude.

**Size** is harder: the Discovery API's venue object has **no `capacity`
field**, so size is inferred from documented proxies, most trustworthy first:

1. an explicit `capacity`, if the feed happens to carry one — scaled as
   `sqrt(capacity / 5000)`, because a 60,000-seat stadium does not send 12x as
   many diners as a 5,000-seat hall; only a spillover fraction eats nearby
2. the classification **segment** (Sports 1.6, Music 1.3, Arts & Theatre 1.0,
   Miscellaneous 0.8, Film 0.5), refined by **genre** (Football ×1.25,
   Festival ×1.3, Classical ×0.85, …)
3. the venue's count of upcoming events, as a tightly bounded nudge (×0.85 to
   ×1.3) — a venue hosting hundreds of shows a year is usually a major one

Each response says which proxy produced the number, via `size_basis`, so a
genre guess is never mistaken for real capacity.

### How it grades, in practice

| Scenario | Distance | Dist. weight | Size weight | Pressure |
| --- | --- | --- | --- | --- |
| NFL game, capacity 65k | 0.3 mi | 0.98 | 3.61 | **3.00** (capped) |
| NFL game, no capacity reported | 0.3 mi | 0.98 | 2.00 | 1.96 |
| NFL game | 12 mi | 0.20 | 2.00 | 0.40 |
| Arena concert | 1 mi | 0.93 | 1.43 | 1.33 |
| Arena concert | 8 mi | 0.47 | 1.43 | 0.67 |
| Theatre show | 0.5 mi | 0.97 | 1.00 | 0.97 |
| Classical recital | 6 mi | 0.60 | 0.85 | 0.51 |
| Film screening | 14 mi | 0.07 | 0.50 | 0.00 (floored) |

A small show next door beats a stadium event across town — which is the whole
point of the grading.

### Where events come from

Three sources, merged and deduplicated into one score. Each scored event
reports its `source`, so you can always see what produced a number.

| Source | Covers | Cost |
| --- | --- | --- |
| `ticketmaster` | Ticketed events: concerts, shows, pro sports | Free key |
| `espn` | Sports fixtures, including college games | Free, no key |
| `recurring` | Calendar facts you state once: a weekly market, school terms | Free |

Ticketmaster alone was a poor fit for this restaurant. Of the 26 event days
flagged in its own history — a city marathon, a street festival, a convention,
a farmers market, a college game day — only one or two are events Ticketmaster
would ever list.

#### ESPN

ESPN's undocumented scoreboard API needs no key. Two limits found by testing
it, both handled:

- **Venues carry a city and state but no coordinates.** Each venue is geocoded
  once and cached in `data/venue_geocode_cache.json`. A transient failure (a
  rate limit) is deliberately *not* cached, since caching it would poison the
  entry permanently after one bad run.
- **`attendance` is a post-game figure** — a future fixture reports `0`. So an
  upcoming game is sized from a documented per-league typical crowd
  (NFL 68,000, college football 40,000, NBA 18,000, MLB 28,000, NHL 17,000),
  and a real attendance figure is used only where the feed has one.

Set `COVERS_RESTAURANT_REGION` to your state. ESPN's scoreboard is nationwide,
and geocoding every stadium in the country to discover Lumen Field is 2,700
miles from Miami both wastes time and gets the geocoder to rate-limit you. The
region is a cheap coarse filter applied *before* geocoding; the distance
grading still does the precise work. Both `Florida` and `FL` are understood.

This two-stage filter matters in practice. For one Sunday it cut 40+ nationwide
venues to a single Florida game — in Jacksonville, 328.8 miles away — which the
distance grading then correctly scored at zero.

It is unofficial, so it can change without notice. Disable it with
`COVERS_ESPN_ENABLED=false`.

#### Recurring events

The cheapest accuracy win available, because a weekly market is a calendar
fact rather than something to discover. Set them up interactively — it asks a
few questions per event:

```bash
python3 scripts/setup_recurring_events.py
```

It asks for the name, how often it recurs (weekly, a monthly date, a given
weekday each month, annually, or a span of dates), how big it is in plain words
(`tiny`/`small`/`medium`/`large`/`huge`), how far away it is, and optionally
the dates between which the rule applies — so a market that started last year
is not applied to older history. It then previews the next 28 days so a
mistyped rule is obvious immediately.

Re-running it is safe: existing rules are listed first and you can add to or
remove them. Results go to `recurring_events.json`, which is plain JSON and
fine to hand-edit, and is read per request so changes need no restart.

Your own sizing always beats the heuristics — `size_basis` reports `operator`
for these.

```bash
curl -s localhost:8000/recurring-events | python3 -m json.tool
```

### Calibration, and why it matters

`is_local_event` in this project's history is **strictly binary**: 706 zeros
and 26 ones. The fitted coefficient therefore means *"the effect of a typical
event day."* So pressure is deliberately calibrated so that **1.0 is a typical
event day** — one reasonably notable event, reasonably close:

- `0.0` → nothing nearby, no effect
- `1.0` → exactly the effect the model was trained on
- `2.0` → about twice that effect

Feeding an uncalibrated score into a coefficient fitted at 0/1 would repeat the
reservations-scaling bug described below. Because the coefficient was only ever
observed at 0 and 1, values far above 1 are linear extrapolation beyond the
fitted range, which is what `COVERS_EVENT_PRESSURE_CAP` bounds. Raise it only
once you have genuinely continuous event history.

Set `COVERS_EVENT_SCORING_ENABLED=false` to revert to the old binary flag.

### Pin down your coordinates first

Grading is only as precise as the origin. Your current config holds
`"Miami, Florida, United States"` — a **city centroid**, which makes a 2-mile
versus 8-mile distinction meaningless. Resolve your actual address once:

```bash
curl -s 'localhost:8000/geocode?address=1601+Biscayne+Blvd,+Miami,+FL+33132'
```

Then put the returned `latitude` / `longitude` in `.env`. Street-level
geocoding uses OpenStreetMap Nominatim, since Open-Meteo's geocoder resolves
place names rather than street addresses.

### Inspecting and overriding

See exactly what the scorer found for a date:

```bash
curl -s 'localhost:8000/events?date=2026-10-10' | python3 -m json.tool
```

Ticketmaster lists only ticketed events, so street festivals, conventions and
private buyouts are invisible to it. Supply them yourself with a graded value
rather than a flag:

```bash
curl -s -X POST localhost:8000/forecast -H 'content-type: application/json' -d '{"days": 7, "overrides": [{"date": "2026-10-10", "is_local_event": 2.0}]}'
```

Setting it skips the lookup for that date. Note that event history still cannot
be backfilled — Ticketmaster has no historical archive — so the coefficient is
only as good as the event days already flagged in your CSV.

## Reservations scaling: a fix

The CLI has a bug this service does not. When reservations are used as a
feature, the CLI estimates a future day's reservations as that weekday's
average **covers** (`covers_forecast_cli.py:862`). Covers and reservations are
on different scales — in this project's data, reservations run about 55% of
covers — so the model, having learned `covers ≈ 1.8 × reservations`, is handed
a number roughly 1.8× too large and inflates every forecast.

On the real data in this repo the effect is about **+35%**:

| | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLI logic | 162 | 178 | 210 | 241 | 314 | 357 | 257 |
| Fixed | 118 | 129 | 151 | 174 | 219 | 248 | 178 |

The service learns the reservations-to-covers ratio at fit time and scales the
estimate by it. The fixed numbers sit sensibly beside the 7-day lag and the
weekday average; the old ones did not.

**The notebooks and the CLI still have this bug.** Worth fixing there too, or
their numbers will disagree with the API's.

## Architecture

```
app/
  main.py              ASGI app, lifespan, CORS
  config.py            Settings (env + .env), falls back to the CLI's JSON config
  schemas.py           Request/response models -> validation + OpenAPI
  service.py           CoversService: wiring, model caching
  store.py             SQLite: daily_covers + forecast_log
  routers/api.py       HTTP endpoints (thin)
  core/
    model.py           Feature building, fit, predict, forecast
    enrichment.py      Weather / holidays / geocoding (degrade, never raise)
    events.py          CandidateEvent + distance x size scoring
    event_sources.py   Ticketmaster / ESPN / recurring, merged and deduplicated
    recurring.py       Recurrence rules and their JSON config
scripts/
  setup_recurring_events.py   Interactive recurring-event setup
  sources/
    base.py            CoversSource protocol
    csv_source.py      CSV export (default, no credentials)
    toast.py           Toast auth + orders -> daily covers
tests/                 61 tests, no network, no credentials
```

Adding another POS means writing one `CoversSource` implementation; nothing
downstream changes.

## Tests

```bash
.venv-api/bin/python -m pytest tests -q
```

All HTTP is faked and enrichment is disabled, so the suite needs no network and
no Toast credentials, and never touches the real data files or database.

## Operational notes

- **Keep it current** by putting `POST /ingest` on a schedule (cron, a worker).
  Re-ingesting a date updates it rather than duplicating it, which matters
  because Toast can revise a business date after the fact via late voids.
- **The model is cached** and refit automatically after an ingest or a recorded
  actual. `?refresh=true` forces a refit.
- **CORS is wide open** (`allow_origins=["*"]`) for local dashboard work.
  Tighten it in `app/main.py` before exposing this publicly.
- **There is no authentication.** Run it on localhost or behind something that
  provides auth; do not put it on a public address as-is.
- `.covers_forecast_config.json` still points `processed_path` at the old
  `~/Downloads/...` location. The service ignores that key, but the CLI does
  not, so it is worth correcting.
