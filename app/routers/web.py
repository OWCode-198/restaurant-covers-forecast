"""The page a manager actually looks at.

One screen, read on a phone at the end of a shift: how many people to expect
tomorrow, whether that is busy or quiet for that weekday, and what is driving
it. Everything else in this service is machinery for producing this number.

Three deliberate choices:

* **A range, not a decimal.** The model's holdout error on this project's own
  data is around 9 covers on a ~45-cover day. Printing "44.6" would claim a
  precision we do not have, and the first time reality lands on 52 the reader
  stops trusting the page. So: a rounded number, with the typical error beside
  it.
* **A comparison, not just a figure.** "53 covers" means little to someone who
  has not memorised their own averages. "About the same as a normal Wednesday"
  is immediately actionable.
* **No external requests.** No CDN, no web fonts, no analytics. The page is one
  self-contained document, so it renders instantly on bad restaurant wifi and
  leaks nothing about the restaurant to anybody.

Served at `/today`, unauthenticated -- put something in front of it before it
reaches a public host.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import logging
import secrets
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.routers.api import get_service
from app.service import CoversService, ServiceError
from app.sources.base import SourceNotConfigured

logger = logging.getLogger(__name__)

router = APIRouter()

# Within this fraction of the weekday average reads as "a normal <weekday>"
# rather than busy or quiet. Below the model's own error there is nothing worth
# reporting, and 10% sits comfortably above the noise on this project's data.
SAME_AS_USUAL_TOLERANCE = 0.10

STYLE = """
:root {
  --bg: #f6f6f4;
  --card: #ffffff;
  --ink: #16160f;
  --muted: #6b6b63;
  --line: #e2e2dc;
  --accent: #b4441f;
  --bar: #d8d8d1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14140f;
    --card: #1e1e18;
    --ink: #f2f2ec;
    --muted: #9b9b91;
    --line: #2e2e26;
    --accent: #e8794b;
    --bar: #33332b;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px 16px 40px;
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 560px; margin: 0 auto; }
.eyebrow {
  font-size: 12px; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 20px;
}
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 14px; padding: 24px; margin-bottom: 16px;
}
.when { font-size: 15px; color: var(--muted); margin: 0 0 4px; }
.big {
  font-size: 76px; line-height: 1; font-weight: 700;
  letter-spacing: -.03em; margin: 8px 0 2px;
}
.unit { font-size: 15px; color: var(--muted); margin: 0 0 18px; }
.compare { font-size: 17px; margin: 0 0 6px; }
.range { font-size: 14px; color: var(--muted); margin: 0; }
.chips { margin: 18px 0 0; padding: 0; list-style: none; display: flex;
         flex-wrap: wrap; gap: 8px; }
.chip {
  font-size: 13px; padding: 5px 11px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--ink); background: var(--bg);
}
.chip.hot { border-color: var(--accent); color: var(--accent); }
h2 { font-size: 13px; font-weight: 600; letter-spacing: .1em;
     text-transform: uppercase; color: var(--muted); margin: 0 0 14px; }
.row { display: flex; align-items: center; gap: 12px; padding: 9px 0;
       border-top: 1px solid var(--line); }
.row:first-of-type { border-top: 0; }
.row .day { width: 104px; flex: none; font-size: 15px; }
.row .track { flex: 1; height: 8px; background: var(--bar); border-radius: 4px; }
.row .fill { display: block; height: 100%; background: var(--accent);
             border-radius: 4px; }
.row .n { width: 42px; flex: none; text-align: right;
          font-variant-numeric: tabular-nums; font-weight: 600; }
footer { font-size: 13px; color: var(--muted); line-height: 1.7; }
.stale { border-color: var(--accent); }
.stale .problem { color: var(--accent); font-weight: 600; }
.problem { font-size: 17px; margin: 0 0 10px; }
.problem + p { font-size: 14px; color: var(--muted); margin: 0; }
"""


def published_accuracy(settings) -> Optional[Dict]:
    """The walk-forward score, if `scripts/evaluate.py --save` published one.

    The model's own `accuracy` field comes from a single hold-out window whose
    features are built from the whole history, so a test day deep inside that
    window can see actuals from inside it. On this project's data that reports
    ~8.8 MAE where honest walk-forward measurement gives ~10.6. Telling a
    manager "usually within 9" when it is really 11 is the kind of small lie
    that costs their trust in the first week, so prefer the honest number
    whenever one is available.
    """
    path = Path(settings.db_path).parent / "walk_forward_accuracy.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "mae" in data else None


def esc(value: Optional[object]) -> str:
    """Escape anything that reaches the page.

    Holiday and event names arrive from third-party feeds, so they are
    untrusted text however innocuous they look.
    """
    return html.escape(str(value), quote=True) if value is not None else ""


def friendly_date(iso: str) -> str:
    """'2026-10-16' -> 'Friday 16 October'. %-d is not portable, so do it here."""
    day = dt.date.fromisoformat(iso)
    return f"{day.strftime('%A')} {day.day} {day.strftime('%B')}"


def short_date(iso: str) -> str:
    """'2026-10-16' -> 'Fri 16 Oct'."""
    day = dt.date.fromisoformat(iso)
    return f"{day.strftime('%a')} {day.day} {day.strftime('%b')}"


def lead_label(iso: str, today: Optional[dt.date] = None) -> str:
    """'Tomorrow', 'Today', or nothing -- never a guess.

    The first forecast day is the day after the last day of *history*, which is
    only tomorrow if ingest is current. When it has fallen behind, calling a
    month-old date "Tomorrow" is actively misleading, so the label is dropped
    and `staleness_note` explains why.
    """
    day = dt.date.fromisoformat(iso)
    today = today or dt.date.today()
    if day == today + dt.timedelta(days=1):
        return "Tomorrow · "
    if day == today:
        return "Today · "
    return ""


def staleness_note(
    last_history: Optional[str],
    today: Optional[dt.date] = None,
    threshold: int = 2,
) -> str:
    """Warn when the history behind the number has gone stale.

    A silently out-of-date forecast is worse than no forecast: it looks
    authoritative and is wrong. Two days of slack absorbs a missed overnight
    run without crying wolf.
    """
    if not last_history:
        return ""
    behind = ((today or dt.date.today()) - dt.date.fromisoformat(last_history)).days
    if behind <= threshold:
        return ""
    return (
        f'<div class="card stale"><p class="problem">Data is {behind} days behind.</p>'
        f"<p>The most recent day recorded is {esc(short_date(last_history))}. "
        "This forecast is based on history up to then, so treat it with care "
        "until the next import runs.</p></div>"
    )


def comparison(forecast: float, typical: Optional[float], weekday: str) -> str:
    """Put the number in terms the reader already has a feel for."""
    if not typical or typical <= 0:
        return f"No {weekday} average to compare against yet."

    shift = (forecast - typical) / typical
    if abs(shift) < SAME_AS_USUAL_TOLERANCE:
        return f"About the same as a normal {weekday} ({round(typical)})."
    direction = "Busier" if shift > 0 else "Quieter"
    updown = "up" if shift > 0 else "down"
    return (
        f"{direction} than a normal {weekday} ({round(typical)}) "
        f"— {abs(shift) * 100:.0f}% {updown}."
    )


def chips_for(day: Dict) -> List[str]:
    """Only the drivers actually in play; an empty list is the normal case."""
    out: List[str] = []
    if day.get("is_holiday"):
        name = day.get("holiday_name")
        out.append(f'<li class="chip hot">{esc(name) if name else "Public holiday"}</li>')
    if day.get("is_local_event"):
        label = esc(day.get("event_name") or "Event nearby")
        miles = day.get("nearest_event_miles")
        if isinstance(miles, (int, float)):
            label += f" · {miles:.1f} mi"
        out.append(f'<li class="chip hot">{label}</li>')
    # Deliberately the heavy-rain flag: `is_rain` is true on four days in five
    # here, so a chip driven by it said "Rain expected" almost every day and
    # taught managers to ignore the row.
    if day.get("is_rain_heavy"):
        out.append('<li class="chip">Heavy rain expected</li>')
    if day.get("is_promo"):
        out.append(f'<li class="chip">{esc(day.get("promo_name") or "Promotion running")}</li>')
    return out


def render_page(body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        "<title>Covers forecast</title>"
        f"<style>{STYLE}</style></head>"
        f'<body><div class="wrap">{body}</div></body></html>'
    )


def render_problem(heading: str, detail: str) -> str:
    return render_page(
        '<p class="eyebrow">Covers forecast</p>'
        f'<div class="card"><p class="problem">{esc(heading)}</p>'
        f"<p>{esc(detail)}</p></div>"
    )


def render_forecast(
    payload: Dict,
    published: Optional[Dict] = None,
    stale_after_days: int = 2,
) -> str:
    days = payload.get("forecast") or []
    if not days:
        return render_problem(
            "No forecast available.",
            "The model produced no days. Check that history has been ingested.",
        )

    first, rest = days[0], days[1:]
    accuracy = payload.get("accuracy") or {}
    # Walk-forward beats the in-process hold-out; see published_accuracy().
    honest = published or {}
    mae = honest.get("mae", accuracy.get("holdout_mae"))
    mape = honest.get("mape", accuracy.get("holdout_mape"))

    raw = float(first["forecast_covers"])
    expected = round(raw)
    weekday = first.get("day_of_week", "day")

    range_line = ""
    if isinstance(mae, (int, float)) and mae > 0:
        margin = round(mae)
        range_line = (
            f'<p class="range">Usually within {margin} either way, so expect '
            f"roughly {max(0, expected - margin)}–{expected + margin}.</p>"
        )

    chips = chips_for(first)
    head = (
        '<p class="eyebrow">Covers forecast</p>'
        '<div class="card">'
        f'<p class="when">{esc(lead_label(first["date"]))}'
        f'{esc(friendly_date(first["date"]))}</p>'
        f'<p class="big">{expected}</p>'
        '<p class="unit">covers expected</p>'
        f'<p class="compare">'
        f'{esc(comparison(raw, first.get("covers_rollavg_4wk_weekday"), weekday))}</p>'
        f"{range_line}"
        + (f'<ul class="chips">{"".join(chips)}</ul>' if chips else "")
        + "</div>"
    )

    week = ""
    if rest:
        peak = max(float(d["forecast_covers"]) for d in rest) or 1.0
        rows = []
        for day in rest:
            value = float(day["forecast_covers"])
            # A floor keeps a quiet day visible rather than rendering nothing.
            width = max(3.0, value / peak * 100.0)
            rows.append(
                '<div class="row">'
                f'<span class="day">{esc(short_date(day["date"]))}</span>'
                f'<span class="track"><span class="fill" style="width:{width:.0f}%"></span></span>'
                f'<span class="n">{round(value)}</span>'
                "</div>"
            )
        week = f'<div class="card"><h2>After that</h2>{"".join(rows)}</div>'

    bits: List[str] = []
    if payload.get("last_history_date"):
        bits.append(f'Data through {short_date(payload["last_history_date"])}')
    if payload.get("trained_days"):
        bits.append(f'{payload["trained_days"]} days of history')
    if isinstance(mae, (int, float)):
        bits.append(
            f"typical error {mae:.0f} covers"
            + (f" ({mape:.0f}%)" if isinstance(mape, (int, float)) else "")
        )
    if payload.get("source"):
        bits.append(f'source: {payload["source"]}')
    footer = f"<footer>{esc(' · '.join(bits))}</footer>" if bits else ""

    stale = staleness_note(payload.get("last_history_date"), threshold=stale_after_days)
    return render_page(stale + head + week + footer)


_basic = HTTPBasic(auto_error=False)


def check_access(
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic),
    service: CoversService = Depends(get_service),
) -> None:
    """Gate the page behind a password, but only once one is configured.

    No password set means open, which keeps local development a single command
    and matches how this runs on a laptop. Once `COVERS_WEB_PASSWORD` is set --
    which any internet-reachable deployment must do, since the page shows a
    real business's daily trade -- the page asks for it. Compared with
    compare_digest so a wrong guess takes the same time as a right one.
    """
    expected = service.settings.web_password
    if not expected:
        return

    supplied_user = credentials.username if credentials else ""
    supplied_pass = credentials.password if credentials else ""
    ok = secrets.compare_digest(
        supplied_user, service.settings.web_username
    ) & secrets.compare_digest(supplied_pass, expected)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Not authorised.",
            headers={"WWW-Authenticate": 'Basic realm="Covers forecast"'},
        )


@router.get("/today", response_class=HTMLResponse, include_in_schema=False)
def today(
    days: int = Query(7, ge=1, le=14),
    service: CoversService = Depends(get_service),
    _: None = Depends(check_access),
) -> HTMLResponse:
    """The one page a manager reads. Never logs: looking is not forecasting."""
    try:
        payload = service.forecast(days=days, log=False)
    except SourceNotConfigured as exc:
        return HTMLResponse(render_problem("Not configured yet.", str(exc)), status_code=503)
    except ServiceError as exc:
        return HTMLResponse(
            render_problem("Not enough history to forecast yet.", str(exc)), status_code=503
        )
    return HTMLResponse(
        render_forecast(
            payload,
            published_accuracy(service.settings),
            stale_after_days=service.settings.stale_after_days,
        )
    )
