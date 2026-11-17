"""Recurring events the operator knows about.

Ticketmaster lists ticketed events; ESPN lists fixtures. Neither knows that a
farmers market fills your street every Saturday morning, that the local school
term starts in August, or that a convention centre two blocks away runs an
expo every March. Those are deterministic calendar facts, not API lookups --
cheaper and more reliable to state once than to discover.

This module holds those rules and answers "does any of them fall on this
date?". Rules live in `recurring_events.json`, which is per-restaurant and
written by `scripts/setup_recurring_events.py`.

Recurrence kinds
----------------
  weekly          every Tuesday
  monthly_date    the 15th of every month
  monthly_weekday the first Saturday of every month
  annual          every 4 July
  date_range      a one-off or seasonal span (a school term, a festival week)

Every rule may carry a `start_date` / `end_date` so a market that began last
spring is not applied to history from before it existed.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

KINDS = ("weekly", "monthly_date", "monthly_weekday", "annual", "date_range")

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Named sizes, so the interactive setup can ask "how big?" in plain words
# rather than demanding a weight. These are multiples of a typical event,
# matching the calibration in app.core.events (1.0 = a typical event day).
SIZE_PRESETS: Dict[str, float] = {
    "tiny": 0.25,
    "small": 0.5,
    "medium": 1.0,
    "large": 1.75,
    "huge": 2.5,
}
DEFAULT_SIZE = "medium"


class RecurringConfigError(Exception):
    """A rule could not be understood."""


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass
class RecurringEvent:
    """One recurring event rule."""

    name: str
    kind: str
    # weekly / monthly_weekday: 0 = Monday ... 6 = Sunday
    weekday: Optional[int] = None
    # monthly_date / annual: day of the month
    day_of_month: Optional[int] = None
    # annual: month number; monthly_weekday: which occurrence (1 = first,
    # -1 = last)
    month: Optional[int] = None
    nth: Optional[int] = None
    # date_range, and optional bounds on any other kind
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    size: str = DEFAULT_SIZE
    # How far away it is. 0 means "at the restaurant", which is the usual case
    # for a market on your own street.
    distance_miles: float = 0.0
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise RecurringConfigError(
                f"{self.name!r}: unknown kind {self.kind!r} (expected one of {', '.join(KINDS)})"
            )
        if self.size not in SIZE_PRESETS:
            raise RecurringConfigError(
                f"{self.name!r}: unknown size {self.size!r} "
                f"(expected one of {', '.join(SIZE_PRESETS)})"
            )
        if self.kind == "weekly" and self.weekday is None:
            raise RecurringConfigError(f"{self.name!r}: a weekly rule needs a weekday")
        if self.kind == "monthly_date" and self.day_of_month is None:
            raise RecurringConfigError(f"{self.name!r}: a monthly_date rule needs day_of_month")
        if self.kind == "monthly_weekday" and (self.weekday is None or self.nth is None):
            raise RecurringConfigError(
                f"{self.name!r}: a monthly_weekday rule needs both weekday and nth"
            )
        if self.kind == "annual" and (self.month is None or self.day_of_month is None):
            raise RecurringConfigError(
                f"{self.name!r}: an annual rule needs both month and day_of_month"
            )
        if self.kind == "date_range" and not (self.start_date and self.end_date):
            raise RecurringConfigError(
                f"{self.name!r}: a date_range rule needs both start_date and end_date"
            )

    @property
    def size_weight(self) -> float:
        return SIZE_PRESETS[self.size]

    def _within_bounds(self, day: dt.date) -> bool:
        if self.start_date and day < dt.date.fromisoformat(self.start_date):
            return False
        if self.end_date and day > dt.date.fromisoformat(self.end_date):
            return False
        return True

    def occurs_on(self, day: dt.date) -> bool:
        """Does this rule fall on `day`?"""
        if not self._within_bounds(day):
            return False

        if self.kind == "date_range":
            # Bounds are required for this kind and already checked above.
            return True
        if self.kind == "weekly":
            return day.weekday() == self.weekday
        if self.kind == "monthly_date":
            return day.day == self.day_of_month
        if self.kind == "annual":
            return day.month == self.month and day.day == self.day_of_month
        if self.kind == "monthly_weekday":
            return self._is_nth_weekday(day)
        return False

    def _is_nth_weekday(self, day: dt.date) -> bool:
        if day.weekday() != self.weekday:
            return False
        if self.nth and self.nth > 0:
            # The 1st..5th occurrence: days 1-7 are the first, 8-14 the second.
            return (day.day - 1) // 7 + 1 == self.nth
        # A negative nth counts back from the end of the month.
        days_in_month = calendar.monthrange(day.year, day.month)[1]
        from_end = (days_in_month - day.day) // 7 + 1
        return from_end == abs(self.nth)

    def describe(self) -> str:
        """Plain-English summary, for listings and the setup script."""
        if self.kind == "weekly":
            base = f"every {WEEKDAY_NAMES[self.weekday]}"
        elif self.kind == "monthly_date":
            base = f"the {_ordinal(self.day_of_month)} of every month"
        elif self.kind == "monthly_weekday":
            which = "last" if self.nth and self.nth < 0 else _ordinal(self.nth or 1)
            base = f"the {which} {WEEKDAY_NAMES[self.weekday]} of every month"
        elif self.kind == "annual":
            base = f"every {self.day_of_month} {calendar.month_name[self.month]}"
        else:
            base = f"{self.start_date} to {self.end_date}"

        extras = [str(self.size)]
        if self.distance_miles:
            extras.append(f"{self.distance_miles:g} mi away")
        if self.kind != "date_range" and (self.start_date or self.end_date):
            extras.append(f"active {self.start_date or '...'} to {self.end_date or '...'}")
        return f"{base} ({', '.join(extras)})"

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RecurringEventSet:
    """All of a restaurant's recurring rules."""

    events: List[RecurringEvent] = field(default_factory=list)

    def occurrences_on(self, day: dt.date) -> List[RecurringEvent]:
        return [e for e in self.events if e.occurs_on(day)]

    def to_list(self) -> List[Dict]:
        return [e.to_dict() for e in self.events]


def load(path: Path) -> RecurringEventSet:
    """Read rules from disk. A missing file is simply an empty set.

    One malformed rule is skipped with a warning rather than taking the whole
    service down -- a typo in an optional config file should not stop
    forecasting.
    """
    path = Path(path)
    if not path.exists():
        return RecurringEventSet()

    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read recurring events from %s: %s", path, exc)
        return RecurringEventSet()

    entries = raw.get("events", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        logger.warning("%s should contain a list of events; ignoring it.", path)
        return RecurringEventSet()

    events = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        try:
            events.append(RecurringEvent(**clean))
        except (RecurringConfigError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid recurring event in %s: %s", path, exc)
    return RecurringEventSet(events=events)


def save(path: Path, event_set: RecurringEventSet) -> None:
    """Write rules to disk, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Recurring events for THIS restaurant. Written by "
            "scripts/setup_recurring_events.py; safe to edit by hand. "
            "size is one of: " + ", ".join(SIZE_PRESETS)
        ),
        "events": event_set.to_list(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
