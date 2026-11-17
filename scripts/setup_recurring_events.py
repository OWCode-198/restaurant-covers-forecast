#!/usr/bin/env python3
"""Interactive setup for recurring events.

Ticketmaster lists ticketed events and ESPN lists fixtures. Neither knows that
a farmers market fills your street every Saturday, that a convention centre
nearby runs an expo each March, or when the local school term starts -- yet in
this project's own history those are most of the flagged event days.

Those are calendar facts, so this asks you for them once and writes
`recurring_events.json`, which the forecaster then reads for every date.

    python3 scripts/setup_recurring_events.py

It is safe to re-run: existing rules are listed first, and you can add to them,
remove them, or leave them alone. The file is plain JSON and fine to hand-edit.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# Allow running this directly from the project root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.recurring import (  # noqa: E402
    DEFAULT_SIZE,
    SIZE_PRESETS,
    WEEKDAY_NAMES,
    RecurringConfigError,
    RecurringEvent,
    RecurringEventSet,
    load,
    save,
)


# --- small prompt helpers --------------------------------------------------


def banner(text: str) -> None:
    print(f"\n{text}")
    print("=" * len(text))


def ask(prompt: str, default: str = None, required: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        if not required:
            return ""
        print("  Please enter a value.")


def ask_yes_no(prompt: str, default: str = "n") -> bool:
    hint = "Y/n" if default == "y" else "y/N"
    while True:
        answer = input(f"{prompt} [{hint}]: ").strip().lower()
        if not answer:
            answer = default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


def ask_choice(prompt: str, options, default: str = None) -> str:
    """Pick one of `options`, accepting a number or the value itself."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = "  <- default" if opt == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        answer = input("Choose a number" + (f" [{default}]" if default else "") + ": ").strip()
        if not answer and default is not None:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options:
            return answer
        print(f"  Please choose 1-{len(options)}.")


def ask_int(prompt: str, low: int, high: int, default: int = None) -> int:
    while True:
        raw = ask(prompt, default=str(default) if default is not None else None, required=True)
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"  Please enter a number between {low} and {high}.")


def ask_float(prompt: str, default: float = None) -> float:
    while True:
        raw = ask(prompt, default=str(default) if default is not None else None)
        if not raw:
            return default if default is not None else 0.0
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number (e.g. 0, 0.5, 2).")


def ask_date(prompt: str, allow_blank: bool = True):
    while True:
        raw = ask(prompt + " (YYYY-MM-DD)" + (" or blank" if allow_blank else ""))
        if not raw and allow_blank:
            return None
        try:
            return dt.date.fromisoformat(raw).isoformat()
        except ValueError:
            print("  Please use YYYY-MM-DD, e.g. 2026-03-15.")


# --- the questions ---------------------------------------------------------

KIND_LABELS = {
    "Every week on the same day (a Saturday market)": "weekly",
    "The same date each month (the 15th)": "monthly_date",
    "A particular weekday each month (first Saturday)": "monthly_weekday",
    "Once a year (a founders' day, an annual parade)": "annual",
    "A span of dates (a school term, a festival week)": "date_range",
}

SIZE_HELP = {
    "tiny": "barely noticeable, a handful of extra people",
    "small": "noticeable but modest",
    "medium": "a typical event day for you",
    "large": "clearly busier than a typical event",
    "huge": "one of your busiest days of the year",
}


def ask_about_one_event() -> RecurringEvent:
    """Walk through one recurring event."""
    name = ask("\n  Name (e.g. 'Saturday Farmers Market')", required=True)

    label = ask_choice("  How often does it happen?", list(KIND_LABELS), None)
    kind = KIND_LABELS[label]

    fields = {"name": name, "kind": kind}

    if kind == "weekly":
        day = ask_choice("  Which day?", WEEKDAY_NAMES, None)
        fields["weekday"] = WEEKDAY_NAMES.index(day)

    elif kind == "monthly_date":
        fields["day_of_month"] = ask_int("  Day of the month (1-31)", 1, 31)

    elif kind == "monthly_weekday":
        which = ask_choice(
            "  Which occurrence?", ["first", "second", "third", "fourth", "last"], "first"
        )
        fields["nth"] = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": -1}[which]
        day = ask_choice("  Which weekday?", WEEKDAY_NAMES, None)
        fields["weekday"] = WEEKDAY_NAMES.index(day)

    elif kind == "annual":
        fields["month"] = ask_int("  Month (1-12)", 1, 12)
        fields["day_of_month"] = ask_int("  Day of the month (1-31)", 1, 31)

    else:  # date_range
        print("  A span of dates, e.g. a school term or a festival week.")
        fields["start_date"] = ask_date("  First day", allow_blank=False)
        fields["end_date"] = ask_date("  Last day", allow_blank=False)

    print("\n  How big is it, relative to a normal event day?")
    for key, help_text in SIZE_HELP.items():
        print(f"    {key:<8} x{SIZE_PRESETS[key]:<5} {help_text}")
    fields["size"] = ask_choice("  Size", list(SIZE_PRESETS), DEFAULT_SIZE)

    print("\n  How far from the restaurant? 0 means on your own block.")
    fields["distance_miles"] = ask_float("  Distance in miles", default=0.0)

    note = ask("  A note to remember what this is (optional)")
    if note:
        fields["note"] = note

    if kind != "date_range":
        print("\n  Optional: limit which dates this applies to. Useful if the")
        print("  market only started last year, so it is not applied to older history.")
        if ask_yes_no("  Set a start or end date?", default="n"):
            fields["start_date"] = ask_date("  Active from")
            fields["end_date"] = ask_date("  Active until")

    return RecurringEvent(**{k: v for k, v in fields.items() if v is not None})


def show(event_set: RecurringEventSet) -> None:
    if not event_set.events:
        print("  (none yet)")
        return
    for i, e in enumerate(event_set.events, 1):
        print(f"  {i}. {e.name} -- {e.describe()}")
        if e.note:
            print(f"       note: {e.note}")


def preview(event_set: RecurringEventSet, days: int = 28) -> None:
    """Show which of the next few weeks' dates the rules land on."""
    if not event_set.events:
        return
    banner(f"Next {days} days, as the forecaster will see them")
    today = dt.date.today()
    any_hit = False
    for i in range(days):
        day = today + dt.timedelta(days=i)
        hits = event_set.occurrences_on(day)
        if hits:
            any_hit = True
            names = ", ".join(f"{h.name} (x{h.size_weight:g})" for h in hits)
            print(f"  {day.isoformat()} {day.strftime('%a')}  {names}")
    if not any_hit:
        print("  No occurrences in that window -- check the days and date bounds above.")


def main() -> None:
    settings = get_settings()
    path = Path(settings.recurring_events_path)

    banner("Recurring events setup")
    print("Events that happen on a schedule, which no ticketing API knows about:")
    print("a weekly market, a monthly art walk, school terms, an annual parade.")
    print(f"\nThese are saved to: {path}")

    event_set = load(path)
    print("\nCurrently configured:")
    show(event_set)

    if event_set.events and ask_yes_no("\nRemove any of these?", default="n"):
        while event_set.events:
            show(event_set)
            raw = ask("  Number to remove, or blank to stop")
            if not raw:
                break
            if raw.isdigit() and 1 <= int(raw) <= len(event_set.events):
                removed = event_set.events.pop(int(raw) - 1)
                print(f"  Removed {removed.name!r}.")
            else:
                print("  Not a listed number.")

    print()
    add_first = "y" if not event_set.events else "n"
    if ask_yes_no("Add a recurring event?", default=add_first):
        while True:
            try:
                event = ask_about_one_event()
            except RecurringConfigError as exc:
                print(f"\n  That rule was not valid: {exc}")
                if ask_yes_no("  Try again?", default="y"):
                    continue
                break
            event_set.events.append(event)
            print(f"\n  Added: {event.name} -- {event.describe()}")
            if not ask_yes_no("\nAdd another?", default="n"):
                break
    elif not event_set.events:
        print("\nNothing configured. You can run this again at any time.")
        return

    save(path, event_set)
    banner("Saved")
    show(event_set)
    preview(event_set)
    print(f"\nWritten to {path}.")
    print("The forecaster picks these up automatically -- the file is read per")
    print("request, so a new forecast needs no restart.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. Nothing was saved.")
        sys.exit(1)
