"""Named calendar effects: holidays, operator-stated occasions, and shoulders.

Why this module exists
----------------------
A single `is_holiday` flag asserts that every holiday moves covers the same
way. In this restaurant's own history that is badly wrong: Mother's Day runs
about +93 covers against its weekday baseline while New Year's Day runs about
-57. Collapsing both into one coefficient produced +12 -- a number that
describes neither day and misleads on both.

So every occasion carries its own effect.

Shoulders
---------
The days *around* an occasion are often a different kind of day from the
occasion itself. The clearest case here is the Friday after Thanksgiving,
which ran +26 and +16 covers in the two years on file while Thanksgiving
itself was a closed day contributing nothing. A chef will also tell you that
a nearby school's parents weekend fills the room on the Thursday before it
officially starts.

Every occasion therefore gets a shoulder window, not just a hand-picked few.
Whether a shoulder *matters* is decided by the data, not by us: a day whose
neighbours behave like ordinary days gets an effect near zero and changes no
forecast. Offering the window costs nothing; withholding it means never
discovering the effect.

Why effects are pooled rather than fitted freely
------------------------------------------------
Two years of history gives two observations per occasion. Fitted freely, a
per-occasion coefficient would memorise those two days. Worse, it would report
the same confidence for MLK Day, whose two years agree within 1 cover, as for
Mother's Day, whose two years disagree by 200.

Each effect is therefore shrunk toward the mean of its pool:

    effect = w * own_mean + (1 - w) * pool_mean,   w = n / (n + k)

An occasion with consistent history keeps close to its own number. One with
noisy or scarce history is pulled toward the pool until it earns otherwise --
automatically, as more years arrive, with no code change.

Generality
----------
Nothing here knows any holiday by name. Anchors arrive from whatever source
the restaurant uses (Nager.Date for its country, plus operator-stated
occasions), pools default to "all holidays" rather than to US-specific
groupings, and the shoulder window is uniform. A restaurant in Lisbon or Osaka
gets the same treatment as one in Miami.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# How far either side of an occasion we are willing to look. Three days
# reaches the Friday after a Thursday holiday and the weekend before a Monday
# one, without inventing day-types for a whole fortnight.
DEFAULT_SHOULDER_BEFORE = 3
DEFAULT_SHOULDER_AFTER = 3

# Pooling strength. k = 2 puts an occasion with two years of history halfway
# between its own average and its pool's. Raise it to be more conservative
# while history is thin.
DEFAULT_POOL_K = 2.0

# Pool names. These are deliberately generic -- an occasion is pooled with
# others of its own structural kind, never by any culture-specific notion of
# what a holiday "is for".
POOL_ANCHOR = "anchor"
POOL_SHOULDER = "shoulder"


@dataclass(frozen=True)
class Occasion:
    """A named day-type the model may learn an effect for.

    `shoulder_before` / `shoulder_after` are per-occasion so an operator can
    widen the window for something they know spills (a multi-day festival)
    or close it entirely for something they know does not.
    """

    name: str
    shoulder_before: int = DEFAULT_SHOULDER_BEFORE
    shoulder_after: int = DEFAULT_SHOULDER_AFTER
    # Optional operator tag. Occasions sharing a tag pool together, which
    # helps a new occasion borrow strength from similar ones. Left unset,
    # everything pools globally.
    tag: Optional[str] = None
    # Set when the restaurant is shut on the occasion itself. The anchor then
    # carries no effect of its own (there are no covers to explain) but its
    # shoulders still do -- which is the whole point for Christmas and
    # Thanksgiving.
    closed: bool = False

    def labels_for_offsets(self) -> List[Tuple[int, str]]:
        """Every (offset, label) this occasion can stamp on a calendar."""
        out: List[Tuple[int, str]] = [(0, self.name)]
        for d in range(1, self.shoulder_before + 1):
            out.append((-d, f"{self.name}-{d}"))
        for d in range(1, self.shoulder_after + 1):
            out.append((d, f"{self.name}+{d}"))
        return out


def label_pool(label: str) -> str:
    """Which pool a label's effect is shrunk toward."""
    return POOL_SHOULDER if ("+" in label or "-" in label) else POOL_ANCHOR


@dataclass
class CalendarIndex:
    """Maps each date to the single day-type label that best describes it.

    Precedence matters, because occasions overlap. Christmas Eve is both a
    holiday in its own right and Christmas minus one; New Year's Eve is a
    holiday and also New Year's Day minus one. Letting a date carry both
    labels would put two near-identical columns in the design matrix and split
    an effect across them.

    The rules, in order:
      1. A date that *is* an occasion takes that occasion's own name.
      2. Otherwise it takes the label of the nearest anchor.
      3. Ties go to the occasion whose anchor falls earlier, so a run of days
         between two occasions is divided predictably rather than arbitrarily.
    """

    labels: Dict[dt.date, str] = field(default_factory=dict)
    # Kept for explanation: which anchor each label came from, so a forecast
    # can say "the Friday after Thanksgiving" rather than just a label.
    anchors: Dict[dt.date, str] = field(default_factory=dict)

    def label_for(self, day: dt.date) -> Optional[str]:
        return self.labels.get(day)


def build_index(
    occurrences: Sequence[Tuple[dt.date, Occasion]],
) -> CalendarIndex:
    """Stamp labels across the calendar from dated occasion occurrences.

    `occurrences` pairs each date an occasion falls on with the occasion
    itself, so a recurring rule contributes one entry per occurrence and a
    holiday one per year.
    """
    index = CalendarIndex()
    # (distance, anchor_date, label) per date, keeping the best claim.
    best: Dict[dt.date, Tuple[int, dt.date, str]] = {}

    for anchor_date, occ in occurrences:
        for offset, label in occ.labels_for_offsets():
            day = anchor_date + dt.timedelta(days=offset)
            claim = (abs(offset), anchor_date, label)
            held = best.get(day)
            # Lower distance wins; an anchor (distance 0) always beats a
            # shoulder. Equal distance goes to the earlier anchor date.
            if held is None or claim[:2] < held[:2]:
                best[day] = claim
                index.anchors[day] = occ.name

    index.labels = {day: claim[2] for day, claim in best.items()}
    return index


def shrink(
    own_sum: float,
    own_n: int,
    pool_mean: float,
    k: float = DEFAULT_POOL_K,
) -> float:
    """Partial-pooling estimate for one label's effect.

    Returns the pool mean outright when the label has no observations of its
    own, which is what makes an unseen occasion safe to forecast: it inherits
    the typical behaviour of its pool rather than contributing a wild guess.
    """
    if own_n <= 0:
        return pool_mean
    w = own_n / (own_n + k)
    own_mean = own_sum / own_n
    return w * own_mean + (1.0 - w) * pool_mean


@dataclass
class EffectTable:
    """Fitted per-label effects, in covers, plus the evidence behind each."""

    effects: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    raw_means: Dict[str, float] = field(default_factory=dict)
    pool_means: Dict[str, float] = field(default_factory=dict)
    k: float = DEFAULT_POOL_K

    def effect_for(self, label: Optional[str]) -> float:
        if not label:
            return 0.0
        if label in self.effects:
            return self.effects[label]
        # An occasion seen for the first time at forecast time still gets the
        # behaviour typical of its pool.
        return self.pool_means.get(label_pool(label), 0.0)

    def explain(self, label: Optional[str]) -> Optional[Dict[str, float]]:
        """Why this label carries the effect it does -- for the API response."""
        if not label or label not in self.effects:
            return None
        n = self.counts.get(label, 0)
        return {
            "label": label,
            "effect_covers": round(self.effects[label], 1),
            "observations": n,
            "own_mean": round(self.raw_means.get(label, 0.0), 1),
            "pool_mean": round(self.pool_means.get(label_pool(label), 0.0), 1),
            # How much of the effect is the label's own evidence rather than
            # borrowed from its pool. Near 0 means "we do not really know yet".
            "self_weight": round(n / (n + self.k), 2) if n else 0.0,
        }


def fit_effects(
    residuals_by_label: Dict[str, List[float]],
    k: float = DEFAULT_POOL_K,
) -> EffectTable:
    """Estimate each label's effect from model residuals on its days.

    Residuals, not raw covers: whatever the day-of-week, lag and weather terms
    already explain must not be re-charged to the occasion. What is left is
    the part the calendar has to answer for.
    """
    table = EffectTable(k=k)

    # Pool means first -- each is the average residual across every label of
    # that kind, so a label with thin evidence falls back on how occasions of
    # its kind behave in general.
    pools: Dict[str, List[float]] = {}
    for label, values in residuals_by_label.items():
        pools.setdefault(label_pool(label), []).extend(values)
    table.pool_means = {
        pool: (sum(v) / len(v) if v else 0.0) for pool, v in pools.items()
    }

    for label, values in residuals_by_label.items():
        n = len(values)
        pool_mean = table.pool_means.get(label_pool(label), 0.0)
        table.counts[label] = n
        table.raw_means[label] = sum(values) / n if n else 0.0
        table.effects[label] = shrink(sum(values), n, pool_mean, k)

    return table
