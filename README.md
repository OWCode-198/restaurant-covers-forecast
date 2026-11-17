# Predictive Covers

Daily covers forecasting for a restaurant. Toast POS history, enriched with weather,
public holidays and nearby ticketed events, fed to an interpretable least-squares
regression. It runs as notebooks, as a FastAPI service, and as a web page a manager can
read on a Monday morning.

The part worth reading is not the forecast. It is how the accuracy number was arrived at,
and the first number I had to throw away to get there.

---

## The number I threw away

The first accuracy figure this project produced was good. A single hold-out window, a
model fit on everything before it, an R² in the low-to-mid eighties. It was also wrong,
and the reason is worth more than the number.

The features were built from the whole history and then split into train and test. One
of them, `covers_rollavg_4wk_weekday`, is a four-week rolling average of covers on the
same weekday. For a test day deep inside the hold-out, that average is computed from
actual covers one, two and three weeks earlier, and those days are themselves inside the
hold-out. They are days I would not have when forecasting forward.

The model was being graded on an exam it had partly seen. The further into the window a
day sat, the more it knew that it should not have known.

There was a second problem underneath the first. One hold-out window is one sample.
Several candidate changes to this project looked like five to eight percent improvements
on that window and were worth nothing across five of them.

## What replaced it

`scripts/evaluate.py` walks an origin forward through the history. At each origin it
refits on data up to that origin only, then forecasts the next seven days through the
same code path the web page uses. Nothing after the origin is visible to the fit.
Historical weather and holidays for the target days are passed as recorded overrides, so
the backtest sees the conditions that actually occurred rather than calling a live API
and getting today's.

It scores two baselines alongside the model, and the choice of baseline is deliberate:

- **Last week, same weekday.** The forecast a manager already makes in their head.
- **Four-week weekday average.** The obvious alternative, and the one most dashboards
  quietly implement.

A model that cannot beat both is not earning its complexity.

## Results

432 forecasts from 62 origins, 9 December 2025 to 16 October 2026, seven-day horizon.

| Method | MAE | MAPE |
|---|---|---|
| **Model** | **10.31** | **17.6%** |
| Four-week weekday average | 11.24 | 19.1% |
| Last week, same weekday | 12.96 | 21.4% |

Point estimates are not enough to claim an improvement, so the harness also runs a paired
comparison per forecast:

| Against | Mean gain | 95% CI | p | Days won |
|---|---|---|---|---|
| Weekday average | +0.93 covers | [+0.40, +1.46] | 0.001 | 57% |
| Last week | +2.72 covers | [+1.76, +3.68] | 0.000 | 59% |

The model is better than both, and the margin over the weekday average is just under one
cover per day. That is a real improvement and a modest one. It wins 57% of days against
that baseline, not all of them.

Accuracy holds across the horizon rather than decaying, because the weekly cycle does
most of the work: MAE is 10.51 at one day ahead and 9.68 at seven. Variance between weeks
is larger than variance across the horizon. Per-origin MAE averages 10.34 with a standard
deviation of 4.82, a best week of 4.56 and a worst of 28.84.

`data/walk_forward_accuracy.json` holds this result, and the running service reads it to
quote its error figure. That file is deliberately not gitignored. Without it, a
deployment falls back to the optimistic in-process hold-out number, which is the number
this README opens by discarding.

## How the features were chosen

Every feature in `NUMERIC_FEATURES` earns its place in the walk-forward backtest. These
were tried and dropped:

| Feature | Why it went |
|---|---|
| `covers_lag_14` | Fitted a negative coefficient on noise. Removing it costs nothing. |
| `temp_high_f` | Coefficient of −0.02. Miami temperature barely moves within a season. |
| `is_rain` | True on 80% of days, because any trace of precipitation counted. |

Replacing `is_rain` with a heavy-rain threshold plus the capped amount was the single
largest accuracy gain available, moving clean MAE from 9.17 to 8.83. A flag that is true
four days in five is not a signal.

Also tried, and neutral or worse: an eight-week weekday average, annual sine and cosine
seasonality, a log target, recency weighting, ridge, Huber and L1 losses, and per-weekday
models.

Two features, `is_local_event` and `is_promo`, are constant zero across the current
history. Their coefficients fit to exactly zero and they cannot move a forecast. They
remain as plumbing for the day that history exists, and the README says so rather than
letting a reader assume the model responds to promotions. It does not yet.

## Why an interpretable regression

A manager staffing a floor has to be able to act on the output, and a coefficient they
can read is worth more than an accuracy point they cannot. Least squares gives a
per-feature number in covers: this much for a Saturday, this much for heavy rain, this
much for the four-week level. A gradient-boosted model would likely fit slightly better
and would not survive the first question anyone asks, which is why.

That is a tradeoff, not a free choice, and it is made deliberately.

## What it cannot do

It predicts whether a day will be busy. It does not tell you how much food to prep or how
many people to staff, and the gap between those two things is wider than it looks.

A seven-day forecast with a mean error of ten covers is useful for deciding whether
Thursday needs a fourth server. It is not precise enough to drive a prep list, and a worst
week at 28.84 says the error is not evenly distributed.

## Running it

```bash
python3 -m venv .venv-api
```

```bash
.venv-api/bin/pip install -r requirements-api.txt
```

```bash
.venv-api/bin/uvicorn app.main:app --reload
```

The default configuration reads `sample_covers_1year.csv`, so no Toast credentials are
needed. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

To reproduce the accuracy figures above:

```bash
.venv-api/bin/python scripts/evaluate.py
```

Full API documentation is in [API_README.md](API_README.md), deployment in
[DEPLOY.md](DEPLOY.md), and the step-by-step method in
[HOW-TO-REBUILD.md](HOW-TO-REBUILD.md).

## Repository map

| Path | What it holds |
|---|---|
| `app/core/model.py` | The model. Source of truth for the feature set. |
| `scripts/evaluate.py` | Walk-forward backtest and the paired comparisons. |
| `app/core/enrichment.py` | Weather, holidays and event lookups. |
| `app/sources/` | Toast API and CSV ingestion. |
| `notebooks/` | The original analysis, 01 through 06. |
| `tests/` | 199 tests. |

## Data

This repository ships synthetic data only. `sample_covers_1year.csv` is a generated year
of covers, and the real sales history it was built from is not included and is not in this
repository's history.

## What is next

The model is accurate enough to be useful and is not used consistently by the restaurant
it was built for. That is not an accuracy problem. The benefit is thin on ordinary days
and concentrated on the few where a human guess goes badly wrong, people do not remember
how much a tool helped them last month, and a daily setup step becomes a minor annoyance
long before it becomes a habit.

The next version has to run without being asked. Scheduled ingestion, a forecast that
arrives rather than one that is requested.

---

A forecast nobody runs is worth nothing, and so is an accuracy figure nobody can trust.
This project spent most of its effort on the second problem and has the apparatus to prove
it: a backtest that refuses to see the future, two baselines chosen because a manager
already uses one of them, and a deployment wired so it cannot quietly report the
flattering number.

The first figure this project produced was the good one. The one it publishes is worse,
and it is the one I would stake something on.
