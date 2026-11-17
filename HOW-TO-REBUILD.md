# How to rebuild the forecast

Enough detail to reproduce this project from a raw POS export without help. The README
says what the result is and why it can be trusted. This says how to get there.

---

## 1. Get daily covers out of the POS

Toast exports a `SalesSummary` folder. The file that matters is `Sales by day.csv`:

```
yyyyMMdd,Net sales,Total orders,Total guests
```

`Total guests` is covers. `Total orders` is checks, which is a different number, and
confusing the two inflates every weekday in different proportions.

```bash
.venv-api/bin/python scripts/import_toast_export.py --path /path/to/SalesSummary_folder
```

Output is `data/processed/covers_daily_toast.csv` with one row per day. Days the
restaurant was closed must be absent rather than zero. A zero is a real observation of
no business and drags every average through it; a missing day is correctly ignored.

## 2. Investigate outliers before touching them

Flag any day more than three standard deviations from its weekday mean, then look at
each one. The categories that came up here:

- **Real and explainable.** Private buyouts, Mother's Day, New Year's Eve. Keep these.
  A holiday feature exists precisely so the model can learn them.
- **Real and unexplainable.** Keep. The error is honest.
- **Data entry errors.** Correct, and write down what was changed.

Deleting outliers because they are inconvenient is how a model comes to look better than
it is. One entry error was corrected in this history. Nothing else was removed.

## 3. Enrich

Three external sources, all free, all keyed on date and location:

| Source | Gives | Notes |
|---|---|---|
| Open-Meteo | Daily high temperature, precipitation | Historical and forecast endpoints differ. Use historical for backfill. |
| Nager.Date | Public holidays by country | Cheap, and holidays matter more than weather here. |
| Ticketmaster | Nearby ticketed events | Needs a key. Distance-weighted into one pressure score. |

Event pressure is capped. An arena show two miles away and three arena shows two miles
away are not proportionally different for a restaurant with sixty seats.

## 4. Build the features

Nine features, in `app/core/model.py`:

**Lag and level**
- `covers_lag_7` — covers on the same weekday last week
- `covers_rollavg_4wk_weekday` — mean of that weekday over four weeks
- `covers_rollavg_7d_trend` — seven-day mean, lagged seven days
- `covers_rollavg_28d_level` — twenty-eight-day mean, the slow level

**Calendar**
- `is_holiday`, plus one-hot day of week

**Conditions**
- `is_rain_heavy`, `precip_in_capped`

**Plumbing, currently inert**
- `is_local_event`, `is_promo` — constant zero in this history, so their coefficients fit
  to zero

Every lagged feature must be computable from data strictly before the day being
predicted. This is the rule the first version of this project broke, and section 7
explains how.

## 5. Fit

Ordinary least squares. With design matrix `X` (features plus an intercept column) and
target `y` (covers), the fitted coefficients are the `β` minimising squared error:

```
β = argmin ‖y − Xβ‖²
```

solved by the normal equations `β = (XᵀX)⁻¹Xᵀy`, which in practice is
`numpy.linalg.lstsq(X, y)` because it uses an SVD and does not fall over when two
features are nearly collinear.

Each coefficient reads directly: `β` for `is_rain_heavy` is the number of covers a heavy
rain day costs, holding everything else constant. That readability is the reason for
choosing this model, so it should be checked. A coefficient with the wrong sign means a
feature is fitting noise, which is exactly how `covers_lag_14` was caught.

## 6. Forecast

For each target day, build the same nine features using only data available at the
origin, then multiply by `β`. Weather comes from the forecast endpoint rather than the
historical one. Holidays are known in advance. Covers lags come from actuals where the
day is in the past and from prior predictions where it is not.

## 7. Backtest — the part that matters

**Do not score on a single hold-out.** Two failure modes, both of which this project hit:

**One window is one sample.** Comparing two feature sets on one hold-out reliably finds
differences that disappear on the next window. Several changes here looked like five to
eight percent wins on one window and were worth nothing across five.

**Leaked features.** This is subtler and it is what invalidated the first result. If you
build the feature matrix from the whole history and then split it, `covers_rollavg_4wk_weekday`
for a test day averages actual covers from one, two and three weeks earlier. For a day
deep inside the hold-out, those earlier days are also inside the hold-out. You would not
have them. The model is reading answers.

The fix is to refit at every origin:

```bash
.venv-api/bin/python scripts/evaluate.py
```

At each origin the model is refit on data up to that origin only, then forecasts forward
through the same code path the web page uses. Weather and holidays for the target days
are passed as recorded overrides, so the backtest sees what actually happened rather than
today's forecast.

Default settings walk the origin in five-day steps. **Avoid multiples of seven**, or
every origin lands on the same weekday and the result describes one day of the week.

## 8. Score against baselines that mean something

An accuracy number alone says nothing. Two baselines, both computed by the harness:

- **Last week, same weekday.** What a manager already does mentally. Beating this is the
  minimum claim worth making.
- **Four-week weekday average.** The obvious alternative, and what most dashboards
  implement without saying so.

Then test whether the difference is real. The harness runs a paired comparison per
forecast and reports a mean difference, a 95% confidence interval, a p-value and the
share of days won. A mean improvement whose interval includes zero is not an improvement,
and the share of days won is the honest version of the headline: this model beats the
weekday average on 57% of days, not on all of them.

## 9. Publish the honest number

```bash
.venv-api/bin/python scripts/evaluate.py --save
```

This writes `data/walk_forward_accuracy.json`, which the service reads at runtime. Without
it the page falls back to the in-process hold-out figure, which is the optimistic one.
That file is deliberately not gitignored for exactly this reason.

## 10. Changing features later

Change them in `app/core/model.py`, which is the source of truth, then measure with
`scripts/evaluate.py` and quote its MAE. A feature that does not improve the walk-forward
score does not go in, however sensible it sounds. Several sensible ones did not survive:
annual seasonality, a log target, recency weighting, ridge, Huber and L1 losses, and
per-weekday models.

---

## The one thing to carry away

The order matters more than any individual step. Build the features so they could only
have used the past, refit at every origin, compare against what someone already does
without your model, and test whether the difference survives a confidence interval.

Do that and a mediocre-sounding number becomes one you can defend. Skip it and a good
number means nothing, which is where this project started.
