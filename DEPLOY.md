# Launching this for the restaurant

The short version: **export a CSV weekly, commit it, redeploy.** No database to
back up, no credentials to rotate, no scheduled job to watch. The data file is
~10 KB, so committing it *is* the refresh mechanism.

That sounds crude, and it is deliberately so. The restaurant forecast changes
once a day at most; a weekly manual export is adequate for a weekly planning
cycle, and every moving part not built now is one that cannot break during the
first month in front of a manager.

## One-time setup

1. **Pick a host.** Any platform that reads a `Procfile` works -- Render,
   Railway, Fly.io. The app binds `$PORT` and needs no persistent disk,
   because the store is rebuilt from the CSV on first request.

2. **Set these environment variables on the host:**

   ```
   COVERS_SOURCE=csv
   COVERS_CSV_PATH=data/processed/covers_daily_toast.csv
   COVERS_ENRICHMENT_ENABLED=true
   COVERS_ESPN_ENABLED=false
   COVERS_WEB_USERNAME=manager
   COVERS_WEB_PASSWORD=<pick something and send it to them>
   COVERS_LATITUDE=25.7278151
   COVERS_LONGITUDE=-80.2423594
   COVERS_TIMEZONE=America/New_York
   COVERS_COUNTRY_CODE=US
   ```

   `COVERS_WEB_PASSWORD` is not optional for anything internet-reachable.
   Without it the page is open to anyone who finds the URL, and it shows a
   real business's daily trade.

3. **Send the manager the `/today` URL and the password.** Tell them to add it
   to their phone's home screen; it is built to be read at arm's length at the
   end of a shift.

## The weekly refresh

```bash
# 1. Restaurant exports Toast Web -> Reports -> Sales summary, full range.
# 2. Import it (drops today's partial day, reports gaps and data quality):
python3 scripts/import_toast_export.py "<new export folder>" --force

# 3. Re-measure honestly, and publish the figure the page quotes:
python3 scripts/evaluate.py --save

# 4. Ship it:
git add -A && git commit -m "Covers data through <date>" && git push
```

Step 3 matters. Without it the page quotes the in-process hold-out score,
which is optimistic by roughly 20% because its features leak across the
hold-out boundary. `--save` publishes the walk-forward number instead.

## What the page tells them

- Tomorrow's expected covers, as a number and a range
- Whether that is busy or quiet **for that weekday**
- A warning banner if the data has gone stale, so a forecast built on
  three-week-old history can never masquerade as current

## Known limits, worth saying out loud

- **Typical error is about 11 covers (18%).** Useful for "busy or quiet",
  not for staffing to the person.
- **It is barely better than a 4-week weekday average** (+0.54 covers,
  p=0.045). It beats "same as last week" clearly (+2.34, p<0.001).
- **The biggest misses are unexplained weekend spikes** -- up to +67 covers.
  Those are local events no public data source knows about, which is what
  `worst_days_to_ask_about.csv` is for.

## Where this goes next

In rough order of value:

1. **The restaurant's answers** to the worst-days sheet, entered via
   `scripts/setup_recurring_events.py`. Forecasting the 10 worst days well
   would take error from 10.6 to 9.2 -- the single largest lever available,
   and it needs a conversation rather than an API.
2. **Toast Standard API access** once the owner clears the 403 (it is a
   permissions wall, not a plan limit). Then `scripts/toast_preflight.py`,
   `scripts/toast_backfill.py`, and a daily scheduled `POST /ingest` replace
   the manual export entirely. The data source is a config switch -- nothing
   else in the codebase changes.
3. **Instagram reach** as a feature, if the export shows an effect. Insights
   expire after 90 days, so export now even if unused.
4. **Multi-restaurant** only when a second one exists. Today a restaurant is
   a set of environment variables; that becomes a record keyed by id.
