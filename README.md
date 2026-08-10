# weatherindex tools

Analysis tools for [weatherindex.ai](https://weatherindex.ai/), Rainbow AI's
open weather-forecast accuracy benchmark. The site ranks precipitation
nowcast providers by their F-score at exactly 60 minutes; these tools instead
compute the **normalized area under the metric-vs-horizon curve** (trapezoidal
mean over a horizon window), which rewards being good across the whole
forecast range rather than at one arbitrary point.

Data comes from the site's undocumented, unauthenticated JSON API
(`https://weatherindex.ai/api/...`). Quirks worth knowing:

- Requests need a normal `User-Agent` header (bare `urllib` gets 403).
- `/api/data/summary` advertises a rolling ~30-day window, but
  `aggregated_metrics` serves history from **2026-04-12** onward via
  `start_timestamp`/`end_timestamp` (unix seconds).
- Providers cover different horizons (vaisala → 60 min, accuweather → 120,
  rainbowai → 240, weathercompany → 420), so scores are only ranked within
  windows a provider fully covers.

## wxindex.py — provider rankings and plots

```sh
uv run wxindex.py --sensor LEBL --start 2026-04-12 --plot
uv run wxindex.py --country ESP
uv run wxindex.py                  # worldwide
```

Prints per-metric tables (F-score, CSI, precision, recall, accuracy,
frequency bias) across horizon windows (10–60/120/240/420 min) and, with
`--plot`, renders small-multiple curves using the site's own provider colors.
Sensor analyses read from the local archive (refreshing it first);
country/world queries hit the API live. Dependencies (matplotlib) are
declared inline (PEP 723) — `uv run` handles the venv.

## archive.py — keep the data before it rolls away

```sh
python3 archive.py LEBL RKSS       # stdlib only
```

Incrementally downloads each sensor's full history into `data/*.json.gz`:
`raw_metrics` (per-observation forecast-vs-observed booleans — everything
else can be recomputed from these), daily-grain `aggregated_metrics`, and
`rain_events`. Re-runs fetch only the last partial week; a manifest tracks
coverage and triggers backfill if the floor moves.

## systemd/ — weekly automatic archival

```sh
./systemd/install.sh               # default sensors: LEBL RKSS
./systemd/install.sh LEBL RJTT    # or your own list
```

Installs a systemd user timer that runs the archiver every Monday morning
(`Persistent=true`, so missed runs catch up at next boot). Uninstall with
`systemctl --user disable --now weatherindex-archive.timer` and remove the
two units from `~/.config/systemd/user/`.
