# wxindex

Analysis tools for [weatherindex.ai](https://weatherindex.ai/), Rainbow AI's
open weather-forecast accuracy benchmark. The site ranks precipitation
nowcast providers by their F-score at exactly 60 minutes; these tools instead
compute the **normalized area under the metric-vs-horizon curve** (trapezoidal
mean over a horizon window), which rewards being good across the whole
forecast range rather than at one arbitrary point.

Data comes from the site's undocumented, unauthenticated JSON API
(`https://weatherindex.ai/api/...`). Quirks worth knowing:

- Requests need a normal `User-Agent` header (bare `urllib` gets 403).
- The API was redesigned in August 2026. Current endpoints:
  `/api/charts/group/{sensor|country}/{id}/{step|total}`,
  `/api/charts/regional` (includes a `world` row), `/api/metadata`,
  `/api/sensors/{id}/{summary|details}` — all accept
  `start_timestamp`/`end_timestamp` (unix seconds) and
  `include_incomplete=true`.
- Summaries advertise a short window, but history is served from
  **2026-02-16** onward.
- Providers cover different horizons (vaisala → 60 min, accuweather → 120,
  rainbowai → 240, weathercompany → 420), so scores are only ranked within
  windows a provider fully covers. Foreca appears in the site palette as of
  August 2026 but serves no data yet.

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
country/world queries hit the API live. Dependencies live in
`pyproject.toml` — `uv run` handles the venv.

## archive.py — keep the data before it rolls away

```sh
python3 archive.py LEBL RKSS       # stdlib only
```

Incrementally downloads each sensor's full history into `data/*.json.gz`:
daily-grain `aggregated_metrics` (confusion counts per provider × horizon)
and `rain_events`, both from one `/step` request per week. Re-runs fetch
only the last partial week; a manifest tracks coverage and triggers
backfill if the floor moves.

## data/ — what each file is

| file | grain | updates | used by |
|---|---|---|---|
| `<S>_aggregated_metrics.json.gz` | daily tp/tn/fp/fn per provider × horizon | weekly (timer) | `wxindex` — all rankings/plots |
| `<S>_rain_events.json.gz` | daily rain-event counts | weekly (timer) | nothing yet; context only |
| `manifest.json` | fetch bookkeeping | every archiver run | `archive.py` |

## systemd/ — weekly automatic archival

```sh
./systemd/install.sh               # default sensors: LEBL RKSS
./systemd/install.sh LEBL RJTT    # or your own list
```

Installs a systemd user timer that runs the archiver every Monday morning
(default sensors: LEBL, RKSS)
(`Persistent=true`, so missed runs catch up at next boot). If the project is
a git repository, each run also commits `data/` when it changed, giving a
versioned record of exactly what the API served each week. Uninstall with
`systemctl --user disable --now wxindex-archive.timer` and remove the
two units from `~/.config/systemd/user/`.

## License

The code is MIT-licensed (see `LICENSE`). The contents of `data/` are not
covered by that license: they are benchmark data produced by
[weatherindex.ai](https://weatherindex.ai/) (Rainbow Technologies),
captured from their public API and redistributed here for archival and
research purposes, with attribution.
