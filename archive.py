#!/usr/bin/env python3
"""Archive weatherindex.ai data locally before it rolls out of their window.

For each sensor, downloads the full available history (weekly chunks) of:
  - raw_metrics: per-observation booleans — what each provider forecast vs
    what the sensor observed, for every timestamp x horizon. Everything else
    can be recomputed from this.
  - aggregated_metrics (timestamp grain): daily confusion counts per
    provider x horizon.
  - rain_events: daily rain-event counts.

Output: data/<SENSOR>_<endpoint>.json.gz (rows deduped, sorted by timestamp)
and data/manifest.json recording ranges and fetch time. Re-running extends
existing archives incrementally and never re-downloads covered weeks.

Usage: ./archive.py LEBL RKSS
"""

import gzip
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://weatherindex.ai/api/data"
DATA_START = 1775952000  # 2026-04-12T00:00Z, earliest data the API holds
WEEK = 7 * 86400
DATA_DIR = Path(__file__).parent / "data"

ENDPOINTS = {
    "raw_metrics": "raw_metrics?sensor_id={sensor}",
    "aggregated_metrics": ("aggregated_metrics?group_by=timestamp,forecast_time,"
                           "forecast_provider&sensor_id={sensor}"),
    "rain_events": "rain_events?sensor_id={sensor}",
}


def fetch(path, start, end):
    url = f"{API}/{path}&start_timestamp={start}&end_timestamp={end}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.9"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def row_key(row):
    return (row["timestamp"], row.get("forecast_time"), row.get("forecast_provider"))


def archive_sensor(sensor, manifest):
    now = int(time.time())
    for name, path_tpl in ENDPOINTS.items():
        out = DATA_DIR / f"{sensor}_{name}.json.gz"
        rows = {}
        if out.exists():
            with gzip.open(out, "rt") as f:
                rows = {row_key(r): r for r in json.load(f)}
        meta = manifest.get(sensor, {}).get(name, {})
        if meta.get("covered_from", DATA_START) > DATA_START:
            start = DATA_START  # backfill: archive misses the earliest data
        else:
            start = max(DATA_START, meta.get("covered_until", DATA_START) - WEEK)
        n_before = len(rows)
        while start < now:
            end = min(start + WEEK, now)
            chunk = fetch(path_tpl.format(sensor=sensor), start, end)
            rows.update({row_key(r): r for r in chunk})
            start = end
        merged = sorted(rows.values(), key=row_key)
        print(f"  {name}: {len(merged)} rows ({len(merged) - n_before:+d})")
        payload = json.dumps(merged).encode()
        # skip the write when nothing changed: gzip embeds an mtime in its
        # header, so rewriting identical data would still dirty the git tree
        if not out.exists() or gzip.decompress(out.read_bytes()) != payload:
            with open(out, "wb") as raw, \
                 gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(payload)
            manifest.setdefault(sensor, {})[name] = {
                "rows": len(merged), "new_rows": len(merged) - n_before,
                "covered_from": DATA_START, "covered_until": now,
                "fetched_at": datetime.now(timezone.utc)
                              .isoformat(timespec="seconds"),
            }


def ensure(sensors):
    """Refresh the local archive for these sensors. Safe to call before analysis."""
    DATA_DIR.mkdir(exist_ok=True)
    mpath = DATA_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for sensor in sensors:
        print(f"== archiving {sensor} ==")
        archive_sensor(sensor, manifest)
    mpath.write_text(json.dumps(manifest, indent=2))


def load(sensor, endpoint="aggregated_metrics"):
    """Read archived rows for a sensor; [] if never archived."""
    out = DATA_DIR / f"{sensor}_{endpoint}.json.gz"
    if not out.exists():
        return []
    with gzip.open(out, "rt") as f:
        return json.load(f)


if __name__ == "__main__":
    ensure(sys.argv[1:] or ["LEBL", "RKSS"])
