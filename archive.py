#!/usr/bin/env python3
"""Archive weatherindex.ai data locally before it rolls out of their window.

For each sensor, downloads the full available history (weekly chunks) from
/api/charts/group/sensor/<id>/step, whose response carries both:
  - aggregated_metrics (timestamp grain): daily confusion counts per
    provider x horizon.
  - rain_events: daily rain-event counts.

(The pre-2026-08 API also served raw_metrics — per-observation booleans of
forecast vs observed. That endpoint was removed in the August 2026 API
redesign; the data/<SENSOR>_raw_metrics.json.gz files preserve what we
captured, 2026-04-12 to 2026-08-12, and can no longer be extended.)

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

API = "https://weatherindex.ai/api"
DATA_START = 1771200000  # 2026-02-16T00:00Z, earliest data the API holds
WEEK = 7 * 86400
DATA_DIR = Path(__file__).parent / "data"

# both files come from one /step response: key -> response field
FILES = {"aggregated_metrics": "metrics", "rain_events": "rain_events"}


def fetch_week(sensor, start, end):
    url = (f"{API}/charts/group/sensor/{sensor}/step"
           f"?start_timestamp={start}&end_timestamp={end}&include_incomplete=true")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.9"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def row_key(row):
    return (row["timestamp"], row.get("forecast_time"), row.get("forecast_provider"))


def archive_sensor(sensor, manifest):
    now = int(time.time())
    store = {name: {row_key(r): r for r in load(sensor, name)} for name in FILES}
    n_before = {name: len(rows) for name, rows in store.items()}
    # both files are filled by the same requests, so coverage is tracked
    # by the aggregated_metrics manifest entry
    meta = manifest.get(sensor, {}).get("aggregated_metrics", {})
    if meta.get("covered_from", DATA_START) > DATA_START:
        start = DATA_START  # backfill: archive misses the earliest data
    else:
        start = max(DATA_START, meta.get("covered_until", DATA_START) - WEEK)
    while start < now:
        end = min(start + WEEK, now)
        chunk = fetch_week(sensor, start, end)
        for name, field in FILES.items():
            store[name].update({row_key(r): r for r in chunk.get(field, [])})
        start = end
    for name in FILES:
        merged = sorted(store[name].values(), key=row_key)
        out = DATA_DIR / f"{sensor}_{name}.json.gz"
        print(f"  {name}: {len(merged)} rows ({len(merged) - n_before[name]:+d})")
        payload = json.dumps(merged).encode()
        # skip the write when nothing changed: gzip embeds an mtime in its
        # header, so rewriting identical data would still dirty the git tree
        if not out.exists() or gzip.decompress(out.read_bytes()) != payload:
            with open(out, "wb") as raw, \
                 gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(payload)
            manifest.setdefault(sensor, {})[name] = {
                "rows": len(merged), "new_rows": len(merged) - n_before[name],
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


def main():
    ensure(sys.argv[1:] or ["LEBL", "RKSS"])


if __name__ == "__main__":
    main()
