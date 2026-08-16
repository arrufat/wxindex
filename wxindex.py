#!/usr/bin/env python3
"""Rank weather providers by area under their metric-vs-horizon curves.

Pulls data from the (undocumented) weatherindex.ai JSON API and computes,
for each metric and horizon window, the trapezoidal mean of the metric
over the window — a normalized AUC that is comparable across providers
with different horizon coverage. A provider is only ranked in a window
it fully covers.

Usage:
  uv run wxindex --sensor LEBL --plot
  uv run wxindex --country ESP
  uv run wxindex            # worldwide
"""

import argparse
import json
import sys
import urllib.request
from collections import defaultdict

API = "https://weatherindex.ai/api"

# metric key -> (label, how to score: "max" = higher is better,
# "one" = closer to 1 is better)
METRICS = {
    "fscore": ("F-score", "max"),
    "csi": ("CSI", "max"),
    "precision": ("Precision", "max"),
    "recall": ("Recall/POD", "max"),
    "accuracy": ("Accuracy", "max"),
    "bias": ("Freq. bias", "one"),
}

WINDOWS = [(10, 60), (10, 120), (10, 240), (10, 420)]

# Fixed color per provider — the same colors weatherindex.ai uses. They are
# designed for a dark surface (yellow/teal are unreadable on white), so the
# chart renders in the site's dark theme.
PROVIDER_COLORS = {
    "accuweather": "#FAC800",
    "rainbowai": "#45D0A2",
    "vaisala": "#0481C8",
    "weathercompany": "#8A2BE2",
    "foreca": "#F76A6A",  # in the site's palette since Aug 2026, no data yet
}
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
SURFACE = "#1a1a19"


def fetch(country=None, start=None, end=None):
    path = f"/charts/group/country/{country}/total" if country else "/charts/regional"
    params = [f"{k}={v}" for k, v in
              (("start_timestamp", start), ("end_timestamp", end)) if v]
    url = f"{API}{path}" + (f"?{'&'.join(params)}" if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.9"})
    with urllib.request.urlopen(req) as r:
        rows = json.load(r)["data"]
    return rows if country else [r for r in rows if r.get("region") == "world"]


def build_curves(rows):
    """provider -> metric -> {minutes: value}"""
    curves = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        minutes = row["forecast_time"] // 60
        for key in METRICS:
            if row.get(key) is not None:
                curves[row["forecast_provider"]][key][minutes] = row[key]
    return curves


def rows_from_archive(sensor, start, end):
    """Sum archived daily confusion counts over the range and recompute metrics,
    yielding rows in the same shape as the API's forecast_time-grain aggregate."""
    import archive
    try:
        archive.ensure([sensor])
    except OSError as e:
        print(f"archive refresh failed ({e}); using cached data", file=sys.stderr)
    counts = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for r in archive.load(sensor):
        if (start is None or r["timestamp"] >= start) and \
           (end is None or r["timestamp"] < end):
            c = counts[(r["forecast_provider"], r["forecast_time"])]
            for k in ("tp", "tn", "fp", "fn"):
                c[k] += r[k]
    rows = []
    for (provider, ft), c in counts.items():
        tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
        if tp + tn + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "forecast_provider": provider, "forecast_time": ft,
            "precision": precision, "recall": recall, "pod": recall,
            "fscore": (2 * precision * recall / (precision + recall)
                       if precision + recall else 0.0),
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "bias": (tp + fp) / (tp + fn) if tp + fn else None,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        })
    return rows


def mean_auc(points, lo, hi):
    """Trapezoidal mean of the curve over [lo, hi]; None if not covered."""
    xs = sorted(t for t in points if lo <= t <= hi)
    if not xs or xs[0] > lo or xs[-1] < hi:
        return None
    area = sum((b - a) * (points[a] + points[b]) / 2 for a, b in zip(xs, xs[1:]))
    return area / (xs[-1] - xs[0])


def plot(curves, scope, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    providers = sorted(curves)

    for ax, (key, (label, mode)) in zip(axes.flat, METRICS.items()):
        logscale = (mode == "one"
                    and all(v > 0 for p in providers for v in curves[p][key].values()))
        for prov in providers:
            pts = curves[prov][key]
            if not pts:
                continue
            xs = sorted(pts)
            ax.plot(xs, [pts[t] for t in xs],
                    color=PROVIDER_COLORS.get(prov, "#898781"),
                    linewidth=2, marker="o", markersize=3.5, label=prov)
        if mode == "one":
            ax.axhline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
            ax.annotate("ideal = 1", xy=(0.99, 1), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8, color=MUTED)
            if logscale:
                ax.set_yscale("log")
                # Plain decimals instead of the log default's 10^n mathtext.
                fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
                ax.yaxis.set_major_formatter(fmt)
                ax.yaxis.set_minor_formatter(fmt)
        ax.axvline(60, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60))
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.02)
        if ax in axes[-1]:
            ax.set_xlabel("forecast horizon, minutes", fontsize=9)

    # Direct labels on the first panel at each line's end; legend covers the rest.
    ax0 = axes.flat[0]
    key0 = next(iter(METRICS))
    for prov in providers:
        pts = curves[prov][key0]
        if pts:
            last = max(pts)
            ax0.annotate(prov, xy=(last, pts[last]), xytext=(5, 0),
                         textcoords="offset points", fontsize=8.5,
                         color=INK2, va="center")

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=len(labels),
               frameon=False, fontsize=10, labelcolor=INK2)
    fig.suptitle(f"{scope} — provider skill vs forecast horizon\n", x=0.01,
                 ha="left", fontsize=14)
    fig.text(0.01, 0.955, "weatherindex.ai data; "
             "dashed vertical line marks the site's 60-minute ranking point",
             fontsize=9, color=INK2)
    fig.savefig(path, dpi=150)
    print(f"\nplot saved to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sensor", help="METAR station id, e.g. LEBL")
    ap.add_argument("--country", help="ISO3 country code, e.g. ESP")
    ap.add_argument("--plot", nargs="?", const=True, default=None, metavar="PATH",
                    help="also write a PNG (default: wxindex_<scope>.png)")
    ap.add_argument("--start", help="YYYY-MM-DD; API has data from 2026-02-16, "
                    "despite the summary endpoint advertising a shorter window")
    ap.add_argument("--end", help="YYYY-MM-DD")
    args = ap.parse_args()

    def ts(day):
        from datetime import datetime, timezone
        return (int(datetime.strptime(day, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc).timestamp()) if day else None)

    scope = args.sensor or args.country or "World"
    if args.start or args.end:
        scope += f" {args.start or '…'}→{args.end or 'now'}"
    if args.sensor:
        rows = rows_from_archive(args.sensor, ts(args.start), ts(args.end))
    else:
        rows = fetch(args.country, ts(args.start), ts(args.end))
    curves = build_curves(rows)
    if not curves:
        sys.exit(f"no data returned for {scope}")

    wins = defaultdict(int)
    for key, (label, mode) in METRICS.items():
        print(f"\n== {label} ({scope}) — mean over window, trapezoid ==")
        header = f"{'provider':<16}" + "".join(f"{f'{lo}-{hi}m':>12}" for lo, hi in WINDOWS)
        print(header)
        table = {p: [mean_auc(c[key], lo, hi) for lo, hi in WINDOWS]
                 for p, c in curves.items()}
        for lo_hi, col in zip(WINDOWS, range(len(WINDOWS))):
            vals = {p: v[col] for p, v in table.items() if v[col] is not None}
            if not vals:
                continue
            best = (min(vals, key=lambda p: abs(vals[p] - 1)) if mode == "one"
                    else max(vals, key=vals.get))
            wins[best] += 1
            vals["__best__"] = best
            table.setdefault("__best__", [None] * len(WINDOWS))
        for prov in sorted(p for p in table if not p.startswith("__")):
            cells = ""
            for col, (lo, hi) in enumerate(WINDOWS):
                v = table[prov][col]
                if v is None:
                    cells += f"{'—':>12}"
                else:
                    col_vals = {p: t[col] for p, t in table.items()
                                if not p.startswith("__") and t[col] is not None}
                    best = (min(col_vals, key=lambda p: abs(col_vals[p] - 1))
                            if mode == "one" else max(col_vals, key=col_vals.get))
                    mark = "*" if prov == best else " "
                    cells += f"{v:>11.3f}{mark}"
            print(f"{prov:<16}{cells}")

    print(f"\n== wins per provider (best-in-window count, all metrics) ==")
    for prov, n in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"{prov:<16}{n}")
    print("\n'*' marks the best provider in that window (for bias: closest to 1).")
    print("'—' means the provider does not cover that window.")

    if args.plot:
        slug = scope.replace(" ", "_").replace("→", "-").replace("…", "start")
        path = args.plot if isinstance(args.plot, str) else f"wxindex_{slug}.png"
        plot(curves, scope, path)


if __name__ == "__main__":
    main()
