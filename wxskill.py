#!/usr/bin/env python3
"""Score precipitation nowcast providers with ETS, bootstrap CIs and seasons.

Reads the local daily archive (see archive.py) for one or more sensors and:
  - pools confusion counts over a horizon window, then scores the pool with
    the Equitable Threat Score (CSI corrected for hits expected by chance);
  - puts 90% confidence intervals on every score with a paired bootstrap that
    resamples whole days (every provider sees the same resampled days, so
    "P(A beats B)" is a fair head-to-head);
  - repeats the ranking per calendar month and meteorological season, flagging
    groups with too few rain days to mean much;
  - draws a performance diagram (POD vs success ratio, CSI contours, bias rays)
    that shows *why* a provider scores as it does.

Usage:
  uv run wxskill --sensor LEBL --plot
  uv run wxskill --sensor LEBL RKSS --html
"""

import argparse
import html
import io
import json
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

import archive
from wxindex import (BASELINE, GRID, INK, INK2, MUTED, PROVIDER_COLORS,
                     SURFACE, theme, ts)

WINDOWS = [(10, 60), (10, 120)]  # 10–60: all four providers; 10–120: three
HEADLINE = WINDOWS[0]
THIN = 5          # groups with fewer rain days than this are flagged
ROLL_DAYS = 30    # trailing window for the rolling score
LEVEL = 90        # confidence level, percent
KEY_HORIZONS = (10, 60, 120, 240)  # labelled on the performance diagram
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
# single-hue sequential ramp (blue 700 → 100): on the dark surface the low
# end recedes toward the background and high skill reads bright
SEQ = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
       "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
       "#cde2fb"]
DAY = 86400


def utc(t):
    return datetime.fromtimestamp(int(t), timezone.utc)


# --- scoring ----------------------------------------------------------------

def scores(S):
    """Scores from summed counts S[..., (tp, fp, fn, tn)]; NaN where undefined."""
    tp, fp, fn, tn = np.moveaxis(np.asarray(S, dtype=float), -1, 0)
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        chance = (tp + fp) * (tp + fn) / n
        # with no observed rain every score is degenerate (ETS would read 0
        # from false alarms alone), so leave it undefined
        return {
            "ets": np.where(tp + fn > 0, (tp - chance) / (tp + fp + fn - chance),
                            np.nan),
            "csi": tp / (tp + fp + fn),
            "pod": tp / (tp + fn),
            "sr": tp / (tp + fp),
            "bias": (tp + fp) / (tp + fn),
        }


def resample(W, boot, rng):
    """Paired day-block bootstrap: W[day, ...] -> summed counts [boot, ...]."""
    days = W.shape[0]
    idx = rng.integers(0, days, (boot, days))
    weights = np.zeros((boot, days))
    np.add.at(weights, (np.arange(boot)[:, None], idx), 1)
    return np.tensordot(weights, W, axes=(1, 0))


def interval(samples):
    """Percentile interval along axis 0, ignoring NaNs."""
    tail = (100 - LEVEL) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanpercentile(samples, tail, axis=0),
                np.nanpercentile(samples, 100 - tail, axis=0))


class Sensor:
    """Daily confusion counts for one sensor, as C[day, provider, horizon, 4]."""

    def __init__(self, sensor, start=None, end=None):
        self.name = sensor
        try:
            archive.ensure([sensor])
        except OSError as e:
            print(f"archive refresh failed ({e}); using cached data", file=sys.stderr)
        rows = [r for r in archive.load(sensor)
                if (start is None or r["timestamp"] >= start)
                and (end is None or r["timestamp"] < end)]
        if not rows:
            sys.exit(f"no archived data for {sensor} in that range")
        self.days = np.array(sorted({r["timestamp"] for r in rows}))
        self.providers = sorted({r["forecast_provider"] for r in rows})
        self.horizons = np.array(sorted({r["forecast_time"] // 60 for r in rows}))
        di = {d: i for i, d in enumerate(self.days)}
        pi = {p: i for i, p in enumerate(self.providers)}
        hi = {h: i for i, h in enumerate(self.horizons)}
        self.C = np.zeros((len(di), len(pi), len(hi), 4))
        self.covers = np.zeros((len(pi), len(hi)), bool)
        for r in rows:
            p, h = pi[r["forecast_provider"]], hi[r["forecast_time"] // 60]
            self.C[di[r["timestamp"]], p, h] = (r["tp"], r["fp"], r["fn"], r["tn"])
            self.covers[p, h] = True
        # a rain day: rain was observed at the shortest horizon, for anyone
        observed = self.C[:, :, 0, 0] + self.C[:, :, 0, 2]
        self.rainy = observed.max(axis=1) > 0
        events = {r["timestamp"]: r["rain_events"]
                  for r in archive.load(sensor, "rain_events")}
        self.rain_events = np.array([events.get(d, 0) for d in self.days])

    def window(self, lo, hi):
        """Counts summed over horizons in [lo, hi] -> W[day, provider, 4], plus
        a mask of the providers that cover the whole window."""
        m = (self.horizons >= lo) & (self.horizons <= hi)
        return self.C[:, :, m].sum(axis=2), self.covers[:, m].all(axis=1)


class Analysis:
    """Everything the tables and figures show, computed once per sensor."""

    def __init__(self, s, boot, seed):
        self.s, self.boot = s, boot
        rng = np.random.default_rng(seed)

        # rankings per horizon window
        self.rankings = {}
        for lo, hi in WINDOWS:
            W, ok = s.window(lo, hi)
            point = scores(W.sum(axis=0))
            sample = scores(resample(W, boot, rng))
            lo_ci, hi_ci = interval(sample["ets"])
            self.rankings[(lo, hi)] = {
                "ok": ok, "point": point, "lo": lo_ci, "hi": hi_ci,
                "beats": beats(sample["ets"]),
            }

        # per month and per season, headline window
        W, ok = s.window(*HEADLINE)
        self.ok = ok
        stamps = [utc(d) for d in s.days]
        months = [t.strftime("%Y-%m") for t in stamps]
        seasons = []
        for t in stamps:
            year = t.year + (t.month == 12)  # December opens next year's DJF
            seasons.append(f"{SEASONS[t.month]} {year}")
        self.groups = []
        for kind, keys in (("month", months), ("season", seasons)):
            for key in dict.fromkeys(keys):
                mask = np.array([k == key for k in keys])
                label = (utc(s.days[mask][0]).strftime("%b") if kind == "month"
                         else key.split()[0])
                full = mask.sum() >= (28 if kind == "month" else 90)
                self.groups.append(self.group(W, mask, rng, label, kind, full))

        # rolling window ending on each day
        self.rolling = [self.group(W, (s.days > d - ROLL_DAYS * DAY) & (s.days <= d),
                                   rng, "", "roll", True) for d in s.days]

        # performance diagram: every horizon pooled over all days
        self.perf = scores(s.C.sum(axis=0))  # [provider, horizon]
        h60 = int(np.flatnonzero(s.horizons == 60)[0])
        sample = scores(resample(s.C[:, :, h60], boot, rng))
        self.perf60 = {k: interval(sample[k]) for k in ("sr", "pod")}

    def group(self, W, mask, rng, label, kind, full):
        sample = scores(resample(W[mask], self.boot, rng))["ets"]
        lo, hi = interval(sample)
        rain = int(self.s.rainy[mask].sum())
        return {"label": label, "kind": kind, "days": int(mask.sum()), "full": full,
                "rain": rain, "thin": rain < THIN,
                "ets": scores(W[mask].sum(axis=0))["ets"], "lo": lo, "hi": hi}


def beats(samples):
    """P(row provider's score > column provider's) over bootstrap samples."""
    n = samples.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = samples[:, i], samples[:, j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if i != j and valid.any():
                out[i, j] = (a[valid] > b[valid]).mean()
    return out


# --- terminal ---------------------------------------------------------------

def fmt(v, spec=".3f"):
    return "—" if v is None or np.isnan(v) else format(v, spec)


def report(an):
    s = an.s
    span = f"{utc(s.days[0]):%Y-%m-%d} → {utc(s.days[-1]):%Y-%m-%d}"
    print(f"\n######## {s.name}  ({span}, {len(s.days)} days, "
          f"{int(s.rainy.sum())} rain days, {an.boot} bootstrap resamples)")
    for (lo, hi), r in an.rankings.items():
        print(f"\n== ETS pooled over {lo}–{hi} min, {LEVEL}% CI ==")
        print(f"{'provider':<16}{'ETS':>7}  {'CI':<16}{'CSI':>7}{'POD':>7}"
              f"{'SR':>7}{'bias':>7}")
        order = sorted(np.flatnonzero(r["ok"]), key=lambda p: -r["point"]["ets"][p])
        for p in order:
            pt = r["point"]
            print(f"{s.providers[p]:<16}{fmt(pt['ets'][p]):>7}  "
                  f"{'[' + fmt(r['lo'][p]) + ', ' + fmt(r['hi'][p]) + ']':<16}"
                  f"{fmt(pt['csi'][p]):>7}{fmt(pt['pod'][p]):>7}"
                  f"{fmt(pt['sr'][p]):>7}{fmt(pt['bias'][p], '.2f'):>7}")

    r = an.rankings[HEADLINE]
    ok = np.flatnonzero(r["ok"])
    print(f"\n== P(row beats column), ETS {HEADLINE[0]}–{HEADLINE[1]} min ==")
    print(f"{'':<16}" + "".join(f"{s.providers[p][:10]:>12}" for p in ok))
    for i in ok:
        print(f"{s.providers[i]:<16}" + "".join(
            f"{fmt(r['beats'][i, j], '.2f'):>12}" for j in ok))

    print(f"\n== ETS {HEADLINE[0]}–{HEADLINE[1]} min by month and season "
          f"(± half CI width; ~ = fewer than {THIN} rain days) ==")
    print(f"{'':<16}" + "".join(f"{g['label'] + ('' if g['full'] else '*'):>12}"
                                 for g in an.groups))
    print(f"{'rain days':<16}" + "".join(f"{g['rain']:>12}" for g in an.groups))
    for p in np.flatnonzero(an.ok):
        cells = ""
        for g in an.groups:
            v = g["ets"][p]
            if np.isnan(v):
                cells += f"{'—':>12}"
            else:
                half = (g["hi"][p] - g["lo"][p]) / 2
                cells += f"{('~' if g['thin'] else '') + f'{v:.2f}±{half:.2f}':>12}"
        print(f"{s.providers[p]:<16}{cells}")
    print("\n'*' = partial month/season in the data.")


# --- figures ----------------------------------------------------------------

def seq_color(v, vmax):
    if np.isnan(v):
        return None
    i = int(round(np.clip(v / vmax if vmax > 0 else 0, 0, 1) * (len(SEQ) - 1)))
    return SEQ[i]


def ink_on(i_color):
    """Text ink for a cell: dark on the light end of the ramp."""
    return SURFACE if i_color in SEQ[8:] else INK


def spread(labels, gap):
    """Nudge [y, ...] label entries apart (in data units) so none overlap."""
    labels.sort(key=lambda e: e[0])
    for a, b in zip(labels, labels[1:]):
        b[0] = max(b[0], a[0] + gap)


def draw_heatmap(ax_h, an, provs):
    from matplotlib.patches import Rectangle
    s, groups = an.s, an.groups
    # heatmap: provider × month, then seasons after a gap column
    cols = [g for g in groups if g["kind"] == "month"] + [None] + \
           [g for g in groups if g["kind"] == "season"]
    vmax = np.nanmax([g["ets"][p] for g in groups for p in provs] + [0.05])
    for x, g in enumerate(cols):
        if g is None:
            continue
        for y, p in enumerate(provs):
            v = g["ets"][p]
            color = seq_color(v, vmax)
            thin = g["thin"] and color
            ax_h.add_patch(Rectangle(
                (x + 0.03, y + 0.05), 0.94, 0.9, facecolor=color or SURFACE,
                alpha=0.3 if thin else 1, linewidth=0))
            if thin or color is None:
                ax_h.add_patch(Rectangle((x + 0.03, y + 0.05), 0.94, 0.9, fill=False,
                                         edgecolor=MUTED if thin else BASELINE,
                                         linestyle=(0, (3, 2)), linewidth=0.9))
            if color:
                half = (g["hi"][p] - g["lo"][p]) / 2
                ax_h.text(x + 0.5, y + 0.5, f"{v:.2f}\n±{half:.2f}", ha="center",
                          va="center", fontsize=8.5,
                          color=INK2 if thin else ink_on(color), linespacing=1.1)
            else:
                ax_h.text(x + 0.5, y + 0.5, "no rain", ha="center", va="center",
                          fontsize=8, color=MUTED)
    ax_h.set_xlim(0, len(cols))
    ax_h.set_ylim(len(provs), 0)
    ax_h.set_xticks([x + 0.5 for x, g in enumerate(cols) if g])
    ax_h.set_xticklabels([f"{g['label']}{'' if g['full'] else '*'}\n{g['rain']} rain d"
                          for g in cols if g], fontsize=8.5)
    ax_h.xaxis.tick_top()
    ax_h.set_yticks([y + 0.5 for y in range(len(provs))])
    ax_h.set_yticklabels([s.providers[p] for p in provs], fontsize=9.5, color=INK2)
    ax_h.tick_params(length=0)
    for side in ax_h.spines.values():
        side.set_visible(False)
    ax_h.set_title(f"ETS {HEADLINE[0]}–{HEADLINE[1]} min by month and season  "
                   f"(± half {LEVEL}% CI; faded, dashed = fewer than {THIN} rain days; "
                   f"* = partial)", fontsize=10.5, loc="left", pad=28)



def fig_time(an, heatmap=True):
    import matplotlib.dates as mdates
    plt = theme()
    s = an.s
    provs = list(np.flatnonzero(an.ok))
    if heatmap:
        fig, (ax_h, ax_r, ax_e) = plt.subplots(
            3, 1, figsize=(13, 9.5), constrained_layout=True,
            gridspec_kw={"height_ratios": [len(provs) * 0.62 + 1.1, 3.2, 0.8]})
        draw_heatmap(ax_h, an, provs)
    else:  # the HTML report shows the heatmap as a table instead
        fig, (ax_r, ax_e) = plt.subplots(
            2, 1, figsize=(13, 5.5), constrained_layout=True,
            gridspec_kw={"height_ratios": [3.2, 0.8]})

    # rolling ETS with CI bands
    dates = [utc(d) for d in s.days]
    ends = []
    for p in provs:
        color = PROVIDER_COLORS.get(s.providers[p], MUTED)
        y = np.array([g["ets"][p] for g in an.rolling])
        lo = np.array([g["lo"][p] for g in an.rolling])
        hi = np.array([g["hi"][p] for g in an.rolling])
        ax_r.fill_between(dates, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax_r.plot(dates, y, color=color, linewidth=2, label=s.providers[p])
        last = np.flatnonzero(~np.isnan(y))
        if last.size:
            ends.append([y[last[-1]], dates[last[-1]], s.providers[p]])
    spread(ends, gap=0.045)
    for y, x, name in ends:
        ax_r.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=8.5, color=INK2)
    ax_r.set_title(f"ETS {HEADLINE[0]}–{HEADLINE[1]} min, trailing {ROLL_DAYS} days, "
                   f"with {LEVEL}% CI", fontsize=10.5, loc="left")
    ax_r.legend(loc="upper left", frameon=False, fontsize=9, ncol=len(provs),
                labelcolor=INK2)
    ax_r.grid(color=GRID, linewidth=0.7)
    ax_r.set_axisbelow(True)
    ax_r.spines[["top", "right"]].set_visible(False)
    ax_r.tick_params(labelsize=9, labelbottom=False)
    ax_r.margins(x=0.08)

    ax_e.bar(dates, s.rain_events, width=0.8, color=MUTED)
    ax_e.set_ylabel("rain events\nper day", fontsize=8.5)
    ax_e.sharex(ax_r)
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_e.spines[["top", "right"]].set_visible(False)
    ax_e.tick_params(labelsize=9)
    ax_e.grid(axis="y", color=GRID, linewidth=0.7)
    ax_e.set_axisbelow(True)

    fig.suptitle(f"{s.name} — provider skill over time", x=0.01, ha="left",
                 fontsize=14)
    return fig


def fig_perf(an):
    from matplotlib import patheffects
    plt = theme()
    s = an.s
    fig, ax = plt.subplots(figsize=(8.5, 8), constrained_layout=True)

    # CSI contours and bias rays
    g = np.linspace(0.005, 1, 300)
    sr, pod = np.meshgrid(g, g)
    csi = 1 / (1 / sr + 1 / pod - 1)
    cs = ax.contour(sr, pod, csi, levels=np.arange(0.1, 1, 0.1), colors=GRID,
                    linewidths=0.9)
    ax.clabel(cs, fmt="%.1f", fontsize=7.5, colors=MUTED)
    for b in (0.5, 1, 1.5, 2, 4):
        end = (1, b) if b <= 1 else (1 / b, 1)
        ax.plot([0, end[0]], [0, end[1]], color=BASELINE,
                linewidth=1.2 if b == 1 else 0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"bias {b:g}", end, xytext=(-2, -2) if b > 1 else (-2, 3),
                    textcoords="offset points", ha="right",
                    va="top" if b > 1 else "bottom", fontsize=7.5, color=MUTED)

    halo = patheffects.withStroke(linewidth=3, foreground=SURFACE)
    h60 = int(np.flatnonzero(s.horizons == 60)[0])
    tags, trail = [], []  # horizon labels to place, every plotted point
    for p, prov in enumerate(s.providers):
        color = PROVIDER_COLORS.get(prov, MUTED)
        m = s.covers[p]
        x, y = an.perf["sr"][p, m], an.perf["pod"][p, m]
        hs = s.horizons[m]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=3.5,
                label=prov, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.6)
        trail += zip(x, y)
        for h in KEY_HORIZONS:
            k = np.flatnonzero(hs == h)
            if k.size:
                ax.plot(x[k], y[k], "o", color=color, markersize=8, zorder=4,
                        markeredgecolor=SURFACE, markeredgewidth=2)
                tags.append((x[k[0]], y[k[0]], f"{h}′", color))
        if s.covers[p, h60]:
            (xlo, xhi), (ylo, yhi) = ((a[p], b[p]) for a, b in
                                      (an.perf60["sr"], an.perf60["pod"]))
            xv, yv = an.perf["sr"][p, h60], an.perf["pod"][p, h60]
            ax.errorbar(xv, yv, xerr=[[xv - xlo], [xhi - xv]],
                        yerr=[[yv - ylo], [yhi - yv]], color=color, alpha=0.55,
                        linewidth=1.2, capsize=0, zorder=2)

    # zoom onto the data (square, to keep the geometry of the guides honest)
    seen = np.concatenate([an.perf[k][s.covers] for k in ("sr", "pod")])
    lo = max(0.0, np.floor((np.nanmin(seen) - 0.05) * 10) / 10)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("success ratio (1 − false alarm ratio)", fontsize=9.5)
    ax.set_ylabel("probability of detection", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2,
              bbox_to_anchor=(0, 1.0), ncol=len(s.providers))
    fig.suptitle(f"{s.name} — performance diagram, all days pooled", x=0.01,
                 ha="left", fontsize=14)
    ax.set_title("each trail runs 10′ → longest horizon; big dots at "
                 f"{', '.join(f'{h}′' for h in KEY_HORIZONS)}; bars = {LEVEL}% CI "
                 "at 60′; top-right is perfect", fontsize=8.5, color=INK2,
                 loc="left", pad=30)
    fig.canvas.draw()  # settle layout so label placement sees final positions
    place_labels(ax, tags, trail, fontsize=8, color=INK2, path_effects=[halo])
    return fig


def place_labels(ax, tags, obstacles, **style):
    """Annotate each (x, y, text, color), trying offsets around the point until
    the label overlaps neither an earlier label nor any plotted point. A short
    leader in the series color ties each label to its dot."""
    renderer = ax.figure.canvas.get_renderer()
    points = ax.transData.transform(np.array(obstacles))
    pad = 4  # pixels of clearance around plotted points
    placed = []
    offsets = [(9, 5), (9, -13), (-9, 5), (-9, -13), (12, -4), (-12, -4),
               (18, 14), (-18, 14), (18, -22), (-18, -22), (0, 16), (0, -24)]
    for x, y, text, color in tags:
        ann = ax.annotate(text, (x, y), textcoords="offset points", va="bottom",
                          arrowprops={"arrowstyle": "-", "color": color,
                                      "linewidth": 0.9, "shrinkA": 1,
                                      "shrinkB": 4},
                          **style)
        for dx, dy in offsets:
            ann.set_position((dx, dy))
            ann.set_horizontalalignment("left" if dx >= 0 else "right")
            box = ann.get_window_extent(renderer)
            anchor = ax.transData.transform((x, y))
            clash = any(box.overlaps(b) for b in placed) or any(
                box.x0 - pad < px < box.x1 + pad and box.y0 - pad < py < box.y1 + pad
                for px, py in points if (px, py) != tuple(anchor))
            if not clash:
                break
        else:  # nowhere is free: fall back to the default spot
            ann.set_position(offsets[0])
            ann.set_horizontalalignment("left")
            box = ann.get_window_extent(renderer)
        placed.append(box)


def figures(an, heatmap=True):
    return {"time": fig_time(an, heatmap), "perf": fig_perf(an)}


def svg(fig):
    import matplotlib.pyplot as plt
    buf = io.StringIO()
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(buf, format="svg")
    text = buf.getvalue()
    return text[text.index("<svg"):]


# --- html -------------------------------------------------------------------

CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {SURFACE}; color: {INK};
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
h2 {{ font-size: 1.15rem; margin: 32px 0 8px; }}
.meta, .note, figcaption {{ color: {INK2}; font-size: .9rem; }}
.muted {{ color: {MUTED}; }}
details {{ margin: 16px 0; color: {INK2}; }}
summary {{ cursor: pointer; color: {INK}; }}
input[name=tab] {{ position: absolute; opacity: 0; }}
nav {{ display: flex; gap: 4px; border-bottom: 1px solid {BASELINE};
  margin-top: 24px; }}
nav label {{ padding: 8px 16px; cursor: pointer; color: {INK2};
  border-bottom: 2px solid transparent; margin-bottom: -1px; }}
.panel {{ display: none; }}
figure {{ margin: 16px 0; }}
figure svg {{ width: 100%; height: auto; display: block; }}
figure.perf {{ max-width: 760px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: separate; border-spacing: 2px; font-variant-numeric:
  tabular-nums; font-size: .9rem; }}
th, td {{ padding: 5px 10px; text-align: right; white-space: nowrap; }}
th {{ color: {INK2}; font-weight: 600; cursor: pointer; user-select: none; }}
th:first-child, td:first-child {{ text-align: left; }}
td.thin {{ opacity: .45; outline: 1px dashed {MUTED}; outline-offset: -3px; }}
td.none {{ color: {MUTED}; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 8px; }}
.sub {{ display: block; font-size: .75rem; color: {MUTED}; font-weight: 400; }}
"""

JS = """
document.querySelectorAll("table.sortable th").forEach((th, col) => {
  th.addEventListener("click", () => {
    const tbody = th.closest("table").tBodies[0];
    const dir = th.dataset.dir === "desc" ? 1 : -1;
    th.closest("tr").querySelectorAll("th").forEach(h => delete h.dataset.dir);
    th.dataset.dir = dir === 1 ? "asc" : "desc";
    const key = r => {
      const c = r.cells[col], v = c.dataset.v ?? c.textContent;
      const n = parseFloat(v);
      return isNaN(n) ? v : n;
    };
    [...tbody.rows].sort((a, b) => {
      const x = key(a), y = key(b);
      if (typeof x !== typeof y) return typeof x === "number" ? -1 : 1;
      return (x > y ? 1 : x < y ? -1 : 0) * dir;
    }).forEach(r => tbody.appendChild(r));
  });
});
"""

EXPLAIN = f"""
<details>
<summary>How to read this</summary>
<p><b>ETS (Equitable Threat Score)</b> is the share of rain events a provider
got right (hits ÷ hits + misses + false alarms, like CSI), after subtracting the
hits a random forecast with the same rain frequency would get. 0 = no skill,
1 = perfect. It ignores the many dry minutes that make plain accuracy look great
for everyone.</p>
<p>Confusion counts are <b>pooled</b> over every minute and horizon in the window
before scoring, so a rainy day with many events weighs more than a dry one.</p>
<p><b>{LEVEL}% CIs</b> come from a paired bootstrap that resamples whole days.
Every provider sees the same resampled days, so <b>P(row beats column)</b> is a
fair head-to-head. Values near 0.5 mean the data can't tell them apart.</p>
<p>Months or seasons with fewer than {THIN} rain days are faded with a dashed edge: their scores
depend on a handful of storms. <b>*</b> marks a partial month or season.</p>
<p><b>Performance diagram:</b> up means detects more rain (POD), right means
fewer false alarms (success ratio). Curves are CSI contours, dashed rays are
frequency bias. Above the bias-1 ray a provider over-forecasts rain; below it,
it under-forecasts.</p>
</details>
"""


def tint(v, vmax):
    color = seq_color(v, vmax)
    return (f' style="background:{color};color:{ink_on(color)}"' if color else "")


def swatch(prov):
    return (f'<span class="sw" style="background:{PROVIDER_COLORS.get(prov, MUTED)}">'
            f'</span>{html.escape(prov)}')


def sensor_html(an, figs):
    s = an.s
    out = []
    span = f"{utc(s.days[0]):%Y-%m-%d} → {utc(s.days[-1]):%Y-%m-%d}"
    out.append(f'<p class="meta">{span} · {len(s.days)} days · '
               f'{int(s.rainy.sum())} rain days</p>')

    for (lo, hi), r in an.rankings.items():
        pt = r["point"]
        vmax = max(np.nanmax(pt["ets"][r["ok"]]), 0.05)
        order = sorted(np.flatnonzero(r["ok"]), key=lambda p: -pt["ets"][p])
        rows = "".join(
            f"<tr><td>{swatch(s.providers[p])}</td>"
            f'<td data-v="{pt["ets"][p]:.4f}"{tint(pt["ets"][p], vmax)}>'
            f"{fmt(pt['ets'][p])}</td>"
            f'<td data-v="{r["lo"][p]:.4f}">{fmt(r["lo"][p])} – {fmt(r["hi"][p])}</td>'
            + "".join(f"<td>{fmt(pt[k][p], '.2f' if k == 'bias' else '.3f')}</td>"
                      for k in ("csi", "pod", "sr", "bias"))
            + "</tr>" for p in order)
        out.append(f"<h2>Ranking, {lo}–{hi} min</h2><div class=scroll>"
                   f"<table class=sortable><thead><tr><th>provider</th><th>ETS</th>"
                   f"<th>{LEVEL}% CI</th><th>CSI</th><th>POD</th><th>SR</th>"
                   f"<th>bias</th></tr></thead><tbody>{rows}</tbody></table></div>")

    r = an.rankings[HEADLINE]
    ok = np.flatnonzero(r["ok"])
    head = "".join(f"<th>{html.escape(s.providers[p])}</th>" for p in ok)
    rows = ""
    for i in ok:
        cells = ""
        for j in ok:
            v = r["beats"][i, j]
            strong = not np.isnan(v) and v >= 0.95
            cells += (f'<td class="none">—</td>' if np.isnan(v) else
                      f"<td>{'<b>' if strong else ''}{v:.2f}{'</b>' if strong else ''}</td>")
        rows += f"<tr><td>{swatch(s.providers[i])}</td>{cells}</tr>"
    out.append(f"<h2>Head to head, {HEADLINE[0]}–{HEADLINE[1]} min</h2>"
               f'<p class="note">Probability that the row provider has the higher '
               f"ETS, over bootstrap resamples. Bold = at least 0.95.</p>"
               f"<div class=scroll><table><thead><tr><th>row beats →</th>{head}"
               f"</tr></thead><tbody>{rows}</tbody></table></div>")

    groups = an.groups
    vmax = max(np.nanmax([g["ets"][p] for g in groups for p in ok] + [0.05]), 0.05)
    head = "".join(
        f"<th>{g['label']}{'' if g['full'] else '*'}"
        f"<span class=sub>{g['rain']} rain d</span></th>" for g in groups)
    rows = ""
    for p in np.flatnonzero(an.ok):
        cells = ""
        for g in groups:
            v = g["ets"][p]
            if np.isnan(v):
                cells += '<td class="none" data-v="">no rain</td>'
                continue
            thin = ' class="thin"' if g["thin"] else ""
            title = (f"{g['label']}: {v:.3f} [{g['lo'][p]:.3f}, {g['hi'][p]:.3f}], "
                     f"{g['rain']} rain days"
                     + (f" — fewer than {THIN}, treat as noise" if g["thin"] else ""))
            cells += (f'<td data-v="{v:.4f}"{thin}{tint(v, vmax)} '
                      f'title="{html.escape(title)}">{v:.2f}'
                      f'<span class=sub style="color:inherit;opacity:.75">'
                      f"±{(g['hi'][p] - g['lo'][p]) / 2:.2f}</span></td>")
        rows += f"<tr><td>{swatch(s.providers[p])}</td>{cells}</tr>"
    out.append(f"<h2>By month and season, ETS {HEADLINE[0]}–{HEADLINE[1]} min</h2>"
               f'<p class="note">± is half the {LEVEL}% CI. Faded, dashed cells rest on '
               f"fewer than {THIN} rain days. Hover a cell for the full interval."
               f"</p><div class=scroll><table class=sortable><thead><tr>"
               f"<th>provider</th>{head}</tr></thead><tbody>{rows}</tbody></table>"
               f"</div>")

    out.append(f"<h2>Over time</h2><figure>{svg(figs['time'])}</figure>")
    out.append(f"<h2>Performance diagram</h2><figure class=perf>{svg(figs['perf'])}"
               f"<figcaption>Up = detects more rain, right = fewer false alarms. "
               f"Dashed rays are frequency bias; curves are CSI.</figcaption></figure>")
    return "\n".join(out)


def write_html(analyses, figs, path, boot, seed):
    manifest = json.loads((archive.DATA_DIR / "manifest.json").read_text())
    fetched = ", ".join(
        f"{a.s.name} {manifest.get(a.s.name, {}).get('aggregated_metrics', {}).get('fetched_at', '?')[:16].replace('T', ' ')} UTC"
        for a in analyses)
    tabs = "".join(f'<input type=radio name=tab id="t-{a.s.name}"'
                   f'{" checked" if i == 0 else ""}>' for i, a in enumerate(analyses))
    labels = "".join(f'<label for="t-{a.s.name}">{a.s.name}</label>' for a in analyses)
    panels = "".join(f'<section class=panel id="p-{a.s.name}">'
                     f"{sensor_html(a, figs[a.s.name])}</section>" for a in analyses)
    rules = "".join(
        f'#t-{a.s.name}:checked ~ #p-{a.s.name} {{ display: block; }}\n'
        f'#t-{a.s.name}:checked ~ nav label[for="t-{a.s.name}"] '
        f"{{ color: {INK}; border-color: {INK}; }}\n"
        f'#t-{a.s.name}:focus-visible ~ nav label[for="t-{a.s.name}"] '
        f"{{ outline: 2px solid {INK2}; }}\n" for a in analyses)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nowcast provider skill</title>
<style>{CSS}{rules}</style></head>
<body><main>
<h1>Nowcast provider skill</h1>
<p class="meta">weatherindex.ai data, local archive fetched {html.escape(fetched)} ·
{boot} bootstrap resamples, seed {seed} · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
{EXPLAIN}
{tabs}<nav>{labels}</nav>
{panels}
</main><script>{JS}</script></body></html>
"""
    with open(path, "w") as f:
        f.write(doc)
    print(f"\nhtml report saved to {path}")


# --- cli --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sensor", nargs="+", required=True, metavar="ID",
                    help="archived METAR station id(s), e.g. LEBL RKSS")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--plot", nargs="?", const=".", default=None, metavar="DIR",
                    help="write wxskill_<sensor>_{time,perf}.png (default dir: .)")
    ap.add_argument("--html", nargs="?", const="wxskill_report.html", default=None,
                    metavar="PATH", help="write a self-contained HTML report "
                    "(default: wxskill_report.html)")
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.boot < 1:
        sys.exit("--boot must be at least 1")

    analyses, figs = [], {}
    for sensor in args.sensor:
        an = Analysis(Sensor(sensor, ts(args.start), ts(args.end)), args.boot,
                      args.seed)
        report(an)
        analyses.append(an)
        if args.plot:
            from pathlib import Path
            for kind, fig in figures(an).items():
                path = Path(args.plot) / f"wxskill_{sensor}_{kind}.png"
                fig.savefig(path, dpi=150)
                print(f"plot saved to {path}")
        if args.html:
            figs[sensor] = figures(an, heatmap=False)
    if args.html:
        write_html(analyses, figs, args.html, args.boot, args.seed)


if __name__ == "__main__":
    main()
