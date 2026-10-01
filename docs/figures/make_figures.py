"""Build every figure for the README and the paper from the repository's own data.

    python docs/figures/make_figures.py      # writes <name>.svg, <name>.png (200 dpi) and <name>.pdf for each figure

One script and one style serve both the README (SVG) and the paper (PDF). Inputs (nothing else is read):
    results/official-scores.csv                  official val_bpb per upload, chip rehearsal, steps, offset
    research/sim-data/runs.jsonl                 the chip-run records the simulator is fitted on
    research/sim-data/validation-pairs.json      the 99 treatment/control pairs used to validate the quality model
    constants copied from named doc tables       docs/FINDINGS.md, docs/GAP-ANALYSIS.md, docs/SIMULATOR.md (cited inline)

Figures (paper numbering in brackets, see paper/OUTLINE.md):
    score_history        [Fig. 1] official val_bpb per upload, with the levers that moved it
    step_counts          [Fig. 2] optimizer steps per 30-minute rehearsal, by chip
    oracle_calibration   [Fig. 3] chip rehearsal vs official score, and the offset per upload
    exchange_rate        [Fig. 4] bpb gained per unit of extra effective compute (0.063 * ln ratio)
    levers               [Fig. 5] measured effects, marked by kind of evidence and n
    gpu_parity           [Fig. 6] GPU proxy vs Trainium on four paired effects
    simulator_pairs      [Fig. 7] the quality surrogate's leave-one-pair-out predictions on all 99 pairs

Requires numpy and matplotlib. Output is deterministic for a given font set: fixed SVG hash salt, no timestamps,
SVG text stored as outlines.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCORES = ROOT / "results" / "official-scores.csv"
RUNS = ROOT / "research" / "sim-data" / "runs.jsonl"
PAIRS = ROOT / "research" / "sim-data" / "validation-pairs.json"

DATES = "15 Sep to 1 Oct 2026 (CDT)"

# ---------------------------------------------------------------------------------------------
# Style: one system for every figure. Colour-blind-safe categorical slots, dark ink on a white
# card so the PNG/SVG read on GitHub light and dark themes and print cleanly in the PDF.
# ---------------------------------------------------------------------------------------------
INK = "#1f1f1d"          # primary text
INK_2 = "#52514e"        # secondary text, tick labels
INK_3 = "#8a8984"        # muted text, leader lines
GRID = "#e7e6e1"         # hairline grid
SURFACE = "#ffffff"
BLUE = "#2a78d6"         # slot 1: the main series, gains
ORANGE = "#eb6834"       # slot 2: the contrast series
AQUA = "#1baf7a"         # slot 3: reference marks
RED = "#d64545"          # losses (diverging partner of BLUE)
GREY = "#b4b3ad"         # de-emphasised marks
BLUE_WASH = "#2a78d61a"  # ~10% area wash
TIE_WASH = "#efeee9"


def _font_family() -> list:
    have = {f.name for f in font_manager.fontManager.ttflist}
    prefs = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Segoe UI", "Liberation Sans", "DejaVu Sans"]
    return [f for f in prefs if f in have] or ["DejaVu Sans"]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _font_family(),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.titlecolor": INK,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": INK_3,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 3,
    "ytick.major.size": 0,
    "xtick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "mathtext.default": "regular",
    "hatch.linewidth": 0.9,
    "svg.fonttype": "path",      # text as outlines: identical rendering wherever the SVG is shown
    "svg.hashsalt": "frontierforge-figures",
    "pdf.fonttype": 42,          # embed TrueType fonts in the PDF (editable, arXiv-safe)
})

LINE_W = 2.0
DOT = 7.5          # marker size in points
RING = 1.6         # surface ring around dots
META = {"svg": {"Date": None}, "png": {"Software": None}, "pdf": {"CreationDate": None, "ModDate": None}}


def save(fig, name: str) -> list:
    out = []
    for ext in ("svg", "png", "pdf"):
        p = HERE / f"{name}.{ext}"
        kw = {"dpi": 200} if ext == "png" else {}
        fig.savefig(p, format=ext, bbox_inches="tight", pad_inches=0.25, metadata=META[ext], **kw)
        out.append(p)
    plt.close(fig)
    return out


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=13.5, fontweight="bold", color=INK,
             transform=fig.transFigure)
    fig.text(0.0, 0.985, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2, transform=fig.transFigure)


def footnote(fig, text: str, y: float = -0.02) -> None:
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8, color=INK_3, transform=fig.transFigure,
             linespacing=1.5)


def label(ax, x, y, text, dx, dy, ha="left", va="center", size=8.5, color=INK, weight="normal"):
    """Direct label with a hairline leader from the mark (offset in points)."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=size, color=color, fontweight=weight,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7, shrinkA=1, shrinkB=4))


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def _f(v):
    v = (v or "").strip()
    return float(v) if v else None


def load_scores() -> list:
    rows = []
    with SCORES.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "name": r["submission"], "date": r["uploaded_cdt"].strip().lstrip("~"),
                "chip": r["rehearsal_chip"].strip(), "rehearsal": _f(r["rehearsal_bpb"]),
                "steps": _f(r["rehearsal_steps"]), "official": _f(r["official_bpb"]),
                "offset": _f(r["offset"]), "best": r["best_so_far"].strip() == "yes",
                "note": r["change_vs_previous_best"],
            })
    return rows


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def day_label(iso: str) -> str:
    _, m, d = iso.split("-")
    return f"{int(d)} {MONTHS[int(m) - 1]}"


def day_number(iso: str) -> int:
    """Days since 1 Sep 2026 (the campaign ran from September into October)."""
    _, m, d = iso.split("-")
    return int(d) - 1 + (30 if int(m) == 10 else 0)


def short(name: str) -> str:
    return name.replace("_min", "")


# What each new best in panel b changed (short form of change_vs_previous_best in results/official-scores.csv).
LEVERS = [
    ("K50", "batch warm-up, multi-token prediction, Muon schedule"),
    ("K51", "key offset, accumulation schedule"),
    ("K53", "ZeRO Muon, EMA prewarm, EMA every 32"),
    ("K54b", "ZeRO-2 + AdamW ZeRO, 3-phase batch ramp"),
    ("K56", "key-offset radius 0.3, accumulation in the graph"),
    ("K57", "seed 73"),
    ("K59", "LeakyReLU(0.35)² MLP"),
    ("K60", "attention-source reuse"),
    ("K63", "EMA off, cooldown 0.7, small-batch LR"),
    ("K70a", "row pool 256 (−0.0022 official)"),
    ("K73s4", "optimizer fusion, XU 48, c_proj LR 0.8, salt 4"),
    ("K77a", "EMA blend 0.6"),
    ("K82s4", "EMA prewarm: best official, 0.96136"),
]


# ---------------------------------------------------------------------------------------------
# Fig. 1  Score history
# ---------------------------------------------------------------------------------------------
def fig_score_history(rows: list) -> list:
    scored = [r for r in rows if r["official"] is not None]
    s = {r["name"]: r for r in scored}
    best_name = min(scored, key=lambda r: r["official"])["name"]

    # Panel a x: calendar time; the uploads of one day are spread evenly across that day.
    per_day: dict = {}
    for r in scored:
        per_day.setdefault(r["date"], []).append(r["name"])
    tx = {}
    for date, names in per_day.items():
        for k, n in enumerate(names):
            tx[n] = day_number(date) + (k + 1) / (len(names) + 1)

    def staircase(names, xof):
        xs, ys, cur = [], [], math.inf
        for n in names:
            cur = min(cur, s[n]["official"])
            xs.append(xof(n))
            ys.append(cur)
        return xs, ys

    def dots(ax, names, xof):
        nb = [n for n in names if not s[n]["best"]]
        bb = [n for n in names if s[n]["best"]]
        ax.scatter([xof(n) for n in nb], [s[n]["official"] for n in nb], s=DOT ** 2 * 0.8, color=GREY,
                   edgecolor=SURFACE, linewidth=RING, zorder=3)
        ax.scatter([xof(n) for n in bb], [s[n]["official"] for n in bb], s=DOT ** 2, color=BLUE,
                   edgecolor=SURFACE, linewidth=RING, zorder=4)

    fig = plt.figure(figsize=(12.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.6, 1.1], width_ratios=[1.0, 1.1], wspace=0.17, hspace=0.34)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    header(fig, "Official score, upload by upload",
           f"Official val_bpb (lower is better) for every scored upload, {DATES}. "
           "Line: best so far. Grey: uploads that did not set a new best.")

    # Panel a: the whole campaign, calendar time
    names = [r["name"] for r in scored]
    xs, ys = staircase(names, tx.get)
    a1.step(xs, ys, where="post", color=BLUE, lw=LINE_W, zorder=2)
    dots(a1, names, tx.get)
    first, last = day_number("2026-09-15"), day_number("2026-10-01")
    a1.set_xlim(first - 0.2, last + 1.2)
    a1.set_xticks(np.arange(first, last + 1, 2), [f"{d + 1} Sep" if d < 30 else f"{d - 29} Oct"
                                                  for d in range(first, last + 1, 2)])
    a1.grid(axis="x", visible=False)
    a1.set_ylim(0.952, 1.158)
    a1.set_yticks(np.arange(0.96, 1.16, 0.02))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    a1.set_ylabel("official val_bpb")
    a1.set_title("a   The whole campaign", color=INK)
    z0 = day_number(s["K50"]["date"])
    a1.axvspan(z0, last + 1.2, color=BLUE_WASH, lw=0, zorder=0)
    a1.text(z0 + 0.25, 1.153, "panel b", fontsize=8, color=INK_3, va="top")
    for n, text, dx, dy in (("F2", "F2  first upload, 1.1463", 12, 4),
                            ("K24", "K24  depth 12, 262k batch, Muon", 12, 8),
                            ("K29", "K29  stream rows (eval layout)", 12, 8),
                            ("K40", "K40  first score below 1.0", 14, 30),
                            ("K44", "K44  0.9888", 16, 22)):
        label(a1, tx[n], s[n]["official"], text, dx, dy)
    label(a1, tx[best_name], s[best_name]["official"], f"{best_name}  {s[best_name]['official']:.4f}", -16, 34,
          ha="right", weight="bold")

    # Panel b: from K50, in upload order; ticks mark the CDT day, hairlines separate the days
    zi = names.index("K50")
    zoom = names[zi:]
    ix = {n: i for i, n in enumerate(zoom)}
    xs, ys = staircase(zoom, ix.get)
    a2.step(xs, ys, where="post", color=BLUE, lw=LINE_W, zorder=2)
    dots(a2, zoom, ix.get)
    groups: dict = {}
    for n in zoom:
        groups.setdefault(s[n]["date"], []).append(ix[n])
    ticks, labs = [], []
    for k, (date, members) in enumerate(groups.items()):
        lo, hi = min(members) - 0.5, max(members) + 0.5
        if k:
            a2.axvline(lo, color=GRID, lw=1.0, ls=(0, (2, 2)), zorder=0)
        ticks.append((lo + hi) / 2)
        labs.append(day_label(date) if date.endswith("-01") else str(int(date[-2:])))
    a2.set_xticks(ticks, labs)
    a2.tick_params(axis="x", length=0)
    a2.set_xlabel("uploads in order (day marked: September unless noted, CDT)")
    a2.grid(axis="x", visible=False)
    a2.set_xlim(-0.6, len(zoom) + 1.0)
    a2.set_ylim(0.9600, 0.9772)
    a2.set_yticks(np.arange(0.960, 0.9775, 0.0025))
    a2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    a2.set_title("b   From K50, in upload order: each step is one change", color=INK)
    below = {"K50", "K73s4", best_name}
    for n in zoom:
        r = s[n]
        if not r["best"]:
            continue
        w = "bold" if n == best_name else "normal"
        if n in below:
            a2.text(ix[n], r["official"] - 0.00045, short(n), ha="center", va="top", fontsize=8, color=INK,
                    fontweight=w)
        else:
            a2.text(ix[n] - 0.28, r["official"], short(n), ha="right", va="center", fontsize=8, color=INK,
                    fontweight=w)
    for n, text, dx, dy in (("K65a", "K65a-d: official tests", 8, 10),
                            ("K73s6", "K73s6: salt 6", 6, 24),
                            ("K82s7", "K82s7: salt 7\n0.96140", 8, 22)):
        if n in ix:
            label(a2, ix[n], s[n]["official"], text, dx, dy, ha="right" if dx < 0 else "left", size=8,
                  color=INK_2)
    handles = [Line2D([], [], color=BLUE, lw=LINE_W, label="best so far"),
               Line2D([], [], lw=0, marker="o", ms=7, mfc=BLUE, mec=SURFACE, label="new best"),
               Line2D([], [], lw=0, marker="o", ms=6, mfc=GREY, mec=SURFACE, label="not a new best")]
    a2.legend(handles=handles, loc="upper right", handlelength=1.6, borderaxespad=0.2)

    # Key: what each new best in panel b changed
    key = fig.add_subplot(gs[1, :])
    key.axis("off")
    key.text(0.0, 1.0, "What each new best in panel b changed", ha="left", va="top", fontsize=9,
             fontweight="bold", color=INK, transform=key.transAxes)
    nrow = 5
    for i, (n, text) in enumerate(LEVERS):
        c, rr = divmod(i, nrow)
        x0, y0 = (0.0, 0.385, 0.655)[c], 0.80 - rr * 0.19
        key.text(x0, y0, n, ha="left", va="top", fontsize=8.5, fontweight="bold", color=INK,
                 transform=key.transAxes)
        key.text(x0 + 0.05, y0, text, ha="left", va="top", fontsize=8.5, color=INK_2, transform=key.transAxes)
    footnote(fig, "Source: results/official-scores.csv. K65a-d were official tests of single changes; K73s6 and "
                  "K82s7 were shuffle-salt draws. Leaderboard positions are not shown.", y=0.035)
    return save(fig, "score_history")


# ---------------------------------------------------------------------------------------------
# Fig. 2  Steps per 30-minute rehearsal, by chip
# ---------------------------------------------------------------------------------------------
def fig_steps(rows: list) -> list:
    pts = [r for r in rows if r["steps"]]
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    header(fig, "Steps were the currency",
           "Optimizer steps in the 30-minute charged budget, from the cold chip rehearsal of each upload. "
           "The letter above each bar is the chip it ran on.")
    x = np.arange(len(pts))
    slow = {"E"}
    colors = [ORANGE if r["chip"] in slow else BLUE for r in pts]
    ax.bar(x, [r["steps"] for r in pts], width=0.72, color=colors, edgecolor=SURFACE, linewidth=0.8, zorder=2)
    for xi, r in zip(x, pts):
        ax.text(xi, r["steps"] + 35, r["chip"], ha="center", va="bottom", fontsize=7.5, color=INK_2)
    for n in ("K24", "K77a", "K82s4", "K82s7"):
        i = next(i for i, r in enumerate(pts) if r["name"] == n)
        ax.text(x[i], pts[i]["steps"] + 160, f"{int(pts[i]['steps']):,}", ha="center", va="bottom", fontsize=8,
                color=INK, fontweight="bold")
    ax.set_xticks(x, [short(r["name"]) for r in pts], rotation=60, ha="right", fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_ylim(0, 2800)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_ylabel("rehearsal steps")
    ax.legend(handles=[matplotlib.patches.Patch(color=BLUE, label="chips A, B, C, G, H"),
                       matplotlib.patches.Patch(color=ORANGE, label="chip E (about 3.5% slower)")],
              loc="upper left", handlelength=1.2, borderaxespad=0.2)
    footnote(fig, "Source: results/official-scores.csv (uploads with a rehearsal; K36 has no step count). Chips "
                  "differ by up to 3.5% in step time, so compare step counts within a chip.\nF2-K15 used a 131k "
                  "batch and K24 onward a 262k batch; from K50 the batch warms up through smaller phases, which adds steps "
                  "early in the run.", y=-0.04)
    return save(fig, "step_counts")


# ---------------------------------------------------------------------------------------------
# Fig. 3  Oracle calibration
# ---------------------------------------------------------------------------------------------
def fig_oracle(rows: list) -> list:
    cal = [r for r in rows if r["official"] is not None and r["rehearsal"] is not None and r["offset"] is not None]
    k50 = next(i for i, r in enumerate(cal) if r["name"] == "K50")
    recent, early = cal[k50:], cal[:k50]
    last = short(recent[-1]["name"])
    off_r = np.array([r["official"] - r["rehearsal"] for r in recent])
    off_e = np.array([r["official"] - r["rehearsal"] for r in early])
    mu, sd = float(off_r.mean()), float(off_r.std(ddof=1))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.7), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.2})
    header(fig, "A deterministic chip rehearsal predicts the official score",
           f"Since K50 the official score has been the chip rehearsal plus a near-constant offset: "
           f"mean +{mu:.4f}, sd {sd:.4f} over {len(recent)} uploads.")

    xr = np.array([r["rehearsal"] for r in recent])
    yr = np.array([r["official"] for r in recent])
    lo, hi = 0.9535, 0.9700
    xx = np.array([lo, hi])
    a1.fill_between(xx, xx + mu - 2 * sd, xx + mu + 2 * sd, color=BLUE_WASH, lw=0, zorder=0)
    a1.plot(xx, xx + mu, color=BLUE, lw=1.2, zorder=1)
    a1.plot(xx, xx, color=GREY, lw=1.0, zorder=1)
    a1.scatter(xr, yr, s=DOT ** 2, color=BLUE, edgecolor=SURFACE, linewidth=RING, zorder=3)
    a1.set_xlim(lo, hi)
    a1.set_ylim(0.9600, 0.9775)
    a1.set_xlabel("chip rehearsal val_bpb (first 2M public tokens)")
    a1.set_ylabel("official val_bpb")
    a1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    a1.set_title(f"a   Rehearsal vs official, K50 to {last} (n = {len(recent)})", color=INK)
    a1.legend(handles=[
        Line2D([], [], lw=0, marker="o", ms=7, mfc=BLUE, mec=SURFACE, label="one upload"),
        Line2D([], [], color=BLUE, lw=1.2, label=f"official = rehearsal + {mu:.4f}"),
        matplotlib.patches.Patch(color=BLUE_WASH, label="±2 sd of the offset"),
        Line2D([], [], color=GREY, lw=1.0, label="official = rehearsal")],
        loc="upper left", handlelength=1.6, borderaxespad=0.4)
    by = {r["name"]: r for r in recent}
    for n, text, dx, dy, ha in (("K50", "K50", -12, 4, "right"), ("K60", "K60", 12, -10, "left"),
                                ("K73s6", "K73s6 (salt-selected)", 18, 24, "left"),
                                ("K82s4", "K77a, K82s4, K82s7", 14, -14, "left")):
        if n in by:
            label(a1, by[n]["rehearsal"], by[n]["official"], text, dx, dy, ha=ha)

    xs = np.arange(len(cal))
    a2.scatter(xs[:k50], off_e * 1e3, s=DOT ** 2 * 0.8, color=GREY, edgecolor=SURFACE, linewidth=RING, zorder=3)
    a2.scatter(xs[k50:], off_r * 1e3, s=DOT ** 2, color=BLUE, edgecolor=SURFACE, linewidth=RING, zorder=4)
    a2.axhspan((mu - sd) * 1e3, (mu + sd) * 1e3, xmin=(k50 - 0.4 + 0.8) / (len(cal) + 0.6), color=BLUE_WASH,
               lw=0, zorder=0)
    a2.plot([k50 - 0.4, len(cal) - 0.6], [mu * 1e3] * 2, color=BLUE, lw=1.2, zorder=1)
    a2.set_xticks(xs, [short(r["name"]) for r in cal], rotation=90, fontsize=7.5)
    a2.grid(axis="x", visible=False)
    a2.set_xlim(-0.8, len(cal) - 0.2)
    a2.set_ylim(4.0, 8.2)
    a2.set_ylabel("offset, official − rehearsal ($\\times10^{-3}$ bpb)")
    a2.set_title("b   Offset per upload, in upload order", color=INK)
    a2.legend([Line2D([], [], lw=0, marker="o", ms=6, mfc=GREY, mec=SURFACE),
               Line2D([], [], lw=0, marker="o", ms=7, mfc=BLUE, mec=SURFACE),
               (matplotlib.patches.Patch(color=BLUE_WASH), Line2D([], [], color=BLUE, lw=1.2))],
              [f"F2 to K46, chips A/B (n = {len(early)})", f"K50 to {last} (n = {len(recent)})",
               f"K50 to {last}: mean {mu * 1e3:.2f} ± {sd * 1e3:.2f} (1 sd)"],
              loc="lower left", handletextpad=0.4, borderaxespad=0.2, handlelength=1.6)
    footnote(fig, "Source: results/official-scores.csv (uploads with a rehearsal; K65a-d were not rehearsed). "
                  "Rehearsals ran on chips A-H with the upload seed and time target; K70a's chip-E rehearsal is "
                  "normalised to chip-C speed.", y=-0.07)
    return save(fig, "oracle_calibration")


# ---------------------------------------------------------------------------------------------
# Fig. 4  Exchange rate
# ---------------------------------------------------------------------------------------------
K = 0.063          # docs/GAP-ANALYSIS.md: delta_bpb ~= 0.063 * ln(effective compute ratio)


def fig_exchange(rows: list) -> list:
    best = min(r["official"] for r in rows if r["official"] is not None)
    best_name = next(r["name"] for r in rows if r["official"] == best)
    rank10, rank1 = 0.9554, 0.9328   # final leaderboard snapshot, by rank (docs/GAP-ANALYSIS.md)
    g = np.linspace(0, 62, 400)
    d = K * np.log1p(g / 100.0) * 1e3

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    header(fig, "What extra compute is worth",
           "Near our operating point, val_bpb falls with the log of effective compute: "
           "Δbpb ≈ 0.063 · ln(compute ratio).")
    ax.axvspan(15, 23, color=AQUA + "22", lw=0, zorder=0)
    ax.text(19, 30.5, "kernel throughput\ntarget, +15-23%", ha="center", va="top", fontsize=8.5, color=INK_2)
    ax.plot(g, d, color=BLUE, lw=LINE_W, zorder=2)

    def gain_for(gap):
        return (math.exp(gap / K) - 1) * 100

    gap10, gap1 = best - rank10, best - rank1
    pts = [(gain_for(gap10), f"gap to #10: {gap10:.4f} bpb\n≈ +{gain_for(gap10):.0f}% effective compute",
            (1.0, 12.6), "left", "bottom", ORANGE, True),
           (28.0, "measured: 2,288 s instead of 1,788 s\n+28% compute, val_bpb −0.0155", (31.0, 12.0),
            "left", "top", BLUE, False),
           (gain_for(gap1), f"gap to #1: {gap1:.4f} bpb\n≈ +{gain_for(gap1):.0f}% effective compute", (54.0, 30.0),
            "right", "bottom", BLUE, False)]
    for gx, text, off, ha, va, color, bold in pts:
        y = K * math.log1p(gx / 100.0) * 1e3
        ax.scatter([gx], [y], s=(DOT * 1.15) ** 2, color=color, edgecolor=SURFACE, linewidth=RING, zorder=4)
        ax.plot([gx, gx], [0, y], color=color, lw=0.8, alpha=0.6, zorder=1)
        ax.annotate(text, xy=(gx, y), xytext=off, textcoords="data", ha=ha, va=va, fontsize=8.5,
                    color=INK, fontweight="bold" if bold else "normal",
                    arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7, shrinkA=2, shrinkB=5))

    small = 0.5   # a late recipe change (~0.0005 bpb) in compute terms
    g_small = (math.exp(small / 1e3 / K) - 1) * 100
    ax.scatter([g_small], [small], s=DOT ** 2, color=INK_3, edgecolor=SURFACE, linewidth=RING, zorder=4)
    ax.annotate(f"a late recipe change,\n~0.0005 bpb ≈ +{g_small:.1f}% compute", xy=(g_small, small),
                xytext=(11.0, 1.0), textcoords="data", ha="left", va="bottom", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.7, shrinkA=2, shrinkB=5))

    ax.set_xlim(0, 62)
    ax.set_ylim(0, 32)
    ax.set_xticks(np.arange(0, 70, 10), [f"+{v}%" if v else "0" for v in range(0, 70, 10)])
    ax.set_xlabel("effective compute gain (more optimizer steps in the same 30-minute budget)")
    ax.set_ylabel("val_bpb reduction ($\\times10^{-3}$)")
    ax.grid(axis="x", visible=False)
    footnote(fig, f"The curve is fitted on one +28% run (24 Sep). Its slope near zero is about 0.00063 bpb per 1%; "
                  "the per-step rate measured directly\non two chips is about 0.00057 bpb per 1% more steps. Gaps "
                  f"are from our best official score, {best:.4f} ({best_name}), to the final leaderboard snapshot,\n"
                  "by rank only (docs/GAP-ANALYSIS.md).")
    return save(fig, "exchange_rate")


# ---------------------------------------------------------------------------------------------
# Fig. 5  Levers: what worked and what lost, by kind of evidence
# ---------------------------------------------------------------------------------------------
# (label, low, high, kind, note) from docs/FINDINGS.md sections 1, 3 and 5; negative = better.
# kind: "chip" = same-seed chip pair(s), raw; "norm" = chip pair(s), step-normalised; "official" = two official
# uploads that differ by this change. note carries n where the repo documents it.
LEVER_ITEMS = [
    ("Multi-token prediction", -0.0043, None, "chip", ""),
    ("Row pool, 256 batches", -0.0023, None, "chip", "n = 1; official −0.0022"),
    ("Batch warm-up 65k → 262k", -0.0019, None, "chip", ""),
    ("LeakyReLU(0.35)²", -0.0017, None, "chip", ""),
    ("Attention-source reuse", -0.0015, None, "chip", "n = 4"),
    ("Key offset", -0.0013, None, "chip", ""),
    ("Cooldown 0.7 + small-batch LR", -0.0006, None, "norm", "n = 3"),
    ("EMA blend 0.6", -0.0005, None, "chip", "n = 1"),
    ("c_proj LR ×0.8", -0.0004, None, "norm", "seed 73"),
    ("EMA prewarm (K82s4 vs K77a)", -0.0003, None, "official", "n = 1"),
    ("Late k=8 batch in cooldown", 0.0005, None, "official", "n = 1"),
    ("Split AdamW cooldown", 0.0006, None, "official", "n = 1"),
    ("AdamW LR −16%", 0.0008, 0.0009, "chip", "n = 2"),
    ("AdamW warm-up reshaping", 0.0010, None, "chip", ""),
    ("Drop block-0 MLP", 0.0019, None, "chip", ""),
    ("NoPE in layers 3-4", 0.0033, None, "chip", ""),
    ("Depth 10 (fewer steps)", 0.0038, 0.0043, "chip", "n = 2"),
    ("Depth 8 (more steps)", 0.0030, 0.0050, "chip", "n = 2"),
    ("2 KV heads", 0.0057, None, "chip", "n = 1"),
]


def fig_levers() -> list:
    items = LEVER_ITEMS[::-1]
    n = len(items)
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    header(fig, "What moved the score, and how we know",
           "Δ val_bpb of each change against its control (negative = better). The fill shows the kind of "
           "evidence; n is the number\nof comparisons where the repo records it.")
    top = n - 0.4
    ax.fill_between([-0.0003, 0.0003], -0.6, top, color=TIE_WASH, lw=0, zorder=0)
    ax.text(0.0, top + 0.15, "tie zone ±0.0003", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.plot([0, 0], [-0.6, top], color=INK_2, lw=0.8, zorder=1)
    for y, (lab, lo, hi, kind, note) in enumerate(items):
        color = BLUE if lo < 0 else RED
        style = {"chip": dict(color=color, edgecolor=SURFACE, linewidth=0.8),
                 "norm": dict(color=SURFACE, edgecolor=color, hatch="//////", linewidth=1.0),
                 "official": dict(color=SURFACE, edgecolor=color, linewidth=1.6)}[kind]
        ax.barh(y, lo, height=0.62, zorder=2, **style)
        end = lo
        if hi is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=1.4, zorder=3)
            ax.plot([hi, hi], [y - 0.18, y + 0.18], color=color, lw=1.4, zorder=3)
            end = hi
        ax.text(-0.0001 if lo > 0 else 0.0001, y, lab, ha="right" if lo > 0 else "left", va="center",
                fontsize=8.3, color=INK)
        val = f"{lo:+.4f}" if hi is None else f"{lo:+.4f} to {hi:+.4f}"
        txt = val.replace("-", "−") + (f"   {note}" if note else "")
        pad = 0.00012
        ax.text(end + (pad if lo > 0 else -pad), y, txt, ha="left" if lo > 0 else "right", va="center",
                fontsize=7.8, color=INK_2)
    ax.set_yticks([])
    ax.set_xlim(-0.0078, 0.0088)
    ax.set_ylim(-0.8, n + 0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.3f}".replace("-", "−") if v else "0"))
    ax.set_xlabel("Δ val_bpb vs its control")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=INK_3, edgecolor=SURFACE, label="chip pair(s), raw"),
                       matplotlib.patches.Patch(facecolor=SURFACE, edgecolor=INK_3, hatch="//////",
                                                label="chip pair(s), step-normalised"),
                       matplotlib.patches.Patch(facecolor=SURFACE, edgecolor=INK_3, linewidth=1.6,
                                                label="two official uploads")],
              loc="lower left", handlelength=1.6, borderaxespad=0.2)
    footnote(fig, "Source: docs/FINDINGS.md sections 1, 3 and 5. Bars with no n are quoted there without a pair "
                  "count: read them as single comparisons, mostly on earlier recipe bases.\nRanges show two "
                  "measurements. Chip pairs are same-seed, same-chip; raw pairs include any step-time cost.",
             y=0.0)
    return save(fig, "levers")


# ---------------------------------------------------------------------------------------------
# Fig. 6  GPU proxy vs Trainium
# ---------------------------------------------------------------------------------------------
def fig_gpu_parity() -> list:
    # docs/SIMULATOR.md, "Route 2 ... What it measured": seed 73, equal steps. (label, gpu, chip, chip source)
    pairs = [("attention-source reuse", -0.00187, -0.00181, "chip"),
             ("LeakyReLU(0.35)² vs relu²", -0.00114, -0.00105, "chip"),
             ("cooldown 0.7 (a tie; chip: mean of 2)", -0.00012, (0.00009 - 0.00037) / 2, "chip"),
             ("split AdamW cooldown 0.4", 0.00045, 0.0006, "official")]
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    header(fig, "GPU proxy vs chip on four paired effects",
           "Each point is one change measured on an L40S GPU with the virtual\ncharged clock and on Trainium, "
           "seed 73, at equal steps.")
    lim = (-2.4, 1.2)
    ax.fill_between([-0.3, 0.3], -0.3, 0.3, color=TIE_WASH, lw=0, zorder=0)
    ax.plot(lim, lim, color=GREY, lw=1.0, zorder=1)
    ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    ax.axvline(0, color=INK_3, lw=0.8, zorder=1)
    offs = {"attention-source reuse": (10, -4, "left"), "LeakyReLU(0.35)² vs relu²": (10, -4, "left"),
            "cooldown 0.7 (a tie; chip: mean of 2)": (-10, 14, "right"), "split AdamW cooldown 0.4": (-10, 10, "right")}
    for lab, gpu, chip, src in pairs:
        x, y = chip * 1e3, gpu * 1e3
        if src == "official":
            ax.scatter([x], [y], s=DOT ** 2 * 1.2, facecolor=SURFACE, edgecolor=BLUE, linewidth=LINE_W, zorder=4)
        else:
            ax.scatter([x], [y], s=DOT ** 2 * 1.2, color=BLUE, edgecolor=SURFACE, linewidth=RING, zorder=4)
        dx, dy, ha = offs[lab]
        label(ax, x, y, lab, dx, dy, ha=ha)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Trainium: Δ val_bpb ($\\times10^{-3}$), chip C pair or official delta")
    ax.set_ylabel("GPU proxy (L40S): Δ val_bpb ($\\times10^{-3}$)")
    ax.legend(handles=[Line2D([], [], lw=0, marker="o", ms=8, mfc=BLUE, mec=SURFACE, label="chip C pair"),
                       Line2D([], [], lw=0, marker="o", ms=8, mfc=SURFACE, mec=BLUE, mew=2,
                              label="official upload delta"),
                       Line2D([], [], color=GREY, lw=1.0, label="GPU = chip"),
                       matplotlib.patches.Patch(color=TIE_WASH, label="tie zone ±0.0003")],
              loc="upper left", handlelength=1.6, borderaxespad=0.2)
    footnote(fig, "Source: docs/SIMULATOR.md, route 2. Absolute level: one K60 run gave 0.959952 on the GPU and "
                  "0.959948 on chip at 2,314 steps;\ndefault-mode CUDA runs of one config differed by 0.00056 "
                  "between boxes, so that 4e-6 match is partly luck. n = 4 is a small sample.", y=-0.01)
    return save(fig, "gpu_parity")


# ---------------------------------------------------------------------------------------------
# Fig. 7  Simulator: quality surrogate, leave-one-pair-out, all 99 pairs
# ---------------------------------------------------------------------------------------------
def fig_simulator() -> list:
    sys.path.insert(0, str(ROOT))
    from ffsim.dataset import load_runs
    from ffsim.quality import QualityModel

    records = load_runs(str(RUNS))
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    qm = QualityModel().fit(records, pairs)
    rows = qm.pair_validation(leave_pair_out=True)["rows"]
    noise = {"seed", "chip"}
    rec = [r for r in rows if r["family"] not in noise]
    nse = [r for r in rows if r["family"] in noise]
    sign = sum(r["sign_ok"] for r in rows) / len(rows)
    mae = sum(r["abs_error"] for r in rows) / len(rows)
    sign_r = sum(r["sign_ok"] for r in rec) / len(rec)
    mae_r = sum(r["abs_error"] for r in rec) / len(rec)
    cover = sum(abs(r["observed"] - r["predicted"]) <= 2 * r["pred_sd"] for r in rows) / len(rows)
    vals = [v * 1e3 for r in rows for v in (r["observed"], r["predicted"])]
    lim = (math.floor(min(vals) * 2) / 2 - 0.4, math.ceil(max(vals) * 2) / 2 + 0.4)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    header(fig, "The simulator's quality model on pairs it did not see",
           f"Each point is one same-seed chip treatment/control pair (all n = {len(rows)}). The model is refitted "
           f"without the pair's\ntwo runs, then predicts their val_bpb difference.")
    ax.fill_between([lim[0], 0], 0, lim[1], color="#f3f2ee", lw=0, zorder=0)
    ax.fill_between([0, lim[1]], lim[0], 0, color="#f3f2ee", lw=0, zorder=0)
    ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    ax.axvline(0, color=INK_3, lw=0.8, zorder=1)
    ax.plot(lim, lim, color=GREY, lw=1.0, zorder=1)
    ax.scatter([r["observed"] * 1e3 for r in rec], [r["predicted"] * 1e3 for r in rec], s=DOT ** 2 * 0.85,
               color=BLUE, edgecolor=SURFACE, linewidth=RING, zorder=4)
    ax.scatter([r["observed"] * 1e3 for r in nse], [r["predicted"] * 1e3 for r in nse], s=DOT ** 2 * 0.85,
               color=ORANGE, edgecolor=SURFACE, linewidth=RING, zorder=4)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured Δ val_bpb, treatment − control ($\\times10^{-3}$)")
    ax.set_ylabel("predicted Δ val_bpb ($\\times10^{-3}$)")
    ax.text(lim[0] + 0.15, lim[1] - 0.15,
            f"all {len(rows)} pairs: sign right {sign:.1%}, MAE {mae:.5f} bpb\n"
            f"{len(rec)} recipe pairs: sign right {sign_r:.1%}, MAE {mae_r:.5f} bpb",
            ha="left", va="top", fontsize=8.5, color=INK)
    ax.text(lim[1] - 0.15, lim[0] + 0.15, "shaded: predicted sign is wrong", ha="right", va="bottom",
            fontsize=8, color=INK_3)
    chip = next((r for r in nse if r["family"] == "chip"), None)
    if chip:
        label(ax, chip["observed"] * 1e3, chip["predicted"] * 1e3, "chip D vs chip C,\nsame recipe", -16, -18,
              ha="right")
    ax.legend([Line2D([], [], lw=0, marker="o", ms=7, mfc=BLUE, mec=SURFACE),
               Line2D([], [], lw=0, marker="o", ms=7, mfc=ORANGE, mec=SURFACE),
               Line2D([], [], color=GREY, lw=1.0)],
              [f"recipe pairs (n = {len(rec)})", f"seed and chip pairs (n = {len(nse)})", "prediction = measurement"],
              loc="lower right", bbox_to_anchor=(1.0, 0.06), handletextpad=0.4, handlelength=1.6)
    footnote(fig, "Source: research/sim-data/runs.jsonl and validation-pairs.json; "
                  "ffsim.quality.QualityModel.pair_validation(leave_pair_out=True).\n"
                  f"{cover:.0%} of measured differences lie within ±2 predicted sd. Seed and chip pairs measure "
                  "run-to-run noise, not a recipe effect.\nPer-subset numbers: research/sim-data/quality-validation.md "
                  "section 3.", y=-0.005)
    print(f"simulator_pairs: n={len(rows)} sign={sign:.3f} mae={mae:.5f}; recipe n={len(rec)} sign={sign_r:.3f} "
          f"mae={mae_r:.5f}; coverage={cover:.2f}")
    return save(fig, "simulator_pairs")


def main() -> int:
    rows = load_scores()
    written = []
    written += fig_score_history(rows)
    written += fig_steps(rows)
    written += fig_oracle(rows)
    written += fig_exchange(rows)
    written += fig_levers()
    written += fig_gpu_parity()
    written += fig_simulator()
    for p in written:
        print(f"{p.relative_to(ROOT).as_posix()}  {p.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
