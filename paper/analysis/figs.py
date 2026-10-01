"""The paper's data figures, drawn at their final printed size (paper Sections 1, 3 and 5).

Revision 1: these replace the README figures that the paper used to shrink to about half size (text at 4-5 pt).
Every figure here is TEXTWIDTH (6.5 in) wide or narrower and is placed at its natural width, so 8 pt in the code is
8 pt on the page. Chip identity and series are carried by marker shape as well as colour (grayscale-safe); the
legend keys that the README versions draw inside the figure are in the LaTeX captions instead.

Input: results/official-scores.csv. Outputs: figures/score_history.pdf, figures/step_counts.pdf,
figures/oracle_calibration.pdf.
"""
from __future__ import annotations

from typing import Dict, List

from common import FIGURES, Numbers, load_scores, mean, sd


def run(N: Numbers) -> Dict[str, object]:
    rows = load_scores()
    _score_history(rows)
    _step_counts(rows)
    _oracle(rows)
    return {}


def _short(n: str) -> str:
    return n.replace("_min", "")


def _score_history(rows: List[dict]) -> None:
    import numpy as np
    from matplotlib.lines import Line2D
    from style import BLUE, BLUE_WASH, GREY, INK, SURFACE, TEXTWIDTH, plt, save_pdf

    scored = [r for r in rows if r["official"] is not None]
    s = {r["name"]: r for r in scored}
    best = min(scored, key=lambda r: r["official"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.9), gridspec_kw={"width_ratios": [1.0, 1.35],
                                                                             "wspace": 0.28})
    # panel a: calendar time; uploads of one day spread across the day
    import datetime as dt
    per_day: Dict[str, List[str]] = {}
    for r in scored:
        per_day.setdefault(r["date"].replace("~", ""), []).append(r["name"])
    d0 = dt.date(2026, 9, 15)
    tx = {}
    for date, names in per_day.items():
        day = (dt.date.fromisoformat(date) - d0).days
        for k, n in enumerate(names):
            tx[n] = day + (k + 1) / (len(names) + 1)

    def stair(ax, names, xof):
        xs, ys, cur = [], [], 9.0
        for n in names:
            cur = min(cur, s[n]["official"])
            xs.append(xof(n))
            ys.append(cur)
        ax.step(xs, ys, where="post", color=BLUE, lw=1.3, zorder=2)
        nb = [n for n in names if not s[n]["best"]]
        bb = [n for n in names if s[n]["best"]]
        ax.scatter([xof(n) for n in nb], [s[n]["official"] for n in nb], s=16, marker="s", color=GREY,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.scatter([xof(n) for n in bb], [s[n]["official"] for n in bb], s=18, marker="o", color=BLUE,
                   edgecolor=SURFACE, linewidth=0.6, zorder=4)

    names = [r["name"] for r in scored]
    stair(a1, names, tx.get)
    a1.set_xlim(-0.3, 17.3)
    a1.set_xticks([0, 4, 8, 12, 16], ["15 Sep", "19 Sep", "23 Sep", "27 Sep", "1 Oct"])
    a1.grid(axis="x", visible=False)
    a1.set_ylabel("official val_bpb")
    a1.set_title("a   whole campaign (calendar time)", loc="left")
    z0 = tx["K50"] - 0.3
    a1.axvspan(z0, 17.3, color=BLUE_WASH, lw=0, zorder=0)
    a1.set_ylim(0.952, 1.162)
    a1.annotate(f"F2 {s['F2']['official']:.4f}", xy=(tx["F2"], s["F2"]["official"]), xytext=(6, 4),
                textcoords="offset points", fontsize=7, va="bottom")
    a1.annotate(f"K44 {s['K44']['official']:.4f}", xy=(tx["K44"], s["K44"]["official"]), xytext=(0.30, 0.42),
                textcoords="axes fraction", fontsize=7, ha="left",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": INK})
    a1.annotate(f"{best['name']} {best['official']:.5f}", xy=(tx[best["name"]], best["official"]), xytext=(0.52, 0.24),
                textcoords="axes fraction", fontsize=7, ha="left", fontweight="bold",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": INK})
    # panel b: from K50 in upload order
    zoom = names[names.index("K50"):]
    ix = {n: i for i, n in enumerate(zoom)}
    stair(a2, zoom, ix.get)
    a2.set_xticks(range(len(zoom)), [_short(n) for n in zoom], rotation=60, ha="right", fontsize=6.5)
    a2.tick_params(axis="x", length=0)
    a2.grid(axis="x", visible=False)
    a2.set_xlim(-0.6, len(zoom) - 0.4)
    a2.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.3f"))
    a2.set_title("b   from K50, in upload order", loc="left")
    a2.legend(handles=[Line2D([], [], lw=0, marker="o", ms=4, mfc=BLUE, mec=SURFACE, label="new best"),
                       Line2D([], [], lw=0, marker="s", ms=4, mfc=GREY, mec=SURFACE, label="not a new best"),
                       Line2D([], [], color=BLUE, lw=1.3, label="best so far")], loc="upper right", handlelength=1.4)
    _ = np
    save_pdf(fig, FIGURES / "score_history.pdf")


def _step_counts(rows: List[dict]) -> None:
    from style import BLUE, CHIP_MARKER, GREY, INK_2, ORANGE, TEXTWIDTH, plt, save_pdf

    pts = [r for r in rows if r["steps"]]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.5))
    names = [r["name"] for r in pts]
    i24, i50 = names.index("K24"), names.index("K50")
    for lo, hi, lab in ((-0.5, i24 - 0.5, "131k-token batch"), (i24 - 0.5, i50 - 0.5, "262k batch"),
                        (i50 - 0.5, len(pts) - 0.5, "batch warm-up 65k/131k $\\to$ 262k")):
        ax.axvspan(lo, hi, color=GREY, alpha=0.10 if lab.startswith("262") else 0.25, lw=0, zorder=0)
        ax.text((lo + hi) / 2, 2870, lab, ha="center", va="top", fontsize=7, color=INK_2)
    for i, r in enumerate(pts):
        slow = r["chip"] == "E"
        ax.bar(i, r["steps"], width=0.7, color=ORANGE if slow else BLUE, hatch="////" if slow else None,
               edgecolor="white", lw=0.4, zorder=2)
        ax.text(i, r["steps"] + 25, r["chip"], ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(len(pts)), [_short(n) for n in names], rotation=60, ha="right", fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_ylim(0, 2900)
    ax.set_ylabel("optimizer steps in 30 min")
    _ = CHIP_MARKER
    save_pdf(fig, FIGURES / "step_counts.pdf")


def _oracle(rows: List[dict]) -> None:
    import numpy as np
    from matplotlib.lines import Line2D
    from style import BLUE, BLUE_WASH, CHIP_MARKER, GREY, INK_2, SURFACE, TEXTWIDTH, chip_s, plt, save_pdf

    cal = [r for r in rows if r["official"] is not None and r["rehearsal"] is not None]
    for r in cal:
        r["off"] = r["official"] - r["rehearsal"]
    k50 = next(i for i, r in enumerate(cal) if r["name"] == "K50")
    since = cal[k50:]
    m, s = mean(r["off"] for r in since), sd([r["off"] for r in since])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8), gridspec_kw={"width_ratios": [1.0, 1.45],
                                                                             "wspace": 0.3})
    x = np.linspace(0.953, 0.970, 10)
    a1.fill_between(x, x + m - 2 * s, x + m + 2 * s, color=BLUE_WASH, lw=0)
    a1.plot(x, x + m, color=BLUE, lw=1.1)
    a1.plot(x, x, color=GREY, lw=0.8, ls=(0, (3, 2)))
    for r in since:
        a1.scatter(r["rehearsal"], r["official"], s=chip_s(r["chip"], 20), marker=CHIP_MARKER[r["chip"]], color=BLUE,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
    a1.set_xlim(0.953, 0.970)
    a1.set_ylim(0.959, 0.977)
    a1.set_xlabel("rehearsal val_bpb (first 2M public tokens)")
    a1.set_ylabel("official val_bpb")
    a1.set_title("a   since K50", loc="left")
    a1.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(0.005))
    a1.text(0.9545, 0.9755, f"official = rehearsal + {m:.4f}", fontsize=7, color=BLUE)
    a1.text(0.9625, 0.9605, "identity", fontsize=7, color=INK_2)
    for i, r in enumerate(cal):
        early = i < k50
        a2.scatter(i, r["off"] * 1e3, s=chip_s(r["chip"], 20), marker=CHIP_MARKER.get(r["chip"], "o"),
                   color=GREY if early else BLUE,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3)
    a2.axhline(m * 1e3, color=BLUE, lw=1.0, xmin=k50 / len(cal))
    a2.set_xticks(range(len(cal)), [_short(r["name"]) for r in cal], rotation=90, fontsize=6)
    a2.tick_params(axis="x", length=0)
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("offset ($10^{-3}$ bpb)")
    a2.set_title("b   offset per upload (grey: chips A/B, before K50)", loc="left")
    chips = sorted({r["chip"] for r in cal})
    a2.legend(handles=[Line2D([], [], lw=0, marker=CHIP_MARKER[c], ms=7 if CHIP_MARKER[c] == "*" else 5, mfc=INK_2,
                              mec=INK_2, label=f"chip {c}")
                       for c in chips], ncol=4, loc="upper left", handletextpad=0.1, columnspacing=0.8,
              borderaxespad=0.2)
    a2.set_ylim(4.8, 9.2)
    save_pdf(fig, FIGURES / "oracle_calibration.pdf")
