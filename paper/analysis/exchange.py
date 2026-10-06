"""The price of a step, and every lever priced in optimizer steps (paper Sections 3.4 and 7).

ONE constant prices everything: kappa = 0.057 bpb per unit of ln(optimizer steps), i.e. 0.00057 bpb per 1% more
steps, the per-step rate measured on same-recipe runs across two chips (docs/GAP-ANALYSIS.md) and the slope the
quality surrogate's prior is centred on (ffsim/quality.py, which therefore is NOT independent evidence for it). The
single long run (+28% compute bought 0.0155, docs/GAP-ANALYSIS.md) calibrates kappa = 0.0629; it is used only as the
upper end of a sensitivity range [0.057, 0.063], never as a second price.

Inputs: docs/GAP-ANALYSIS.md (anchor, per-step rate, leaderboard references by rank), results/official-scores.csv
(our best score), docs/FINDINGS.md sections 1, 3 and 5 (the levers), docs/first-principles-note (LeakyReLU).
Outputs: tables/gap.tex, tables/levers.tex, tables/negative.tex, figures/exchange_rate.pdf.
"""
from __future__ import annotations

import math
from typing import Dict

from common import FIGURES, TABLES, Numbers, load_scores, milli, num, table

GAP = "docs/GAP-ANALYSIS.md"
FIND = "docs/FINDINGS.md"

ANCHOR_GAIN = 0.0155          # val_bpb improvement of the 2,288 s run over 1,788 s (24 Sep)
ANCHOR_LONG, ANCHOR_SHORT = 2288.0, 1788.0
KAPPA = 0.057                 # THE price: bpb per unit ln(steps) (= 0.00057 per 1% more steps)
PER_STEP_PER_PCT = KAPPA / 100
CURV = 0.026                  # ffsim/quality.py curvature prior: kappa(S) = KAPPA - 2 CURV ln(S / 2300)
RANK10, RANK1 = 0.9554, 0.9328
SALT_SD = 0.00035             # docs/FINDINGS.md s3: order re-draw SD 0.0003-0.0004 (midpoint)
OFF_SD = 0.00038              # since-K50 offset SD (results/official-scores.csv; oracle.py)

# (lever, delta, delta_hi or None, evidence, n, note)
# evidence: chip | norm (chip pair, step-normalised) | official | log (campaign log, no run IDs in the repository)
LEVERS = [
    ("Multi-token prediction (3 tokens, faded)", -0.0043, None, "log", "", ""),
    ("Row pool, 256 micro-batches", -0.0023, None, "chip", "1", "chip E; official $-$2.2; $-$0.5\\% steps"),
    ("Batch warm-up 65k$\\to$131k$\\to$262k", -0.0019, None, "log", "", "changes steps itself$^{\\ddagger}$"),
    ("LeakyReLU(0.35)$^2$ MLP", -0.0017, None, "log", "", "chip C vs ReLU$^2$ $-$1.05; official $-$1.5"),
    ("Attention-source reuse (6--8 reuse 5)", -0.0015, None, "chip", "4", "3 seeds, 2 chips; 5th pair (s97) $+$1.15 raw; step cost $<$0.1\\%"),
    ("Rotary key offset", -0.0013, None, "log", "", ""),
    ("Cooldown 0.7 + small-batch LR", -0.0006, None, "norm", "3", "all 3 negative; seeds differ"),
    ("EMA blend 0.6 (no prewarm)", -0.0005, None, "chip", "1", "official $-$0.3 (K77a$-$K73s4)"),
    ("MLP output-projection LR $\\times$0.8", -0.0004, None, "norm", "1", "seed 73 only"),
    ("EMA prewarm (K82s4 vs K77a)", -0.0003, None, "official", "1", "recovers 15--30 steps"),
]
NEGATIVE = [
    ("Depth 10 (bigger; 6--11\\% fewer steps)", 0.0038, 0.0043, "chip", "2"),
    ("Two KV heads (bigger)", 0.0057, None, "chip", "1"),
    ("Depth 8 (smaller; +10.6\\% steps)", 0.0030, 0.0050, "chip", "2"),
    ("Late $k{=}8$ batch in the cooldown (K65b)", 0.0005, None, "official", "1"),
    ("Split AdamW cooldown 0.4 (K65d)", 0.0006, None, "official", "1"),
    ("EMA-Nesterov lookahead 0.3 (K65c)", 0.00004, None, "official", "1"),
    ("AdamW LR $-$16\\%", 0.0008, 0.0009, "chip", "2"),
    ("AdamW warm-up reshaping (steps 0--19)", 0.0010, None, "chip", "1"),
    ("Drop the block-0 MLP", 0.0019, None, "chip", "1"),
    ("No positional encoding in layers 3--4", 0.0033, None, "chip", "1"),
    ("Attention-input shift, blocks 6--8 (K65a)$^{\\S}$", 0.0034, None, "official", "1"),
]
EVIDENCE = {"chip": "chip pair", "norm": "chip, step-norm.", "official": "official pair",
            "log": "campaign log$^{\\dagger}$"}


def steps_equiv(delta: float, kappa: float = KAPPA) -> float:
    """% more optimizer steps worth the same |delta| at price kappa (log form)."""
    return (math.exp(abs(delta) / kappa) - 1.0) * 100.0


def pair_sd(evidence: str, n: int) -> float:
    """SD of a measured difference under the order-redraw noise model (Section 7.6): sqrt(2) x salt SD per chip
    pair, plus the offset noise of two uploads for an official pair; divided by sqrt(n) for n pairs."""
    s = math.sqrt(2) * SALT_SD
    if evidence == "official":
        s = math.sqrt(2) * math.sqrt(SALT_SD ** 2 + OFF_SD ** 2)
    return s / math.sqrt(max(n, 1))


def run(N: Numbers) -> Dict[str, object]:
    k_anchor = ANCHOR_GAIN / math.log(ANCHOR_LONG / ANCHOR_SHORT)
    best = min(r["official"] for r in load_scores() if r["official"] is not None)
    k44 = next(r for r in load_scores() if r["name"] == "K44")
    N.add("xr-k", num(k_anchor, 4), f"{GAP}: 0.0155 / ln(2288/1788)")
    N.add("xr-k-anchor", num(k_anchor, 3), GAP)
    N.add("kappa", num(KAPPA, 3), f"{GAP}: per-step rate x 100 (doc)")
    N.add("xr-anchor-pct", num(100 * (ANCHOR_LONG / ANCHOR_SHORT - 1), 0) + "\\%", GAP)
    N.add("xr-anchor-gain", num(ANCHOR_GAIN, 4), GAP + ": gain of the long run (doc)")
    N.add("xr-anchor-long", "2{,}288", GAP + ": long run, seconds (doc)")
    N.add("xr-anchor-short", "1{,}788", GAP + ": short run, seconds (doc)")
    N.add("xr-slope-pct", num(k_anchor * math.log(1.01), 5), f"{GAP}: k ln(1.01)")
    N.add("xr-step-rate", num(PER_STEP_PER_PCT, 5), GAP + ": per-step rate, bpb per 1% more steps (doc)")
    N.add("q-prior-b", num(-KAPPA, 3), "ffsim/quality.py slope prior mean")
    N.add("q-prior-b-sd", num(0.005, 3), "ffsim/quality.py slope prior SD")
    N.add("q-prior-c", num(CURV, 3), "ffsim/quality.py curvature prior mean")
    N.add("q-prior-c-sd", num(0.01, 2), "ffsim/quality.py curvature prior SD")
    N.add("q-prior-knob", num(0.004, 3), "ffsim/quality.py knob prior SD")
    N.add("q-prior-chip", num(0.003, 3), "ffsim/quality.py chip prior SD")
    N.add("q-prior-seed", num(0.0006, 4), "ffsim/quality.py seed prior SD")
    N.add("q-sigma-floor", num(0.0003, 4), "ffsim/quality.py sigma floor")
    N.add("slope-900", "0.095--0.115", "ffsim/quality.py docstring: slope magnitude at about 900 steps (doc)")
    # the curvature prior's local slope at the anchor run's mid-point, if it ran at K44's step count
    s_mid = k44["steps"] * math.sqrt(ANCHOR_LONG / ANCHOR_SHORT)
    N.add("k44-steps", f"{int(k44['steps']):,}".replace(",", "{,}"), "results/official-scores.csv")
    N.add("kappa-at-anchor", num(KAPPA - 2 * CURV * math.log(s_mid / 2300.0), 3),
          "kappa(S) = 0.057 - 2 x 0.026 ln(S/2300) at the anchor's geometric mid-point")
    N.add("rank10", num(RANK10, 4), f"{GAP}: #10 on the final snapshot (doc)")
    N.add("rank1", num(RANK1, 4), f"{GAP}: #1 on the final snapshot (doc)")
    body = []
    k_print = round(k_anchor, 3)    # the table header prints kappa to 3 decimals; price the column at that value
    for label, target, key in (("\\#10 on the final snapshot", RANK10, "ten"), ("\\#1 on the final snapshot", RANK1, "one")):
        gap = best - target
        lo = (math.exp(gap / k_print) - 1.0) * 100.0
        hi = (math.exp(gap / KAPPA) - 1.0) * 100.0
        N.add(f"gap-{key}", num(gap, 4), GAP)
        N.add(f"gap-{key}-pct", num(hi, 0) + "\\%", GAP + f" at kappa {KAPPA}")
        N.add(f"gap-{key}-range", f"{num(lo, 0)}--{num(hi, 0)}\\%", GAP + " kappa in [0.057, 0.063]")
        body.append(f"{label} & {num(target, 4)} & {num(gap, 4)} & +{num(hi, 0)}\\% & +{num(lo, 0)}\\% \\\\")
    table(TABLES / "gap.tex", body, "@{}lrrrr@{}",
          f"Target & val\\_bpb & Gap & Compute, $\\kappa={num(KAPPA, 3)}$ & Compute, $\\kappa={num(k_anchor, 3)}$",
          "compute needed = exp(gap / kappa) - 1 (exchange.py)")
    lo = KAPPA * math.log(1.15)
    hi = k_anchor * math.log(1.23)
    N.add("tput-range", f"{num(lo, 3)}--{num(hi, 3)}", GAP + ": 15-23% tokens/s at kappa 0.057-0.063")
    N.add("tput-lo-pct", "15", GAP)
    N.add("tput-hi-pct", "23", GAP)
    N.add("late-change-pct", num(steps_equiv(0.0005), 1) + "\\%", "0.0005 bpb priced at kappa")

    noise_sds = []
    body = []
    for lab, d, dh, ev, n, note in LEVERS:
        val = milli(d, 1, sign=True) if dh is None else f"{milli(d, 1, sign=True)} to {milli(dh, 1, sign=True)}"
        s = pair_sd(ev, int(n) if n else 1)
        noise_sds.append(s)
        eq = "n/a" if "changes steps" in note else num(steps_equiv(d), 1) + "\\%"
        # campaign-log rows have no recorded comparison, so no ratio to a noise scale is printed for them
        ratio = "--" if ev == "log" else num(abs(d) / s, 1)
        body.append(f"{lab} & {val} & {milli(s, 2)} & {ratio} & {EVIDENCE[ev]} & {n or '--'} & "
                    f"{eq} & {note} \\\\")
    table(TABLES / "levers.tex", body, "@{}p{4.2cm}rrrlrrp{3.3cm}@{}",
          "Lever & $\\Delta$ & Noise SD & $|\\Delta|/\\text{SD}$ & Evidence & $n$ & Steps & Note",
          f"levers that worked, {FIND} sections 1 and 3; step equivalent at kappa {KAPPA}", size="\\footnotesize",
          tabcolsep="4pt")
    N.add("noise-chip-pair", num(pair_sd("chip", 1), 5), "sqrt(2) x salt SD 0.00035")
    N.add("noise-official-pair", num(pair_sd("official", 1), 5), "sqrt(2) x sqrt(salt SD^2 + offset SD^2)")
    N.add("rowpool-z", num(0.0023 / pair_sd("chip", 1), 1), "row pool |delta| / chip-pair noise SD")
    N.add("lever-leaky", num(-0.0017, 4), f"{FIND} s1 (doc)")
    body = []
    for lab, d, dh, ev, n in NEGATIVE:
        val = milli(d, 2 if abs(d) < 0.0001 else 1, sign=True) if dh is None else \
            f"{milli(d, 1, sign=True)} to {milli(dh, 1, sign=True)}"
        body.append(f"{lab} & {val} & {EVIDENCE[ev]} & {n} \\\\")
    body.append("Scalar retunes (LR scales, betas, WD, init, momentum, soft cap, RoPE base, cooldown $\\pm$0.1) & "
                "0 of 61 keys won & chip pairs & 61 \\\\")
    body.append("Bigger pools, epoch reshuffles, pool-aware accumulation, document reordering & 0 $\\pm$ 0.3 & "
                "chip pairs & -- \\\\")
    table(TABLES / "negative.tex", body, "@{}p{7.6cm}rlr@{}", "Change & $\\Delta$ ($10^{-3}$) & Evidence & $n$",
          f"what lost, {FIND} section 5 and results/official-scores.csv (K65a-d)", size="\\small")
    N.add("rowpool-steps-equiv", num(steps_equiv(-0.0023), 1) + "\\%", "row pool priced in steps")
    N.add("mtp-steps-equiv", num(steps_equiv(-0.0043), 1) + "\\%", "MTP priced in steps")
    late = [abs(d) for lab, d, _, ev, _, _ in LEVERS if ev != "log" and abs(d) < 0.001]
    N.add("late-lever-range", f"{num(min(late), 4)}--{num(max(late), 4)}", FIND + ": late levers in tab:levers")
    N.add("n-levers", str(len(LEVERS)), FIND)
    N.add("n-negative", str(len(NEGATIVE) + 2), FIND)
    _figure(best, k_anchor)
    return {"kappa": KAPPA, "k_anchor": k_anchor}


def vague_prices(N: Numbers, k_lo: float, k_hi: float) -> None:
    """The gap to #10 and a typical late change priced at the ends of the vague-prior kappa interval (simulator.py)."""
    best = min(r["official"] for r in load_scores() if r["official"] is not None)
    gap = best - RANK10
    src = f"{GAP}: gap to #10 priced at the vague-prior kappa interval (simulator.py kappa-vague-lo/hi)"
    pcts = sorted((math.exp(gap / k) - 1.0) * 100.0 for k in (k_lo, k_hi))
    N.add("gap-ten-vague-range", f"{num(pcts[0], 0)}--{num(pcts[1], 0)}\\%", src)
    late = sorted(steps_equiv(0.0005, k) for k in (k_lo, k_hi))
    N.add("late-change-vague-range", f"{num(late[0], 1)}--{num(late[1], 1)}\\%",
          "0.0005 bpb priced at the vague-prior kappa interval")


def _figure(best: float, k_anchor: float) -> None:
    from style import AQUA, BLUE, BLUE_WASH, INK, ORANGE, TEXTWIDTH, plt, save_pdf
    import numpy as np

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.62, 2.6))
    r = np.linspace(0.0, 70.0, 300)
    g_lo = KAPPA * np.log1p(r / 100) * 1e3
    g_hi = k_anchor * np.log1p(r / 100) * 1e3
    ax.fill_between(r, g_lo, g_hi, color=BLUE_WASH, lw=0)
    ax.plot(r, g_lo, color=BLUE, lw=1.3, label=f"$\\kappa$ = {KAPPA:.3f} (price used)")
    ax.plot(r, g_hi, color=BLUE, lw=1.0, ls=(0, (4, 2)), label=f"$\\kappa$ = {k_anchor:.3f} (long-run anchor)")
    ax.scatter([100 * (ANCHOR_LONG / ANCHOR_SHORT - 1)], [ANCHOR_GAIN * 1e3], marker="s", s=24, color=ORANGE,
               zorder=4, label="anchor: +28% compute")
    for target, lab in ((RANK10, "#10"), (RANK1, "#1")):
        gap = (best - target) * 1e3
        x_lo = (math.exp(gap / 1e3 / k_anchor) - 1) * 100
        x_hi = (math.exp(gap / 1e3 / KAPPA) - 1) * 100
        ax.plot([x_lo, x_hi], [gap, gap], color=INK, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.annotate(f"gap to {lab}: {gap:.1f}", xy=(x_hi, gap), xytext=(4, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=7.5, color=INK)
    late = 0.5
    ax.scatter([steps_equiv(0.0005)], [late], marker="v", s=24, color=AQUA, zorder=4,
               label=f"late recipe change ({steps_equiv(0.0005):.1f}% steps)")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 32)
    ax.set_xlabel("extra effective compute (% more optimizer steps)")
    ax.set_ylabel("bpb gained ($10^{-3}$)")
    ax.legend(loc="lower right", handlelength=1.6)
    save_pdf(fig, FIGURES / "exchange_rate.pdf")
