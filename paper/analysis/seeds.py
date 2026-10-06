"""Warm-up path statistics (the paper's warm-up analysis) from runs.jsonl loss curves.

The seed sweeps are the full chip runs of one fixed recipe that differ only in FF_SEED: run names
``HHMM_R_g7_s<seed>`` (K57 recipe, code M9, chip C) and ``HHMM_R_g7lk35_s<seed>`` (LeakyReLU(0.35)^2, code M12,
chips C and D). The logged loss curve has points at steps 0-4, 10 and 20. A run is on the SPIKE path when its
training loss at step 10 is at least ``SPIKE_L10`` nats: good-path runs of these recipes sit at 13.11-13.23, spike
runs at 13.69-14.27 (the threshold sits in the empty gap; one run at 13.39 is reported as intermediate).

Outputs: tables/seed_paths.tex, figures/warmup_paths.pdf.
"""
from __future__ import annotations

import collections
import math
import re
from typing import Dict, List

from common import (FIGURES, RUNS, TABLES, Numbers, load_scores, mean, milli, num, pct, sd, t_quantile, table,
                    wilson)

SPIKE_L10 = 13.5
INTERMEDIATE = (13.3, SPIKE_L10)
GROUPS = [("C", "g7", "K57 recipe (M9), chip C"),
          ("C", "g7lk35", "LeakyReLU(0.35)$^2$ (M12), chip C"),
          ("D", "g7lk35", "LeakyReLU(0.35)$^2$ (M12), chip D")]
SRC = "research/sim-data/runs.jsonl loss_curve (seeds.py)"


def _welch(a: List[float], b: List[float]):
    ma, mb = mean(a), mean(b)
    va, vb = sd(a) ** 2 / len(a), sd(b) ** 2 / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = t_quantile(0.975, max(1, int(math.floor(df))))
    d = ma - mb
    return d, d - t * se, d + t * se


def run(N: Numbers) -> Dict[str, object]:
    from ffsim.dataset import load_runs
    recs = load_runs(str(RUNS))
    groups: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in recs:
        m = re.match(r"^\d{4}_R_(g7|g7lk35)_s(\d+)$", r.run or "")
        if not m or r.is_screen or r.bpb_2m is None:
            continue
        curve = {int(s): float(v) for s, v in (r.loss_curve or [])}
        groups[(r.chip, m.group(1))].append({"seed": r.seed, "bpb": float(r.bpb_2m), "steps": r.steps,
                                             "curve": curve, "L10": curve.get(10)})
    body = []
    all_k = all_n = 0
    deltas = []
    pooled_good, pooled_spike = [], []
    sds_all, sds_good = [], []
    for chip, fam, label in GROUPS:
        g = groups[(chip, fam)]
        with_c = [x for x in g if x["L10"] is not None]
        spike = [x for x in with_c if x["L10"] >= SPIKE_L10]
        good = [x for x in with_c if x["L10"] < INTERMEDIATE[0]]
        mid = [x for x in with_c if INTERMEDIATE[0] <= x["L10"] < SPIKE_L10]
        all_k += len(spike)
        all_n += len(with_c)
        d, lo, hi = _welch([x["bpb"] for x in spike], [x["bpb"] for x in good])
        deltas.append((len(spike), len(good), d))
        # within-group centring removes the chip/recipe level before pooling
        gm = mean([x["bpb"] for x in good])
        pooled_good += [x["bpb"] - gm for x in good]
        pooled_spike += [x["bpb"] - gm for x in spike]
        sds_all.append(sd([x["bpb"] for x in g]))
        sds_good.append(sd([x["bpb"] for x in good]))
        body.append(f"{label} & {len(g)} & {len(with_c)} & {len(spike)} & {len(mid)} & "
                    f"{milli(d, 2, sign=True)} [{milli(lo, 2, sign=True)}, {milli(hi, 2, sign=True)}] & "
                    f"{milli(sd([x['bpb'] for x in g]), 2)} & {milli(sd([x['bpb'] for x in good]), 2)} \\\\")
    lo_w, hi_w = wilson(all_k, all_n)
    dp, dlo, dhi = _welch(pooled_spike, pooled_good)
    body.append("\\midrule")
    body.append(f"pooled (centred on each group's good-path mean) & {sum(len(groups[(c, f)]) for c, f, _ in GROUPS)} & "
                f"{all_n} & {all_k} & & {milli(dp, 2, sign=True)} [{milli(dlo, 2, sign=True)}, {milli(dhi, 2, sign=True)}] & & \\\\")
    table(TABLES / "seed_paths.tex", body, "@{}p{4.0cm}rrrrlrr@{}",
          "Seed sweep & Seeds & Curve & Spike & Interm. & Spike $-$ good [95\\% CI] & SD all & SD good",
          "warm-up path per seed sweep, 1e-3 bpb (seeds.py)", size="\\footnotesize", tabcolsep="4pt")
    N.add("spike-k", str(all_k), SRC)
    N.add("spike-n", str(all_n), SRC)
    N.add("spike-frac", pct(all_k / all_n, 0), SRC)
    N.add("spike-wilson-lo", pct(lo_w, 0), SRC)
    N.add("spike-wilson-hi", pct(hi_w, 0), SRC)
    N.add("spike-delta", num(dp, 4, sign=True), SRC)
    N.add("spike-delta-lo", num(dlo, 4, sign=True), SRC)
    N.add("spike-delta-hi", num(dhi, 4, sign=True), SRC)
    N.add("spike-l10", num(SPIKE_L10, 1), SRC)
    N.add("seed-sd-all-lo", num(min(sds_all), 5), SRC)
    N.add("seed-sd-all-hi", num(max(sds_all), 5), SRC)
    N.add("seed-sd-good-lo", num(min(sds_good), 5), SRC)
    N.add("seed-sd-good-hi", num(max(sds_good), 5), SRC)
    # step-count jitter: the seed changes neither the program nor the step time, so within a sweep (one recipe, one
    # chip) the spread of the step count is run-to-run jitter of the charged clock.
    jit_sd, jit_rng, jit_rel = [], [], []
    for chip, fam, _ in GROUPS:
        st = [x["steps"] for x in groups[(chip, fam)] if x["steps"]]
        jit_sd.append(sd(st))
        jit_rel.append(sd(st) / mean(st))
        jit_rng.append(max(st) - min(st))
    N.add("jitter-sd-lo", num(min(jit_sd), 1), SRC + ": SD of steps within a seed sweep")
    N.add("jitter-sd-hi", num(max(jit_sd), 1), SRC)
    N.add("jitter-range-lo", str(min(jit_rng)), SRC + ": max - min steps within a seed sweep")
    N.add("jitter-range-hi", str(max(jit_rng)), SRC)
    # the chip-G step drop without the EMA prewarm (K73s6 - K77a), in single-run jitter SDs
    sc = {r["name"]: r for r in load_scores()}
    # the official K65a - K63 delta (an attention-input shift that may have re-rolled the warm-up path) in units of
    # the controlled spike-path penalty
    N.add("xshift-spike-ratio", num((sc["K65a"]["official"] - sc["K63"]["official"]) / dp, 0),
          "results/official-scores.csv K65a - K63 over the spike-path penalty (seeds.py)")
    drop = sc["K73s6"]["steps"] - sc["K77a"]["steps"]
    N.add("prewarm-g-drop-sds", f"{num(drop / max(jit_sd), 0)}--{num(drop / min(jit_sd), 0)}",
          "results/official-scores.csv: K73s6 - K77a steps over the within-sweep step SDs (seeds.py)")
    # step-10 loss ranges by path (the threshold sits in the empty gap; chosen after seeing the data)
    l10 = [x["L10"] for c, f, _ in GROUPS for x in groups[(c, f)] if x["L10"] is not None]
    good_l = [v for v in l10 if v < INTERMEDIATE[0]]
    spike_l = [v for v in l10 if v >= SPIKE_L10]
    mid_l = [v for v in l10 if INTERMEDIATE[0] <= v < SPIKE_L10]
    N.add("l10-good", f"{num(min(good_l), 2)}--{num(max(good_l), 2)}", SRC)
    N.add("l10-spike", f"{num(min(spike_l), 2)}--{num(max(spike_l), 2)}", SRC)
    N.add("l10-mid", ", ".join(num(v, 2) for v in mid_l), SRC)
    N.add("l10-interm-lo", num(INTERMEDIATE[0], 1), SRC)
    N.add("spike-gap-nats", num(mean(spike_l) - mean(good_l), 1), SRC + ": mean step-10 loss, spike - good")
    # seeds shared between sweeps (the path may be a property of seed + recipe + hardware: not independent draws)
    seedsets = [{x["seed"] for x in groups[(c, f)] if x["L10"] is not None} for c, f, _ in GROUPS]
    shared = (seedsets[0] & seedsets[1]) | (seedsets[0] & seedsets[2]) | (seedsets[1] & seedsets[2])
    distinct = set().union(*seedsets)
    N.add("spike-distinct-seeds", str(len(distinct)), SRC)
    N.add("spike-shared-seeds", str(len(shared)), SRC + ": seeds that appear in more than one sweep")
    # a conservative count: one observation per distinct seed (spike if ANY of its runs took the spike path)
    seed_spike = collections.defaultdict(bool)
    for c, f, _ in GROUPS:
        for x in groups[(c, f)]:
            if x["L10"] is not None:
                seed_spike[x["seed"]] |= x["L10"] >= SPIKE_L10
    ks = sum(seed_spike.values())
    lo_s, hi_s = wilson(ks, len(seed_spike))
    N.add("spike-seed-k", str(ks), SRC)
    N.add("spike-seed-wilson", f"{pct(lo_s, 0)}--{pct(hi_s, 0)}", SRC + ": Wilson over distinct seeds")
    g7 = groups[("C", "g7")]
    N.add("g7-nseeds", str(len(g7)), SRC)
    s73 = next(x for x in g7 if x["seed"] == 73)
    rank = sorted(x["bpb"] for x in g7).index(s73["bpb"]) + 1
    N.add("g7-s73-rank", str(rank), SRC)
    N.add("g7-s73-gap", num(mean([x["bpb"] for x in g7 if x["L10"] is not None and x["L10"] < INTERMEDIATE[0]]) - s73["bpb"], 4),
          SRC + ": good-path mean minus seed 73")
    _figure(groups)
    return {"groups": groups, "jitter_rel": jit_rel}


def _figure(groups) -> None:
    from style import BLUE, GREY, INK_2, ORANGE, SURFACE, TEXTWIDTH, plt, save_pdf
    import numpy as np

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.32})
    g7 = groups[("C", "g7")]
    steps = [0, 1, 2, 3, 4, 10, 20]
    for x in g7:
        if x["L10"] is None:
            continue
        ys = [x["curve"].get(s) for s in steps]
        if any(v is None for v in ys):
            continue
        spike = x["L10"] >= SPIKE_L10
        mid = INTERMEDIATE[0] <= x["L10"] < SPIKE_L10
        a1.plot(steps, ys, color=ORANGE if spike else (GREY if mid else BLUE), lw=1.1, alpha=0.85,
                ls="--" if spike else (":" if mid else "-"), marker="^" if spike else "o", ms=2.5,
                zorder=3 if spike else 2)
    a1.set_xlabel("optimizer step")
    a1.set_ylabel("training loss (nats)")
    a1.set_title("a   Seed sweep of one recipe (chip C)", fontsize=8.5)
    a1.set_xticks(steps)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], color=BLUE, lw=1.4, label="good path"),
                       Line2D([], [], color=ORANGE, lw=1.4, ls="--", label="spike path"),
                       Line2D([], [], color=GREY, lw=1.4, ls=":", label="intermediate")],
              loc="upper right", handlelength=1.5)

    labels = []
    for i, (chip, fam, lab) in enumerate(GROUPS):
        g = groups[(chip, fam)]
        rng = np.random.default_rng(i)
        for x in g:
            if x["L10"] is None:
                col, edge, mk = SURFACE, INK_2, "o"
            elif x["L10"] >= SPIKE_L10:
                col, edge, mk = ORANGE, SURFACE, "^"
            elif x["L10"] >= INTERMEDIATE[0]:
                col, edge, mk = GREY, SURFACE, "D"
            else:
                col, edge, mk = BLUE, SURFACE, "o"
            a2.scatter(i + rng.uniform(-0.12, 0.12), x["bpb"], s=22, marker=mk, color=col, edgecolor=edge,
                       linewidth=0.9, zorder=3)
        labels.append({"g7": "K57, C", "g7lk35": f"LK0.35, {chip}"}[fam])
    a2.set_xticks(range(len(GROUPS)), labels)
    a2.set_xlim(-0.5, len(GROUPS) - 0.5)
    a2.set_ylabel("val_bpb (first 2M public tokens)")
    a2.set_title("b   Final score by path", fontsize=8.5)
    a2.grid(axis="x", visible=False)
    a2.tick_params(axis="x", length=0)
    a2.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.3f"))
    a2.legend(handles=[Line2D([], [], lw=0, marker="o", ms=5, mfc=BLUE, mec=SURFACE, label="good path"),
                       Line2D([], [], lw=0, marker="^", ms=5, mfc=ORANGE, mec=SURFACE, label="spike path"),
                       Line2D([], [], lw=0, marker="D", ms=4, mfc=GREY, mec=SURFACE, label="intermediate"),
                       Line2D([], [], lw=0, marker="o", ms=5, mfc=SURFACE, mec=INK_2, label="no curve logged")],
              loc="upper left", handlelength=1.0)
    save_pdf(fig, FIGURES / "warmup_paths.pdf")
