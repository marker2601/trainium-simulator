"""Stress test of the community-contribution refit guard (paper Section 8).

Synthetic contributed records are pushed through the real pipeline in memory (``ffsim.contrib.validate_record`` ->
``fit_quality_safely`` -> ``model_guard``); nothing is written to the contribution file. Scenarios:

* honest: runs on a published base recipe with knob values the data has seen, scored as the built-in model's
  prediction plus a hardware bias and N(0, 0.0004) noise (the size of seed noise), from three contributors;
* outlier: one honest-looking record whose score is 0.01 bpb worse than predicted;
* attack-k: the coordinated slope attack of tests/test_contrib.py with k sock-puppet records (half claim cooldown
  0.3 is very good, half claim 0.8 is very bad).

Output: tables/guard.tex and the numbers it prints.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from common import PAIRS, RUNS, TABLES, UPLOADS, Numbers, milli, num, table

BASE = {"schema_version": "1", "hardware": "trn2.3xlarge", "time_budget_s": 1800, "consent": True}


def _honest(q, base: str, n: int, rng, bias: float, start: int = 0) -> List[dict]:
    import ffsim.contrib as C
    cv, knobs = C.base_recipe(base)
    choices = [{"FF_COOLDOWN_FRAC": "0.65"}, {"FF_COOLDOWN_FRAC": "0.6"}, {"FF_QK_GAIN": "1.6"}, {},
               {"FF_WD": "0.03"}, {"FF_LEAKY_RELU2": "0.25"}, {"FF_MOM_PEAK": "0.97"}, {"FF_ROPE_BASE": "300000"}]
    out = []
    for i in range(n):
        change = choices[(start + i) % len(choices)]
        seed = [73, 58, 67][i % 3]
        steps = int(2360 + rng.integers(-12, 13))
        kk = dict(knobs)
        kk.update(change)
        mean, _ = q.predict(kk, steps, cv, "C", seed)
        if base in C.BASE_ANCHORS:
            bpb0, st0, sd0 = C.BASE_ANCHORS[base]
            mean += bpb0 - q.predict(knobs, st0, cv, "C", sd0)[0]
        rec = dict(copy.deepcopy(BASE), base_recipe=base, recipe=change, seed=seed, steps=steps,
                   chip_val_bpb=round(float(mean + bias + rng.normal(0.0, 0.0004)), 6),
                   contributor=f"member{i % 3}")
        out.append(rec)
    return out


def _attack(k: int) -> List[dict]:
    half = k // 2
    plan = [(0.3, 0.950)] * half + [(0.8, 0.975)] * (k - half)
    return [dict(copy.deepcopy(BASE), base_recipe="K60", recipe={"FF_COOLDOWN_FRAC": cd}, seed=[73, 58, 67][i % 3],
                 chip_val_bpb=round(bpb + 0.0001 * i, 6), steps=2380 + i, contributor=f"sock{i}")
            for i, (cd, bpb) in enumerate(plan)]


def _evaluate(records: List[dict], ctx, q_b) -> Dict[str, object]:
    import ffsim.contrib as C
    rows = []
    flags = 0
    hashes: set = set()
    for raw in records:
        res = C.validate_record(raw, quality=q_b, existing_hashes=hashes, builtin=ctx.builtin,
                                offset_mean=ctx.offset_mean, contributor_counts=C.contributor_counts(rows))
        if not res.ok:
            continue
        hashes.add(res.content_hash)
        flags += int("outlier" in res.flags)
        rows.append((res.run_record, {"content_hash": res.content_hash, "submission": res.record}))
    ctx2 = C.Context(ctx.builtin, rows, ctx.pairs, ctx.offset_mean)
    q_full, _ = C.fit_quality_safely(ctx.builtin, C.contrib_fit_records(rows), ctx.pairs)
    g = C.model_guard(ctx2, q_full, q_b)
    return {"n": len(rows), "outliers": flags,
            "d_insample": g["builtin_insample_mae_after"] - g["builtin_insample_mae_before"],
            "d_pairs": g["pair_mae_after"] - g["pair_mae_before"],
            "shift": g["base_prediction_shift"], "ok": g["ok"]}


def run(N: Numbers) -> Dict[str, object]:
    import ffsim.contrib as C
    ctx = C.load_context(RUNS, RUNS.parent / "contrib" / "__no_such_file__.jsonl", PAIRS, UPLOADS)
    q_b = ctx.quality()
    rng = np.random.default_rng(0)
    scenarios = [
        ("honest, K60 base, 8 runs, bias +0.0015", _honest(q_b, "K60", 8, rng, 0.0015), "honest-a"),
        ("honest, K82s4 base, 8 runs, bias $-$0.0010", _honest(q_b, "K82s4", 8, rng, -0.0010, start=3), "honest-b"),
        ("honest, both bases, 24 runs, bias +0.0015", _honest(q_b, "K60", 12, rng, 0.0015) +
         _honest(q_b, "K82s4", 12, rng, 0.0015, start=5), "honest-c"),
        ("one outlier (+0.01 bpb)", [dict(r, chip_val_bpb=round(r["chip_val_bpb"] + 0.01, 6))
                                     for r in _honest(q_b, "K60", 1, rng, 0.0)], "outlier"),
        ("slope attack, 2 sock puppets", _attack(2), "attack-two"),
        ("slope attack, 4 sock puppets", _attack(4), "attack-four"),
        ("slope attack, 8 sock puppets", _attack(8), "attack-eight"),
    ]
    body = []
    for label, recs, key in scenarios:
        r = _evaluate(recs, ctx, q_b)
        verdict = "accepted" if r["ok"] else "\\textbf{refused}"
        body.append(f"{label} & {r['n']} & {r['outliers']} & {milli(r['d_insample'], 3, sign=True)} & "
                    f"{milli(r['d_pairs'], 3, sign=True)} & {milli(r['shift'], 2)} & {verdict} \\\\")
        N.add(f"guard-{key}-shift", num(r["shift"], 4), "contrib_guard.py")
        N.add(f"guard-{key}-dpairs", num(r["d_pairs"], 6, sign=True), "contrib_guard.py")
        N.add(f"guard-{key}-ok", "accepted" if r["ok"] else "refused", "contrib_guard.py")
        N.add(f"guard-{key}-outliers", str(r["outliers"]), "contrib_guard.py")
    table(TABLES / "guard.tex", body, "@{}p{4.6cm}rrrrrl@{}",
          "Scenario & Records & Flagged & $\\Delta$ in-sample & $\\Delta$ pairs & Base shift & Guard",
          "refit-guard stress test, all deltas x 1e-3 bpb (contrib_guard.py)", size="\\footnotesize")
    N.add("guard-max-worse", num(C.GUARD_MAX_WORSE, 4), "ffsim/contrib.py GUARD_MAX_WORSE")
    N.add("guard-max-shift", num(C.GUARD_MAX_BASE_SHIFT, 3), "ffsim/contrib.py GUARD_MAX_BASE_SHIFT")
    N.add("guard-cap", str(C.MAX_FIT_PER_CONTRIBUTOR), "ffsim/contrib.py MAX_FIT_PER_CONTRIBUTOR")
    N.add("guard-outlier-z", num(C.OUTLIER_Z, 0), "ffsim/contrib.py OUTLIER_Z")
    N.add("guard-min-steps", str(C.MIN_FIT_STEPS), "ffsim/contrib.py MIN_FIT_STEPS")
    N.add("guard-anchor", num(C.BASE_ANCHORS["K82s4"][0], 6), "ffsim/contrib.py BASE_ANCHORS")
    return {}
