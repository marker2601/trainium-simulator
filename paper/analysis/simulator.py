"""Simulator validation (paper Section 6 and Appendix B), recomputed from the shipped data with the shipped code.

Inputs: research/sim-data/runs.jsonl, validation-pairs.json, official-uploads.csv; ffsim.quality / steptime / offset.
Outputs: tables/sim_calibration.tex, sim_other.tex, postfit.tex, lopo_family.tex, temporal.tex, coefficients.tex,
features.tex, gpu_parity.tex, dataset.tex, and figures/simulator_pairs.pdf.

Revision 1 (review response): trivial baselines for the quality surrogate (leave-one-out majority sign, the zero
predictor, and a steps + era + chip + seed model with every knob effect switched off), all scored with the same
leave-one-pair-out machinery and a cluster bootstrap; sign agreement on prediction-decisive pairs; the residual SD
without the empirical-Bayes floor; the temporal hold-out split into supported and prior-only pairs; the 15 post-fit
pairs scored here (with a Wilson interval and the zero-predictor baseline) from the per-pair values of
docs/simulator-report-20260929.md section 5; the compute the dataset represents.

Numbers quoted from documents rather than recomputed are marked ``(doc)`` in their source comment in numbers.tex.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import math
import statistics
from typing import Any, Dict, List

import numpy as np

from common import (FIGURES, PAIRS, RUNS, TABLES, UPLOADS, Numbers, intc, load_pairs, mean, milli, num, pct, table,
                    tex_escape, wilson)

NOISE_FAMILIES = {"seed", "chip"}
SIMDOC = "docs/SIMULATOR.md"
REPORT = "docs/simulator-report-20260929.md"
NOISE = 0.0003

# docs/simulator-report-20260929.md section 5: (pair, observed, predicted) for the 15 pairs that finished after the
# release fit. "-0.00000" is recorded as a tiny negative (the report scores it as a wrong-sign tie).
POSTFIT = [
    ("C41 K60 vs C40 K59 (cold rehearsals)", -0.00182, -0.00210),
    ("AS-a s67 (C)", -0.00136, -0.00204),
    ("AS-a s58 (C)", -0.00106, -0.00171),
    ("AS-a s97 (D)", 0.00115, -0.00099),
    ("AS-b 4:5,6,7,8 s73 (D)", -0.00086, -0.00229),
    ("K60 + ALR s73 (C) vs AS-a", 0.00081, 0.00167),
    ("K60 + ALR s67 (C) vs AS-a", 0.00095, 0.00224),
    ("ALR s97 (D) vs LK0.35", -0.00078, 0.00173),
    ("ALR s113 (D) vs LK0.35", -0.00009, 0.00181),
    ("ALR s131 (D) vs LK0.35", 0.00044, 0.00213),
    ("AS-d 5:7,8 s73 (C) vs AS-a", -0.000001, 0.00015),
    ("AS-d 5:7,8 s67 (C) vs AS-a", 0.00033, 0.00039),
    ("AS-d 5:7,8 s73 (D) vs AS-a", -0.00020, 0.00002),
    ("AS-c 6:7,8 s73 (D) vs AS-a", 0.00076, -0.00007),
    ("K60 + softcap 16.5 s73 (C) vs AS-a", 0.00068, -0.00022),
]

FAMILY_NAMES = {
    "leaky_slope": "LeakyReLU slope", "seed": "seed (noise)", "g5_retune": "scalar retunes, K57 base",
    "accum_sched": "batch warm-up schedule", "rehearsal": "cold upload rehearsals", "leaky_form": "LeakyReLU form",
    "key_offset_rad": "key-offset radius", "wd_sched": "weight-decay schedule", "acc_in_graph": "accumulation in graph",
    "qk_gain": "QK gain", "relu2_fn": "ReLU$^2$ hand-written backward", "beta2": "AdamW $\\beta_2$",
    "lr": "learning rate", "mom_peak": "Muon momentum peak", "softcap_asym": "asymmetric soft cap",
    "xsa": "attention-input shift", "zero2_adamw_zero": "ZeRO-2 + AdamW ZeRO", "chip": "chip (noise)",
    "m13_flags": "code-M13 flags", "m14_attn_src": "attention-source reuse", "mtp": "multi-token prediction",
    "rope": "RoPE base", "softcap": "soft cap", "warmup": "LR warm-up",
}


def _family(f: Any) -> str:
    return FAMILY_NAMES.get(str(f), tex_escape(str(f).replace("_", " ")))


def _stats(rows: List[dict]) -> Dict[str, float]:
    n = len(rows)
    dec = [r for r in rows if abs(r["observed"]) >= NOISE]
    pdec = [r for r in rows if abs(r["predicted"]) >= NOISE]
    return {
        "n": n,
        "sign": sum(r["sign_ok"] for r in rows) / n if n else float("nan"),
        "mae": mean(r["abs_error"] for r in rows) if n else float("nan"),
        "rmse": math.sqrt(mean(r["error"] ** 2 for r in rows)) if n else float("nan"),
        "n_dec": len(dec),
        "sign_dec": (sum(r["sign_ok"] for r in dec) / len(dec)) if dec else float("nan"),
        "n_pdec": len(pdec),
        "sign_pdec": (sum(r["sign_ok"] for r in pdec) / len(pdec)) if pdec else float("nan"),
        "cover": (sum(abs(r["observed"] - r["predicted"]) <= 2 * r["pred_sd"] for r in rows) / n) if n else float("nan"),
    }


def _sign_ok(pred: float, obs: float) -> bool:
    return (pred > 0) == (obs > 0) if obs != 0 else abs(pred) < NOISE


def _cluster_bootstrap(rows: List[dict], key: str = "control", reps: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = list(groups)
    signs, maes = [], []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        rs = [r for i in pick for r in groups[keys[i]]]
        signs.append(sum(r["sign_ok"] for r in rs) / len(rs))
        maes.append(mean(r["abs_error"] for r in rs))
    return (float(np.percentile(signs, 2.5)), float(np.percentile(signs, 97.5)),
            float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5)), len(keys))


def _paired_bootstrap(a: List[dict], b: List[dict], reps: int = 4000, seed: int = 1):
    """CI of sign(a) - sign(b) and MAE(a) - MAE(b) over the same pairs, resampling control-run clusters."""
    rng = np.random.default_rng(seed)
    idx = collections.defaultdict(list)
    for i, r in enumerate(a):
        idx[r["control"]].append(i)
    keys = list(idx)
    ds, dm = [], []
    for _ in range(reps):
        pick = [j for k in rng.integers(0, len(keys), len(keys)) for j in idx[keys[k]]]
        ds.append(mean(a[j]["sign_ok"] for j in pick) - mean(b[j]["sign_ok"] for j in pick))
        dm.append(mean(a[j]["abs_error"] for j in pick) - mean(b[j]["abs_error"] for j in pick))
    return (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)),
            float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5)))


@contextlib.contextmanager
def _knobs_off():
    """Switch every knob effect off: prior mean 0 and a vanishing prior SD for every knob feature, bowls included.
    What is left is era + chip + steps (with its prior) + seed: the 'steps-only' baseline."""
    import ffsim.quality as Q
    saved = Q.FEATURES
    Q.FEATURES = tuple(dataclasses.replace(f, prior_sd=1e-9) for f in saved)
    try:
        yield
    finally:
        Q.FEATURES = saved


def _majority_rows(rows: List[dict]) -> List[dict]:
    """Leave-one-out majority sign: predict, for each pair, the more common sign among the OTHER pairs."""
    out = []
    for i, r in enumerate(rows):
        others = [o["observed"] for j, o in enumerate(rows) if j != i]
        up = sum(1 for o in others if o > 0)
        s = 1.0 if up >= len(others) - up else -1.0
        out.append(dict(r, predicted=s * 1e-9, sign_ok=_sign_ok(s, r["observed"])))
    return out


def _zero_rows(rows: List[dict]) -> List[dict]:
    return [dict(r, predicted=0.0, error=-r["observed"], abs_error=abs(r["observed"])) for r in rows]


def _temporal(qm, pairs: List[dict], cutoff: str) -> Dict[str, Any]:
    """Fit only on runs dated before ``cutoff`` (UTC date), predict every pair dated on or after it."""
    from ffsim.quality import FEATURES, knob_features
    late = [r.run_id for r in qm.fit_records if (r.date_utc or "") >= cutoff]
    child = qm.refit_without(late)
    rows = []
    unsupported = 0
    for p in pairs:
        if (p.get("date_utc") or "") < cutoff:
            continue
        t_id, c_id = qm._pair_ids(p)
        t = qm._resolve_arm(t_id, p, "treatment")
        c = qm._resolve_arm(c_id, p, "control")
        if t is None or c is None:
            continue
        tk, _ = qm._arm_knobs(t, p.get("knob_changes"))
        ck, _ = qm._arm_knobs(c, p.get("control_knobs"))
        pred, psd, _ = child.predict_delta(t, c, tk, ck)
        obs = p.get("delta_raw")
        obs = float(t.bpb_2m) - float(c.bpb_2m) if obs is None else float(obs)
        diff = knob_features(tk) - knob_features(ck)
        support = np.any(child._X[:, child._i_knob0:child._i_seed0] != 0.0, axis=0)
        unsup = any(diff[j] != 0.0 and not support[j] for j in range(len(FEATURES)))
        unsupported += int(unsup)
        err = pred - obs
        rows.append({"observed": obs, "predicted": pred, "pred_sd": psd, "error": err, "abs_error": abs(err),
                     "sign_ok": _sign_ok(pred, obs), "family": p.get("family"), "unsupported": unsup})
    st = _stats(rows)
    st.update({"n_fit": child.n, "n_late_runs": len(late), "n_unsupported": unsupported})
    sup = [r for r in rows if not r["unsupported"]]
    uns = [r for r in rows if r["unsupported"]]
    st["sup"] = _stats(sup) if sup else None
    st["uns"] = _stats(uns) if uns else None
    return st


def run(N: Numbers) -> Dict[str, Any]:
    from ffsim.dataset import load_runs
    from ffsim.offset import OffsetModel
    from ffsim.quality import FEATURES, QualityConfig, QualityModel
    from ffsim.steptime import StepTimeModel

    records = load_runs(str(RUNS))
    pairs = load_pairs()
    qm = QualityModel().fit(records, pairs)
    ds = "research/sim-data/runs.jsonl"

    # ---------------------------------------------------------------- dataset
    n_screen = sum(1 for r in records if r.is_screen)
    chips = collections.Counter(r.chip for r in records)
    N.add("ds-n", intc(len(records)), ds)
    N.add("ds-full", intc(len(records) - n_screen), ds)
    N.add("ds-screens", intc(n_screen), ds)
    N.add("ds-loss-curves", intc(sum(1 for r in records if r.loss_curve)), ds)
    N.add("ds-chips", str(len([c for c in chips if c and c != "?"])), ds)
    N.add("ds-pairs", str(len(pairs)), "research/sim-data/validation-pairs.json")
    N.add("ds-pairs-recipe", str(sum(1 for p in pairs if p.get("family") not in NOISE_FAMILIES)), "validation-pairs.json")
    N.add("ds-pairs-noise", str(sum(1 for p in pairs if p.get("family") in NOISE_FAMILIES)), "validation-pairs.json")
    N.add("ds-pair-families", str(len({p.get('family') for p in pairs})), "validation-pairs.json")
    body = []
    for c in sorted(k for k in chips if k):
        rs = [r for r in records if r.chip == c]
        full = [r for r in rs if not r.is_screen]
        body.append(f"{tex_escape(c) if c != '?' else 'unknown'} & {intc(len(rs))} & {intc(len(full))} & "
                    f"{intc(sum(1 for r in rs if r.is_screen))} & {intc(sum(1 for r in full if r.bpb_2m is not None))} & "
                    f"{intc(sum(1 for r in rs if r.loss_curve))} \\\\")
    body.append("\\midrule")
    body.append(f"all & {intc(len(records))} & {intc(len(records) - n_screen)} & {intc(n_screen)} & "
                f"{intc(sum(1 for r in records if not r.is_screen and r.bpb_2m is not None))} & "
                f"{intc(sum(1 for r in records if r.loss_curve))} \\\\")
    table(TABLES / "dataset.tex", body, "@{}lrrrrr@{}",
          "Chip & Records & Full runs & Screens & Full, with bpb & Loss curves",
          "runs.jsonl by chip (simulator.py)")
    # compute represented by the dataset: charged + start-up seconds, the median where a record lacks them
    full = [r for r in records if not r.is_screen]
    scr = [r for r in records if r.is_screen]

    def hours(rs):
        ch = [r.charged_seconds for r in rs if r.charged_seconds]
        su = [r.startup_seconds for r in rs if r.startup_seconds]
        mc, ms = statistics.median(ch), statistics.median(su) if su else 0.0
        return sum((r.charged_seconds or mc) + (r.startup_seconds or ms) for r in rs) / 3600.0, mc, ms

    h_full, mc_full, ms_full = hours(full)
    su_all = [r.startup_seconds for r in records if r.startup_seconds]
    ch_scr = [r.charged_seconds for r in scr if r.charged_seconds]
    h_scr = sum((r.charged_seconds or statistics.median(ch_scr)) + (r.startup_seconds or statistics.median(su_all))
                for r in scr) / 3600.0
    N.add("trn-hours-full", intc(round(h_full, -1)), ds + ": charged + start-up, medians imputed")
    N.add("trn-hours-screens", intc(round(h_scr, -1)), ds)
    N.add("trn-hours-total", intc(round(h_full + h_scr, -1)), ds)
    N.add("trn-charged-median", intc(mc_full), ds)
    N.add("trn-startup-median", intc(statistics.median(su_all)), ds)

    # ---------------------------------------------------------------- quality fit
    N.add("q-nfit", str(qm.n), "ffsim.quality fit on runs.jsonl")
    N.add("q-sigma", num(qm.sigma, 4), "ffsim.quality fit")
    q_nofloor = QualityModel(QualityConfig(sigma_floor=1e-7)).fit(records, pairs)
    N.add("q-sigma-nofloor", num(q_nofloor.sigma, 5), "ffsim.quality fit with sigma_floor ~0")
    N.add("q-edf", num(qm.edf, 1), "ffsim.quality fit")
    N.add("q-ncols", str(len(qm.columns)), "ffsim.quality fit")
    N.add("q-nfeatures", str(len(FEATURES)), "ffsim.quality FEATURES")
    N.add("q-nbase-features", str(sum(1 for f in FEATURES if not f.bowl)), "ffsim.quality FEATURES")
    N.add("q-nbowl-features", str(sum(1 for f in FEATURES if f.bowl)), "ffsim.quality FEATURES")
    N.add("q-neras", str(len(qm.eras)), "ffsim.quality fit")
    N.add("q-nseeds", str(len(qm.seeds)), "ffsim.quality fit")
    N.add("q-slope", num(qm.steps_slope(), 4), "ffsim.quality fit, d bpb / d ln steps at 2300")
    N.add("q-slope-pct", num(-qm.steps_slope() * math.log(1.01), 5), "ffsim.quality fit, per 1% steps")
    N.add("q-dropped-noknobs", str(qm.dropped.get("no_effective_knobs", 0)), "ffsim.quality fit")
    N.add("q-dropped-nocode", str(qm.dropped.get("no_code_version", 0)), "ffsim.quality fit")

    lopo = qm.pair_validation(leave_pair_out=True)
    lokvo = qm.pair_validation(leave_value_out=True)
    ins = qm.pair_validation(leave_pair_out=False)
    with _knobs_off():
        qs = QualityModel(QualityConfig(bowl_prior_mean=0.0)).fit(records, pairs)
        steps_only = qs.pair_validation(leave_pair_out=True)
    res = {}
    for tag, rep in (("lopo", lopo), ("lokvo", lokvo), ("ins", ins), ("steps", steps_only)):
        rows = rep["rows"]
        rec = [r for r in rows if r["family"] not in NOISE_FAMILIES]
        res[tag] = (_stats(rows), _stats(rec))
        for sub, st in (("all", res[tag][0]), ("rec", res[tag][1])):
            N.add(f"{tag}-{sub}-n", str(st["n"]), "pair_validation")
            N.add(f"{tag}-{sub}-sign", pct(st["sign"], 1), "pair_validation")
            N.add(f"{tag}-{sub}-mae", num(st["mae"], 5), "pair_validation")
            N.add(f"{tag}-{sub}-rmse", num(st["rmse"], 5), "pair_validation")
            N.add(f"{tag}-{sub}-ndec", str(st["n_dec"]), "pair_validation")
            N.add(f"{tag}-{sub}-signdec", pct(st["sign_dec"], 1), "pair_validation")
            N.add(f"{tag}-{sub}-npdec", str(st["n_pdec"]), "pair_validation")
            N.add(f"{tag}-{sub}-signpdec", pct(st["sign_pdec"], 1), "pair_validation")
            N.add(f"{tag}-{sub}-cover", pct(st["cover"], 0), "pair_validation")
    lo, hi, mlo, mhi, ncl = _cluster_bootstrap(lopo["rows"])
    N.add("lopo-boot-lo", pct(lo, 0), "cluster bootstrap over control runs, 4000 draws, rng 0")
    N.add("lopo-boot-hi", pct(hi, 0), "cluster bootstrap")
    N.add("lopo-boot-mae-lo", num(mlo, 5), "cluster bootstrap")
    N.add("lopo-boot-mae-hi", num(mhi, 5), "cluster bootstrap")
    N.add("lopo-boot-clusters", str(ncl), "cluster bootstrap")
    N.add("pair-noise-floor", num(math.sqrt(2) * qm.sigma * math.sqrt(2 / math.pi), 5),
          "mean |N(0, 2 sigma^2)| at the fitted sigma")
    N.add("pair-noise-floor-nofloor", num(math.sqrt(2) * q_nofloor.sigma * math.sqrt(2 / math.pi), 5),
          "mean |N(0, 2 sigma^2)| at the unfloored sigma")

    # ---------------------------------------------------------------- baselines (same pairs, same LOPO protocol)
    def rec_only(rows):
        return [r for r in rows if r["family"] not in NOISE_FAMILIES]

    base_rows = {"full": lopo["rows"], "steps": steps_only["rows"], "major": _majority_rows(lopo["rows"]),
                 "zero": _zero_rows(lopo["rows"])}
    assert [r["name"] for r in base_rows["full"]] == [r["name"] for r in base_rows["steps"]]
    boot = {}
    for key in ("full", "steps", "major", "zero"):
        for sub, rows in (("all", base_rows[key]), ("rec", rec_only(base_rows[key]))):
            b = _cluster_bootstrap(rows)
            boot[(key, sub)] = b
            st = _stats(rows)
            N.add(f"base-{key}-{sub}-sign", pct(st["sign"], 1), "baseline, LOPO pairs")
            N.add(f"base-{key}-{sub}-mae", num(st["mae"], 5), "baseline, LOPO pairs")
            N.add(f"base-{key}-{sub}-signci", f"{pct(b[0], 0)}--{pct(b[1], 0)}", "cluster bootstrap")
    dlo, dhi, dmlo, dmhi = _paired_bootstrap(rec_only(base_rows["full"]), rec_only(base_rows["steps"]))
    N.add("base-diff-sign-ci", f"{num(100 * dlo, 0, sign=True)} to {num(100 * dhi, 0, sign=True)} points",
          "paired cluster bootstrap, recipe pairs, full - steps-only")
    N.add("base-diff-mae-ci", f"{milli(dmlo, 2, sign=True)} to {milli(dmhi, 2, sign=True)}",
          "paired cluster bootstrap, recipe pairs, full - steps-only, 1e-3 bpb")

    # by family (appendix)
    fam_l = collections.defaultdict(list)
    fam_v = collections.defaultdict(list)
    for r in lopo["rows"]:
        fam_l[r["family"]].append(r)
    for r in lokvo["rows"]:
        fam_v[r["family"]].append(r)
    body = []
    for fam in sorted(fam_l, key=lambda f: (-len(fam_l[f]), str(f))):
        a, b = _stats(fam_l[fam]), _stats(fam_v[fam])
        body.append(f"{_family(fam)} & {a['n']} & {milli(mean(r['observed'] for r in fam_l[fam]), 2, sign=True)} & "
                    f"{a['sign']:.0%} & {milli(a['mae'], 2)} & {b['sign']:.0%} & {milli(b['mae'], 2)} \\\\"
                    .replace("%", "\\%"))
    table(TABLES / "lopo_family.tex", body, "@{}p{4.4cm}rrrrrr@{}",
          "Pair family & $n$ & Mean $\\Delta$ & LOPO sign & LOPO MAE & LOKVO sign & LOKVO MAE",
          "per-family pair validation, 1e-3 bpb (simulator.py)", size="\\footnotesize")

    # ---------------------------------------------------------------- temporal hold-out
    tbody = []
    for cutoff, key in (("2026-09-28", "a"), ("2026-09-29", "b")):
        st = _temporal(qm, pairs, cutoff)
        d = cutoff[-2:]
        N.add(f"temp-{key}-cutoff", f"{int(d)} Sep", "temporal hold-out")
        N.add(f"temp-{key}-nfit", str(st["n_fit"]), "temporal hold-out")
        N.add(f"temp-{key}-n", str(st["n"]), "temporal hold-out")
        N.add(f"temp-{key}-sign", pct(st["sign"], 0), "temporal hold-out")
        N.add(f"temp-{key}-mae", num(st["mae"], 5), "temporal hold-out")
        N.add(f"temp-{key}-signdec", pct(st["sign_dec"], 0), "temporal hold-out")
        N.add(f"temp-{key}-ndec", str(st["n_dec"]), "temporal hold-out")
        N.add(f"temp-{key}-nunsup", str(st["n_unsupported"]), "temporal hold-out")
        s2, s3 = st["sup"], st["uns"]
        if s2:
            N.add(f"temp-{key}-sup-sign", pct(s2["sign"], 0), "temporal hold-out")
            N.add(f"temp-{key}-sup-mae", num(s2["mae"], 5), "temporal hold-out")
            N.add(f"temp-{key}-sup-n", str(s2["n"]), "temporal hold-out")
        if s3:
            N.add(f"temp-{key}-uns-sign", pct(s3["sign"], 0), "temporal hold-out")
            N.add(f"temp-{key}-uns-n", str(s3["n"]), "temporal hold-out")
        sup_cell = f"{s2['n']}: {pct(s2['sign'], 0)}" if s2 else "--"
        uns_cell = f"{s3['n']}: {pct(s3['sign'], 0)}" if s3 else "--"
        tbody.append(f"runs before {int(d)} Sep & {st['n_fit']} & {st['n']} & {pct(st['sign'], 0)} & "
                     f"{milli(st['mae'], 2)} & {sup_cell} & {uns_cell} \\\\")
    table(TABLES / "temporal.tex", tbody, "@{}lrrrrrr@{}",
          "Fit on & Runs & Pairs & Sign & MAE ($10^{-3}$) & Supported & Prior-only",
          "temporal hold-out (simulator.py); supported / prior-only cells are n: sign agreement")

    # ---------------------------------------------------------------- post-fit pairs (new mechanisms)
    pf = [{"observed": o, "predicted": p, "error": p - o, "abs_error": abs(p - o), "sign_ok": _sign_ok(p, o)}
          for _, o, p in POSTFIT]
    k = sum(r["sign_ok"] for r in pf)
    wl, wh = wilson(k, len(pf))
    N.add("postfit-n", str(len(pf)), f"{REPORT} section 5 (doc), scored in simulator.py")
    N.add("postfit-k", str(k), "simulator.py")
    N.add("postfit-sign", pct(k / len(pf), 0), "simulator.py")
    N.add("postfit-wilson", f"{pct(wl, 0)}--{pct(wh, 0)}", "Wilson 95% interval")
    N.add("postfit-mae", num(mean(r["abs_error"] for r in pf), 5), "simulator.py")
    N.add("postfit-zero-mae", num(mean(abs(r["observed"]) for r in pf), 5), "zero predictor on the 15 pairs")
    dec = [r for r in pf if abs(r["observed"]) >= NOISE]
    N.add("postfit-ndec", str(len(dec)), "simulator.py")
    N.add("postfit-signdec", pct(sum(r["sign_ok"] for r in dec) / len(dec), 0), "simulator.py")
    body = [f"{tex_escape(lab)} & {milli(o, 2, sign=True)} & {milli(p, 2, sign=True)} & "
            f"{'yes' if r['sign_ok'] else 'no'} \\\\" for (lab, o, p), r in zip(POSTFIT, pf)]
    body.append("\\midrule")
    body.append(f"sign agreement {k}/{len(pf)} (Wilson 95\\%: {pct(wl, 0)}--{pct(wh, 0)}) & & "
                f"MAE {milli(mean(r['abs_error'] for r in pf), 2)} & \\\\")
    table(TABLES / "postfit.tex", body, "@{}p{6.4cm}rrc@{}",
          "Pair (report label) & Observed & Predicted & Sign", "15 post-fit pairs, 1e-3 bpb (simulator.py)",
          size="\\footnotesize")

    # ---------------------------------------------------------------- step time, offset model, anchors
    stm = StepTimeModel().fit(records)
    loo = stm.loo_report("k4")
    N.add("st-lolo-reach", num(loo["mean_abs_pct_reachable_by_run"], 3) + "\\%", "StepTimeModel.loo_report('k4')")
    N.add("st-lolo-reach-lineage", num(loo["mean_abs_pct_reachable_by_lineage"], 2) + "\\%", "loo_report")
    N.add("st-lolo-all", num(loo["mean_abs_pct_by_run"], 2) + "\\%", "loo_report")
    N.add("st-lolo-nreach", str(loo["n_reachable"]), "loo_report")
    N.add("st-lolo-nruns", str(loo["n_runs"]), "loo_report")
    N.add("st-lolo-nnovel", str(loo["n_novel"]), "loo_report")
    N.add("st-lolo-nversions", str(loo["n_versions"]), "loo_report")
    N.add("st-lolo-max", num(loo["max_abs_pct"], 0) + "\\%", "loo_report")
    N.add("st-lolo-rmse-reach", num(loo["rmse_pct_reachable"], 2) + "\\%", "loo_report")

    om = OffsetModel().fit(str(UPLOADS))
    N.add("om-mean", num(om.mean, 5, sign=True), "ffsim.offset on official-uploads.csv")
    N.add("om-sd", num(om.sd, 5), "ffsim.offset")
    N.add("om-n", str(om.n), "ffsim.offset")
    N.add("om-psd", num(om.predictive_sd, 5), "ffsim.offset")
    lo_sd, hi_sd = om.sd_ci95
    N.add("om-sd-lo", num(lo_sd, 5), "ffsim.offset")
    N.add("om-sd-hi", num(hi_sd, 5), "ffsim.offset")

    N.add("search-n", "282", f"{SIMDOC} (doc)")
    N.add("walk-phase-err", "6--10", f"{SIMDOC} (doc)")
    N.add("walk-total-err", "0.3\\%", f"{SIMDOC} (doc)")
    N.add("reserve-s", "5.5", "research/sim-data/simulate-validation.md section 2 (doc)")
    N.add("reserve-n", "136", "research/sim-data/simulate-validation.md section 2 (doc)")
    N.add("overhead", "0.57\\%", "ffsim/steptime.py MEDIAN_TO_MEAN_OVERHEAD")

    body = [
        f"Step time, k4 phase, leave one code lineage out: runs whose mechanisms another lineage ran & "
        f"{loo['n_reachable']} runs & MAE {num(loo['mean_abs_pct_reachable_by_run'], 2)}\\% & target 0.5\\% \\\\",
        f"\\quad all held-out runs, including novel mechanisms & {loo['n_runs']} runs & "
        f"MAE {num(loo['mean_abs_pct_by_run'], 2)}\\% & up to {num(loo['max_abs_pct'], 0)}\\% \\\\",
        "\\midrule",
        "End to end, \\upl{K59}: steps / rehearsal / official$^{\\dagger}$ & 1 upload & 2{,}369 / 0.96082 / 0.9674 & "
        "2{,}357 / 0.96092 / 0.9671 \\\\",
        "End to end, \\upl{K60}: steps / rehearsal / official$^{\\dagger}$ & 1 upload & 2{,}365 / 0.95888 / 0.9654 & "
        "2{,}361 / 0.95910 / 0.9655 \\\\",
        "\\midrule",
        f"Offset model (chip-C uploads): mean and SD & {om.n} uploads & {num(om.mean, 5, sign=True)}, {num(om.sd, 5)} & "
        f"SD 95\\% CI {num(lo_sd, 5)}--{num(hi_sd, 5)} \\\\",
    ]
    table(TABLES / "sim_other.tex", body, "@{}p{6.2cm}rll@{}", "Check & Data & Predicted & Measured / note",
          "step time, end-to-end anchors and the offset model (simulator.py); anchors from docs/SIMULATOR.md",
          size="\\footnotesize")

    # ---------------------------------------------------------------- the quality calibration table (main text)
    lp, lr = res["lopo"]
    vp, vr = res["lokvo"]
    ip, ir = res["ins"]

    def row(label, n, st, ci=""):
        return f"{label} & {n} & {pct(st['sign'], 1)} & {ci} & {milli(st['mae'], 2)} \\\\"

    sa, sr = res["steps"]
    za, zr = _stats(base_rows["zero"]), _stats(rec_only(base_rows["zero"]))
    ma, mr = _stats(base_rows["major"]), _stats(rec_only(base_rows["major"]))
    ci = lambda key, sub: f"{pct(boot[(key, sub)][0], 0)}--{pct(boot[(key, sub)][1], 0)}"  # noqa: E731
    body = [
        "\\multicolumn{5}{@{}l}{\\emph{Full surrogate}} \\\\",
        row("\\quad in-sample (reference only)", ip["n"], ip),
        row("\\quad leave one pair out (LOPO), all pairs", lp["n"], lp, ci("full", "all")),
        row("\\quad LOPO, recipe pairs only", lr["n"], lr, ci("full", "rec")),
        f"\\quad LOPO, recipe pairs with $|\\hat\\Delta|\\ge0.3$ (decision-relevant) & {lr['n_pdec']} & "
        f"{pct(lr['sign_pdec'], 1)} & & \\\\",
        f"\\quad LOPO, recipe pairs with $|\\Delta|\\ge0.3$ (outcome-decisive) & {lr['n_dec']} & "
        f"{pct(lr['sign_dec'], 1)} & & \\\\",
        row("\\quad leave one knob value out (LOKVO), recipe pairs", vr["n"], vr),
        "\\multicolumn{5}{@{}l}{\\emph{Baselines, same pairs and protocol}} \\\\",
        row("\\quad steps + era + chip + seed, knob effects off (LOPO), recipe", sr["n"], sr, ci("steps", "rec")),
        f"\\quad \\quad recipe pairs with $|\\hat\\Delta|\\ge0.3$ & {sr['n_pdec']} & {pct(sr['sign_pdec'], 1)} & & \\\\",
        row("\\quad leave-one-out majority sign, recipe pairs", mr["n"], mr, ci("major", "rec")).rsplit("&", 1)[0] + "& -- \\\\",
        f"\\quad zero predictor ($\\hat\\Delta=0$), recipe pairs & {zr['n']} & -- & & {milli(zr['mae'], 2)} \\\\",
        "\\multicolumn{5}{@{}l}{\\emph{Forward in time and new mechanisms}} \\\\",
        f"\\quad temporal hold-out, fit before 28 Sep & {N.values['temp-a-n']} & {N.values['temp-a-sign']} & & "
        f"{milli(float(N.values['temp-a-mae']), 2)} \\\\",
        f"\\quad temporal hold-out, fit before 29 Sep & {N.values['temp-b-n']} & {N.values['temp-b-sign']} & & "
        f"{milli(float(N.values['temp-b-mae']), 2)} \\\\",
        f"\\quad pairs finished after the release fit$^{{\\dagger}}$ & {len(pf)} & {N.values['postfit-sign']} & "
        f"{N.values['postfit-wilson']} & {milli(mean(r['abs_error'] for r in pf), 2)} \\\\",
        f"\\quad \\quad zero predictor on the same pairs & {len(pf)} & -- & & "
        f"{milli(mean(abs(r['observed']) for r in pf), 2)} \\\\",
    ]
    table(TABLES / "sim_calibration.tex", body, "@{}p{7.6cm}rrcr@{}",
          "Check & Pairs & Sign & 95\\% CI & MAE ($10^{-3}$)",
          "quality surrogate calibration with baselines (simulator.py)", size="\\footnotesize")

    # ---------------------------------------------------------------- coefficients (appendix)
    coefs = [c for c in qm.coefficients() if c["name"].startswith("knob:") and not c["pinned"]]
    desc = {f.name: f for f in FEATURES}
    ranked = sorted(coefs, key=lambda c: -abs(c["estimate"]) / max(c["sd"], 1e-12))
    N.add("coef-ntop", "16", "simulator.py")
    body = []
    fsup = qm.feature_support()
    for c in ranked[:16]:
        fname = c["name"][5:]
        f = desc[fname]
        sup = fsup.get(fname, {})
        eras = ", ".join(sorted(sup.get("eras_nonzero", {})))
        knobs = ", ".join(k.replace("FF_", "") for k in f.knobs)
        body.append(f"\\texttt{{{tex_escape(fname)}}} & {tex_escape(knobs)} & "
                    f"{milli(c['estimate'], 2, sign=True)} & {milli(c['sd'], 2)} & {milli(c['prior_sd'], 1)} & "
                    f"{sup.get('n_nonzero', '')} & {tex_escape(eras)} \\\\")
    table(TABLES / "coefficients.tex", body, "@{}lp{3.6cm}rrrrp{2.6cm}@{}",
          "Feature & Knob(s) & Est. & Post.\\ SD & Prior SD & Runs $\\neq0$ & Code eras",
          "the 16 knob coefficients furthest from zero in posterior-SD units, x 1e-3 bpb per scaled unit",
          size="\\footnotesize", tabcolsep="4pt")

    body = []
    for f in FEATURES:
        if f.bowl and f.name != "leaky_sq":
            continue
        body.append(f"\\texttt{{{tex_escape(f.name)}}} & {tex_escape(f.transform)} & {f.scale:.3g} \\\\")
    table(TABLES / "features.tex", body, "@{}lp{10.6cm}r@{}", "Feature & Encoding (centred on the K59 recipe) & Scale",
          "ffsim.quality FEATURES (simulator.py)", size="\\scriptsize")

    # ---------------------------------------------------------------- GPU proxy parity (docs/SIMULATOR.md route 2)
    gpu = [("Attention-source reuse (5:6,7,8)", -0.00187, -0.00181, "chip C pair", True),
           ("LeakyReLU(0.35)$^2$ vs ReLU$^2$", -0.00114, -0.00105, "chip C pair", True),
           ("Cooldown 0.7", -0.00012, (0.00009 - 0.00037) / 2, "mean of 2 chip pairs of opposite sign", False),
           ("Split AdamW cooldown 0.4", 0.00045, 0.0006, "official upload delta, other steps", False)]
    body = []
    diffs, diffs_inf = [], []
    for lab, g, c, src, informative in gpu:
        diffs.append(abs(g - c))
        if informative:
            diffs_inf.append(abs(g - c))
        body.append(f"{lab}{'' if informative else '$^{*}$'} & {milli(g, 2, sign=True)} & {milli(c, 2, sign=True)} & "
                    f"{milli(g - c, 2, sign=True)} & {src} \\\\")
    body.append("\\midrule")
    body.append("\\upl{K60} absolute val\\_bpb at 2{,}314 steps (bpb) & 0.959952 & 0.959948 & "
                f"{milli(0.959952 - 0.959948, 3, sign=True)} & one run each \\\\")
    table(TABLES / "gpu_parity.tex", body, "@{}p{4.6cm}rrrp{3.6cm}@{}",
          "Comparison (seed 73, equal steps) & GPU & Trainium & Diff. & Trainium evidence",
          f"GPU proxy vs chip, 1e-3 bpb except the last row; values from {SIMDOC} route 2", size="\\footnotesize")
    N.add("gpu-mae", num(mean(diffs), 5), f"{SIMDOC} route 2, computed")
    N.add("gpu-maxdiff", num(max(diffs), 5), f"{SIMDOC} route 2, computed")
    N.add("gpu-maxdiff-inf", num(max(diffs_inf), 5), f"{SIMDOC} route 2, the two equal-step chip pairs")
    N.add("gpu-n", str(len(gpu)), SIMDOC)
    N.add("gpu-n-inf", str(len(diffs_inf)), SIMDOC)
    N.add("gpu-patches", "32", f"{SIMDOC} (doc)")
    N.add("gpu-box-diff", "0.00056", f"{SIMDOC} (doc)")
    N.add("gpu-tps-det", "76k", f"{SIMDOC} (doc)")
    N.add("gpu-tps-default", "104k", f"{SIMDOC} (doc)")
    N.add("gpu-hours", "1.35", f"{SIMDOC} (doc)")
    N.add("gpu-cost-run", "2.5", f"{SIMDOC} (doc)")
    N.add("gpu-cost-total", "185", f"{SIMDOC} (doc)")

    _figure(lopo["rows"])
    return {"qm": qm, "lopo": lopo}


def _figure(rows: List[dict]) -> None:
    from style import BLUE, GREY, INK_2, ORANGE, SURFACE, TEXTWIDTH, TIE_WASH, plt, save_pdf
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, TEXTWIDTH * 0.5))
    lim = 4.0
    ax.fill_between([0, lim], [-lim, -lim], [0, 0], color=TIE_WASH, lw=0, zorder=0)
    ax.fill_between([-lim, 0], [0, 0], [lim, lim], color=TIE_WASH, lw=0, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, zorder=1)
    for r in rows:
        noise = r["family"] in NOISE_FAMILIES
        x, y = max(-lim, min(lim, 1e3 * r["observed"])), max(-lim, min(lim, 1e3 * r["predicted"]))
        ax.scatter([x], [y], s=16 if not noise else 20, marker="s" if noise else "o",
                   color=ORANGE if noise else BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("measured $\\Delta$ ($10^{-3}$ bpb)")
    ax.set_ylabel("predicted $\\Delta$, pair held out ($10^{-3}$)")
    ax.set_aspect("equal")
    ax.legend(handles=[Line2D([], [], lw=0, marker="o", ms=4.5, mfc=BLUE, mec=SURFACE, label="recipe pair"),
                       Line2D([], [], lw=0, marker="s", ms=4.5, mfc=ORANGE, mec=SURFACE, label="seed / chip pair"),
                       Line2D([], [], color=GREY, lw=0.8, label="perfect prediction")],
              loc="upper left", handlelength=1.2)
    ax.text(lim * 0.95, -lim * 0.95, "wrong sign", ha="right", va="bottom", fontsize=7, color=INK_2)
    ax.text(-lim * 0.95, lim * 0.45, "wrong sign", ha="left", va="top", fontsize=7, color=INK_2)
    save_pdf(fig, FIGURES / "simulator_pairs.pdf")
