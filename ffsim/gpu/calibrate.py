"""Calibrate the GPU proxy against known Trainium pairs, then turn a GPU delta into
P(Trainium delta < 0).

Inputs: results.jsonl (train_gpu.py rows, possibly partial), the fleet manifest, the gate
config (its ``pairs`` section maps GPU arms to Trainium pair ids) and
research/sim-data/validation-pairs.json.

Per row: GPU delta = treatment - control at matched (seed, steps); Trainium delta = the
pair's raw equal-time delta corrected to equal steps,
``delta_eq = delta_raw + k * ln(steps_T / steps_C)`` with k = d bpb / d ln steps = 0.057 at
~2300 steps (CONTRACT: 1% of steps ~ 0.00057 bpb), carried over the range 0.045-0.070.

THE RULE IS FROZEN IN THIS FILE (section "the frozen rule" below).  It is the one rule:
PROTOCOL.md 3.3 and RUNBOOK.md quote these constants, they do not define them, and nothing in a
config, in results.jsonl or on the command line can change a threshold, a pair's role or the
set of gated rows after results exist.  ``--freeze`` writes the resolved rule (pair list, seeds,
Trainium targets, thresholds, digest) at plan time; ``--rule`` makes a verdict refuse to decide
when that file no longer matches the rule recomputed from the config and the Trainium data.

Statistics (numpy + stdlib only): sign agreement on the frozen decisive rows, Spearman rank
correlation, a scale factor delta_trn ~ a * delta_gpu (least squares through the origin,
bootstrap CI), per-family offsets, and the three noise terms of PROTOCOL.md section 3.3:
GPU seed noise (pooled paired-delta sd, base-arm run sd, repeat nulls), Trainium seed noise
(0.0006 per run) and the calibration error (residual after the fit).

Decision: gates 0-5 of PROTOCOL.md 3.3, all of them, on the complete frozen row set.  Status is
``pending`` until every frozen row, both nulls and both absolute-level runs have data; there is
no early accept on a subset.  Then for a new candidate: P(Trainium delta < 0) from its GPU
deltas, the fitted a, the family offset and the calibrated noise, plus the cost of the seeds a
resolved verdict needs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------------- the frozen rule
# Every number here is the one PROTOCOL.md section 3.3 states.  Change the rule here or nowhere.
RULE_VERSION = "gate-k60 / PROTOCOL.md 3.3 / 2026-09-29"
RULE_MODE = "a: every GPU run at equal steps; the Trainium target is delta_eq"

STEP_ELASTICITY_K = 0.057          # d bpb / d ln steps at ~2300 steps (CONTRACT.md)
STEP_ELASTICITY_RANGE = (0.045, 0.070)
DECISIVE_ABS = 0.0003              # gate 3: |delta_eq| of a decisive row, for every k in the range
SIGN_AGREEMENT_MIN = 0.80          # gate 3: agreeing decisive rows / decisive rows
SCALE_BAND = (0.5, 2.0)            # gate 4: a of delta_trn ~ a * delta_gpu over the decisive rows
NULL_MAX_ABS = 0.0006              # gate 1: worst run-to-run null |delta|
TIE_BAND_MIN = 0.0006              # gate 5: band = max(TIE_BAND_MIN, TIE_BAND_NULL_MULT x null rms)
TIE_BAND_NULL_MULT = 3.0
TIE_MAX_OUTSIDE = 2                # gate 5: "at least 11 of 13" = at most 2 null/tie rows outside the band
EVAL_TOKENS = 2_097_152            # gate 0
EVAL_BYTES = 8_387_440
ABS_LEVEL_TOL = 0.01               # gate 2
ABS_LEVEL_STRUCTURAL = 0.03        # gate 2: beyond this the setup is structurally wrong
ABS_LEVEL_SEED = 73
ABS_LEVEL_RUNS = {                 # gate 2: the config's dedicated runs (_absolute_level), each at its reference's own schedule
    "abs_asa": {"steps": 2361, "ref": 0.95910, "what": "K60 as scored: C41_cold_K60, chip C s73, TT 1793"},
    "abs_base": {"steps": 2357, "ref": 0.96092, "what": "K59 lineage (FF_ATTN_SRC off): C40_cold_K59, chip C s73, TT 1793"},
}
GATING_CHIP = "C"                  # chip-D references are reported, never gated (D is judged only against D)
NULL_ARMS = ("repeat", "repeat2")
TRAINIUM_RUN_SD = 0.0006           # seed sd at 2M tokens (CONTRACT.md)
TRAINIUM_PAIR_SD_DEFAULT = TRAINIUM_RUN_SD * math.sqrt(2.0)
BOOTSTRAP_N = 2000
MDE_SEEDS = (1, 2, 3, 6)
MDE_Z = 1.6449                     # one-sided 95%

# The gated rows, named by (config pair id, Trainium seed, role).  "decisive" rows carry gates 3
# and 4; "null" and "tie" rows carry gate 5.  A row is resolvable only when the config plans
# both arms at that seed and validation-pairs.json holds a chip-C reference with both step
# counts at that seed; an unresolvable row keeps the verdict pending for good.
FROZEN_ROWS: Tuple[Tuple[str, int, str], ...] = (
    ("lk35_vs_relu2", 73, "decisive"), ("lk35_vs_relu2", 58, "decisive"), ("lk35_vs_relu2", 67, "decisive"),
    ("lk35_vs_rf", 73, "decisive"), ("lk35_vs_rf", 58, "decisive"), ("lk35_vs_rf", 67, "decisive"),
    ("lk35_vs_lk05", 73, "decisive"), ("lk35_vs_lk05", 58, "decisive"),
    ("asa", 73, "decisive"),
    ("mp97", 73, "decisive"), ("mp93", 73, "decisive"),
    ("rf_vs_relu2", 73, "null"), ("rf_vs_relu2", 58, "null"), ("rf_vs_relu2", 67, "null"),
    ("wd_sched1", 73, "tie"), ("wd_sched1", 58, "tie"),
    ("softcap165", 73, "tie"), ("softcap165", 58, "tie"),
    ("qk16", 73, "tie"), ("qk16", 58, "tie"),
)
# Config pairs that are reported, never gated, and why.
REPORTED_PAIRS: Dict[str, str] = {
    "asb": "single chip-D reference without a treatment step count: no equal-step target",
    "ramp3": "tests the equal-step correction, not a model change (treatment ran +6% steps)",
    "r3p30": "tests the equal-step correction, not a model change (treatment ran +6.5% steps)",
    "ko03": "|delta_eq| < 0.0003 on two of three seeds: ambiguous on Trainium",
    "warmup40": "the correction flips its sign (PROTOCOL 3.1: reported, not gated)",
    "acc_in_graph": "Neuron graph-capture flag; PROTOCOL 3.1 gives it no GPU pair",
    "emb_lr036": "not in PROTOCOL 3.3's null/tie list; single seed",
}
# PROTOCOL 3.1 sign-gate rows the gate-k60 config cannot enforce (so the code does not pretend to).
PROTOCOL_ROWS_UNENFORCED = (
    "asa s67: config reference lacks step counts (PROTOCOL 3.1: 2326 / 2325)",
    "asa s58: not in the config's trainium list (PROTOCOL 3.1: -0.00106 raw / -0.00140 eq)",
    "lk35_vs_lk05 s67: no lk05 s67 GPU run planned",
    "seed luck s281 / s283 / s293 / s307: no GPU runs planned",
)


def pair_role(pair_id: str) -> str:
    roles = {pid: role for pid, _seed, role in FROZEN_ROWS}
    return roles.get(pair_id, "reported")


def rule_thresholds() -> Dict[str, Any]:
    return {
        "k": STEP_ELASTICITY_K, "k_range": list(STEP_ELASTICITY_RANGE), "decisive_abs": DECISIVE_ABS,
        "sign_agreement_min": SIGN_AGREEMENT_MIN, "scale_band": list(SCALE_BAND), "null_max_abs": NULL_MAX_ABS,
        "tie_band_min": TIE_BAND_MIN, "tie_band_null_mult": TIE_BAND_NULL_MULT, "tie_max_outside": TIE_MAX_OUTSIDE,
        "eval_tokens": EVAL_TOKENS, "eval_bytes": EVAL_BYTES, "abs_level_tol": ABS_LEVEL_TOL,
        "abs_level_structural": ABS_LEVEL_STRUCTURAL, "abs_level_seed": ABS_LEVEL_SEED,
        "abs_level_runs": {k: dict(v) for k, v in ABS_LEVEL_RUNS.items()}, "gating_chip": GATING_CHIP,
        "null_arms": list(NULL_ARMS), "trainium_run_sd": TRAINIUM_RUN_SD,
    }


# ----------------------------------------------------------------------------- helpers

def equal_step_delta(delta_raw: float, steps_t: Optional[float], steps_c: Optional[float],
                     k: float = STEP_ELASTICITY_K) -> Tuple[float, bool]:
    if not steps_t or not steps_c:
        return float(delta_raw), False
    return float(delta_raw) + k * math.log(float(steps_t) / float(steps_c)), True


def summarize(xs: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray([float(x) for x in xs], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None}
    sd = float(arr.std(ddof=1)) if n > 1 else None
    return {"n": n, "mean": float(arr.mean()), "sd": sd, "se": (sd / math.sqrt(n)) if sd is not None else None}


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if xa.size < 3:
        return None
    rx, ry = _ranks(xa), _ranks(ya)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else None


def fit_scale(x: Sequence[float], y: Sequence[float], seed: int = 0) -> Dict[str, Any]:
    """y ~ a x through the origin; analytic se and a bootstrap-over-pairs percentile CI."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    n = int(xa.size)
    if n == 0 or float((xa * xa).sum()) == 0.0:
        return {"a": None, "se": None, "ci95": None, "n": n, "rms_residual": None}
    a = float((xa * ya).sum() / (xa * xa).sum())
    resid = ya - a * xa
    rms = float(math.sqrt((resid * resid).mean())) if n else None
    se = float(math.sqrt((resid * resid).sum() / (n - 1) / (xa * xa).sum())) if n > 1 else None
    ci = None
    if n >= 3:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n, n)
            bx, by = xa[idx], ya[idx]
            s = float((bx * bx).sum())
            if s > 0:
                boots.append(float((bx * by).sum() / s))
        if boots:
            ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    return {"a": a, "se": se, "ci95": ci, "n": n, "rms_residual": rms}


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pooled_sd(groups: Sequence[Sequence[float]]) -> Optional[float]:
    ss, df = 0.0, 0
    for g in groups:
        arr = np.asarray(g, float)
        if arr.size > 1:
            ss += float(((arr - arr.mean()) ** 2).sum())
            df += arr.size - 1
    return math.sqrt(ss / df) if df > 0 else None


def _sign(v: float) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _ok_row(r: Dict[str, Any]) -> bool:
    return r.get("status") == "ok" and isinstance(r.get("val_bpb"), (int, float))


# ----------------------------------------------------------------------------- Trainium side

def load_validation_pairs(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("pairs", [])
    return {p["pair_id"]: p for p in data if isinstance(p, dict) and p.get("pair_id")}


def trainium_reference(pair_def: Dict[str, Any], pairs_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve the config pair's Trainium references: by pair_id from validation-pairs.json,
    or inline (delta_raw, treatment_steps, control_steps, seed, chip, source).  ``gates`` says
    whether the reference may carry a gate (chip C, seed known, both step counts known);
    ``excluded`` says why not."""
    out = []
    for ref in pair_def.get("trainium", []):
        if isinstance(ref, str):
            ref = {"pair_id": ref}
        src = pairs_index.get(ref.get("pair_id", ""), {})
        merged = {**src, **{k: v for k, v in ref.items() if v is not None}}
        if merged.get("delta_raw") is None:
            out.append({"pair_id": ref.get("pair_id"), "missing": True})
            continue
        st, sc = merged.get("treatment_steps"), merged.get("control_steps")
        d_raw = float(merged["delta_raw"])
        d_eq, corrected = equal_step_delta(d_raw, st, sc)
        d_lo, _ = equal_step_delta(d_raw, st, sc, STEP_ELASTICITY_RANGE[0])
        d_hi, _ = equal_step_delta(d_raw, st, sc, STEP_ELASTICITY_RANGE[1])
        chip = merged.get("chip")
        excluded = None
        if chip != GATING_CHIP:
            excluded = f"chip {chip}: reported, not gated"
        elif merged.get("seed") is None:
            excluded = "no seed: cannot be matched to a GPU row"
        elif not corrected:
            excluded = "step counts missing: no equal-step target"
        out.append({
            "pair_id": merged.get("pair_id"), "seed": merged.get("seed"), "chip": chip,
            "delta_raw": d_raw, "delta_eq": d_eq, "delta_eq_range": [min(d_lo, d_hi), max(d_lo, d_hi)],
            "steps_t": st, "steps_c": sc, "corrected": corrected,
            "changes_step_time": merged.get("changes_step_time"),
            "source": merged.get("source") or ("validation-pairs.json" if src else "inline"),
            "gates": excluded is None, "excluded": excluded,
        })
    return out


def row_decisive(delta_eq: float, lo: float, hi: float) -> Tuple[bool, str]:
    """|delta_eq| >= DECISIVE_ABS with the same sign for every k in STEP_ELASTICITY_RANGE."""
    if abs(delta_eq) < DECISIVE_ABS:
        return False, f"|delta_eq| {abs(delta_eq):.5f} < {DECISIVE_ABS}"
    if lo * hi <= 0 or min(abs(lo), abs(hi)) < DECISIVE_ABS:
        return False, f"sign or size not stable over k in {STEP_ELASTICITY_RANGE}: [{lo:+.5f}, {hi:+.5f}]"
    return True, "decisive"


def is_decisive(refs: List[Dict[str, Any]], gating: bool = True) -> Tuple[bool, str]:
    """The row rule applied to the mean of a reference set (a reporting helper; the gate itself
    runs on the FROZEN_ROWS, never on a config flag)."""
    good = [r for r in refs if not r.get("missing")]
    if not good:
        return False, "no Trainium reference"
    if not gating:
        return False, "marked non-gating"
    return row_decisive(float(np.mean([r["delta_eq"] for r in good])),
                        float(np.mean([r["delta_eq_range"][0] for r in good])),
                        float(np.mean([r["delta_eq_range"][1] for r in good])))


# ----------------------------------------------------------------------------- freezing

def planned_keys(index: Dict[str, Dict[str, Any]]) -> Set[Tuple[Any, Any, Any]]:
    """(arm, seed, steps) of every run the index knows about, whatever its status."""
    return {(r.get("arm"), r.get("seed"), r.get("steps")) for r in index.values() if r.get("arm")}


def config_keys(config: Dict[str, Any]) -> Set[Tuple[Any, Any, Any]]:
    return {(r.get("arm"), r.get("seed"), r.get("steps")) for r in config.get("runs", [])}


def freeze_rule(pair_defs: List[Dict[str, Any]], pairs_index: Dict[str, Dict[str, Any]],
                planned: Iterable[Tuple[Any, Any, Any]], base_arm: str = "k60") -> Dict[str, Any]:
    """Resolve FROZEN_ROWS against the config's pairs, its planned runs and the Trainium data.
    Depends on nothing measured on the GPU, so it can be written at plan time and re-derived at
    verdict time; ``digest`` is what --rule compares."""
    by_id = {pd["pair_id"]: pd for pd in pair_defs}
    planned = set(planned)

    def matched_steps(t_arm: str, c_arm: str, seed: int) -> List[Any]:
        return sorted({st for (a, s, st) in planned if a == t_arm and s == seed and (c_arm, seed, st) in planned},
                      key=lambda v: v or 0)

    rows: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for pair_id, seed, role in FROZEN_ROWS:
        pd = by_id.get(pair_id)
        if pd is None:
            unresolved.append({"pair_id": pair_id, "seed": seed, "role": role, "why": "pair not in this config"})
            continue
        refs = [r for r in trainium_reference(pd, pairs_index) if not r.get("missing") and r.get("seed") == seed]
        gating = [r for r in refs if r["gates"]]
        steps = matched_steps(pd["treatment_arm"], pd["control_arm"], seed)
        why = None
        if not steps:
            why = f"arms {pd['treatment_arm']} / {pd['control_arm']} not both planned at seed {seed}"
        elif role == "decisive" and not gating:
            why = ("no gating Trainium reference at this seed"
                   + (": " + "; ".join(str(r.get("excluded")) for r in refs) if refs else ""))
        row: Dict[str, Any] = {
            "pair_id": pair_id, "family": pd.get("family", ""), "role": role, "seed": seed,
            "treatment_arm": pd["treatment_arm"], "control_arm": pd["control_arm"], "steps": steps,
            "trainium_refs": [r["pair_id"] for r in gating], "trainium_n": len(gating),
            "delta_eq": float(np.mean([r["delta_eq"] for r in gating])) if gating else None,
            "delta_raw": float(np.mean([r["delta_raw"] for r in gating])) if gating else None,
            "delta_eq_range": ([float(np.mean([r["delta_eq_range"][0] for r in gating])),
                                float(np.mean([r["delta_eq_range"][1] for r in gating]))] if gating else None),
        }
        if why is None and role == "decisive":
            ok, dwhy = row_decisive(row["delta_eq"], row["delta_eq_range"][0], row["delta_eq_range"][1])
            if not ok:
                why = f"Trainium target not decisive: {dwhy}"
        if why:
            unresolved.append({"pair_id": pair_id, "seed": seed, "role": role, "why": why})
        else:
            rows.append(row)
    rule = {
        "version": RULE_VERSION, "mode": RULE_MODE, "thresholds": rule_thresholds(),
        "pair_roles": {pd["pair_id"]: pair_role(pd["pair_id"]) for pd in pair_defs},
        "reported_pairs": REPORTED_PAIRS, "protocol_rows_unenforced": list(PROTOCOL_ROWS_UNENFORCED),
        "rows": rows, "unresolved": unresolved,
        "n_decisive": sum(1 for r in rows if r["role"] == "decisive"),
        "n_null_tie": sum(1 for r in rows if r["role"] in ("null", "tie")),
        "abs_level": {arm: {"seed": ABS_LEVEL_SEED, "tol": ABS_LEVEL_TOL, **spec,
                            "planned": (arm, ABS_LEVEL_SEED, spec["steps"]) in planned}
                      for arm, spec in ABS_LEVEL_RUNS.items()},
        "base_arm": base_arm, "null_arms": list(NULL_ARMS),
    }
    rule["digest"] = hashlib.sha256(json.dumps(rule, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return rule


# ----------------------------------------------------------------------------- GPU side

def gpu_deltas(index: Dict[str, Dict[str, Any]], treatment_arm: str, control_arm: str) -> List[Dict[str, Any]]:
    """treatment - control at matched (seed, steps); only rows with status ok and a val_bpb."""
    t: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    c: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for r in index.values():
        if not _ok_row(r):
            continue
        key = (r.get("seed"), r.get("steps"))
        if r.get("arm") == treatment_arm:
            t[key] = r
        elif r.get("arm") == control_arm:
            c[key] = r
    out = []
    for key in sorted(set(t) & set(c), key=lambda k: (k[0] or 0, k[1] or 0)):
        out.append({"seed": key[0], "steps": key[1], "delta": float(t[key]["val_bpb"]) - float(c[key]["val_bpb"]),
                    "labels": [t[key]["label"], c[key]["label"]]})
    return out


def null_deltas(index: Dict[str, Dict[str, Any]], base_arm: str) -> List[Dict[str, Any]]:
    base = {(r.get("seed"), r.get("steps")): r for r in index.values() if r.get("arm") == base_arm and _ok_row(r)}
    out = []
    for r in index.values():
        if r.get("arm") in NULL_ARMS and _ok_row(r):
            b = base.get((r.get("seed"), r.get("steps")))
            if b:
                out.append({"arm": r["arm"], "seed": r.get("seed"), "delta": float(r["val_bpb"]) - float(b["val_bpb"]),
                            "labels": [r["label"], b["label"]]})
    return out


def token_set_ok(r: Dict[str, Any]) -> bool:
    """Gate 0 for one scored row: train_gpu.py's eval_denominator_ok, or the raw counts."""
    if r.get("eval_denominator_ok") is True:
        return True
    return r.get("eval_num_tokens") == EVAL_TOKENS and r.get("eval_num_bytes") == EVAL_BYTES


# ----------------------------------------------------------------------------- calibration

def calibrate(index: Dict[str, Dict[str, Any]], pair_defs: List[Dict[str, Any]],
              pairs_index: Dict[str, Dict[str, Any]], base_arm: str = "k60",
              frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rule = freeze_rule(pair_defs, pairs_index, planned_keys(index), base_arm)
    rule_mismatch = None
    if frozen is not None and frozen.get("digest") != rule["digest"]:
        rule_mismatch = (f"frozen rule {str(frozen.get('digest', '?'))[:12]} (written {frozen.get('written', '?')}) "
                         f"!= rule recomputed from the config and the Trainium data {rule['digest'][:12]}: "
                         "the row set or a threshold changed after planning; no verdict")

    # --- pair-level table (reported) and seed rows (gated)
    pairs_out: List[Dict[str, Any]] = []
    deltas_by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for pd in pair_defs:
        refs = trainium_reference(pd, pairs_index)
        good = [r for r in refs if not r.get("missing")]
        gating = [r for r in good if r["gates"]]
        g = gpu_deltas(index, pd["treatment_arm"], pd["control_arm"])
        deltas_by_pair[pd["pair_id"]] = g
        gsum = summarize([d["delta"] for d in g])
        tsum_eq = summarize([r["delta_eq"] for r in gating])
        tsum_raw = summarize([r["delta_raw"] for r in gating])
        role = pair_role(pd["pair_id"])
        row = {
            "pair_id": pd["pair_id"], "family": pd.get("family", ""), "treatment_arm": pd["treatment_arm"],
            "control_arm": pd["control_arm"], "role": role, "decisive": role == "decisive",
            "why": REPORTED_PAIRS.get(pd["pair_id"], role),
            "trainium": refs, "trainium_eq": tsum_eq, "trainium_raw": tsum_raw,
            "gpu": g, "gpu_summary": gsum, "sign_agree": None,
        }
        if gsum["n"] and tsum_eq["n"]:
            row["sign_agree"] = _sign(gsum["mean"]) == _sign(tsum_eq["mean"])
        pairs_out.append(row)

    rows: List[Dict[str, Any]] = []
    for fr in rule["rows"]:
        g = [d for d in deltas_by_pair.get(fr["pair_id"], []) if d["seed"] == fr["seed"] and d["steps"] in fr["steps"]]
        row = dict(fr)
        row["gpu"] = float(np.mean([d["delta"] for d in g])) if g else None
        row["gpu_n"] = len(g)
        row["gpu_labels"] = [d["labels"] for d in g]
        row["agree"] = (_sign(row["gpu"]) == _sign(row["delta_eq"])) if (g and row["delta_eq"] is not None) else None
        rows.append(row)
    decisive_rows = [r for r in rows if r["role"] == "decisive"]
    tie_rows = [r for r in rows if r["role"] in ("null", "tie")]
    dec_data = [r for r in decisive_rows if r["gpu"] is not None]
    tie_data = [r for r in tie_rows if r["gpu"] is not None]

    # --- statistics
    x = [r["gpu"] for r in dec_data]
    y = [r["delta_eq"] for r in dec_data]
    fit = fit_scale(x, y)
    rho = spearman(x, y)
    all_rows_data = [(p["gpu_summary"]["mean"], p["trainium_eq"]["mean"], p["family"], p["sign_agree"])
                     for p in pairs_out if p["gpu_summary"]["n"] and p["trainium_eq"]["n"]]
    fit_all = fit_scale([v[0] for v in all_rows_data], [v[1] for v in all_rows_data])
    sign_rate_all = (sum(1 for v in all_rows_data if v[3]) / len(all_rows_data)) if all_rows_data else None
    n_agree = sum(1 for r in dec_data if r["agree"])
    sign_rate = n_agree / len(dec_data) if dec_data else None

    families: Dict[str, List[float]] = {}
    if fit["a"] is not None:
        for gm, tm, fam, _ in all_rows_data:
            families.setdefault(fam, []).append(tm - fit["a"] * gm)
    family_offsets = {f: summarize(v) for f, v in families.items()}

    # --- noise terms
    gpu_pair_sd = pooled_sd([[d["delta"] for d in p["gpu"]] for p in pairs_out])
    base_runs = [float(r["val_bpb"]) for r in index.values() if r.get("arm") == base_arm and _ok_row(r)]
    base_sd = summarize(base_runs)["sd"] if len(base_runs) > 1 else None
    nulls = null_deltas(index, base_arm)
    null_worst = max((abs(n["delta"]) for n in nulls), default=None)
    null_rms = math.sqrt(float(np.mean([n["delta"] ** 2 for n in nulls]))) if nulls else None
    trn_pair_sd = pooled_sd([[r["delta_eq"] for r in p["trainium"] if not r.get("missing") and r["gates"]] for p in pairs_out])
    trn_pair_sd_used = trn_pair_sd if trn_pair_sd is not None else TRAINIUM_PAIR_SD_DEFAULT
    gpu_pair_sd_used = gpu_pair_sd if gpu_pair_sd is not None else (base_sd * math.sqrt(2) if base_sd else None)
    cal_err = None
    cal_err_raw = fit["rms_residual"]
    if fit["a"] is not None and gpu_pair_sd_used is not None and dec_data:
        expected = float(np.mean([trn_pair_sd_used ** 2 / max(1, r["trainium_n"])
                                  + (fit["a"] ** 2) * gpu_pair_sd_used ** 2 / max(1, r["gpu_n"]) for r in dec_data]))
        cal_err = math.sqrt(max(0.0, cal_err_raw ** 2 - expected))
    noise = {
        "gpu_pair_sd_pooled": gpu_pair_sd, "gpu_base_run_sd": base_sd, "gpu_base_n": len(base_runs),
        "gpu_null_deltas": nulls, "gpu_null_worst_abs": null_worst, "gpu_null_rms": null_rms,
        "gpu_pair_sd_used": gpu_pair_sd_used,
        "trainium_run_sd": TRAINIUM_RUN_SD, "trainium_pair_sd_pooled": trn_pair_sd, "trainium_pair_sd_used": trn_pair_sd_used,
        "calibration_rms_residual": cal_err_raw, "calibration_error_sd": cal_err,
    }
    mde = {str(n): (MDE_Z * gpu_pair_sd_used / math.sqrt(n)) if gpu_pair_sd_used else None for n in MDE_SEEDS}

    # --- gates (PROTOCOL.md 3.3), evaluated in order; ok is True / False / None (pending)
    gates: List[Dict[str, Any]] = []
    pending: List[str] = []

    scored = [r for r in index.values() if _ok_row(r)]
    bad0 = sorted(r["label"] for r in scored if not token_set_ok(r))
    gates.append({"n": 0, "name": "token set", "ok": not bad0 if scored else None,
                  "detail": (f"{len(scored)} scored runs at {EVAL_TOKENS:,} tokens / {EVAL_BYTES:,} bytes" if not bad0
                             else f"{len(bad0)} scored run(s) without the {EVAL_TOKENS:,}-token / {EVAL_BYTES:,}-byte "
                                  f"denominator: {', '.join(bad0[:6])}") if scored else "no scored runs"})

    if nulls:
        g1 = null_worst <= NULL_MAX_ABS
        d1 = f"worst null |delta| {null_worst:.5f} {'<=' if g1 else '>'} {NULL_MAX_ABS} over {len(nulls)} null(s), rms {null_rms:.5f}"
        if not g1:
            d1 += "; set CUBLAS_WORKSPACE_CONFIG=:4096:8 and torch.use_deterministic_algorithms(True) and re-measure"
    else:
        g1, d1 = None, f"no null run ({' / '.join(NULL_ARMS)} vs {base_arm} at seed {ABS_LEVEL_SEED}) scored yet"
        pending.append("nulls (gate 1)")
    gates.append({"n": 1, "name": "noise floor", "ok": g1, "detail": d1})

    lvl_parts, g2, lvl = [], True, {}
    for arm, spec in ABS_LEVEL_RUNS.items():
        cands = [r for r in index.values()
                 if r.get("arm") == arm and r.get("seed") == ABS_LEVEL_SEED and r.get("steps") == spec["steps"] and _ok_row(r)]
        if not cands:
            g2 = None if g2 is not False else False
            lvl_parts.append(f"{arm} s{ABS_LEVEL_SEED} @ {spec['steps']} not scored")
            pending.append(f"{arm} s{ABS_LEVEL_SEED} @ {spec['steps']} (gate 2)")
            continue
        level = float(cands[0]["val_bpb"])
        dist = abs(level - spec["ref"])
        ok2 = dist <= ABS_LEVEL_TOL
        lvl[arm] = {"val_bpb": level, "steps": spec["steps"], "ref": spec["ref"], "distance": dist, "ok": ok2,
                    "label": cands[0].get("label")}
        if not ok2:
            g2 = False
        lvl_parts.append(f"{arm} s{ABS_LEVEL_SEED} {level:.5f} @ {spec['steps']} vs {spec['ref']}: |d| {dist:.5f}"
                         + ("" if ok2 else (" STRUCTURAL: check data-world 8, eval ranks 8, FF_LAMBDA_0D=1"
                                            if dist > ABS_LEVEL_STRUCTURAL else " > tol")))
    gates.append({"n": 2, "name": "absolute level", "ok": g2, "detail": "; ".join(lvl_parts) + f" (tol {ABS_LEVEL_TOL})",
                  "levels": lvl})

    missing3 = [f"{r['pair_id']}@s{r['seed']}" for r in decisive_rows if r["gpu"] is None]
    fam_stats: Dict[str, Dict[str, int]] = {}
    for r in dec_data:
        fs = fam_stats.setdefault(r["family"], {"n": 0, "agree": 0})
        fs["n"] += 1
        fs["agree"] += 1 if r["agree"] else 0
    fam_fail = sorted(f for f, s in fam_stats.items() if s["agree"] == 0)
    if not decisive_rows:
        g3, d3 = None, "no frozen decisive row is resolvable in this config"
    elif missing3:
        g3, d3 = None, f"{len(dec_data)} of {len(decisive_rows)} decisive rows scored; waiting on {', '.join(missing3)}"
        pending.extend(f"{m} (gate 3)" for m in missing3)
    else:
        g3 = sign_rate >= SIGN_AGREEMENT_MIN and not fam_fail
        d3 = (f"{n_agree}/{len(dec_data)} = {sign_rate:.0%} {'>=' if sign_rate >= SIGN_AGREEMENT_MIN else '<'} "
              f"{SIGN_AGREEMENT_MIN:.0%}; families " + ", ".join(f"{f} {s['agree']}/{s['n']}" for f, s in sorted(fam_stats.items()))
              + (f"; family with no agreeing seed: {', '.join(fam_fail)}" if fam_fail else ""))
    gates.append({"n": 3, "name": "sign agreement", "ok": g3, "detail": d3, "families": fam_stats})

    if g3 is None:
        g4, d4 = None, "waits on gate 3's rows"
    else:
        a = fit["a"]
        g4 = a is not None and SCALE_BAND[0] <= a <= SCALE_BAND[1]
        d4 = (f"a = {'-' if a is None else round(a, 3)} {'in' if g4 else 'outside'} [{SCALE_BAND[0]}, {SCALE_BAND[1]}] "
              f"over {fit['n']} decisive rows (CI95 {fit['ci95']}, Spearman rho {'-' if rho is None else round(rho, 2)})")
    gates.append({"n": 4, "name": "scale", "ok": g4, "detail": d4})

    band = max(TIE_BAND_MIN, TIE_BAND_NULL_MULT * null_rms) if null_rms is not None else TIE_BAND_MIN
    missing5 = [f"{r['pair_id']}@s{r['seed']}" for r in tie_rows if r["gpu"] is None]
    for r in tie_data:
        r["in_band"] = abs(r["gpu"]) <= band
    outside = [f"{r['pair_id']}@s{r['seed']} {r['gpu']:+.5f}" for r in tie_data if not r["in_band"]]
    if not tie_rows:
        g5, d5 = None, "no frozen null/tie row is resolvable in this config"
    elif missing5 or null_rms is None:
        g5 = None
        d5 = f"{len(tie_data)} of {len(tie_rows)} null/tie rows scored" + (f"; waiting on {', '.join(missing5)}" if missing5 else "") \
            + ("; band needs the nulls" if null_rms is None else "")
        pending.extend(f"{m} (gate 5)" for m in missing5)
    else:
        g5 = len(outside) <= TIE_MAX_OUTSIDE
        d5 = (f"{len(outside)} of {len(tie_data)} null/tie rows outside band {band:.5f} "
              f"(= max({TIE_BAND_MIN}, {TIE_BAND_NULL_MULT:g} x null rms); <= {TIE_MAX_OUTSIDE} allowed)"
              + (": " + ", ".join(outside) if outside else ""))
    gates.append({"n": 5, "name": "nulls and ties", "ok": g5, "detail": d5, "band": band})

    fam_notes = {}
    for r in dec_data:
        fam_notes.setdefault(r["family"], {"gpu": [], "trainium_eq": []})
        fam_notes[r["family"]]["gpu"].append(r["gpu"])
        fam_notes[r["family"]]["trainium_eq"].append(r["delta_eq"])
    fam_notes = {f: {"gpu": summarize(v["gpu"]), "trainium_eq": summarize(v["trainium_eq"])} for f, v in fam_notes.items()}
    gates.append({"n": 6, "name": "notes (not gated)", "ok": None,
                  "detail": "MDE at n seed pairs " + ", ".join(f"{n}: {'-' if v is None else format(v, '.5f')}" for n, v in mde.items())
                  + f"; unenforced PROTOCOL rows: {len(PROTOCOL_ROWS_UNENFORCED)}",
                  "families": fam_notes, "mde": mde})

    # --- decision
    if rule_mismatch:
        status, reasons = "rejected", [rule_mismatch]
    elif gates[0]["ok"] is False:
        status, reasons = "rejected", ["gate 0 token set: " + gates[0]["detail"] + " (nothing else is read)"]
    elif rule["unresolved"]:
        status = "pending"
        reasons = [f"frozen row {u['pair_id']}@s{u['seed']} ({u['role']}) unresolvable: {u['why']}" for u in rule["unresolved"]]
    elif pending or gates[0]["ok"] is None:
        status = "pending"
        reasons = ["waiting on " + ", ".join(pending)] if pending else ["no scored runs"]
    else:
        failed = [f"gate {g['n']} {g['name']}: {g['detail']}" for g in gates[1:6] if g["ok"] is False]
        status = "rejected" if failed else "accepted"
        reasons = failed or [f"gates 0-5 pass on {len(dec_data)} decisive + {len(tie_data)} null/tie rows: "
                             f"sign {sign_rate:.0%}, a = {fit['a']:.3f} (CI {fit['ci95']})"]
    return {
        "status": status, "reasons": reasons, "gates": gates, "rule": rule, "rows": rows,
        "n_pairs": len(pairs_out), "n_with_data": len(all_rows_data),
        "n_decisive_rows": len(decisive_rows), "n_decisive_with_data": len(dec_data),
        "pending_decisive": missing3, "pending": pending,
        "sign_agreement_decisive": sign_rate, "sign_agreement_all": sign_rate_all, "spearman": rho,
        "scale": fit, "scale_all": fit_all, "family_offsets": family_offsets, "noise": noise, "mde": mde,
        "pairs": pairs_out,
    }


# ----------------------------------------------------------------------------- prediction

def predict(gpu_deltas_new: Sequence[float], cal: Dict[str, Any], family: Optional[str] = None,
            seconds_per_step: float = 4.5, steps: int = 2361, base_banked: bool = True,
            targets: Sequence[float] = (0.90, 0.95)) -> Dict[str, Any]:
    """P(Trainium delta < 0) for a candidate from its GPU deltas and the calibration."""
    from ffsim.gpu import fleet as _fleet
    xs = [float(v) for v in gpu_deltas_new]
    s = summarize(xs)
    a = cal["scale"]["a"] if cal.get("scale") else None
    if s["n"] == 0 or a is None:
        return {"error": "no GPU deltas or no fitted scale", "gpu": s}
    se_a = cal["scale"]["se"] or 0.0
    noise = cal["noise"]
    sd_gpu = noise.get("gpu_pair_sd_used") or (s["sd"] if s["sd"] else 0.0)
    if s["sd"] is not None and s["n"] >= 3:
        sd_gpu = max(sd_gpu, s["sd"])
    sd_cal = noise.get("calibration_error_sd")
    if sd_cal is None:
        sd_cal = noise.get("calibration_rms_residual") or 0.0
    offset = 0.0
    off = cal.get("family_offsets", {}).get(family or "", {})
    if off and off.get("mean") is not None:
        offset = float(off["mean"])
    mean_pred = a * s["mean"] + offset

    def sd_at(n: int) -> float:
        return math.sqrt((a * sd_gpu) ** 2 / n + sd_cal ** 2 + (se_a * s["mean"]) ** 2)

    sd_now = sd_at(s["n"])
    p_neg = norm_cdf(-mean_pred / sd_now) if sd_now > 0 else (1.0 if mean_pred < 0 else 0.0)
    warning = None
    if cal.get("status") != "accepted":
        warning = f"proxy status is {cal.get('status')!r}: this probability is NOT decision-grade"
    # cost table: seeds needed so that |mean_pred| / sd(n) >= z(target)
    cost_rows = []
    for tgt in targets:
        z = -_inv_norm_cdf(1.0 - tgt)  # e.g. 1.2816 for 0.90, 1.6449 for 0.95
        need = None
        for n in range(1, 65):
            if abs(mean_pred) / sd_at(n) >= z:
                need = n
                break
        if need is None:
            cost_rows.append({"target": tgt, "seed_pairs": None, "reason": "unreachable: the calibration error "
                              "or the scale uncertainty dominates at any seed count (or the effect is ~0)"})
            continue
        extra = max(0, need - s["n"])
        runs = extra if base_banked else 2 * extra
        fake = [{"steps": steps} for _ in range(runs)]
        tbl = _fleet.cost_table(fake, seconds_per_step) if runs else None
        cost_rows.append({"target": tgt, "seed_pairs": need, "extra_seed_pairs": extra, "extra_runs": runs,
                          "gpu_hours": tbl["gpu_hours"] if tbl else 0.0,
                          "usd": {r["instance"]: r["usd"] for r in tbl["instances"]} if tbl else {}})
    return {"gpu": s, "a": a, "se_a": se_a, "family": family, "family_offset": offset, "sd_gpu_pair": sd_gpu,
            "sd_calibration": sd_cal, "predicted_trainium_delta": mean_pred, "sd_prediction": sd_now,
            "p_trainium_delta_negative": p_neg, "warning": warning, "cost": cost_rows,
            "note": "delta = treatment - control at equal steps; negative is better; this ignores any "
                    "step-time change of the candidate (a Trainium property the GPU cannot see)"}


def _inv_norm_cdf(p: float) -> float:
    """Acklam's rational approximation (|error| < 1.2e-9), stdlib only."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ----------------------------------------------------------------------------- report

def _f(v: Any, fmt: str = "{:+.5f}") -> str:
    return fmt.format(v) if isinstance(v, (int, float)) else "-"


def rule_text() -> str:
    """The frozen rule in one paragraph; PROTOCOL.md / RUNBOOK.md quote this."""
    gate2 = ", ".join(f"{a} s{ABS_LEVEL_SEED} @ {s['steps']} within {ABS_LEVEL_TOL} of {s['ref']}"
                      for a, s in ABS_LEVEL_RUNS.items())
    return (f"rule {RULE_VERSION}; mode {RULE_MODE}. delta_eq = delta_raw + {STEP_ELASTICITY_K} ln(steps_T/steps_C); "
            f"decisive row iff |delta_eq| >= {DECISIVE_ABS} for every k in {STEP_ELASTICITY_RANGE}, chip {GATING_CHIP} only. "
            f"Gates: 0 every scored run at {EVAL_TOKENS:,} tokens / {EVAL_BYTES:,} bytes; 1 worst null |delta| <= {NULL_MAX_ABS}; "
            f"2 {gate2}; "
            f"3 sign agreement >= {SIGN_AGREEMENT_MIN:.0%} on all {sum(1 for _r in FROZEN_ROWS if _r[2] == 'decisive')} frozen "
            f"decisive rows and >= 1 agreeing seed per family; 4 a in [{SCALE_BAND[0]}, {SCALE_BAND[1]}]; "
            f"5 at most {TIE_MAX_OUTSIDE} of the {sum(1 for _r in FROZEN_ROWS if _r[2] in ('null', 'tie'))} null/tie rows "
            f"outside max({TIE_BAND_MIN}, {TIE_BAND_NULL_MULT:g} x null rms). Pending until every frozen row, both nulls and "
            f"both level runs are scored; no accept on a subset.")


def render(cal: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append(f"GPU proxy calibration: {cal['status'].upper()}  ({'; '.join(cal['reasons'])})")
    L.append(f"pairs {cal['n_pairs']}, with GPU data {cal['n_with_data']}; frozen decisive rows {cal['n_decisive_rows']} "
             f"(scored {cal['n_decisive_with_data']})" + (f", waiting on: {', '.join(cal['pending'])}" if cal["pending"] else "")
             + f"; rule digest {cal['rule']['digest'][:12]}")
    for g in cal["gates"]:
        mark = "PASS" if g["ok"] is True else ("FAIL" if g["ok"] is False else "....")
        L.append(f"gate {g['n']} {g['name']:<18} {mark}  {g['detail']}")
    if cal["rule"]["unresolved"]:
        L.append("unresolved frozen rows: " + "; ".join(f"{u['pair_id']}@s{u['seed']} {u['why']}" for u in cal["rule"]["unresolved"]))
    L.append(f"sign agreement: decisive rows {_f(cal['sign_agreement_decisive'], '{:.0%}')}, all pairs {_f(cal['sign_agreement_all'], '{:.0%}')}; "
             f"Spearman rho {_f(cal['spearman'], '{:.2f}')}")
    sc = cal["scale"]
    L.append(f"scale delta_trn ~ a * delta_gpu: a = {_f(sc['a'], '{:.3f}')} se {_f(sc['se'], '{:.3f}')} "
             f"CI95 {sc['ci95']} (n {sc['n']} decisive rows); all pairs a = {_f(cal['scale_all']['a'], '{:.3f}')}")
    nz = cal["noise"]
    L.append(f"noise: GPU paired-delta sd {_f(nz['gpu_pair_sd_pooled'])} (base-run sd {_f(nz['gpu_base_run_sd'])}, n {nz['gpu_base_n']}; "
             f"null worst |d| {_f(nz['gpu_null_worst_abs'])}, rms {_f(nz['gpu_null_rms'])}); Trainium run sd {nz['trainium_run_sd']} "
             f"(paired pooled {_f(nz['trainium_pair_sd_pooled'])}, used {_f(nz['trainium_pair_sd_used'])}); "
             f"calibration residual rms {_f(nz['calibration_rms_residual'])}, error sd {_f(nz['calibration_error_sd'])}")
    if cal["family_offsets"]:
        L.append("family offsets (trn - a*gpu): " + ", ".join(
            f"{f} {_f(s['mean'])} (n {s['n']})" for f, s in sorted(cal["family_offsets"].items())))
    L.append("")
    L.append(f"{'row':<22}{'family':<14}{'role':<9}{'seed':>5}{'gpu':>10}{'n':>3}{'trn eq':>10}{'trn raw':>10}{'verdict':>9}")
    for r in cal["rows"]:
        if r["role"] == "decisive":
            v = "-" if r["agree"] is None else ("ok" if r["agree"] else "FLIP")
        else:
            v = "-" if r.get("in_band") is None else ("ok" if r["in_band"] else "OUTSIDE")
        L.append(f"{r['pair_id']:<22}{r['family']:<14}{r['role']:<9}{r['seed']:>5}{_f(r['gpu']):>10}{r['gpu_n']:>3}"
                 f"{_f(r['delta_eq']):>10}{_f(r['delta_raw']):>10}{v:>9}")
    L.append("")
    L.append(f"{'pair':<22}{'family':<14}{'role':<9}{'gpu mean':>10}{'sd':>9}{'n':>3}{'trn eq':>10}{'trn raw':>10}{'n':>3}{'sign':>6}")
    for p in cal["pairs"]:
        g, t, tr = p["gpu_summary"], p["trainium_eq"], p["trainium_raw"]
        sign = "-" if p["sign_agree"] is None else ("ok" if p["sign_agree"] else "FLIP")
        L.append(f"{p['pair_id']:<22}{p['family']:<14}{p['role']:<9}{_f(g['mean']):>10}{_f(g['sd']):>9}{g['n']:>3}"
                 f"{_f(t['mean']):>10}{_f(tr['mean']):>10}{t['n']:>3}{sign:>6}")
    L.append("")
    L.append(rule_text())
    return "\n".join(L)


def render_prediction(pred: Dict[str, Any]) -> str:
    if "error" in pred:
        return f"prediction: {pred['error']}"
    L = [f"candidate: GPU delta mean {_f(pred['gpu']['mean'])} sd {_f(pred['gpu']['sd'])} n {pred['gpu']['n']}"
         f" -> predicted Trainium delta {_f(pred['predicted_trainium_delta'])} +/- {_f(pred['sd_prediction'], '{:.5f}')}"
         f" (a {pred['a']:.3f}, family offset {_f(pred['family_offset'])}, sd_gpu {_f(pred['sd_gpu_pair'], '{:.5f}')}, "
         f"sd_cal {_f(pred['sd_calibration'], '{:.5f}')})",
         f"P(Trainium delta < 0) = {pred['p_trainium_delta_negative']:.3f}"]
    if pred.get("warning"):
        L.append("WARNING: " + pred["warning"])
    for c in pred["cost"]:
        if c.get("seed_pairs") is None:
            L.append(f"  resolve at {c['target']:.0%}: {c['reason']}")
        else:
            L.append(f"  resolve at {c['target']:.0%}: {c['seed_pairs']} seed pairs ({c['extra_runs']} more runs, "
                     f"{c['gpu_hours']:.2f} GPU-h): " + ", ".join(f"{k} ${v:.2f}" for k, v in c["usd"].items()))
    L.append(pred["note"])
    return "\n".join(L)


# ----------------------------------------------------------------------------- main

def main(argv: Optional[Sequence[str]] = None) -> int:
    from ffsim.gpu import fleet as _fleet
    ap = argparse.ArgumentParser(prog="ffsim.gpu.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=None, help="results.jsonl (required unless only --freeze is wanted)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--configs", default=str(_fleet.CONFIG_DIR / "gate-k60.json"))
    ap.add_argument("--pairs", default=str(_fleet.REPO / "research" / "sim-data" / "validation-pairs.json"))
    ap.add_argument("--base-arm", default="k60")
    ap.add_argument("--freeze", default=None, help="write the frozen rule (rows, targets, thresholds, digest) "
                                                   "resolved from the config and the Trainium data; needs no results")
    ap.add_argument("--rule", default=None, help="a --freeze file; the verdict is refused when it no longer matches")
    ap.add_argument("--json", default=None)
    ap.add_argument("--predict", default=None, help="comma-separated GPU deltas of a new candidate")
    ap.add_argument("--family", default=None)
    ap.add_argument("--seconds-per-step", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=_fleet.K60_STEPS)
    args = ap.parse_args(argv)

    config = _fleet.load_config(Path(args.configs))
    pairs_index = load_validation_pairs(Path(args.pairs))
    if args.freeze:
        rule = freeze_rule(config["pairs"], pairs_index, config_keys(config), args.base_arm)
        rule["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rule["config"] = str(args.configs)
        Path(args.freeze).write_text(json.dumps(rule, indent=1, default=str), encoding="utf-8")
        print(f"frozen rule {rule['digest'][:12]}: {rule['n_decisive']} decisive + {rule['n_null_tie']} null/tie rows, "
              f"{len(rule['unresolved'])} unresolved -> {args.freeze}")
        for u in rule["unresolved"]:
            print(f"  unresolved {u['pair_id']}@s{u['seed']} ({u['role']}): {u['why']}")
        if not args.results:
            return 0 if not rule["unresolved"] else 2
    if not args.results:
        ap.error("--results is required (or --freeze alone)")
    frozen = json.loads(Path(args.rule).read_text(encoding="utf-8")) if args.rule else None
    results = Path(args.results)
    manifest = Path(args.manifest) if args.manifest else results.with_name(results.stem + "-manifest.jsonl")
    index = _fleet.merge_results(results, manifest, config)
    cal = calibrate(index, config["pairs"], pairs_index, base_arm=args.base_arm, frozen=frozen)
    print(render(cal))
    out: Dict[str, Any] = {"calibration": cal}
    if args.predict:
        deltas = [float(v) for v in args.predict.split(",") if v.strip()]
        pred = predict(deltas, cal, args.family, args.seconds_per_step, args.steps)
        print("\n" + render_prediction(pred))
        out["prediction"] = pred
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    # exit codes as the RUNBOOK states them: 0 accepted, 1 rejected, 2 incomplete/pending
    return 0 if cal["status"] == "accepted" else (2 if cal["status"] == "pending" else 1)


if __name__ == "__main__":
    sys.exit(main())
