"""ffsim.quality: Part 2 of the simulator, the quality surrogate (spec route 1, statistical surrogate).

Target
------
``bpb_2m``: val_bpb on the public shard, first 2,097,152 tokens, trained weights, RAW (never step-normalised),
for a recipe (FF_* knobs), an optimizer-step count, a code version, a chip and a seed.

Model
-----
::

    bpb = era[code_version] + chip[chip] + b * L + c * L**2 + sum_j beta_j * x_j(knobs) + u[seed] + eps
    L   = ln(steps / 2300)

* ``era``  one fixed effect per code ERA. A record's code version is its ``lineage:`` tag when the dataset has one
  (chip dir ``code_k12off`` -> ``M12``), else its ``code_version`` canonicalised (``code_m9`` -> ``M9``). Versions are
  then grouped into eras (``QualityConfig.era_groups``): by default the flag-gated M9 family (M9, M10, M10b, M11,
  M11c, M12, M13, M14: every one of them is M9 plus mechanisms that are OFF unless a knob turns them on, "M12
  flags-off is bit-identical to M9") is ONE era ``M9+`` and its knobs carry every effect; older versions stay their
  own eras (M3 .. M7: the base env moved between them). LEAKAGE GUARD: an era seen in fewer than ``min_era_runs``
  runs (default 2) is merged into its parent lineage (nearest older fitted version of the same family), so a
  single-run treatment can never be absorbed by "its" era effect (M8 -> M7; with grouping off, M13 -> M12,
  M11c -> M11, M10b -> M10). ``era_merges`` records what was merged where. Fitted as a grand intercept plus a per-era
  delta shrunk toward 0 with prior sd 0.02 bpb. The placeholder version ``code`` (an unknown era: the default chip
  dir was redeployed many times) is dropped.
* ``chip`` a fixed effect for every chip other than C that appears in the data (D: runtime 2.33.10, ~7-12% slower in
  full runs, so at EQUAL steps it is a small residual effect; the contract's rule that D runs are never controls for
  C is honoured because the D indicator absorbs the C/D level difference). Prior mean 0, sd 0.003.
* ``b``    the steps slope, NEGATIVE: more steps, lower bpb. Prior mean -0.057 bpb per ln(steps) (campaign: 1%
  more steps ~ -0.00057 bpb at ~2300 steps), prior sd 0.005: a STRONG prior, because within an era the step count
  varies mostly through mechanisms that also change quality (never step-normalise a mechanism that changes step
  time), so the data alone would confound b with those knobs.
* ``c``    curvature. The slope magnitude was 0.095-0.115 at ~900 steps and 0.057 at ~2300: d(slope)/dL ~ +0.051,
  so the Taylor prior is c = +0.026 (sd 0.01), i.e. bpb = era - 0.057 L + 0.026 L**2. It only matters for records
  far from 2300 steps (at 1700 steps the term is +0.0023).
* ``x_j``  the knob features of ``FEATURES`` (documented table below), each centred on the K59 reference recipe
  so that x(K59) = 0 and ``era[M9+]`` is literally "K59's recipe on the M9+ code at 2300 steps on chip C,
  seed-averaged". Ridge prior mean 0, sd ``knob_prior_sd`` = 0.004 bpb per scaled unit (one scaled unit is a
  typical campaign move, e.g. QK gain 1.44 -> 1.60, warm-up 20 -> 40, rope 100k -> 300k).
* ``u``    per-seed random effect: a column per seed shrunk toward 0 with prior sd 0.0006 (the contract's seed sd
  at 2M tokens). Jointly estimated with the fixed effects, i.e. a BLUP; seeds seen many times keep most of their
  mean residual, seeds seen once keep about 60% of it. Predicting with ``seed=None`` gives the seed-averaged mean
  and adds 0.0006**2 to the variance; an unseen seed does the same.
* ``eps``  residual, sd ``sigma`` estimated from the data (residual sum of squares over n - effective dof, after
  removing seed effects), clamped to [0.0003, 0.01]. It is model misfit + chip jitter and is added to every
  predictive variance.
* BOWL CONSTRAINT (``bowl_nonneg``, default on): a ``*_sq`` coefficient is the curvature of a TUNED knob and is
  constrained to be >= 0. Two single runs on either side of the incumbent (FF_COOLDOWN_FRAC 0.5 / 0.7, each
  -0.0001 vs its control, i.e. noise) would otherwise fit a CONCAVE bowl in which BOTH directions "improve"; a
  negative estimate is pinned at 0 and the system re-solved (active set), so the knob is then carried by its linear
  term alone (~0 +- its sd). ``pinned`` lists the columns; ``coefficients()`` flags them.

Everything is one closed-form ridge / MAP solve in numpy:
``(X'X + sigma^2 D) beta = X'y + sigma^2 D m`` with ``D = diag(1/tau_j^2)`` and prior means ``m``; the posterior
covariance ``sigma^2 (X'X + sigma^2 D)^-1`` gives the parameter part of every prediction's sd. ``sigma`` is
re-estimated for three passes (empirical Bayes for the noise only). The fit is milliseconds, so leave-one-pair-out
validation refits the model for every pair.

Knob conventions (two, deliberately)
------------------------------------
* RECORDS (fit, ``predict_record``, pair validation): a knob absent from a run's effective env (base.env +
  overrides, or the config echo of a hand-launched rehearsal) was at train_ff.py's OWN default on the chip code
  dirs, ``CODE_DEFAULTS`` = the K57 recipe with every opt-in mechanism off (LK 0, KEY_OFFSET 0, ACC_IN_GRAPH 0,
  MTP 0). The queue envs spell FF_MTP=1 / FF_KEY_OFFSET=1 / FF_ACC_IN_GRAPH=1 explicitly and never spell
  FF_LEAKY_RELU2 for a K57-recipe run, which is how a K57 control differs from an LK arm. Two single-file code
  variants carry their leaky form in the code, not the env (``LINEAGE_IMPLIED_KNOBS``: M10b = abs, M11c = max).
* CANDIDATES (``predict``): a recipe is specified relative to K59, so a missing knob is the K59 default
  (``knob_features({})`` is all zeros). Pass full knob dicts to avoid any ambiguity.

Data used
---------
FULL runs only: ``is_screen`` records are dropped (a 120-step screen predicts early loss only), as are records
without ``bpb_2m``, ``steps`` or ``code_version``, runs shorter than ``min_steps`` (a screen that was not flagged),
unknown eras, and records whose knobs are only the per-run overrides of a monitor / experiments.csv row (no chip
log, ``knobs_complete`` False): their base recipe is unknown, so their features would be wrong. Those records are
still available to ``pair_validation`` as out-of-sample arms (``holdout``).

Pairs
-----
``pair_validation`` predicts ``delta_raw = bpb(treatment) - bpb(control)`` for each validation pair with each arm's
own steps, seed, chip and code version, and reports sign agreement and mean absolute error against the spec
targets (>= 80% sign agreement, <= 0.0003 MAE), plus z-scores / 2-sd coverage and a per-family breakdown. With
fewer than ``lopo_max_pairs`` pairs it refits the model WITHOUT the two runs of the pair before predicting them
(honest leave-one-pair-out); above that it reports in-sample (say so). Pair format: ``{"treatment": run_id,
"control": run_id, "name": str, "delta_raw": float}`` where run_ids match ``RunRecord.run_id`` (``chip:run``);
``research/sim-data/validation-pairs.json`` rows (``treatment_run`` + ``chip``, ``control_run`` + ``control_chip``,
``pair_id``, ``knob_changes`` / ``control_knobs``, ``family``, ``in_task_list``) are accepted directly. A wildcard
run name (``A:*_R_g4_s67``) is resolved by chip + bpb + seed. For an arm whose record has no effective knobs, the
pair's ``knob_changes`` / ``control_knobs`` are laid over the record's knobs. ``pairs_from_records`` builds pairs
from ``control_of``. Pairs passed to ``fit`` are only remembered as the default validation set: the mixed model
already exploits the pairing (same seed + same era cancel in the difference).

Leave-one-pair-out measures REPLICATE prediction: the pair's two runs go, but the same knob values usually stay in
the fit through other seeds and pairs (0626_R_g7_s73 is the control of 12 pairs; 19 other LK=0.35 runs remain when
an lk35 pair is left out). ``pair_validation(..., leave_value_out=True)`` is the leave-one-knob-value-out (LOKVO)
report, which measures what the surrogate is used for: predicting a knob value that was NEVER run. For every knob a
feature reads that differs between the arms, the untried value is the arm's value that is not K59's reference
(the treatment's when neither is), and every fitted run carrying that value is dropped before the refit. Rows say
which values were untried, how many runs went, and whether the untried value was inside the remaining range
(``extrapolated_features`` empty: interpolation) or outside it.

Support
-------
``support`` is knob -> effective values seen (the shape ``ffsim.search`` reads); ``support_detail`` adds, per value,
the number of fitted runs and the eras they come from; ``value_support(knob, value, code_version)`` grades one
candidate value: ``seen`` (>= ``min_value_runs`` runs, at least one in the candidate's era), ``weak`` (a single run,
or runs only from other eras: a value the fit knows from M3-era runs says nothing about M12), ``interp`` (numeric,
inside the seen range, never run), ``unseen``, ``never``. ``feature_support`` flags ``era_confounded`` features: ones
that never vary INSIDE an era (FF_ACC_IN_GRAPH is off in every M3-M7 run and on in every M9+ run; MUON_ZERO2 /
ADAMW_ZERO likewise mark M6 -> M7), whose measured effect therefore lives in the era difference and whose
coefficient is the prior. ``design_row`` notes a move on such a feature. ``QualityConfig.fit_eras`` restricts the
fit to named era groups (e.g. ``("M9+",)``); the excluded runs become holdout arms.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .schema import RunRecord

__all__ = [
    "K59_DEFAULTS",
    "CODE_DEFAULTS",
    "LINEAGE_IMPLIED_KNOBS",
    "STEPS_REF",
    "Feature",
    "FEATURES",
    "FEATURE_NAMES",
    "QualityConfig",
    "QualityModel",
    "canonical_version",
    "default_era_group",
    "knob_feature_dict",
    "knob_features",
    "parse_accum_sched",
    "parse_version",
    "pairs_from_records",
    "record_knobs",
]

STEPS_REF = 2300.0

# The K59 reference recipe: the _env_* defaults of the K59 upload train.py (M12 code; not shipped) (the then
# best, official 0.9671; cold rehearsal C40 0.96092 @2357 on chip C, seed 73, code M12). Strings exactly as
# train_ff.py reads them. Every feature is centred here, so knob_features(K59_DEFAULTS) is all zeros.
K59_DEFAULTS: Dict[str, str] = {
    "FF_DEPTH": "9",
    "FF_ASPECT_RATIO": "113",
    "FF_HEAD_DIM": "256",
    "FF_MB": "8",
    "FF_TOTAL_BATCH": "262144",
    "FF_SEED": "73",
    "FF_TIME_TARGET": "1793.0",
    "FF_COOLDOWN_FRAC": "0.6",
    "FF_SOFTCAP": "15.0",
    "FF_SOFTCAP_A": "0.0",
    "FF_SOFTCAP_B": "0.0",
    "FF_EMB_LR": "0.30",
    "FF_UNEMB_LR": "0.006",
    "FF_SCALAR_LR": "0.20",
    "FF_MATRIX_LR_SCALE": "2.0",
    "FF_ADAMW_LR_SCALE": "1.4142",
    "FF_ROPE_BASE": "100000.0",
    "FF_SCALAR_BETA2": "0.95",
    "FF_WD": "0.02",
    "FF_WD_SCHED": "0.0",
    "FF_WARMUP": "20",
    "FF_MLP_MULT": "4",
    "FF_LEAKY_RELU2": "0.35",
    "FF_LEAKY_FORM": "fnm",
    "FF_RELU2_FN": "0",
    "FF_QK_GAIN": "1.44",
    "FF_MUON_BETA2": "0.9",
    "FF_KEY_OFFSET": "1",
    "FF_KEY_OFFSET_RAD": "0.3",
    "FF_MTP": "1",
    "FF_MTP_PHASES": "0.2,0.45",
    "FF_MTP_W": "1,0.5,0.25",
    "FF_XSA_LAYERS": "",
    "FF_ATTN_SRC": "",
    "FF_ROPE_FN": "0",
    "FF_RMSNORM_FN": "0",
    "FF_QK_GAIN_FOLD": "0",
    "FF_ACCUM_SCHED": "1:0.06,2:0.2,4",
    "FF_ACCUM_LR": "0.5,0.7071",
    "FF_ACC_IN_GRAPH": "1",
    "FF_MOM_PEAK": "0.95",
    "FF_MOM_RAMP": "300.0",
    "FF_CE_BF16": "0",
    "FF_PTP_W": "0",
}

# train_ff.py's OWN defaults on the chip code dirs (code_m6 .. code_k14off): the K57 recipe with every opt-in
# mechanism off. A knob absent from a record's effective env was at this value (see the module docstring).
CODE_DEFAULTS: Dict[str, str] = dict(K59_DEFAULTS)
CODE_DEFAULTS.update({"FF_LEAKY_RELU2": "0", "FF_KEY_OFFSET": "0", "FF_ACC_IN_GRAPH": "0", "FF_MTP": "0"})

# Code defaults that moved between versions. FF_KEY_OFFSET_RAD: K51 (M4) introduced the rotary key offset with
# radius 0.1; K55 (M8) moved the default to 0.3, and from M8 on every run spells 0.3 explicitly (validation-pairs
# ko0x controls: KEY_OFFSET_RAD 0.1; the M5 g1ko sweep 0.03 / 1.0 / 3.0 brackets 0.1).
VERSION_CODE_DEFAULTS: Tuple[Tuple[str, int, int, Dict[str, str]], ...] = (
    ("M", 0, 7, {"FF_KEY_OFFSET_RAD": "0.1"}),
)


def code_defaults_for(version: Optional[str]) -> Dict[str, str]:
    """``CODE_DEFAULTS`` as they stood for one code version (see ``VERSION_CODE_DEFAULTS``)."""
    pv = parse_version(version) if version else None
    if pv is None:
        return CODE_DEFAULTS
    out = dict(CODE_DEFAULTS)
    for fam, lo, hi, over in VERSION_CODE_DEFAULTS:
        if pv[0] == fam and lo <= pv[1] <= hi:
            out.update(over)
    return out


# Single-file code variants whose mechanism lives in the code, not in a knob: spell it so the features see it.
LINEAGE_IMPLIED_KNOBS: Dict[str, Dict[str, str]] = {
    "M10b": {"FF_LEAKY_FORM": "abs"},   # leaky relu^2 in the abs form (1722_R_g7lrb_s73)
    "M11c": {"FF_LEAKY_FORM": "max"},   # torch.maximum form (1939_R_g7lrm_s73)
}

_LN2 = math.log(2.0)


@dataclass(frozen=True)
class Feature:
    """One row of the knob-feature table.

    ``scale`` divides the raw (centred) transform so that one scaled unit is a typical campaign move; the ridge
    prior sd applies per scaled unit. ``prior_sd`` None means the model's ``knob_prior_sd`` (``bowl_prior_sd``
    for a bowl). A ``bowl`` feature is the square of its ``base`` feature: the curvature of bpb in a TUNED scalar
    knob, given a positive prior mean (``bowl_prior_mean``: the incumbent value sits near an optimum, so moving
    either way costs) instead of 0.
    """
    name: str
    knobs: Tuple[str, ...]
    reference: str
    transform: str
    scale: float
    prior_sd: Optional[float] = None
    bowl: bool = False
    base: Optional[str] = None


# The documented feature table: the knobs that actually varied in the last two weeks of full runs (chips A-D,
# K44 .. LK0.35 eras), with numeric encodings centred on K59. Order = column order in the design matrix.
FEATURES: Tuple[Feature, ...] = (
    Feature("leaky", ("FF_LEAKY_RELU2",), "0.35", "LK - 0.35 (leaky relu^2 slope; 0 = plain relu^2)", 0.15),
    Feature("leaky_sq", ("FF_LEAKY_RELU2",), "0.35", "(LK - 0.35)^2: the slope curve is a bowl (0 / 0.25 / 0.35 / 0.5 "
            "gave +0.0013 / +0.0003 / 0 / +0.0014 at s73)", 0.15 ** 2),
    Feature("leaky_form_abs", ("FF_LEAKY_FORM", "FF_LEAKY_RELU2"), "fnm", "1 if LK > 0 and form is 'abs' (M10b); fn/fnm = 0", 1.0),
    Feature("leaky_form_max", ("FF_LEAKY_FORM", "FF_LEAKY_RELU2"), "fnm", "1 if LK > 0 and form is 'max' (M11c, torch.maximum)", 1.0),
    Feature("relu2_fn", ("FF_RELU2_FN",), "0", "1 if FF_RELU2_FN (the RF arm: relu^2 with a hand-written backward, LK = 0)", 1.0),
    Feature("softcap_log", ("FF_SOFTCAP",), "15.0", "ln(SOFTCAP / 15) when SOFTCAP > 0, else 0", math.log(18.0 / 15.0)),
    Feature("softcap_off", ("FF_SOFTCAP",), "15.0", "1 if SOFTCAP == 0 (no logit cap)", 1.0),
    Feature("softcap_a", ("FF_SOFTCAP_A", "FF_SOFTCAP"), "0.0", "A_eff / SOFTCAP - 1 with A_eff = SOFTCAP_A if > 0 else "
            "SOFTCAP (asymmetric cap A*tanh(z/C): 16.5 on 15 -> 0.1)", 0.1),
    Feature("softcap_b", ("FF_SOFTCAP_B",), "0.0", "SOFTCAP_B (logit shift inside the tanh)", 5.0),
    Feature("wd_sched", ("FF_WD_SCHED",), "0.0", "WD_SCHED power (muon wd = WD * lrm ** WD_SCHED; 0 = constant wd; "
            "non-numeric -> 1)", 1.0),
    Feature("mom_peak", ("FF_MOM_PEAK",), "0.95", "MOM_PEAK - 0.95 (Muon momentum peak; 0.93 / 0.97 were tried)", 0.02),
    Feature("muon_beta2", ("FF_MUON_BETA2",), "0.9", "MUON_BETA2 - 0.9 (0.8 and 0.98 were tried)", 0.05),
    Feature("rope_log", ("FF_ROPE_BASE",), "100000.0", "ln(ROPE_BASE / 1e5) (300k tried)", math.log(3.0)),
    Feature("scalar_beta2", ("FF_SCALAR_BETA2",), "0.95", "SCALAR_BETA2 - 0.95 (0.99 tried)", 0.04),
    Feature("qk_gain", ("FF_QK_GAIN",), "1.44", "QK_GAIN - 1.44 (1.6 tried)", 0.16),
    Feature("warmup_log", ("FF_WARMUP",), "20", "ln(WARMUP / 20) (40 tried)", _LN2),
    Feature("emb_lr_log", ("FF_EMB_LR",), "0.30", "ln(EMB_LR / 0.30) (0.36 tried)", math.log(1.2)),
    Feature("unemb_lr_log", ("FF_UNEMB_LR",), "0.006", "ln(UNEMB_LR / 0.006) (0.0072 tried)", math.log(1.2)),
    Feature("scalar_lr_log", ("FF_SCALAR_LR",), "0.20", "ln(SCALAR_LR / 0.20) (0.1664 tried with ADAMW_LR 1.7)", math.log(1.2)),
    Feature("matrix_lr_log", ("FF_MATRIX_LR_SCALE",), "2.0", "ln(MATRIX_LR_SCALE / 2.0) (1.75 / 2.3 tried)", math.log(1.15)),
    Feature("adamw_lr_log", ("FF_ADAMW_LR_SCALE",), "1.4142", "ln(ADAMW_LR_SCALE / 1.4142) (1.7 tried)", math.log(1.2)),
    Feature("wd_log", ("FF_WD",), "0.02", "ln(WD / 0.02) (0.012 / 0.03 tried)", math.log(1.5)),
    Feature("key_offset_off", ("FF_KEY_OFFSET",), "1", "1 if the rotary key offset is off (pre-K51 recipes)", 1.0),
    Feature("key_offset_log", ("FF_KEY_OFFSET_RAD", "FF_KEY_OFFSET"), "0.3", "ln(radius / 0.3) when the key offset is on "
            "(0.03 / 0.1 / 0.2 / 0.5 / 1.0 / 3.0 were tried: two decades, so a log scale), 0 when off", math.log(3.0)),
    Feature("acc_in_graph", ("FF_ACC_IN_GRAPH",), "1", "ACC_IN_GRAPH - 1 (K56 mechanism: -0.0006 over 3 pairs, +1% steps)", 1.0),
    Feature("accum_until", ("FF_ACCUM_SCHED",), "1:0.06,2:0.2,4", "charged-time fraction at which the batch warm-up "
            "reaches the final k, minus 0.2 (plain 'K' -> -0.2; '2:0.2,4' and RAMP3 -> 0)", 0.2),
    Feature("accum_k1_until", ("FF_ACCUM_SCHED",), "1:0.06,2:0.2,4", "first-phase (k1) end fraction minus 0.06 "
            "(RAMP3 -> 0; two-phase or none -> -0.06)", 0.06),
    Feature("accum_lr0", ("FF_ACCUM_LR", "FF_ACCUM_SCHED"), "0.5,0.7071", "LR scale of the first small-batch phase "
            "minus 0.5 (0 when no batch warm-up)", 0.25),
    Feature("accum_lr1", ("FF_ACCUM_LR", "FF_ACCUM_SCHED"), "0.5,0.7071", "LR scale of the second small-batch phase "
            "minus 0.7071 (0 unless RAMP3)", 0.25),
    Feature("cooldown_frac", ("FF_COOLDOWN_FRAC",), "0.6", "COOLDOWN_FRAC - 0.6 (0.45 / 0.55 / 0.65 tried)", 0.1),
    Feature("mtp_off", ("FF_MTP",), "1", "1 if FF_MTP == 0 (no multi-token head)", 1.0),
    Feature("mtp_p1", ("FF_MTP_PHASES", "FF_MTP"), "0.2,0.45", "MTP fade phase 1 - 0.2 (0 when MTP off)", 0.1),
    Feature("mtp_p2", ("FF_MTP_PHASES", "FF_MTP"), "0.2,0.45", "MTP fade phase 2 - 0.45 (0 when MTP off; 0.5 tried)", 0.1),
    Feature("mtp_w_tail", ("FF_MTP_W", "FF_MTP"), "1,0.5,0.25", "sum of the MTP head weights after the first, minus 0.75 "
            "('1,0.6,0.3' -> 0.15; 0 when MTP off)", 0.15),
    Feature("xsa_on", ("FF_XSA_LAYERS",), "", "1 if FF_XSA_LAYERS is non-empty", 1.0),
    Feature("xsa_frac", ("FF_XSA_LAYERS", "FF_DEPTH"), "", "fraction of blocks with XSA ('all' -> 1; '0,1,2,3' on d9 -> 0.44)", 1.0),
    Feature("attn_src", ("FF_ATTN_SRC",), "", "1 if FF_ATTN_SRC is non-empty (attention-source reuse, M14 '5:6,7,8')", 1.0),
    Feature("fn_flags", ("FF_ROPE_FN", "FF_RMSNORM_FN", "FF_QK_GAIN_FOLD"), "0", "fraction of the M13 autograd.Function "
            "flags on (step-time mechanisms, quality-neutral by design: prior sd 0.001)", 1.0, 0.001),
    Feature("ce_bf16", ("FF_CE_BF16",), "0", "1 if the cross-entropy runs in bf16 (G3 probe, rejected)", 1.0),
    Feature("ptp_w", ("FF_PTP_W",), "0", "PTP_W (previous-token-prediction head weight; 0.25 tried)", 0.25),
    Feature("mom_ramp_log", ("FF_MOM_RAMP",), "300.0", "ln(MOM_RAMP / 300) (450 tried)", math.log(1.5)),
    Feature("depth", ("FF_DEPTH",), "9", "DEPTH - 9 (8 / 10 / 12 tried)", 1.0),
    Feature("aspect_log", ("FF_ASPECT_RATIO",), "113", "ln(ASPECT_RATIO / 113) (width = aspect * depth)", math.log(1.1)),
    Feature("total_batch_log2", ("FF_TOTAL_BATCH",), "262144", "log2(TOTAL_BATCH / 262144)", 1.0),
    Feature("mlp_mult", ("FF_MLP_MULT",), "4", "MLP_MULT - 4", 1.0),
    Feature("time_target_log", ("FF_TIME_TARGET",), "1793.0", "ln(TIME_TARGET / 1793): mostly absorbed by ln(steps), so "
            "its prior sd is 0.001 and it only catches what the step count misses", math.log(1.1), 0.001),
)

# Tuned scalar knobs get a curvature ("bowl") term: the square of the centred feature, per scaled unit^2. The data
# shows the pattern wherever both directions were tried (LK 0 and 0.5, MOM_PEAK 0.93 and 0.97, WD 0.012 and 0.03,
# MATRIX_LR 1.75 and 2.3 are all worse than the incumbent), and a linear term cannot express it.
BOWL_BASES: Tuple[str, ...] = (
    "mom_peak", "qk_gain", "cooldown_frac", "muon_beta2", "scalar_beta2", "rope_log", "warmup_log", "emb_lr_log",
    "unemb_lr_log", "scalar_lr_log", "matrix_lr_log", "adamw_lr_log", "wd_log", "key_offset_log", "softcap_log",
)
_BASE_FEATURES = {f.name: f for f in FEATURES}
FEATURES = tuple(f if f.name != "leaky_sq" else replace(f, bowl=True, base="leaky") for f in FEATURES) + tuple(
    Feature(f"{b}_sq", _BASE_FEATURES[b].knobs, _BASE_FEATURES[b].reference, f"({b})^2: curvature of a tuned knob",
            _BASE_FEATURES[b].scale ** 2, None, True, b) for b in BOWL_BASES)
FEATURE_NAMES: Tuple[str, ...] = tuple(f.name for f in FEATURES)
_FEATURE_INDEX = {f.name: i for i, f in enumerate(FEATURES)}
FEATURE_KNOBS: Tuple[str, ...] = tuple(sorted({k for f in FEATURES for k in f.knobs}))


# ----------------------------------------------------------------------------- knob reading


def _raw(knobs: Dict[str, Any], name: str, defaults: Dict[str, str]) -> Optional[str]:
    v = knobs.get(name)
    if v is None:
        v = defaults.get(name)
    if v is None:
        return None
    return str(v).strip()


def _float(knobs: Dict[str, Any], name: str, default: float, defaults: Dict[str, str]) -> float:
    v = _raw(knobs, name, defaults)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _flag(knobs: Dict[str, Any], name: str, default: bool, defaults: Dict[str, str]) -> bool:
    """train_ff.py _env_flag semantics: 1/true/yes/on are True, everything else False."""
    v = _raw(knobs, name, defaults)
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on", "y", "t")


def parse_accum_sched(raw: Optional[str]) -> Tuple[Optional[int], float, Optional[int], float, int]:
    """Parse FF_ACCUM_SCHED ('K', 'k0:u0,K' or 'k0:u0,k1:u1,K') -> (k0, u0, k1, u1, K).

    k0/k1 are None when that phase does not exist (u = 0.0 then). Unparseable input is read as the plain final
    batch 'K=4' (no warm-up).
    """
    if raw is None:
        raw = K59_DEFAULTS["FF_ACCUM_SCHED"]
    parts = [p.strip() for p in str(raw).replace(" ", "").split(",") if p.strip()]
    try:
        if len(parts) == 1:
            return None, 0.0, None, 0.0, int(float(parts[0]))
        if len(parts) == 2:
            k0, u0 = parts[0].split(":")
            return int(k0), float(u0), None, 0.0, int(float(parts[1]))
        if len(parts) == 3:
            k0, u0 = parts[0].split(":")
            k1, u1 = parts[1].split(":")
            return int(k0), float(u0), int(k1), float(u1), int(float(parts[2]))
    except ValueError:
        pass
    return None, 0.0, None, 0.0, 4


def _xsa_fraction(raw: Optional[str], depth: int) -> Tuple[float, float]:
    s = (raw or "").replace(" ", "").lower()
    if not s:
        return 0.0, 0.0
    if s == "all":
        return 1.0, 1.0
    layers = {p for p in s.split(",") if p}
    return 1.0, min(1.0, len(layers) / max(depth, 1))


def _floats(raw: Optional[str], fallback: List[float]) -> List[float]:
    try:
        vals = [float(v) for v in (raw or "").split(",") if v.strip()]
    except ValueError:
        return list(fallback)
    return vals if vals else list(fallback)


def knob_feature_dict(knobs: Optional[Dict[str, Any]], defaults: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    """Raw (centred, unscaled) feature values for one knob dict.

    ``defaults`` supplies missing knobs: ``K59_DEFAULTS`` (the default: a candidate relative to K59) or
    ``CODE_DEFAULTS`` (a record's effective env, see the module docstring).
    """
    k = knobs or {}
    d = K59_DEFAULTS if defaults is None else defaults
    out: Dict[str, float] = {}

    leaky = _float(k, "FF_LEAKY_RELU2", 0.35, d)
    form = (_raw(k, "FF_LEAKY_FORM", d) or "fnm").lower()
    out["leaky"] = leaky - 0.35
    out["leaky_sq"] = (leaky - 0.35) ** 2
    out["leaky_form_abs"] = 1.0 if (leaky > 0.0 and form == "abs") else 0.0
    out["leaky_form_max"] = 1.0 if (leaky > 0.0 and form == "max") else 0.0
    out["relu2_fn"] = 1.0 if _flag(k, "FF_RELU2_FN", False, d) else 0.0

    softcap = _float(k, "FF_SOFTCAP", 15.0, d)
    out["softcap_log"] = math.log(softcap / 15.0) if softcap > 0 else 0.0
    out["softcap_off"] = 1.0 if softcap <= 0 else 0.0
    cap_a = _float(k, "FF_SOFTCAP_A", 0.0, d)
    out["softcap_a"] = (cap_a / softcap - 1.0) if (softcap > 0 and cap_a > 0) else 0.0
    out["softcap_b"] = _float(k, "FF_SOFTCAP_B", 0.0, d)

    wd_raw = _raw(k, "FF_WD_SCHED", d) or "0"
    try:
        out["wd_sched"] = float(wd_raw)
    except ValueError:
        out["wd_sched"] = 1.0
    out["mom_peak"] = _float(k, "FF_MOM_PEAK", 0.95, d) - 0.95
    out["muon_beta2"] = _float(k, "FF_MUON_BETA2", 0.9, d) - 0.9
    out["rope_log"] = math.log(max(_float(k, "FF_ROPE_BASE", 1e5, d), 1.0) / 1e5)
    out["scalar_beta2"] = _float(k, "FF_SCALAR_BETA2", 0.95, d) - 0.95
    out["qk_gain"] = _float(k, "FF_QK_GAIN", 1.44, d) - 1.44
    out["warmup_log"] = math.log(max(_float(k, "FF_WARMUP", 20.0, d), 1.0) / 20.0)
    out["emb_lr_log"] = math.log(max(_float(k, "FF_EMB_LR", 0.30, d), 1e-9) / 0.30)
    out["unemb_lr_log"] = math.log(max(_float(k, "FF_UNEMB_LR", 0.006, d), 1e-9) / 0.006)
    out["scalar_lr_log"] = math.log(max(_float(k, "FF_SCALAR_LR", 0.20, d), 1e-9) / 0.20)
    out["matrix_lr_log"] = math.log(max(_float(k, "FF_MATRIX_LR_SCALE", 2.0, d), 1e-9) / 2.0)
    out["adamw_lr_log"] = math.log(max(_float(k, "FF_ADAMW_LR_SCALE", 1.4142, d), 1e-9) / 1.4142)
    out["wd_log"] = math.log(max(_float(k, "FF_WD", 0.02, d), 1e-9) / 0.02)
    ko_on = _flag(k, "FF_KEY_OFFSET", True, d)
    out["key_offset_off"] = 0.0 if ko_on else 1.0
    out["key_offset_log"] = math.log(max(_float(k, "FF_KEY_OFFSET_RAD", 0.3, d), 1e-3) / 0.3) if ko_on else 0.0
    out["acc_in_graph"] = _float(k, "FF_ACC_IN_GRAPH", 1.0, d) - 1.0

    k0, u0, k1, u1, _kf = parse_accum_sched(_raw(k, "FF_ACCUM_SCHED", d))
    until_final = (u1 if k1 is not None else u0) if k0 is not None else 0.0
    out["accum_until"] = until_final - 0.2
    out["accum_k1_until"] = (u0 if k1 is not None else 0.0) - 0.06
    lrs = _floats(_raw(k, "FF_ACCUM_LR", d) or "0.5,0.7071", [0.5, 0.7071])
    if k0 is None:
        out["accum_lr0"] = 0.0
        out["accum_lr1"] = 0.0
    else:
        out["accum_lr0"] = lrs[0] - 0.5
        out["accum_lr1"] = ((lrs[1] if len(lrs) > 1 else 0.7071) - 0.7071) if k1 is not None else 0.0
    out["cooldown_frac"] = _float(k, "FF_COOLDOWN_FRAC", 0.6, d) - 0.6

    mtp = _float(k, "FF_MTP", 1.0, d)
    out["mtp_off"] = 1.0 if mtp == 0 else 0.0
    p = _floats(_raw(k, "FF_MTP_PHASES", d) or "0.2,0.45", [0.2, 0.45])
    if mtp == 0 or len(p) < 2:
        out["mtp_p1"], out["mtp_p2"] = 0.0, 0.0
    else:
        out["mtp_p1"], out["mtp_p2"] = p[0] - 0.2, p[1] - 0.45
    w = _floats(_raw(k, "FF_MTP_W", d) or "1,0.5,0.25", [1.0, 0.5, 0.25])
    out["mtp_w_tail"] = 0.0 if mtp == 0 else (sum(w[1:]) - 0.75)

    depth = int(_float(k, "FF_DEPTH", 9.0, d))
    out["xsa_on"], out["xsa_frac"] = _xsa_fraction(_raw(k, "FF_XSA_LAYERS", d), depth)
    out["attn_src"] = 1.0 if (_raw(k, "FF_ATTN_SRC", d) or "").strip() else 0.0
    out["fn_flags"] = sum(1.0 for n in ("FF_ROPE_FN", "FF_RMSNORM_FN", "FF_QK_GAIN_FOLD") if _flag(k, n, False, d)) / 3.0
    out["ce_bf16"] = 1.0 if _flag(k, "FF_CE_BF16", False, d) else 0.0
    out["ptp_w"] = _float(k, "FF_PTP_W", 0.0, d)
    out["mom_ramp_log"] = math.log(max(_float(k, "FF_MOM_RAMP", 300.0, d), 1.0) / 300.0)
    out["depth"] = depth - 9.0
    out["aspect_log"] = math.log(max(_float(k, "FF_ASPECT_RATIO", 113.0, d), 1.0) / 113.0)
    out["total_batch_log2"] = math.log2(max(_float(k, "FF_TOTAL_BATCH", 262144.0, d), 1.0) / 262144.0)
    out["mlp_mult"] = _float(k, "FF_MLP_MULT", 4.0, d) - 4.0
    out["time_target_log"] = math.log(max(_float(k, "FF_TIME_TARGET", 1793.0, d), 1.0) / 1793.0)
    for f in FEATURES:
        if f.bowl and f.base and f.name not in out:
            out[f.name] = out[f.base] ** 2
    return out


def knob_features(knobs: Optional[Dict[str, Any]], defaults: Optional[Dict[str, str]] = None) -> np.ndarray:
    """Scaled feature vector in FEATURES order (raw / scale). All zeros for the K59 recipe."""
    d = knob_feature_dict(knobs, defaults)
    return np.array([d[f.name] / f.scale for f in FEATURES], dtype=float)


# ----------------------------------------------------------------------------- code-version lineage

_VERSION_RE = re.compile(r"^(?:code[_-]?)?([A-Za-z]+)[_-]?(\d+)([A-Za-z0-9_-]*)$")


def parse_version(code_version: Optional[str]) -> Optional[Tuple[str, int, str]]:
    """'M12' -> ('M', 12, ''); 'M11c' -> ('M', 11, 'c'); 'code_m9' -> ('M', 9, ''); 'K57' -> ('K', 57, '').

    None when the string has no family+number shape.
    """
    if not code_version:
        return None
    m = _VERSION_RE.match(str(code_version).strip())
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3).lower()


def canonical_version(code_version: Optional[str], tags: Optional[Iterable[str]] = None) -> Optional[str]:
    """The train_ff.py lineage name of a record: its ``lineage:`` tag (chip dir code_k12off -> M12) when present,
    else the M-family spelling of ``code_version`` (code_m9 -> M9, M11c stays), else the raw string."""
    for t in tags or ():
        if isinstance(t, str) and t.startswith("lineage:") and t[8:]:
            return t[8:]
    if not code_version:
        return None
    s = str(code_version).strip()
    pv = parse_version(s)
    if pv is not None and pv[0] == "M":
        return f"M{pv[1]}{pv[2]}"
    return s


def default_era_group(version: Optional[str]) -> Optional[str]:
    """The default era grouping: every M-version from M9 on is the flag-gated ``M9+`` family; others are themselves."""
    if version is None:
        return None
    pv = parse_version(version)
    if pv is not None and pv[0] == "M" and pv[1] >= 9:
        return "M9+"
    return version


def record_knobs(record: RunRecord, code_version: Optional[str] = None) -> Dict[str, str]:
    """A record's explicit knobs plus what its code lineage implies (``LINEAGE_IMPLIED_KNOBS``); explicit wins."""
    version = code_version if code_version is not None else canonical_version(record.code_version, record.tags)
    out: Dict[str, str] = dict(LINEAGE_IMPLIED_KNOBS.get(version or "", {}))
    out.update({str(k): str(v) for k, v in (record.knobs or {}).items() if v is not None})
    return out


def pairs_from_records(records: Iterable[RunRecord]) -> List[Dict[str, Any]]:
    """Validation pairs from ``RunRecord.control_of`` (treatment -> its control run_id)."""
    by_id = {r.run_id: r for r in records}
    pairs: List[Dict[str, Any]] = []
    for r in by_id.values():
        if r.control_of and r.control_of in by_id:
            c = by_id[r.control_of]
            pair: Dict[str, Any] = {"name": f"{r.run} vs {c.run}", "treatment": r.run_id, "control": c.run_id}
            if r.bpb_2m is not None and c.bpb_2m is not None:
                pair["delta_raw"] = r.bpb_2m - c.bpb_2m
            pairs.append(pair)
    return pairs


# ----------------------------------------------------------------------------- the model


@dataclass
class QualityConfig:
    """Hyperparameters (prior means / sds in bpb units). See the module docstring for the reasoning."""
    steps_ref: float = STEPS_REF
    slope_prior_mean: float = -0.057    # d bpb / d ln(steps): negative, more steps -> lower bpb
    slope_prior_sd: float = 0.005
    curvature_prior_mean: float = 0.026  # d2 bpb / d ln(steps)^2 / 2: the slope flattens with more steps
    curvature_prior_sd: float = 0.01
    knob_prior_sd: float = 0.004
    bowl_prior_mean: float = 0.0005    # curvature of bpb in a tuned scalar knob, per scaled unit^2: an optimum
    bowl_prior_sd: float = 0.001
    bowl_nonneg: bool = True           # a bowl is >= 0: a concave estimate (both directions improve) is pinned at 0
    era_prior_sd: float = 0.02
    chip_prior_sd: float = 0.003
    seed_prior_sd: float = 0.0006
    sigma_init: float = 0.001
    sigma_floor: float = 0.0003
    sigma_ceiling: float = 0.01
    sigma_iters: int = 3
    unseen_era_sd: float = 0.0015      # per generation of lineage distance (variance scales with distance)
    unseen_family_sd: float = 0.004    # no version of the same family was ever fitted
    min_steps: int = 500               # shorter runs are screens whatever their flag says
    reference_chip: str = "C"
    lopo_max_pairs: int = 120          # leave-one-pair-out below this many pairs
    noise_floor: float = 0.0003        # |delta| below this is inside seed noise ("decisive" pairs are above it)
    # era handling
    era_groups: Optional[Dict[str, str]] = None   # None = default_era_group (M9.. -> "M9+"); {} = every version its own era
    min_era_runs: int = 2              # leakage guard: an era with fewer runs is merged into its parent lineage
    unknown_eras: Tuple[str, ...] = ("code",)     # placeholder versions that are not one era: dropped
    fit_eras: Optional[Tuple[str, ...]] = None    # restrict the fit to these era groups (("M9+",)); others -> holdout
    min_value_runs: int = 2            # a knob value seen in fewer fitted runs is 'weak' support for a candidate
    # knob conventions
    record_defaults: str = "code"      # "code": a record's missing knob = CODE_DEFAULTS; "k59": = K59_DEFAULTS
    require_effective_knobs: bool = True  # drop monitor-/csv-only records (per-run overrides, unknown base recipe)


@dataclass
class _Row:
    record: RunRecord
    y: float
    version: str
    era: str
    chip: str
    seed: Optional[int]
    L: float
    x: np.ndarray
    knobs: Dict[str, str]        # explicit knobs (+ lineage-implied)
    effective: Dict[str, str]    # knobs with every feature knob filled from the record's code defaults


def _effective_knobs(knobs: Dict[str, str], defaults: Dict[str, str]) -> Dict[str, str]:
    eff = dict(knobs)
    for k in FEATURE_KNOBS:
        v = _raw(knobs, k, defaults)
        if v is not None:
            eff[k] = v
    return eff


_IGNORED_KNOBS = ("FF_SEED", "FF_CODE_DIR", "FF_TIMEOUT", "FF_CC_FLAGS")
_PIN_SD = 1e-8   # prior sd that pins a constrained bowl coefficient at 0 (sigma^2 / _PIN_SD^2 >> any X'X entry)


def _same_knob_value(a: Optional[Any], b: Optional[Any]) -> bool:
    """Two knob spellings of the same value ('0.60' == '0.6', '1' == '1.0', '2:0.2,4' == '2:0.2, 4')."""
    if a is None or b is None:
        return a is b
    sa, sb = str(a).strip(), str(b).strip()
    if sa == sb:
        return True
    try:
        return math.isclose(float(sa), float(sb), rel_tol=1e-9, abs_tol=1e-12)
    except ValueError:
        return sa.replace(" ", "").lower() == sb.replace(" ", "").lower()


class QualityModel:
    """Ridge/MAP surrogate for raw bpb_2m. ``fit`` then ``predict`` / ``pair_validation`` / ``summary``."""

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.fitted = False
        self.columns: List[str] = []
        self.beta = np.zeros(0)
        self.cov = np.zeros((0, 0))
        self.sigma = self.config.sigma_init
        self.eras: Dict[str, int] = {}
        self.chips: Dict[str, int] = {}
        self.seeds: Dict[int, int] = {}
        self.era_dates: Dict[str, str] = {}
        self.era_counts: Dict[str, int] = {}
        self.era_members: Dict[str, List[str]] = {}
        self.era_merges: Dict[str, str] = {}
        self.version_to_era: Dict[str, str] = {}
        self.version_aliases: Dict[str, str] = {}
        self.n = 0
        self.edf = 0.0
        self.pinned: List[str] = []              # bowl columns pinned at 0 by the non-negativity constraint
        self._era_confounded: List[str] = []     # features that never vary inside an era (see feature_support)
        self.dropped: Dict[str, int] = {}
        self.fit_records: List[RunRecord] = []
        self.records_by_id: Dict[str, RunRecord] = {}
        self.holdout: Dict[str, RunRecord] = {}
        self.pairs: List[Dict[str, Any]] = []
        self._rows: List[_Row] = []
        self._prior_mean = np.zeros(0)
        self._prior_sd = np.zeros(0)
        self._i_intercept = 0
        self._i_era0 = 0
        self._i_chip0 = 0
        self._i_L = 0
        self._i_L2 = 0
        self._i_knob0 = 0
        self._i_seed0 = 0

    # ------------------------------------------------------------------ data selection

    def _record_defaults(self, version: Optional[str] = None) -> Dict[str, str]:
        if self.config.record_defaults == "k59":
            return K59_DEFAULTS
        return code_defaults_for(version) if version else CODE_DEFAULTS

    def _group_of(self, version: str) -> str:
        groups = self.config.era_groups
        if groups is None:
            return default_era_group(version) or version
        return groups.get(version, version)

    def _select(self, records: Iterable[RunRecord]) -> List[_Row]:
        cfg = self.config
        rows: List[_Row] = []
        dropped = {"is_screen": 0, "no_bpb": 0, "no_steps": 0, "short": 0, "no_code_version": 0, "unknown_era": 0,
                   "no_effective_knobs": 0, "era_excluded": 0}
        self.holdout = {}
        for r in records:
            if r.is_screen:
                dropped["is_screen"] += 1
                continue
            if r.bpb_2m is None or not math.isfinite(float(r.bpb_2m)):
                dropped["no_bpb"] += 1
                continue
            if r.steps is None or r.steps <= 0:
                dropped["no_steps"] += 1
                continue
            if r.steps < cfg.min_steps:
                dropped["short"] += 1
                continue
            version = canonical_version(r.code_version, r.tags)
            if not version:
                dropped["no_code_version"] += 1
                continue
            if version in cfg.unknown_eras or str(r.code_version) in cfg.unknown_eras:
                dropped["unknown_era"] += 1
                continue
            if cfg.require_effective_knobs and r.sources and not r.knobs_complete and "chip_log" not in r.sources:
                dropped["no_effective_knobs"] += 1
                self.holdout[r.run_id] = r
                continue
            if cfg.fit_eras is not None and self._group_of(version) not in cfg.fit_eras and version not in cfg.fit_eras:
                dropped["era_excluded"] += 1        # fit restricted to named eras: the run is a holdout arm
                self.holdout[r.run_id] = r
                continue
            knobs = record_knobs(r, version)
            effective = _effective_knobs(knobs, self._record_defaults(version))
            rows.append(_Row(r, float(r.bpb_2m), version, self._group_of(version), str(r.chip or cfg.reference_chip),
                             int(r.seed) if r.seed is not None else None,
                             math.log(float(r.steps) / cfg.steps_ref), knob_features(effective), knobs, effective))
        self.dropped = dropped
        return rows

    @staticmethod
    def _version_distance(a: str, b: str) -> Optional[float]:
        """Distance between two versions of the same family: generations apart, with a suffix-only difference
        (M11c vs M11) at 0.5 so that it ranks closer than the previous generation but still counts as one
        generation when reported; None across families."""
        pa, pb = parse_version(a), parse_version(b)
        if pa is None or pb is None or pa[0] != pb[0]:
            return None
        d = abs(pa[1] - pb[1])
        if d == 0 and pa[2] != pb[2]:
            return 0.5
        return float(d)

    def _nearest_era(self, version: str, eras: Dict[str, List[str]]) -> Optional[Tuple[str, int]]:
        """(era, generations) of the fitted member version nearest to ``version`` in the same family, preferring the
        older neighbour; None when no era has a member of that family."""
        pv = parse_version(version)
        best: Optional[Tuple[Tuple[float, int, int, str], str, int]] = None
        for era, members in eras.items():
            for m in members:
                dist = self._version_distance(version, m)
                if dist is None:
                    continue
                pm = parse_version(m)
                key = (dist, 0 if (pm is not None and pv is not None and pm[1] <= pv[1]) else 1,
                       0 if (pm is not None and pv is not None and pm[2] == pv[2]) else 1, era)
                if best is None or key < best[0]:
                    best = (key, era, max(1, int(math.ceil(dist))))
        return None if best is None else (best[1], best[2])

    def _assign_eras(self, rows: List[_Row]) -> None:
        """Group versions into eras, then merge every era with < min_era_runs runs into its parent lineage."""
        cfg = self.config
        members: Dict[str, List[str]] = {}
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row.era] = counts.get(row.era, 0) + 1
            if row.version not in members.setdefault(row.era, []):
                members[row.era].append(row.version)
        merges: Dict[str, str] = {}
        big = {e: list(ms) for e, ms in members.items() if counts[e] >= cfg.min_era_runs}
        for era in sorted(members):
            if counts[era] >= cfg.min_era_runs or not big:
                continue
            target = None
            for v in members[era]:
                near = self._nearest_era(v, big)
                if near is not None:
                    target = near[0]
                    break
            if target is None:
                continue          # no fitted relative: the thin era stays its own (flagged in era_counts)
            merges[era] = target
        for row in rows:
            if row.era in merges:
                row.era = merges[row.era]
        self.era_merges = merges
        self.era_members = {}
        self.version_to_era = {}
        self.version_aliases = {}
        for row in rows:
            ms = self.era_members.setdefault(row.era, [])
            if row.version not in ms:
                ms.append(row.version)
            self.version_to_era[row.version] = row.era
            raw = str(row.record.code_version or "")
            if raw and raw != row.version:
                self.version_aliases[raw] = row.version

    # ------------------------------------------------------------------ fit

    def fit(self, records: Sequence[RunRecord], pairs: Optional[List[Dict[str, Any]]] = None) -> "QualityModel":
        """Fit on the FULL runs among ``records``. ``pairs`` are stored as the default validation set only."""
        rows = self._select(records)
        if len(rows) < 2:
            raise ValueError(f"QualityModel.fit needs at least 2 full runs with bpb_2m, got {len(rows)} "
                             f"(dropped: {self.dropped})")
        return self._fit_rows(rows, pairs)

    def refit_without(self, run_ids: Iterable[str]) -> "QualityModel":
        """A fresh model of the same config fitted on this model's rows minus ``run_ids`` (identical to
        ``QualityModel(config).fit(kept records)``, but the rows' features are reused, which makes the
        leave-one-pair-out loop ~50x faster). Holdout records are carried over; pairs are not."""
        if not self.fitted:
            raise RuntimeError("QualityModel.refit_without before fit")
        drop = set(run_ids)
        rows = [replace(row, era=self._group_of(row.version)) for row in self._rows if row.record.run_id not in drop]
        if len(rows) < 2:
            raise ValueError(f"QualityModel.refit_without leaves {len(rows)} runs: need at least 2")
        child = QualityModel(replace(self.config))
        child.dropped = dict(self.dropped)
        child.holdout = dict(self.holdout)
        return child._fit_rows(rows, [])

    def _fit_rows(self, rows: List[_Row], pairs: Optional[List[Dict[str, Any]]]) -> "QualityModel":
        cfg = self.config
        self._assign_eras(rows)
        self._rows = rows
        self.fit_records = [row.record for row in rows]
        self.records_by_id = {r.run_id: r for r in self.fit_records}
        self.pairs = list(pairs) if pairs is not None else pairs_from_records(self.fit_records)

        eras = sorted({row.era for row in rows})
        chips = sorted({row.chip for row in rows if row.chip != cfg.reference_chip})
        seeds = sorted({row.seed for row in rows if row.seed is not None})
        self.eras = {e: i for i, e in enumerate(eras)}
        self.chips = {c: i for i, c in enumerate(chips)}
        self.seeds = {s: i for i, s in enumerate(seeds)}
        self.era_counts = {e: sum(1 for row in rows if row.era == e) for e in eras}
        self.era_dates = {}
        for row in rows:
            d = row.record.date_utc
            if d and (row.era not in self.era_dates or d > self.era_dates[row.era]):
                self.era_dates[row.era] = d

        n_knob = len(FEATURES)
        columns = ["intercept"] + [f"era:{e}" for e in eras] + [f"chip:{c}" for c in chips] + ["ln_steps", "ln_steps_sq"] \
            + [f"knob:{f.name}" for f in FEATURES] + [f"seed:{s}" for s in seeds]
        p = len(columns)
        self.columns = columns
        self._i_intercept = 0
        i_era0 = 1
        i_chip0 = i_era0 + len(eras)
        self._i_L = i_chip0 + len(chips)
        self._i_L2 = self._i_L + 1
        self._i_knob0 = self._i_L2 + 1
        i_seed0 = self._i_knob0 + n_knob
        self._i_era0, self._i_chip0, self._i_seed0 = i_era0, i_chip0, i_seed0

        n = len(rows)
        X = np.zeros((n, p))
        y = np.array([row.y for row in rows])
        for i, row in enumerate(rows):
            X[i, 0] = 1.0
            X[i, i_era0 + self.eras[row.era]] = 1.0
            if row.chip in self.chips:
                X[i, i_chip0 + self.chips[row.chip]] = 1.0
            X[i, self._i_L] = row.L
            X[i, self._i_L2] = row.L * row.L
            X[i, self._i_knob0:i_seed0] = row.x
            if row.seed is not None:
                X[i, i_seed0 + self.seeds[row.seed]] = 1.0

        m = np.zeros(p)
        tau = np.zeros(p)
        m[0], tau[0] = float(np.mean(y)), 1.0                     # grand intercept: effectively unpenalised
        tau[i_era0:i_chip0] = cfg.era_prior_sd
        tau[i_chip0:self._i_L] = cfg.chip_prior_sd
        m[self._i_L], tau[self._i_L] = cfg.slope_prior_mean, cfg.slope_prior_sd
        m[self._i_L2], tau[self._i_L2] = cfg.curvature_prior_mean, cfg.curvature_prior_sd
        for j, f in enumerate(FEATURES):
            if f.bowl:
                m[self._i_knob0 + j] = cfg.bowl_prior_mean
            tau[self._i_knob0 + j] = f.prior_sd if f.prior_sd is not None else (cfg.bowl_prior_sd if f.bowl else cfg.knob_prior_sd)
        tau[i_seed0:] = cfg.seed_prior_sd
        self._prior_mean, self._prior_sd = m.copy(), tau.copy()   # the priors as specified (pinning edits m / tau)

        # features that never take two values INSIDE one era: their effect is the era difference (FF_ACC_IN_GRAPH is
        # off in every M3-M7 run and on in every M9+ run), so the coefficient stays at its prior. Reported, not fixed.
        Xk = X[:, self._i_knob0:i_seed0]
        era_idx = np.array([self.eras[row.era] for row in rows])
        confounded: List[str] = []
        for j, f in enumerate(FEATURES):
            col = Xk[:, j]
            if not np.any(col != 0.0):
                continue
            if not any(np.unique(np.round(col[era_idx == e], 9)).size > 1 for e in range(len(eras))):
                confounded.append(f.name)
        self._era_confounded = confounded

        XtX = X.T @ X
        Xty = X.T @ y
        bowl_cols = [self._i_knob0 + j for j, f in enumerate(FEATURES) if f.bowl] if cfg.bowl_nonneg else []
        pinned: List[int] = []

        def solve(sig: float) -> Tuple[np.ndarray, np.ndarray]:
            """MAP solve at noise sd ``sig``. A bowl coefficient that comes out NEGATIVE (a concave 'optimum' in which
            both directions improve) is pinned at 0 (prior mean 0, sd _PIN_SD) and the system re-solved until every
            bowl is >= 0: an active-set solution of the non-negativity constraint."""
            while True:
                D = 1.0 / (tau ** 2)
                A = XtX + (sig ** 2) * np.diag(D)
                b = np.linalg.solve(A, Xty + (sig ** 2) * D * m)
                neg = [j for j in bowl_cols if j not in pinned and b[j] < 0.0]
                if not neg:
                    return A, b
                for j in neg:
                    pinned.append(j)
                    m[j], tau[j] = 0.0, _PIN_SD

        sigma = cfg.sigma_init
        beta = m.copy()
        edf = 0.0
        for _ in range(max(1, cfg.sigma_iters)):
            A, beta = solve(sigma)
            resid = y - X @ beta
            edf = float(np.trace(np.linalg.solve(A, XtX)))
            rss = float(resid @ resid)
            sigma = math.sqrt(rss / max(n - edf, 1.0))
            sigma = min(max(sigma, cfg.sigma_floor), cfg.sigma_ceiling)
        A, self.beta = solve(sigma)
        self.cov = (sigma ** 2) * np.linalg.inv(A)
        self.pinned = [columns[j] for j in sorted(pinned)]
        self.sigma = sigma
        self.edf = edf
        self.n = n
        self._X, self._y = X, y
        self.fitted = True
        return self

    # ------------------------------------------------------------------ era lineage

    def resolve_era(self, code_version: Optional[str], tags: Optional[Iterable[str]] = None) -> Tuple[Optional[str], int, bool]:
        """(era used, lineage distance in generations, cross_family).

        Distance 0 = the version (or its lineage / group / alias: code_k12off == M12, M13 in ``M9+``) was fitted.
        Same family: the era holding the nearest fitted version, preferring the older one (M15 -> M14's era; M11c
        <-> M11 counts as one generation). No fitted version of that family (or an unparseable name): the most
        recently dated fitted era, flagged cross_family.
        """
        if not self.eras:
            return None, 0, False
        if code_version in self.eras:
            return str(code_version), 0, False
        version = canonical_version(code_version, tags)
        if version is not None:
            version = self.version_aliases.get(version, version)
            if version in self.version_to_era:
                return self.version_to_era[version], 0, False
            # a never-run version (even one whose group was fitted) is judged by its distance to the nearest member
            near = self._nearest_era(version, self.era_members)
            if near is not None:
                return near[0], max(near[1], 1), False
        # cross-family fallback: the latest era by date, else the highest number, else the last name
        def latest_key(era: str) -> Tuple[str, int, str]:
            pv = parse_version(era)
            return (self.era_dates.get(era, ""), pv[1] if pv else -1, era)
        era = max(self.eras, key=latest_key)
        return era, 1, True

    # ------------------------------------------------------------------ predict

    def design_row(self, knobs: Optional[Dict[str, Any]], steps: float, code_version: Optional[str],
                   chip: str = "C", seed: Optional[int] = None, defaults: Optional[Dict[str, str]] = None,
                   tags: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, float, List[str]]:
        """(x, extra_variance, notes) for one prediction. ``extra_variance`` covers unseen seed/era/chip.

        ``defaults`` is the missing-knob convention (K59_DEFAULTS for a candidate, CODE_DEFAULTS for a record).
        """
        if not self.fitted:
            raise RuntimeError("QualityModel.predict before fit")
        cfg = self.config
        if steps is None or steps <= 0:
            raise ValueError(f"steps must be positive, got {steps!r}")
        x = np.zeros(len(self.columns))
        extra = 0.0
        notes: List[str] = []
        x[0] = 1.0
        era, dist, cross = self.resolve_era(code_version, tags)
        if era is not None:
            x[self._i_era0 + self.eras[era]] = 1.0
        if dist > 0:
            if cross:
                extra += cfg.unseen_family_sd ** 2
                notes.append(f"code_version {code_version!r} unseen: using {era} (different lineage)")
            else:
                extra += (cfg.unseen_era_sd ** 2) * dist
                notes.append(f"code_version {code_version!r} unseen: using {era} ({dist} generation(s) away)")
        chip = str(chip or cfg.reference_chip)
        if chip in self.chips:
            x[self._i_chip0 + self.chips[chip]] = 1.0
        elif chip != cfg.reference_chip:
            extra += cfg.chip_prior_sd ** 2
            notes.append(f"chip {chip!r} unseen: treated as {cfg.reference_chip} with sd {cfg.chip_prior_sd}")
        L = math.log(float(steps) / cfg.steps_ref)
        x[self._i_L] = L
        x[self._i_L2] = L * L
        x[self._i_knob0:self._i_seed0] = knob_features(knobs, defaults)
        if defaults is None and self._era_confounded:      # a CANDIDATE moving an era-confounded feature
            xk = x[self._i_knob0:self._i_seed0]
            moved = [f.name for j, f in enumerate(FEATURES) if f.name in self._era_confounded and xk[j] != 0.0]
            if moved:
                notes.append(f"era-confounded feature(s) moved: {', '.join(moved)} (never varied inside an era; "
                             "the coefficient is the prior, the measured effect sits in the era difference)")
        if seed is not None and int(seed) in self.seeds:
            x[self._i_seed0 + self.seeds[int(seed)]] = 1.0
        else:
            extra += cfg.seed_prior_sd ** 2
            if seed is not None:
                notes.append(f"seed {seed} unseen: seed-averaged mean, sd + {cfg.seed_prior_sd}")
        return x, extra, notes

    def predict(self, knobs: Optional[Dict[str, Any]], steps: float, code_version: Optional[str],
                chip: str = "C", seed: Optional[int] = None) -> Tuple[float, float]:
        """(mean, sd) of raw bpb_2m for a CANDIDATE recipe (missing knobs = K59 defaults) at ``steps`` optimizer
        steps on ``chip`` with ``seed``.

        sd = sqrt(parameter uncertainty + residual sigma^2 + unseen seed/era/chip variance). ``seed=None`` is
        the seed-averaged prediction (its sd includes the 0.0006 seed spread).
        """
        x, extra, _ = self.design_row(knobs, steps, code_version, chip, seed)
        mean = float(x @ self.beta)
        var = float(x @ self.cov @ x) + self.sigma ** 2 + extra
        return mean, math.sqrt(max(var, 0.0))

    def _record_row(self, record: RunRecord, knobs: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, float, List[str]]:
        version = canonical_version(record.code_version, record.tags)
        return self.design_row(knobs if knobs is not None else record_knobs(record, version), record.steps,
                               version, record.chip, record.seed, defaults=self._record_defaults(version), tags=record.tags)

    def predict_record(self, record: RunRecord, knobs: Optional[Dict[str, str]] = None) -> Tuple[float, float]:
        """(mean, sd) for a RECORD read with the record convention (missing knobs = code defaults, lineage tags)."""
        x, extra, _ = self._record_row(record, knobs)
        mean = float(x @ self.beta)
        var = float(x @ self.cov @ x) + self.sigma ** 2 + extra
        return mean, math.sqrt(max(var, 0.0))

    def predict_delta(self, treatment: RunRecord, control: RunRecord, treatment_knobs: Optional[Dict[str, str]] = None,
                      control_knobs: Optional[Dict[str, str]] = None) -> Tuple[float, float, List[str]]:
        """Predicted bpb(treatment) - bpb(control), each arm with its own knobs/steps/chip/seed/code version
        (record convention). ``treatment_knobs`` / ``control_knobs`` override the records' knobs."""
        xt, et, nt = self._record_row(treatment, treatment_knobs)
        xc, ec, nc = self._record_row(control, control_knobs)
        d = xt - xc
        mean = float(d @ self.beta)
        var = float(d @ self.cov @ d) + 2.0 * self.sigma ** 2 + et + ec
        notes = nt + nc
        if self._era_confounded:
            dk = d[self._i_knob0:self._i_seed0]
            moved = [f.name for j, f in enumerate(FEATURES) if f.name in self._era_confounded and dk[j] != 0.0]
            if moved:
                notes.append(f"era-confounded feature(s) differ between the arms: {', '.join(moved)} "
                             "(the prediction comes from the era effects, not the coefficient)")
        return mean, math.sqrt(max(var, 0.0)), notes

    def extrapolated_features(self, knobs: Optional[Dict[str, Any]], defaults: Optional[Dict[str, str]] = None) -> List[str]:
        """Features whose scaled value for ``knobs`` lies outside the range seen in the fitted runs (the prediction
        there leans on the prior / the linear-quadratic form, not on data). Empty when everything is inside."""
        if not self.fitted:
            return []
        x = knob_features(knobs, defaults)
        Xk = self._X[:, self._i_knob0:self._i_seed0]
        lo, hi = Xk.min(axis=0), Xk.max(axis=0)
        eps = 1e-9
        return [f.name for j, f in enumerate(FEATURES) if x[j] < lo[j] - eps or x[j] > hi[j] + eps]

    def steps_slope(self, steps: float = STEPS_REF) -> float:
        """d bpb / d ln(steps) at ``steps`` (b + 2 c L); the simulator's local slope."""
        if not self.fitted:
            return self.config.slope_prior_mean
        L = math.log(float(steps) / self.config.steps_ref)
        return float(self.beta[self._i_L] + 2.0 * self.beta[self._i_L2] * L)

    # ------------------------------------------------------------------ paired validation

    @staticmethod
    def _pair_ids(pair: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        t = pair.get("treatment", pair.get("treatment_id", pair.get("arm", pair.get("t"))))
        c = pair.get("control", pair.get("control_id", pair.get("ctl", pair.get("c"))))
        if t is None and pair.get("treatment_run") is not None:
            t = f"{pair.get('chip') or 'C'}:{pair['treatment_run']}"
        if c is None and pair.get("control_run") is not None:
            c = f"{pair.get('control_chip') or pair.get('chip') or 'C'}:{pair['control_run']}"
        return (str(t) if t is not None else None), (str(c) if c is not None else None)

    def _resolve_arm(self, run_id: Optional[str], pair: Dict[str, Any], role: str) -> Optional[RunRecord]:
        """A pair arm by run_id; a wildcard name (``A:*_R_g4_s67``) by chip + bpb (+ seed) among all known runs."""
        if not run_id:
            return None
        r = self.records_by_id.get(run_id) or self.holdout.get(run_id)
        if r is not None or "*" not in run_id:
            return r
        chip, _, pattern = run_id.partition(":")
        pattern = pattern.split(":")[-1]
        bpb = pair.get(f"{role}_bpb")
        seed = pair.get("seed" if role == "treatment" else "control_seed")
        suffix = pattern.split("*")[-1]
        cands = []
        for rec in list(self.records_by_id.values()) + list(self.holdout.values()):
            if rec.chip != chip or rec.bpb_2m is None or bpb is None:
                continue
            if abs(float(rec.bpb_2m) - float(bpb)) > 2e-5:
                continue
            if seed is not None and rec.seed is not None and int(rec.seed) != int(seed):
                continue
            if suffix and not rec.run.endswith(suffix) and not rec.run.startswith("desc:"):
                continue
            cands.append(rec)
        return cands[0] if len(cands) == 1 else None

    def _arm_knobs(self, rec: RunRecord, overlay: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], bool]:
        knobs = record_knobs(rec)
        if overlay and not rec.knobs_complete:
            knobs.update({str(k): str(v) for k, v in overlay.items() if v is not None})
            return knobs, True
        return knobs, False

    def _untried_values(self, tk: Dict[str, str], vt: Optional[str], ck: Dict[str, str],
                        vc: Optional[str]) -> Dict[str, str]:
        """LOKVO: for every feature knob whose effective value differs between the arms, the value that is NOT
        K59's reference (the treatment's, unless the treatment sits at the reference: then the control's)."""
        et = _effective_knobs(tk, self._record_defaults(vt))
        ec = _effective_knobs(ck, self._record_defaults(vc))
        out: Dict[str, str] = {}
        for k in FEATURE_KNOBS:
            a, b = et.get(k), ec.get(k)
            if a is None or b is None or _same_knob_value(a, b):
                continue
            ref = K59_DEFAULTS.get(k)
            out[k] = b if (_same_knob_value(a, ref) and not _same_knob_value(b, ref)) else a
        return out

    def _runs_with_values(self, values: Dict[str, str], exclude: Iterable[str] = ()) -> List[str]:
        """run_ids of the fitted runs whose effective env carries ANY of ``values`` (knob -> value)."""
        if not values:
            return []
        ex = set(exclude)
        return [row.record.run_id for row in self._rows if row.record.run_id not in ex
                and any(_same_knob_value(row.effective.get(k), v) for k, v in values.items())]

    def pair_validation(self, pairs: Optional[List[Dict[str, Any]]] = None,
                        leave_pair_out: Optional[bool] = None, leave_value_out: bool = False) -> Dict[str, Any]:
        """Predict delta_raw for every validation pair; report sign agreement and MAE against the spec targets.

        ``leave_pair_out`` None = automatic: refit without the pair's two runs when there are fewer than
        ``config.lopo_max_pairs`` pairs (honest), else in-sample. ``leave_value_out`` True = leave-one-knob-value-out
        (implies leave-pair-out): the refit also drops every fitted run that carries the pair's untried knob
        value(s) (``_untried_values``), so the pair is predicted for a value the model has never seen; rows then
        carry ``untried`` (knob -> value), ``n_value_runs_dropped`` and ``value_refit``. Pairs whose runs cannot be
        found (screens, missing bpb, unknown run_id) are listed under ``skipped``. Keys: n, sign_agreement, mae,
        rmse, rows, mode, targets, pass, n_decisive, sign_agreement_decisive, mae_decisive, coverage_2sd,
        by_family, n_skipped, skipped, plus n_unsupported / sign_agreement_supported / mae_supported: a
        leave-out row is "unsupported" when a knob feature that differs between its arms is zero in every
        remaining fitted run, i.e. the pair was the ONLY evidence for that knob and the refit can only answer
        with the prior (0). Those rows are a coin toss by construction, not a model failure; the supported subset
        is the fair read. n_extrapolated / *_interior / *_extrapolated split the rows by whether the moved
        features stayed inside the refit's value range (interpolation) or not.
        """
        if not self.fitted:
            raise RuntimeError("QualityModel.pair_validation before fit")
        cfg = self.config
        pairs = list(self.pairs if pairs is None else pairs)
        resolved: List[Tuple[Dict[str, Any], RunRecord, RunRecord, float]] = []
        skipped: List[Dict[str, Any]] = []
        for pair in pairs:
            t_id, c_id = self._pair_ids(pair)
            t = self._resolve_arm(t_id, pair, "treatment")
            c = self._resolve_arm(c_id, pair, "control")
            if t is None or c is None:
                skipped.append({"name": pair.get("name") or pair.get("pair_id"), "treatment": t_id, "control": c_id,
                                "reason": "run not in the fitted full-run set"})
                continue
            obs = pair.get("delta_raw")
            if obs is None:
                obs = float(t.bpb_2m) - float(c.bpb_2m)
            resolved.append((pair, t, c, float(obs)))

        lokvo = bool(leave_value_out)
        lopo = True if lokvo else ((len(resolved) < cfg.lopo_max_pairs) if leave_pair_out is None else bool(leave_pair_out))
        rows: List[Dict[str, Any]] = []
        for pair, t, c, obs in resolved:
            model = self
            notes: List[str] = []
            in_fit = [rid for rid in (t.run_id, c.run_id) if rid in self.records_by_id]
            refit = False
            tk, t_over = self._arm_knobs(t, pair.get("knob_changes"))
            ck, c_over = self._arm_knobs(c, pair.get("control_knobs"))
            vt = canonical_version(t.code_version, t.tags)
            vc = canonical_version(c.code_version, c.tags)
            untried: Dict[str, str] = {}
            value_runs: List[str] = []
            if lokvo:
                untried = self._untried_values(tk, vt, ck, vc)
                value_runs = self._runs_with_values(untried, exclude=(t.run_id, c.run_id))
            value_refit = False
            if lopo and (in_fit or value_runs):
                try:
                    model = self.refit_without((t.run_id, c.run_id, *value_runs))
                    refit = True
                    value_refit = bool(value_runs)
                except ValueError as exc:
                    notes.append(f"leave-out refit impossible: {exc}")
                    model = self
                    if value_runs and in_fit:
                        try:
                            model = self.refit_without((t.run_id, c.run_id))
                            refit = True
                            notes.append("fell back to leave-pair-out (the untried value's runs were kept)")
                        except ValueError:
                            model = self
            if untried:
                notes.append("untried value(s): " + ", ".join(f"{k}={v}" for k, v in sorted(untried.items()))
                             + f"; {len(value_runs)} other fitted run(s) carrying them dropped")
            if t_over or c_over:
                notes.append("pair knob_changes/control_knobs laid over an arm without effective knobs")
            holdout = [rid for rid in (t.run_id, c.run_id) if rid in self.holdout]
            if holdout:
                notes.append(f"out-of-sample arm(s) not in the fit: {', '.join(holdout)}")
            pred, pred_sd, pnotes = model.predict_delta(t, c, tk, ck)
            unsupported: List[str] = []
            if refit or holdout:
                diff = knob_features(tk, self._record_defaults(vt)) - knob_features(ck, self._record_defaults(vc))
                support = np.any(model._X[:, model._i_knob0:model._i_seed0] != 0.0, axis=0)
                unsupported = [f.name for j, f in enumerate(FEATURES) if diff[j] != 0.0 and not support[j]]
            differs = knob_features(tk, self._record_defaults(vt)) != knob_features(ck, self._record_defaults(vc))
            moving = {f.name for j, f in enumerate(FEATURES) if differs[j]}
            extrap = sorted((set(model.extrapolated_features(tk, self._record_defaults(vt)))
                             | set(model.extrapolated_features(ck, self._record_defaults(vc)))) & moving)
            if extrap:
                notes.append(f"outside the fitted value range: {', '.join(extrap)}")
            err = pred - obs
            sign_ok = (pred > 0) == (obs > 0) if obs != 0 else abs(pred) < cfg.noise_floor
            rows.append({
                "name": pair.get("name") or pair.get("pair_id") or f"{t.run} vs {c.run}",
                "family": pair.get("family"), "in_task_list": pair.get("in_task_list"),
                "treatment": t.run_id, "control": c.run_id,
                "observed": obs, "predicted": pred, "pred_sd": pred_sd, "error": err, "abs_error": abs(err),
                "z": (err / pred_sd) if pred_sd > 0 else float("nan"),
                "sign_ok": bool(sign_ok), "decisive": abs(obs) >= cfg.noise_floor,
                "steps_t": t.steps, "steps_c": c.steps, "seed_t": t.seed, "seed_c": c.seed,
                "chip_t": t.chip, "chip_c": c.chip, "code_t": t.code_version, "code_c": c.code_version,
                "era_t": canonical_version(t.code_version, t.tags), "era_c": canonical_version(c.code_version, c.tags),
                "lopo_refit": refit, "holdout": bool(holdout),
                "untried": untried, "n_value_runs_dropped": len(value_runs), "value_refit": value_refit,
                "unsupported_features": unsupported, "extrapolated_features": extrap,
                "notes": notes + pnotes + ([f"only evidence for {', '.join(unsupported)}: refit predicts the prior"]
                                           if unsupported else []),
            })
        mode = "leave-one-knob-value-out" if lokvo else ("leave-one-pair-out" if lopo else "in-sample")
        return self._pair_report(rows, skipped, mode)

    def _pair_report(self, rows: List[Dict[str, Any]], skipped: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
        cfg = self.config

        def stats(rs: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
            if not rs:
                return None, None
            return sum(r["sign_ok"] for r in rs) / len(rs), sum(r["abs_error"] for r in rs) / len(rs)

        n = len(rows)
        dec = [r for r in rows if r["decisive"]]
        sup = [r for r in rows if not r["unsupported_features"]]
        interior = [r for r in rows if not r["extrapolated_features"]]
        extrap = [r for r in rows if r["extrapolated_features"]]
        sign, mae = stats(rows)
        sign_d, mae_d = stats(dec)
        sign_s, mae_s = stats(sup)
        sign_i, mae_i = stats(interior)
        sign_x, mae_x = stats(extrap)
        fams = sorted({str(r["family"]) for r in rows if r.get("family")})
        by_family = {}
        for fam in fams:
            rs = [r for r in rows if str(r.get("family")) == fam]
            s, m = stats(rs)
            by_family[fam] = {"n": len(rs), "sign_agreement": s, "mae": m,
                              "mean_observed": float(np.mean([r["observed"] for r in rs])),
                              "mean_predicted": float(np.mean([r["predicted"] for r in rs]))}
        out: Dict[str, Any] = {
            "n": n,
            "sign_agreement": sign,
            "mae": mae,
            "rmse": math.sqrt(float(np.mean([r["error"] ** 2 for r in rows]))) if rows else None,
            "coverage_2sd": (sum(1 for r in rows if abs(r["z"]) <= 2.0) / n) if n else None,
            "n_decisive": len(dec),
            "sign_agreement_decisive": sign_d,
            "mae_decisive": mae_d,
            "n_unsupported": n - len(sup),
            "sign_agreement_supported": sign_s,
            "mae_supported": mae_s,
            "n_extrapolated": len(extrap),
            "sign_agreement_interior": sign_i,
            "mae_interior": mae_i,
            "sign_agreement_extrapolated": sign_x,
            "mae_extrapolated": mae_x,
            "n_value_refit": sum(1 for r in rows if r.get("value_refit")),
            "by_family": by_family,
            "rows": rows,
            "mode": mode,
            "targets": {"sign_agreement": 0.8, "mae": 0.0003},
            "noise_floor": cfg.noise_floor,
            "n_skipped": len(skipped),
            "skipped": skipped,
        }
        out["pass"] = {
            "sign_agreement": (sign is not None and sign >= 0.8),
            "mae": (mae is not None and mae <= 0.0003),
        }
        return out

    # ------------------------------------------------------------------ reporting

    def coefficients(self) -> List[Dict[str, Any]]:
        """Every column with its posterior estimate/sd and prior. Knob coefficients are per SCALED unit."""
        if not self.fitted:
            return []
        sd = np.sqrt(np.clip(np.diag(self.cov), 0.0, None))
        return [{"name": c, "estimate": float(self.beta[i]), "sd": float(sd[i]),
                 "prior_mean": float(self._prior_mean[i]), "prior_sd": float(self._prior_sd[i]),
                 "pinned": c in self.pinned}
                for i, c in enumerate(self.columns)]

    def era_confounded_features(self) -> List[str]:
        """Features that never take two values inside one fitted era (their effect is an era difference)."""
        return list(self._era_confounded)

    def era_effects(self) -> Dict[str, Tuple[float, float]]:
        """Fitted bpb of the K59 recipe at steps_ref on chip C, seed-averaged, per era: (mean, sd)."""
        out: Dict[str, Tuple[float, float]] = {}
        for era in self.eras:
            x = np.zeros(len(self.columns))
            x[0] = 1.0
            x[self._i_era0 + self.eras[era]] = 1.0
            out[era] = (float(x @ self.beta), math.sqrt(max(float(x @ self.cov @ x), 0.0)))
        return out

    @property
    def support(self) -> Dict[str, List[str]]:
        """knob -> sorted distinct EFFECTIVE values among the fitted runs (feature knobs: raw or code default;
        other knobs: explicit values only). The shape ``ffsim.search.support_status`` reads."""
        if not self.fitted:
            return {}
        seen: Dict[str, set] = {}
        for row in self._rows:
            for k, v in row.effective.items():
                if v is not None and (k in FEATURE_KNOBS or str(v) != ""):
                    seen.setdefault(k, set()).add(str(v))
        return {k: sorted(v) for k, v in sorted(seen.items())}

    def support_detail(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """knob -> effective value -> {"n": fitted runs at that value, "eras": {era: n}}: the same knobs and values
        as ``support``, with how many runs and which eras stand behind each value a candidate could take."""
        if not self.fitted:
            return {}
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in self._rows:
            for k, v in row.effective.items():
                if v is None or not (k in FEATURE_KNOBS or str(v) != ""):
                    continue
                cell = out.setdefault(k, {}).setdefault(str(v), {"n": 0, "eras": {}})
                cell["n"] += 1
                cell["eras"][row.era] = cell["eras"].get(row.era, 0) + 1
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    def value_support(self, knob: str, value: Any, code_version: Optional[str] = None) -> Dict[str, Any]:
        """Grade one candidate knob value against the fitted runs (module docstring, 'Support'): ``status`` is
        'seen' (>= config.min_value_runs runs, and at least one in the candidate's era when ``code_version`` names
        a fitted era), 'weak' (fewer runs, or runs only from other eras), 'interp' (numeric, never run, inside the
        seen range), 'unseen' (outside it / a category never run) or 'never' (the knob never varied). Also ``n``,
        ``n_in_era``, ``eras`` (era -> runs at that value) and a ``note``."""
        detail = self.support_detail().get(knob)
        none: Dict[str, Any] = {"n": 0, "n_in_era": 0, "eras": {}}
        if not detail:
            return dict(none, status="never", note=f"{knob} never varied in a fitted run")
        era: Optional[str] = None
        if code_version is not None:
            e, dist, cross = self.resolve_era(code_version)
            era = e if (e is not None and dist == 0 and not cross) else None
        hit = [c for v, c in detail.items() if _same_knob_value(v, value)]
        if hit:
            n = sum(c["n"] for c in hit)
            eras: Dict[str, int] = {}
            for c in hit:
                for e, k in c["eras"].items():
                    eras[e] = eras.get(e, 0) + k
            n_in_era = eras.get(era, 0) if era is not None else n
            if n < self.config.min_value_runs:
                st, note = "weak", f"{knob}={value}: only {n} fitted run(s) ({', '.join(sorted(eras))})"
            elif n_in_era == 0:
                st, note = "weak", (f"{knob}={value}: {n} run(s), none in era {era} "
                                    f"({', '.join(sorted(eras))} only: the effect there is an era difference)")
            else:
                st = "seen"
                note = f"{knob}={value}: {n} run(s)" + (f", {n_in_era} in era {era}" if era is not None else "")
            return {"status": st, "n": n, "n_in_era": n_in_era, "eras": eras, "note": note}
        seen = ", ".join(detail)
        if len(detail) < 2:
            return dict(none, status="never", note=f"{knob} never varied in a fitted run (always {seen})")
        try:
            fv = float(str(value).strip())
            nums = [float(v) for v in detail]
        except ValueError:
            return dict(none, status="unseen", note=f"{knob}={value} never run (seen: {seen})")
        if nums and min(nums) <= fv <= max(nums):
            return dict(none, status="interp", note=f"{knob}={value} never run; inside the seen range (seen: {seen})")
        return dict(none, status="unseen", note=f"{knob}={value} never run; outside the seen range (seen: {seen})")

    def feature_support(self) -> Dict[str, Dict[str, Any]]:
        """Per feature: how many fitted runs have it non-zero (and in which eras), the distinct scaled values and
        effective knob values. A feature with n_nonzero == 0 is pure extrapolation (its coefficient is the prior);
        one whose posterior sd stayed at its prior sd (``identified`` False) is collinear with the eras and not
        separately estimated; ``era_confounded`` = it never varies inside an era (the measured effect IS an era
        difference); ``pinned`` = a bowl held at 0 by the non-negativity constraint."""
        if not self.fitted:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        Xk = self._X[:, self._i_knob0:self._i_seed0]
        sd = np.sqrt(np.clip(np.diag(self.cov), 0.0, None))
        for j, f in enumerate(FEATURES):
            col = Xk[:, j]
            nz = col != 0.0
            values: Dict[str, List[str]] = {}
            eras_nz: Dict[str, int] = {}
            for row, v in zip(self._rows, col):
                if v != 0.0:
                    eras_nz[row.era] = eras_nz.get(row.era, 0) + 1
                for k in f.knobs:
                    kv = row.effective.get(k)
                    if kv is not None:
                        values.setdefault(k, [])
                        if kv not in values[k]:
                            values[k].append(kv)
            post_sd = float(sd[self._i_knob0 + j])
            prior_sd = float(self._prior_sd[self._i_knob0 + j])
            pinned = f"knob:{f.name}" in self.pinned
            out[f.name] = {
                "n_nonzero": int(nz.sum()), "n_rows": int(col.size),
                "eras_nonzero": eras_nz,
                "scaled_values": sorted({round(float(v), 4) for v in col}),
                "scaled_range": (float(col.min()), float(col.max())) if col.size else (0.0, 0.0),
                "knob_values": {k: sorted(v) for k, v in values.items()},
                "supported": bool(nz.any()),
                "identified": bool(nz.any() and post_sd < 0.9 * prior_sd and not pinned),
                "era_confounded": f.name in self._era_confounded,
                "pinned": pinned,
                "posterior_sd": post_sd, "prior_sd": prior_sd,
            }
        return out

    def unmodelled_knobs(self) -> Dict[str, Dict[str, int]]:
        """Explicit knobs that take more than one value inside at least one fitted era and are read by no feature:
        their effect sits in the residual (and in the era effect when they moved between eras). Absence counts
        as a value only in records with complete knobs (a hand-launched rehearsal's env is baked into its
        train.py, so a knob it does not spell is unknown, not off)."""
        if not self.fitted:
            return {}
        out: Dict[str, Dict[str, int]] = {}
        by_era: Dict[str, List[_Row]] = {}
        for row in self._rows:
            by_era.setdefault(row.era, []).append(row)
        for era, rs in by_era.items():
            keys = {k for r in rs for k in r.knobs if k not in FEATURE_KNOBS and k not in _IGNORED_KNOBS
                    and not k.startswith("NEURON_")}
            for k in keys:
                vals: Dict[str, int] = {}
                for r in rs:
                    v = r.knobs.get(k)
                    if v is None:
                        if not r.record.knobs_complete:
                            continue
                        v = "<absent>"
                    vals[v] = vals.get(v, 0) + 1
                if len(vals) > 1:
                    cur = out.setdefault(k, {})
                    for v, c in vals.items():
                        cur[v] = cur.get(v, 0) + c
        return out

    def summary(self) -> Dict[str, Any]:
        """Plain-data fit report for report.py: sizes, sigma, slope, era/chip/seed/knob effects."""
        if not self.fitted:
            return {"fitted": False}
        coefs = {c["name"]: c for c in self.coefficients()}
        return {
            "fitted": True,
            "n": self.n,
            "edf": self.edf,
            "sigma": self.sigma,
            "dropped": dict(self.dropped),
            "slope": (coefs["ln_steps"]["estimate"], coefs["ln_steps"]["sd"]),
            "curvature": (coefs["ln_steps_sq"]["estimate"], coefs["ln_steps_sq"]["sd"]),
            "eras": self.era_effects(),
            "era_counts": dict(self.era_counts),
            "era_members": {e: list(m) for e, m in self.era_members.items()},
            "era_merges": dict(self.era_merges),
            "era_grouping": "default (M9.. -> M9+)" if self.config.era_groups is None else
                            ("none" if not self.config.era_groups else dict(self.config.era_groups)),
            "chips": {c: (coefs[f"chip:{c}"]["estimate"], coefs[f"chip:{c}"]["sd"]) for c in self.chips},
            "seeds": {s: (coefs[f"seed:{s}"]["estimate"], coefs[f"seed:{s}"]["sd"]) for s in self.seeds},
            "seed_counts": {s: int(self._X[:, self._i_seed0 + i].sum()) for s, i in self.seeds.items()},
            "knobs": {f.name: (coefs[f"knob:{f.name}"]["estimate"], coefs[f"knob:{f.name}"]["sd"]) for f in FEATURES},
            "features": [{"name": f.name, "knobs": list(f.knobs), "reference": f.reference,
                          "transform": f.transform, "scale": f.scale,
                          "prior_sd": f.prior_sd if f.prior_sd is not None else self.config.knob_prior_sd}
                         for f in FEATURES],
            "feature_support": self.feature_support(),
            "unmodelled_knobs": self.unmodelled_knobs(),
            "era_confounded": list(self._era_confounded),
            "bowls_pinned": list(self.pinned),
            "fit_eras": None if self.config.fit_eras is None else list(self.config.fit_eras),
            "record_defaults": self.config.record_defaults,
            "n_pairs": len(self.pairs),
            "n_holdout": len(self.holdout),
        }

    def residuals(self) -> List[Dict[str, Any]]:
        """Per fitted run: run_id, observed, fitted, residual (for the report's outlier table)."""
        if not self.fitted:
            return []
        fitted = self._X @ self.beta
        return [{"run_id": r.run_id, "observed": float(self._y[i]), "fitted": float(fitted[i]),
                 "residual": float(self._y[i] - fitted[i]), "era": self._rows[i].era, "chip": r.chip,
                 "seed": r.seed, "steps": r.steps} for i, r in enumerate(self.fit_records)]
