"""Part 1 of the Trainium run simulator: the step-time model (ffsim/CONTRACT.md).

    steps = charged_seconds / step_time(code, hardware, runtime)

What this module is
-------------------
A REGRESSION ON MEASURED RUNS, not a first-principles op-count model. The spec's hard lessons
(torch.maximum 5.9% slower than a 5-op formula, QKV fusion +3.2% despite fewer ops, a 96-parameter
gate +7.6%, chip D +12% in full runs but +1% in 60-step screens) show that neuronx-cc is not
linear in op count, so the model learns per-lineage baselines and per-mechanism multipliers from
train.log medians and predicts by lineage similarity when a code version is unseen.

Model
-----
For each accumulation phase p in (k1, k2, k4):

    log(step_time_p) = b0 + lineage[code_version] + chip terms + screen terms + sum_j coef_j * feature_j

fitted by ridge regression (numpy lstsq on the augmented system, closed form, intercept
unpenalised), weighted by n_k through w = n / (n + N_HALF): 1/variance for a median whose noise is
a per-step term (~3% sd) plus a between-run chip-jitter floor (~0.5%), so a 1,500-step full run
weighs about 3x a 40-step screen rather than 40x.

Lineages, not code dirs. The chips' code dirs are flag-gated bakes of a few train_ff.py lineages
(`code_k12off` = the K57 recipe baked on M12 with the new flags off, bit-identical to M9 at the
graph level; `code_k57` = K57 on M9; `code_m9off` = M9 with the flags off; ...), and the monitor
names the same files "M12", "K57". LINEAGE_ALIASES maps every spelling to one lineage key, so the
lineage effect is shared and the mechanisms (leaky form, RF, XSA, ...) are carried by knob
features. Leave-one-code-version-out therefore means leave-one-LINEAGE-out.

Knob features are CENTRED ON THE K57-RECIPE REFERENCE (KNOB_DEFAULTS: relu^2 MLP, no custom
backward, FF_ACC_IN_GRAPH=1, FF_MUON_ZERO=1, EMA every 32, fused CE 8, depth 9 x 113, chip C, full
run): a reference run contributes zero on every knob feature, so intercept + lineage[...] is "this
lineage at the reference recipe" and explain() reads as deviations from it. A knob the record's env
does not state is read from CODE_DIR_KNOBS (what that code dir bakes in or lacks: the harvested
`overrides` of a screen only lists the arm's extras) and then from KNOB_DEFAULTS. K59 is the
reference plus FF_LEAKY_RELU2=0.35 / FF_LEAKY_FORM=fnm, so predict("M12", K59 env, "C") is
lineage M12 + the leaky_fnm multiplier.

Records the fit ignores (counted in `skipped`): no code_version; the `code` placeholder dir (the
default /root/ff-claude/code/ was re-deployed as M1..M9 and its 71 driver-era runs are a
depth-12 x head-128 model: not one lineage, see EXCLUDE_VERSIONS); chips outside CHIPS; a phase
with fewer than MIN_N logged steps (5-step probes, partial logs).

Public API
----------
StepTimeModel.fit(records) / predict(code_version, knobs, chip, is_screen=False) ->
    {"k1", "k2", "k4", "sd_k4", "version_used", "notes"}  (notes non-empty on a fallback or a
    mechanism no fitted run carries; such a mechanism is charged NOVEL_PRIOR_LOG (+3%) per
    mechanism, never 0, and sd_k4 is widened)
StepTimeModel.loo_report(phase="k4")   leave-one-lineage-out error (target <= 0.5% on k4), split
    into runs whose mechanisms the training fold has seen ("reachable") and runs carrying a
    mechanism no other lineage ran ("novel": unpredictable by construction, reported, not hidden).
    Both averaging conventions are reported (`*_by_run`: every held-out run counts once;
    `*_by_lineage`: the mean of the per-lineage means); the unsuffixed keys are the run-weighted
    values and `passes` judges `mean_abs_pct_reachable_by_run`, the convention `rmse_pct` uses.
StepTimeModel.loro_report(phase="k4")  leave-one-run-out error (within-lineage noise floor)
StepTimeModel.known_versions() / nearest_version(v) / explain(code_version, knobs, chip)
StepTimeModel.coefficient_table(phase) feature -> coefficient, %, number of fitted runs carrying it
steps_from_step_time(step_time, charged_seconds, accum_sched, warmup_steps) -> int
simulate_phases(...) -> dict with the phase switch steps (the campaign's FF_ACCUM_SCHED logic)
features_from_code(train_py_path) -> {}  (hook for a later op-count model; TODO)
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .schema import CHIPS, K_PHASES, RunRecord

# ----------------------------------------------------------------------------------------------
# Knob vocabulary. The reference is the K57 recipe (G7 arm on M9 / code_k12off flags off).
# ----------------------------------------------------------------------------------------------
KNOB_DEFAULTS: Dict[str, str] = {
    "FF_LEAKY_RELU2": "0",        # leaky ReLU^2 slope; 0 = plain relu^2 (M9/K57); K59 = 0.35
    "FF_LEAKY_FORM": "fnm",       # abs | max | fn | fnm (fn ~ fnm in step time: 0.9173 vs 0.9160)
    "FF_RELU2_FN": "0",           # custom relu^2 backward (RF): one elementwise pass fewer, -1.9% k4
    "FF_FUSE_QKV": "0",           # flag; +3.2% when on (pre-K44 era; NO chip C/D run carries it)
    "FF_XSA_LAYERS": "",          # "" | "all" | "5,6,7,8"; XA (all 9) +9.5-10%, 4 layers +3.2-3.4%
    "FF_ATTN_SRC": "",            # attention-source reuse "5:6,7,8" (AS-a) | "4:5,6,7,8" (AS-b)
    "FF_LANES": "1",              # parallel lanes (2: +3.1%, 3: +4.7% in S60)
    "FF_POOL_LAST": "0",          # pool the last n blocks (3: +0.9% S60)
    "FF_ROPE_FN": "0",            # M13 custom autograd Functions
    "FF_QK_GAIN_FOLD": "0",
    "FF_RMSNORM_FN": "0",
    "FF_CE_ONEHOT": "0",          # one-hot CE variant (+1.7% S60)
    "FF_MTP_LOSS_NEXT": "0",      # (+0.2% S60)
    "FF_MTP": "1",                # 0 off | 1 one [chunk,3] scatter | 2 three [chunk,1] scatters
    "FF_ACC_IN_GRAPH": "1",       # 0 off | 1 every leaf | 2 not the embeddings; on = -2.1% k4
    "FF_MUON_ZERO": "1",          # flag; on = -3.1% k4 (Z1/Z2 vs controls on M6)
    "FF_EMA": "1",                # EMA on (screens run it off)
    "FF_EMA_EVERY": "32",         # EMA update period in steps (cost per step ~ 1/period)
    "FF_VALUE_EMBED": "1",        # 0 off | 1 | 3 gated (the 96-parameter gate, +7.6%, code era only)
    "FF_DEPTH": "9",              # transformer blocks
    "FF_ASPECT_RATIO": "113",     # base_dim = depth * aspect_ratio
    "FF_MB": "8",                 # micro-batch sequences per rank
    "FF_FUSED_CE": "8",           # 0 off | n row chunks; F4 +0.2%, F16 +2.7% vs 8
    "FF_KV_HEADS": "1",           # 0 = n_kv_heads == n_head (no GQA)
    "FF_HEAD_DIM": "256",
    "FF_TOTAL_BATCH": "262144",   # tokens per optimizer step in the final (k4) phase
}

# Knobs a code dir bakes in (or lacks) that the harvested env of its runs does not state. Keyed by
# the code_version spelling in runs.jsonl. Sources: known-facts.json queue_environment.code_dirs,
# dataset-qa.md section 8, the S60 batches in the campaign ledger.
# - M1..M5 predate FF_MUON_ZERO (G3) and FF_ACC_IN_GRAPH (G7): an env that does not set them ran
#   without them. M6..M8 (and their code dirs) predate FF_ACC_IN_GRAPH; every harvested M6+ run
#   states FF_MUON_ZERO explicitly, and every harvested M9..M11 run states both, so those lineages
#   take no fill and predict("M9", {}, "C") means "M9 at the K57-recipe reference".
# - The K5x bakes carry the K57 recipe (the reference) unless listed; code_m9off/ln/oh are M9 with
#   the K54c recipe (accumulation off-graph); code_m10b / code_m11c bake the abs / max leaky form.
_PRE_G3 = {"FF_ACC_IN_GRAPH": "0", "FF_MUON_ZERO": "0"}
_PRE_G7 = {"FF_ACC_IN_GRAPH": "0"}
CODE_DIR_KNOBS: Dict[str, Dict[str, str]] = {
    "M1": _PRE_G3, "M2": _PRE_G3, "M3": _PRE_G3, "M4": _PRE_G3, "M5": _PRE_G3,
    "M6": _PRE_G7, "M7": _PRE_G7, "M7a": _PRE_G7, "M8": _PRE_G7,
    "code_m6": _PRE_G7, "code_m7": _PRE_G7, "code_m7a": _PRE_G7,
    "M10b": {"FF_LEAKY_FORM": "abs"}, "code_m10b": {"FF_LEAKY_FORM": "abs"},
    "M11c": {"FF_LEAKY_FORM": "max"}, "code_m11c": {"FF_LEAKY_FORM": "max"},
    "code_k52": {"FF_ACC_IN_GRAPH": "0", "FF_MUON_ZERO": "0", "FF_EMA_EVERY": "8"},
    "code_k52m6": {"FF_ACC_IN_GRAPH": "0", "FF_MUON_ZERO": "0", "FF_EMA_EVERY": "8"},
    "code_k53": {"FF_ACC_IN_GRAPH": "0", "FF_EMA_EVERY": "4"},
    "code_k54b": _PRE_G7, "code_k54bmin": _PRE_G7, "code_k54cmin": _PRE_G7,
    "code_m9off": _PRE_G7, "code_m9ln": _PRE_G7, "code_m9oh": _PRE_G7,
    "code_m10xa": {"FF_XSA_LAYERS": "all"}, "code_m10xl": {"FF_XSA_LAYERS": "5,6,7,8"},
}
# Screens on code_m10xa / code_m10xl set FF_XSA_LAYERS=1 as an on-flag for the baked layer list.
_XSA_FLAG_DIRS = {"code_m10xa": "all", "code_m10xl": "5,6,7,8"}

# The `code` placeholder is not a lineage (dataset-qa.md section 8).
EXCLUDE_VERSIONS: Tuple[str, ...] = ("code",)

TRUE_STRINGS = ("1", "true", "yes", "on")

# Feature -> one-line meaning (value at the reference in parentheses). Used by explain().
FEATURE_DOC: Dict[str, str] = {
    "intercept": "log step time of the reference lineage mix at the K57-recipe reference, chip C, full run",
    "chip_D": "chip D (runtime 2.33.10) vs chip C, full run",
    "chip_AB": "chips A/B vs chip C, full run",
    "screen": "60/120-step screen (forced phases, FF_HOST_TIMES=1) vs full run, chip C",
    "chip_D_x_screen": "extra chip-D effect in a screen (D is +12% in full runs, +1% in screens)",
    "chip_AB_x_screen": "extra chip-A/B effect in a screen",
    "leaky_fnm": "leaky relu^2 (FF_LEAKY_RELU2 > 0) in the fn/fnm form; K59 uses 0.35 fnm (0)",
    "leaky_abs": "leaky relu^2 in the abs form (M10b) (0)",
    "leaky_max": "leaky relu^2 in the torch.maximum form (M11c) (0)",
    "relu2_fn": "FF_RELU2_FN custom relu^2 backward: one elementwise pass fewer over the 4C tensor (0)",
    "fuse_qkv": "FF_FUSE_QKV on (0); no chip C/D run carries it",
    "xsa_layers": "XSA gated blocks / depth; 'all' = 1.0 (0)",
    "attn_src_layers": "blocks reading a shared attention source / depth: '5:6,7,8' = 3/9 (0)",
    "lanes_extra": "FF_LANES - 1 (0)",
    "pool_last": "FF_POOL_LAST > 0 (0)",
    "rope_fn": "FF_ROPE_FN custom Function (0)",
    "qk_gain_fold": "FF_QK_GAIN_FOLD custom Function (0)",
    "rmsnorm_fn": "FF_RMSNORM_FN custom Function (0)",
    "ce_onehot": "FF_CE_ONEHOT != 0 (0)",
    "mtp_loss_next": "FF_MTP_LOSS_NEXT on (0)",
    "mtp_off": "FF_MTP == 0 (0)",
    "mtp_2": "FF_MTP == 2, three [chunk,1] scatters (0)",
    "acc_in_graph_off": "FF_ACC_IN_GRAPH == 0 (0)",
    "muon_zero_off": "FF_MUON_ZERO off (0)",
    "ema_off": "FF_EMA == 0 (0)",
    "ema_rate": "32 / FF_EMA_EVERY - 1: relative EMA update rate (0)",
    "value_embed_off": "FF_VALUE_EMBED == 0 (0)",
    "value_embed_gate": "FF_VALUE_EMBED == 3, the gated value embeddings (0)",
    "depth_rel": "FF_DEPTH / 9 - 1 (0)",
    "log_aspect": "ln(FF_ASPECT_RATIO / 113) (0)",
    "log2_mb": "log2(FF_MB / 8) (0)",
    "fused_ce_off": "FF_FUSED_CE == 0, the unfused head (0)",
    "log2_fused_ce": "log2(FF_FUSED_CE / 8) when fused (0)",
    "log2_kv_heads": "log2(n_kv_heads), 0 -> n_head (0)",
    "log2_head_dim": "log2(FF_HEAD_DIM / 256) (0)",
    "log2_total_batch": "log2(FF_TOTAL_BATCH / 262144) (0)",
}

KNOB_FEATURES: Tuple[str, ...] = (
    "leaky_fnm", "leaky_abs", "leaky_max", "relu2_fn", "fuse_qkv", "xsa_layers", "attn_src_layers",
    "lanes_extra", "pool_last", "rope_fn", "qk_gain_fold", "rmsnorm_fn", "ce_onehot", "mtp_loss_next",
    "mtp_off", "mtp_2", "acc_in_graph_off", "muon_zero_off", "ema_off", "ema_rate",
    "value_embed_off", "value_embed_gate", "depth_rel", "log_aspect", "log2_mb",
    "fused_ce_off", "log2_fused_ce", "log2_kv_heads", "log2_head_dim", "log2_total_batch",
)
# Named subsets for the validation ladder ("fewer knob features").
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "full": KNOB_FEATURES,
    "mechanisms": ("leaky_fnm", "leaky_abs", "leaky_max", "relu2_fn", "xsa_layers", "attn_src_layers",
                   "lanes_extra", "pool_last", "rope_fn", "qk_gain_fold", "rmsnorm_fn", "ce_onehot",
                   "mtp_loss_next", "acc_in_graph_off", "muon_zero_off", "ema_off", "ema_rate",
                   "fused_ce_off", "log2_fused_ce"),
    "core": ("leaky_fnm", "leaky_abs", "leaky_max", "relu2_fn", "xsa_layers", "attn_src_layers",
             "acc_in_graph_off", "muon_zero_off", "ema_off"),
    "minimal": ("leaky_fnm", "relu2_fn", "acc_in_graph_off", "muon_zero_off"),
}
STRUCT_FEATURES: Tuple[str, ...] = ("chip_D", "chip_AB", "screen", "chip_D_x_screen", "chip_AB_x_screen")

# Phase ratios used ONLY when a phase has no fitted data at all (K56 C35: 0.2946/0.5056/0.9277;
# K57 M9 S60: 0.2953/0.5104/0.9315 -> k1/k4 ~0.32, k2/k4 ~0.55).
PHASE_RATIO_FALLBACK: Dict[str, float] = {"k1": 0.32, "k2": 0.55, "k4": 1.0}

N_HALF = 36.0            # weight w = n / (n + N_HALF): (3% per-step sd)^2 / (0.5% jitter floor)^2
MIN_N = 10               # a phase median from fewer logged steps is a probe, not a measurement
DEFAULT_LAMBDA = 1e-3    # ridge penalty on every coefficient but the intercept (features are O(1))
FALLBACK_SD_LOG = 0.01   # sd (log) added in quadrature for an unseen lineage before loo_report ran
NOVEL_SD_LOG = 0.03      # sd (log) added for a mechanism no fitted run carries (mechanisms cost 1-10%)
# Prior cost predict() charges for EACH knob mechanism no fitted run carries (+3% per mechanism).
# Centring an unknown mechanism on 0 is not neutral downstream: ffsim.simulate draws a symmetric
# step-time jitter factor (1 + s z) with the widened sd, and E[1 / (1 + s z)] > 1, so a recipe with
# an unknown-cost mechanism came out with MORE steps than its base (FF_FUSE_QKV=1 / FF_VALUE_EMBED=3
# / FF_DEPTH=12: 2390.6 vs 2385.4 at an identical k4). The contract's two unmeasured mechanisms
# cost +3.2% (QKV fusion) and +7.6% (the 96-parameter gate); +3% is the low end of that, the note
# says it was charged, and one S60 screen of the mechanism replaces it with a fitted coefficient.
NOVEL_PRIOR_LOG = math.log(1.03)
VERSION_PREFIX = "ver:"

# Campaign facts (CONTRACT.md): steps 0-4 are outside the charged clock; warm-up 20 steps at k = K.
STARTUP_STEPS_EXCLUDED = 5
DEFAULT_ACCUM_SCHED = "1:0.06,2:0.2,4"
DEFAULT_WARMUP_STEPS = 20
# Medians under-count the mean step (slow-step tail: EMA every 32 steps, insurance saves, phase
# switches), so the phase walk on the logged medians over-predicts the step count. The constant
# below is the K59-ERA CHIP-C THREE-PHASE value: it was calibrated on the twelve hand-launched
# rehearsals C28..C40 (C30..C40: raw walk +0.32..+0.47%, corrected -0.27..-0.09%; K51/K52 on
# 2:0.12,4: +1.18% / +0.59%). Over the population it is a bias TABLE, not a constant (129 chip C/D
# full runs of a fitted lineage with n_k4 >= 100, each walked from its own medians; mean +- sd of
# predicted / measured - 1; research/sim-data/steptime-validation.md section 6):
#   all runs                              n=129  raw +0.78 +- 0.33   corrected +0.20 +- 0.33  (MAE 0.30%,
#                                                                     worst +2.16% C:1100_R_g7wu40_s58)
#   chip C  FF_ACCUM_SCHED=1:0.06,2:0.2,4 n=85   raw +0.73 +- 0.29   corrected +0.15 +- 0.29
#   chip C  2:0.2,4                       n=25   raw +1.04 +- 0.26   corrected +0.46 +- 0.25
#   chip C  2:0.12,4                      n=4    raw +1.16 +- 0.04   corrected +0.59 +- 0.05
#   chip C  2:0.3,4                       n=3    raw +0.78 +- 0.10   corrected +0.21 +- 0.10
#   chip D  1:0.06,2:0.2,4                n=8    raw +0.27 +- 0.24   corrected -0.33 +- 0.24
# The +-0.3% sd is the per-run jitter floor no constant removes; a two-phase schedule or chip D
# carries a bias of its own (+0.3..0.6% / -0.3%) on top. Opt in per call (overhead_frac=...).
MEDIAN_TO_MEAN_OVERHEAD = 0.0057


# ----------------------------------------------------------------------------------------------
# Knob parsing
# ----------------------------------------------------------------------------------------------
def _get(env: Dict[str, Any], name: str) -> str:
    v = env.get(name)
    if v is None or (isinstance(v, str) and v.strip() == "" and name not in ("FF_XSA_LAYERS", "FF_ATTN_SRC")):
        return KNOB_DEFAULTS[name]
    return str(v).strip()


def _flag(s: str) -> bool:
    # train_ff._env_flag: default when empty; else anything but "0"/"false"/"False" is on.
    return s.strip().lower() not in ("", "0", "false", "no", "off")


def _num(s: str, default: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _norm_version(code_version: Optional[str]) -> str:
    return (code_version or "").strip()


def effective_env(knobs: Optional[Dict[str, Any]], code_version: Optional[str] = None) -> Dict[str, str]:
    """KNOB_DEFAULTS, then what the code dir bakes in (CODE_DIR_KNOBS), then the record's env.
    The env wins whenever it states a knob; the flag-form FF_XSA_LAYERS=1 of the code_m10xa /
    code_m10xl screens is translated to the layer list those dirs bake."""
    env: Dict[str, str] = dict(KNOB_DEFAULTS)
    cv = _norm_version(code_version)
    env.update(CODE_DIR_KNOBS.get(cv, {}))
    for k, v in (knobs or {}).items():
        if v is None:
            continue
        env[str(k)] = str(v)
    if cv in _XSA_FLAG_DIRS and env.get("FF_XSA_LAYERS", "").strip() in ("1", "true", "on"):
        env["FF_XSA_LAYERS"] = _XSA_FLAG_DIRS[cv]
    return env


def _layer_count(spec: str, depth: float) -> float:
    spec = spec.replace(" ", "").lower()
    if spec == "all":
        return 1.0
    if not spec:
        return 0.0
    n = len({v for v in spec.split(",") if v.strip().isdigit()})
    return n / depth


def knob_features(knobs: Optional[Dict[str, Any]], code_version: Optional[str] = None) -> Dict[str, float]:
    """Step-time-relevant features of an FF_* environment, centred on KNOB_DEFAULTS (all zero for a
    reference-recipe run). Unknown or unparsable values fall back to the reference. With
    code_version the knobs that dir bakes in are applied first (CODE_DIR_KNOBS)."""
    k = effective_env(knobs, code_version)
    f: Dict[str, float] = {name: 0.0 for name in KNOB_FEATURES}

    leaky = _num(_get(k, "FF_LEAKY_RELU2"), 0.0)
    form = _get(k, "FF_LEAKY_FORM").lower()
    if leaky > 0.0:
        if form == "abs":
            f["leaky_abs"] = 1.0
        elif form == "max":
            f["leaky_max"] = 1.0
        else:
            f["leaky_fnm"] = 1.0
    f["relu2_fn"] = 1.0 if _flag(_get(k, "FF_RELU2_FN")) else 0.0
    f["fuse_qkv"] = 1.0 if _flag(_get(k, "FF_FUSE_QKV")) else 0.0

    depth = max(1.0, _num(_get(k, "FF_DEPTH"), 9.0))
    f["xsa_layers"] = _layer_count(_get(k, "FF_XSA_LAYERS"), depth)
    src = _get(k, "FF_ATTN_SRC").replace(" ", "")
    if src:
        consumers = src.split(":", 1)[1] if ":" in src else src
        f["attn_src_layers"] = _layer_count(consumers, depth)
    lanes = _num(_get(k, "FF_LANES"), 1.0)
    f["lanes_extra"] = max(0.0, lanes - 1.0)
    f["pool_last"] = 1.0 if _num(_get(k, "FF_POOL_LAST"), 0.0) > 0 else 0.0
    for feat, knob in (("rope_fn", "FF_ROPE_FN"), ("qk_gain_fold", "FF_QK_GAIN_FOLD"),
                       ("rmsnorm_fn", "FF_RMSNORM_FN"), ("mtp_loss_next", "FF_MTP_LOSS_NEXT")):
        f[feat] = 1.0 if _flag(_get(k, knob)) else 0.0
    f["ce_onehot"] = 1.0 if _num(_get(k, "FF_CE_ONEHOT"), 0.0) != 0 else 0.0

    mtp = int(_num(_get(k, "FF_MTP"), 1.0))
    f["mtp_off"] = 1.0 if mtp == 0 else 0.0
    f["mtp_2"] = 1.0 if mtp == 2 else 0.0

    acc = int(_num(_get(k, "FF_ACC_IN_GRAPH"), 1.0))
    f["acc_in_graph_off"] = 1.0 if acc == 0 else 0.0
    f["muon_zero_off"] = 0.0 if _flag(_get(k, "FF_MUON_ZERO")) else 1.0

    ema_on = _flag(_get(k, "FF_EMA"))
    f["ema_off"] = 0.0 if ema_on else 1.0
    ema_every = max(1.0, _num(_get(k, "FF_EMA_EVERY"), 32.0))
    f["ema_rate"] = (32.0 / ema_every - 1.0) if ema_on else 0.0

    ve = int(_num(_get(k, "FF_VALUE_EMBED"), 1.0))
    f["value_embed_off"] = 1.0 if ve == 0 else 0.0
    f["value_embed_gate"] = 1.0 if ve == 3 else 0.0

    f["depth_rel"] = depth / 9.0 - 1.0
    aspect = max(1.0, _num(_get(k, "FF_ASPECT_RATIO"), 113.0))
    f["log_aspect"] = math.log(aspect / 113.0)
    mb = max(1.0, _num(_get(k, "FF_MB"), 8.0))
    f["log2_mb"] = math.log2(mb / 8.0)

    fused = int(_num(_get(k, "FF_FUSED_CE"), 8.0))
    if fused <= 0:
        f["fused_ce_off"] = 1.0
    else:
        f["log2_fused_ce"] = math.log2(fused / 8.0)

    head_dim = max(1.0, _num(_get(k, "FF_HEAD_DIM"), 256.0))
    kv = int(_num(_get(k, "FF_KV_HEADS"), 1.0))
    if kv <= 0:  # 0 = n_kv_heads == n_head; n_head ~ base_dim / head_dim
        kv = max(1, int(round(depth * aspect / head_dim)))
    f["log2_kv_heads"] = math.log2(kv)
    f["log2_head_dim"] = math.log2(head_dim / 256.0)

    total = max(1.0, _num(_get(k, "FF_TOTAL_BATCH"), 262144.0))
    f["log2_total_batch"] = math.log2(total / 262144.0)
    return f


def structural_features(chip: str, is_screen: bool) -> Dict[str, float]:
    chip = (chip or "C").strip().upper()
    d = 1.0 if chip == "D" else 0.0
    ab = 1.0 if chip in ("A", "B") else 0.0
    s = 1.0 if is_screen else 0.0
    return {"chip_D": d, "chip_AB": ab, "screen": s, "chip_D_x_screen": d * s, "chip_AB_x_screen": ab * s}


# ----------------------------------------------------------------------------------------------
# Code-version keys and lineage similarity
# ----------------------------------------------------------------------------------------------
_VER_TOKEN = re.compile(r"([a-z]+)(\d+)")

# code dir / submission spelling (after the code_ prefix is dropped and lower-cased) -> lineage key.
# From known-facts.json queue_environment.code_dirs and dataset-qa.md section 8.
LINEAGE_ALIASES: Dict[str, str] = {
    "k12off": "m12", "k13off": "m13", "k14off": "m14", "k15off": "m15",
    "k52": "m5", "k52m6": "m6", "k53": "m6", "k54b": "m7", "k54bmin": "m7", "k54cmin": "m8",
    "k56": "m9", "k57": "m9", "m9acc1": "m9", "m9ln": "m9", "m9off": "m9", "m9oh": "m9",
    "m10b": "m10", "m10off": "m10", "m10xa": "m10", "m10xl": "m10", "m10lr": "m10", "m11c": "m11",
    # submissions (known-facts k57_k59, offsets per_upload): K50 M3, K51 M4, K52 M5, K53 M6,
    # K54a/b M7, K55 M8, K56/K57/K57T M9, K58/K59 M12
    "k50": "m3", "k51": "m4", "k53_": "m6", "k54a": "m7", "k55": "m8", "k57t": "m9",
    "k58": "m12", "k58lk": "m12", "k58rf": "m12", "k59": "m12",
}


def version_key(code_version: str) -> str:
    """Case-insensitive LINEAGE key: 'code_m9' -> 'm9', 'M12' -> 'm12', 'code_k12off' -> 'm12'
    (K57's recipe baked on M12 with the new flags off), 'code_k57' -> 'm9', 'K59' -> 'm12'."""
    v = (code_version or "").strip().lower()
    if v.startswith("code_"):
        v = v[len("code_"):]
    if v in LINEAGE_ALIASES:
        return LINEAGE_ALIASES[v]
    for suffix in ("_off", "off"):
        if v.endswith(suffix) and len(v) > len(suffix):
            v = v[: -len(suffix)]
            break
    return LINEAGE_ALIASES.get(v, v)


def lineage_label(key: str) -> str:
    """Canonical spelling of a lineage key for reports: 'm12' -> 'M12', 'm7a' -> 'M7a'."""
    m = _VER_TOKEN.match(key)
    if m:
        return m.group(1).upper() + m.group(2) + key[m.end():]
    return key.upper()


def _lineage_parts(key: str) -> Optional[Tuple[str, int]]:
    m = _VER_TOKEN.search(key)
    return (m.group(1), int(m.group(2))) if m else None


def version_similarity(a: str, b: str) -> float:
    """3.0 for the same key; same family letter(s) score 0.5 + 1 / (1 + |n_a - n_b|) (M12 vs M13 = 1.0,
    M12 vs M14 = 0.83); otherwise a difflib ratio in [0, 1], so a close string match beats a
    far-off same-family number."""
    ka, kb = version_key(a), version_key(b)
    if ka == kb:
        return 3.0
    pa, pb = _lineage_parts(ka), _lineage_parts(kb)
    if pa and pb and pa[0] == pb[0]:
        return 0.5 + 1.0 / (1.0 + abs(pa[1] - pb[1]))
    return difflib.SequenceMatcher(None, ka, kb).ratio()


def nearest_version(code_version: str, known: Iterable[str]) -> Optional[str]:
    known = list(known)
    if not known:
        return None
    key = version_key(code_version)
    for v in known:
        if version_key(v) == key:
            return v
    # Best similarity; ties -> the higher lineage number (more recent), then alphabetical.
    def rank(v: str):
        p = _lineage_parts(version_key(v))
        return (version_similarity(code_version, v), p[1] if p else -1, -ord(v[0]) if v else 0)
    return max(known, key=rank)


# ----------------------------------------------------------------------------------------------
# Ridge helpers
# ----------------------------------------------------------------------------------------------
def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, penalize: np.ndarray) -> np.ndarray:
    """Weighted ridge via lstsq on the augmented system [sqrt(W) X; sqrt(lam) P] beta = [sqrt(W) y; 0]."""
    sw = np.sqrt(np.asarray(w, dtype=float))[:, None]
    a = np.asarray(X, dtype=float) * sw
    b = np.asarray(y, dtype=float) * sw[:, 0]
    pen = np.diag(np.sqrt(lam) * np.asarray(penalize, dtype=float))
    a_aug = np.vstack([a, pen])
    b_aug = np.concatenate([b, np.zeros(X.shape[1])])
    beta, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
    return beta


def record_weight(n: int, n_half: float = N_HALF) -> float:
    n = max(int(n or 0), 1)
    if n_half <= 0:
        return 1.0
    return n / (n + n_half)


@dataclass
class _PhaseFit:
    phase: str
    columns: List[str]
    coef: np.ndarray
    n: int
    rmse_log: float
    versions: List[str]                       # lineage labels with data in this phase
    per_version_n: Dict[str, int] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)   # feature -> fitted runs with a non-zero value
    X: Optional[np.ndarray] = None            # training design (for the residual kNN term)
    resid: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None

    def coef_of(self, col: str) -> float:
        try:
            return float(self.coef[self.columns.index(col)])
        except ValueError:
            return 0.0


# ----------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------
class StepTimeModel:
    """Per-phase ridge regression of log(step time) on lineage, chip, screen and knob features.

    Options (all keyword, all with the validated defaults):
      lam        ridge penalty; n_half  weight half-point (0 = equal weights, large = weight ~ n);
      features   knob feature names or a FEATURE_SETS key; min_n  minimum logged steps per phase;
      exclude_versions  code_version spellings never fitted (the `code` placeholder);
      knn_k / knn_bandwidth  residual nearest-neighbour term over the fitted runs (0 = off).
    """

    def __init__(self, lam: float = DEFAULT_LAMBDA, n_half: float = N_HALF, phases: Sequence[str] = K_PHASES,
                 features: Any = None, min_n: int = MIN_N, exclude_versions: Sequence[str] = EXCLUDE_VERSIONS,
                 knn_k: int = 0, knn_bandwidth: float = 0.5):
        self.lam = float(lam)
        self.n_half = float(n_half)
        self.phases = tuple(phases)
        if features is None:
            self.features: Tuple[str, ...] = KNOB_FEATURES
        elif isinstance(features, str):
            self.features = FEATURE_SETS[features]
        else:
            self.features = tuple(features)
        self.min_n = int(min_n)
        self.exclude_versions = tuple(str(v).strip() for v in exclude_versions)
        self.knn_k = int(knn_k)
        self.knn_bandwidth = float(knn_bandwidth)
        self._fits: Dict[str, _PhaseFit] = {}
        self._versions: Dict[str, str] = {}     # key -> lineage label
        self._records: List[RunRecord] = []
        self._fallback_sd_log: Optional[float] = None
        self.skipped: Dict[str, int] = {}

    # -- data -----------------------------------------------------------------------------------
    def _usable(self, rec: RunRecord, phase: str) -> bool:
        st = rec.step_time
        v = getattr(st, phase, None)
        n = int(getattr(st, "n_" + phase, 0) or 0)
        return (
            rec.code_version is not None and str(rec.code_version).strip() != ""
            and str(rec.code_version).strip() not in self.exclude_versions
            and (rec.chip or "").upper() in CHIPS
            and isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            and n >= self.min_n
        )

    def _feature_dict(self, code_version: str, knobs: Optional[Dict[str, Any]], chip: str,
                      is_screen: bool) -> Dict[str, float]:
        feats = {"intercept": 1.0}
        feats.update(structural_features(chip, is_screen))
        kf = knob_features(knobs, code_version)
        feats.update({name: kf[name] for name in self.features})
        feats[VERSION_PREFIX + version_key(code_version)] = 1.0
        return feats

    def _row(self, code_version: str, knobs: Optional[Dict[str, Any]], chip: str, is_screen: bool,
             columns: List[str]) -> np.ndarray:
        feats = self._feature_dict(code_version, knobs, chip, is_screen)
        return np.array([feats.get(c, 0.0) for c in columns], dtype=float)

    def fit(self, records: List[RunRecord]) -> "StepTimeModel":
        self._records = list(records)
        self._fits = {}
        self._versions = {}
        self._fallback_sd_log = None
        self.skipped = {"no_code_version": 0, "excluded_version": 0, "bad_chip": 0, "low_n": 0}
        feats: List[Optional[Dict[str, float]]] = []
        for r in records:
            cv = _norm_version(r.code_version)
            if cv == "":
                self.skipped["no_code_version"] += 1
                feats.append(None)
                continue
            if cv in self.exclude_versions:
                self.skipped["excluded_version"] += 1
                feats.append(None)
                continue
            if (r.chip or "").upper() not in CHIPS:
                self.skipped["bad_chip"] += 1
                feats.append(None)
                continue
            if not any(self._usable(r, p) for p in self.phases):
                if any(isinstance(getattr(r.step_time, p, None), (int, float)) for p in self.phases):
                    self.skipped["low_n"] += 1
                feats.append(None)
                continue
            self._versions.setdefault(version_key(cv), lineage_label(version_key(cv)))
            feats.append(self._feature_dict(cv, r.knobs, r.chip, bool(r.is_screen)))
        ver_cols = [VERSION_PREFIX + k for k in sorted(self._versions)]
        columns = ["intercept", *STRUCT_FEATURES, *self.features, *ver_cols]
        penalize = np.array([0.0] + [1.0] * (len(columns) - 1))

        for phase in self.phases:
            rows, ys, ws, vers = [], [], [], {}
            for r, fd in zip(records, feats):
                if fd is None or not self._usable(r, phase):
                    continue
                rows.append([fd.get(c, 0.0) for c in columns])
                ys.append(math.log(getattr(r.step_time, phase)))
                ws.append(record_weight(getattr(r.step_time, "n_" + phase, 0), self.n_half))
                key = version_key(r.code_version)
                vers[key] = vers.get(key, 0) + 1
            if not rows:
                continue
            X, y, w = np.array(rows, dtype=float), np.array(ys), np.array(ws)
            beta = _ridge(X, y, w, self.lam, penalize)
            resid = y - X @ beta
            n_fixed = len(columns) - len(ver_cols)
            rmse = float(math.sqrt(np.sum(w * resid ** 2) / np.sum(w))) if len(rows) > n_fixed else 0.0
            support = {c: int(np.count_nonzero(X[:, i])) for i, c in enumerate(columns)}
            self._fits[phase] = _PhaseFit(
                phase=phase, columns=columns, coef=beta, n=len(rows), rmse_log=rmse,
                versions=[self._versions[k] for k in sorted(vers)],
                per_version_n={self._versions[k]: n for k, n in vers.items()},
                support=support, X=X, resid=resid, w=w,
            )
        return self

    # -- introspection --------------------------------------------------------------------------
    def known_versions(self, phase: Optional[str] = None) -> List[str]:
        if phase is None:
            return [self._versions[k] for k in sorted(self._versions)]
        f = self._fits.get(phase)
        return list(f.versions) if f else []

    def nearest_version(self, code_version: str, phase: Optional[str] = None) -> Optional[str]:
        return nearest_version(code_version, self.known_versions(phase))

    def phase_fit_summary(self) -> Dict[str, Dict[str, Any]]:
        return {p: {"n": f.n, "rmse_pct": 100.0 * f.rmse_log, "versions": list(f.versions),
                    "per_version_n": dict(f.per_version_n)}
                for p, f in self._fits.items()}

    def coefficient_table(self, phase: str = "k4") -> List[Dict[str, Any]]:
        """Every fitted column: coefficient (log), % multiplier and the number of fitted runs that
        carry a non-zero value ('support'; 0 = the feature is not identifiable from this data)."""
        f = self._fits.get(phase)
        if f is None:
            return []
        out = []
        for col, c in zip(f.columns, f.coef):
            out.append({"feature": col, "coef": float(c), "pct": 100.0 * (math.exp(c) - 1.0),
                        "support": f.support.get(col, 0),
                        "doc": FEATURE_DOC.get(col, "lineage baseline" if col.startswith(VERSION_PREFIX) else "")})
        return out

    def unsupported_features(self, code_version: str, knobs: Optional[Dict[str, Any]], phase: str = "k4",
                             chip: str = "C", is_screen: bool = False) -> List[str]:
        """Knob and structural features this prediction needs that NO fitted run of the phase carries
        (e.g. a mechanism only one lineage ran, or a chip-D screen when none was fitted)."""
        f = self._fits.get(phase)
        if f is None:
            return []
        kf = knob_features(knobs, code_version)
        out = [name for name in self.features if kf[name] != 0.0 and f.support.get(name, 0) == 0]
        sf = structural_features(chip, is_screen)
        out += [name for name in STRUCT_FEATURES if sf[name] != 0.0 and f.support.get(name, 0) == 0]
        return out

    def support(self, phase: str = "k4") -> Dict[str, Any]:
        """Knob -> values the fitted runs of the phase used (effective env, so baked knobs count):
        {"values": [...], "min", "max"} for numeric knobs, a list for categorical ones. Consumed by
        ffsim.search.support_check to tag a candidate knob value the step-time fit never saw."""
        f = self._fits.get(phase)
        if f is None:
            return {}
        seen: Dict[str, set] = {k: set() for k in KNOB_DEFAULTS}
        for r in self._records:
            if not self._usable(r, phase):
                continue
            env = effective_env(r.knobs, r.code_version)
            for k in seen:
                seen[k].add(env.get(k, KNOB_DEFAULTS[k]))
        out: Dict[str, Any] = {}
        for k, vals in seen.items():
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    nums = None
                    break
            if nums:
                out[k] = {"values": sorted(nums), "min": min(nums), "max": max(nums)}
            else:
                out[k] = sorted(vals)
        return out

    # -- prediction -----------------------------------------------------------------------------
    def _resolve(self, code_version: str, phase: str) -> Tuple[Optional[str], str]:
        """(lineage label to use for this phase, note). None when the phase has no data."""
        f = self._fits.get(phase)
        if f is None or not f.versions:
            return None, f"{phase}: no fitted data, used PHASE_RATIO_FALLBACK[{phase}] x k4"
        key = version_key(code_version)
        for v in f.versions:
            if version_key(v) == key:
                return v, ""
        near = nearest_version(code_version, f.versions)
        return near, f"{phase}: fallback {code_version!r} -> nearest known lineage {near!r}"

    def _knn_log(self, phase: str, row: np.ndarray, chip: str, is_screen: bool) -> float:
        """Residual nearest-neighbour term: kernel-weighted mean residual of the k closest fitted
        runs in knob-feature space with the same chip and screen flag (0 when knn_k == 0)."""
        f = self._fits[phase]
        if self.knn_k <= 0 or f.X is None or f.resid is None:
            return 0.0
        cols = [i for i, c in enumerate(f.columns) if c in self.features]
        sidx = [f.columns.index(c) for c in STRUCT_FEATURES]
        same = np.all(f.X[:, sidx] == row[sidx], axis=1)
        if not np.any(same):
            return 0.0
        d = np.sqrt(np.sum((f.X[same][:, cols] - row[cols]) ** 2, axis=1))
        order = np.argsort(d)[: self.knn_k]
        kw = np.exp(-0.5 * (d[order] / self.knn_bandwidth) ** 2) * f.w[same][order]
        if kw.sum() <= 0:
            return 0.0
        return float(np.sum(kw * f.resid[same][order]) / (np.sum(kw) + 0.5))

    def _predict_log(self, phase: str, version: str, knobs: Optional[Dict[str, Any]], chip: str,
                     is_screen: bool, code_version: Optional[str] = None) -> float:
        f = self._fits[phase]
        # knobs are filled from the CANDIDATE's code dir, the lineage column from the resolved version
        row = self._row(code_version or version, knobs, chip, is_screen, f.columns)
        vcol = VERSION_PREFIX + version_key(version)
        for i, c in enumerate(f.columns):
            if c.startswith(VERSION_PREFIX):
                row[i] = 1.0 if c == vcol else 0.0
        return float(row @ f.coef) + self._knn_log(phase, row, chip, is_screen)

    def predict(self, code_version: str, knobs: Optional[Dict[str, Any]], chip: str = "C",
                is_screen: bool = False) -> Dict[str, Any]:
        """Median seconds per step in each phase plus sd_k4 (seconds), 'version_used' and 'notes'
        (empty unless a fallback happened or a mechanism is unsupported by the fitted runs)."""
        if not self._fits:
            raise RuntimeError("StepTimeModel.predict before fit (or fit saw no usable records)")
        out: Dict[str, Any] = {}
        notes: List[str] = []
        version_used: Optional[str] = None
        fallback = False
        # k4 first: the other phases may need it as a base.
        order = ["k4"] + [p for p in self.phases if p != "k4"]
        for phase in order:
            v, note = self._resolve(code_version, phase)
            if note:
                notes.append(note)
            if v is None:
                out[phase] = None
                continue
            if phase == "k4" or version_used is None:
                version_used = v
            fallback = fallback or version_key(v) != version_key(code_version)
            out[phase] = math.exp(self._predict_log(phase, v, knobs, chip, is_screen, code_version))
        if out.get("k4") is None:
            raise RuntimeError("StepTimeModel: no k4 data was fitted; cannot predict")
        for phase in self.phases:
            if out.get(phase) is None:
                out[phase] = out["k4"] * PHASE_RATIO_FALLBACK.get(phase, 1.0)
        sd_log = self._fits["k4"].rmse_log
        if fallback:
            extra = self._fallback_sd_log if self._fallback_sd_log is not None else FALLBACK_SD_LOG
            sd_log = math.sqrt(sd_log ** 2 + extra ** 2)
        novel = self.unsupported_features(code_version, knobs, "k4", chip, is_screen)
        if novel:
            note = "k4: feature(s) no fitted run carries, cost unknown: " + ",".join(novel)
            # Knob mechanisms are charged the +3% prior each; a structural gap (a chip-D screen when
            # no D screen was fitted) is not a graph change and is not.
            n_prior = sum(1 for name in novel if name in self.features)
            if n_prior:
                for phase in self.phases:
                    out[phase] *= math.exp(NOVEL_PRIOR_LOG * n_prior)
                note += f"; charged NOVEL_PRIOR +{100.0 * (math.exp(NOVEL_PRIOR_LOG) - 1.0):.1f}% x{n_prior}"
            notes.append(note)
            sd_log = math.sqrt(sd_log ** 2 + NOVEL_SD_LOG ** 2)
        out["sd_k4"] = out["k4"] * sd_log
        out["version_used"] = version_used
        out["notes"] = "; ".join(notes)
        return out

    def explain(self, code_version: str, knobs: Optional[Dict[str, Any]], chip: str = "C",
                is_screen: bool = False, phase: str = "k4") -> Dict[str, Any]:
        """Feature contributions (log units and %) for one prediction, largest |effect| first."""
        v, note = self._resolve(code_version, phase)
        if v is None:
            return {"phase": phase, "version_used": None, "notes": note, "contributions": [], "predicted": None}
        f = self._fits[phase]
        row = self._row(code_version, knobs, chip, is_screen, f.columns)
        vcol = VERSION_PREFIX + version_key(v)
        for i, c in enumerate(f.columns):
            if c.startswith(VERSION_PREFIX):
                row[i] = 1.0 if c == vcol else 0.0
        contribs = []
        for col, x, c in zip(f.columns, row, f.coef):
            if x == 0.0:
                continue
            contribs.append({
                "feature": col, "value": float(x), "coef": float(c), "contribution_log": float(x * c),
                "pct": 100.0 * (math.exp(x * c) - 1.0), "support": f.support.get(col, 0),
                "doc": FEATURE_DOC.get(col, "lineage baseline" if col.startswith(VERSION_PREFIX) else ""),
            })
        total = float(row @ f.coef)
        knn = self._knn_log(phase, row, chip, is_screen)
        if knn != 0.0:
            contribs.append({"feature": "knn_residual", "value": 1.0, "coef": knn, "contribution_log": knn,
                             "pct": 100.0 * (math.exp(knn) - 1.0), "support": self.knn_k,
                             "doc": "residual nearest-neighbour correction"})
            total += knn
        for name in self.unsupported_features(code_version, knobs, "k4", chip, is_screen):
            if name not in self.features:
                continue
            contribs.append({"feature": "novel_prior:" + name, "value": 1.0, "coef": NOVEL_PRIOR_LOG,
                             "contribution_log": NOVEL_PRIOR_LOG, "pct": 100.0 * (math.exp(NOVEL_PRIOR_LOG) - 1.0),
                             "support": 0, "doc": "prior cost of a mechanism no fitted run carries (NOVEL_PRIOR_LOG)"})
            total += NOVEL_PRIOR_LOG
        contribs.sort(key=lambda d: (d["feature"] != "intercept", -abs(d["contribution_log"])))
        return {"phase": phase, "version_used": v, "notes": note, "predicted": math.exp(total),
                "log_total": total, "contributions": contribs,
                "effective_env": effective_env(knobs, code_version)}

    # -- validation -----------------------------------------------------------------------------
    def _clone(self) -> "StepTimeModel":
        return StepTimeModel(lam=self.lam, n_half=self.n_half, phases=self.phases, features=self.features,
                             min_n=self.min_n, exclude_versions=self.exclude_versions, knn_k=self.knn_k,
                             knn_bandwidth=self.knn_bandwidth)

    def loo_report(self, phase: str = "k4", target_pct: float = 0.5) -> Dict[str, Any]:
        """Leave-one-lineage-out: refit without a lineage, predict its runs through the
        nearest-lineage fallback, report the relative error. This is the spec's acceptance test
        (<= 0.5% on k4) and also sets the sd added to predictions for unseen lineages.

        A held-out run that carries a mechanism NO training-fold run carries (its feature column
        is all zero in the fold) is 'novel': the regression cannot know its cost, so it is scored
        separately. `mean_abs_pct` is over every held-out run; `mean_abs_pct_reachable` over the
        runs whose mechanisms the fold had seen. Each comes in both averaging conventions:
        `_by_run` (every held-out run counts once, the convention `rmse_pct` uses) and
        `_by_lineage` (the mean of the per-lineage means, where a 3-run lineage cancels a 48-run
        one). The unsuffixed keys are the run-weighted values and `passes` judges
        `mean_abs_pct_reachable_by_run`. Novel runs are scored by the regression alone (predict()'s
        NOVEL_PRIOR is not applied here): their error is the raw cost of the unseen mechanism."""
        f = self._fits.get(phase)
        if f is None:
            return {"phase": phase, "n_versions": 0, "n_scored": 0, "per_version": {}, "mean_abs_pct": None,
                    "mean_abs_pct_by_run": None, "mean_abs_pct_by_lineage": None,
                    "mean_abs_pct_reachable": None, "mean_abs_pct_reachable_by_run": None,
                    "mean_abs_pct_reachable_by_lineage": None, "max_abs_pct": None, "rmse_pct": None,
                    "rmse_pct_reachable": None, "n_runs": 0, "n_novel": 0, "n_reachable": 0,
                    "n_reachable_above_target": 0, "target_pct": target_pct,
                    "judged_on": "mean_abs_pct_reachable_by_run", "passes": False}
        per_version: Dict[str, Dict[str, Any]] = {}
        all_errs: List[float] = []
        reach_errs: List[float] = []
        for held in f.versions:
            hk = version_key(held)
            train = [r for r in self._records if version_key(r.code_version or "") != hk]
            test = [r for r in self._records if version_key(r.code_version or "") == hk and self._usable(r, phase)]
            sub = self._clone().fit(train)
            if phase not in sub._fits or not sub._fits[phase].versions or not test:
                per_version[held] = {"n": len(test), "fallback_to": None, "mean_abs_pct": None,
                                     "max_abs_pct": None, "bias_pct": None, "n_novel": 0, "novel_features": [],
                                     "mean_abs_pct_reachable": None, "runs": [], "skipped": "no training data"}
                continue
            near = nearest_version(held, sub._fits[phase].versions)
            errs, reach, runs, novel_feats = [], [], [], set()
            for r in test:
                pred = math.exp(sub._predict_log(phase, near, r.knobs, r.chip, bool(r.is_screen), r.code_version))
                err = 100.0 * (pred / getattr(r.step_time, phase) - 1.0)
                novel = sub.unsupported_features(r.code_version, r.knobs, phase, r.chip, bool(r.is_screen))
                errs.append(err)
                runs.append({"run_id": r.run_id, "is_screen": bool(r.is_screen), "chip": r.chip,
                             "measured": float(getattr(r.step_time, phase)), "predicted": pred,
                             "err_pct": err, "novel": list(novel)})
                if novel:
                    novel_feats.update(novel)
                else:
                    reach.append(err)
            e = np.array(errs)
            per_version[held] = {
                "n": len(errs), "fallback_to": near,
                "mean_abs_pct": float(np.mean(np.abs(e))), "max_abs_pct": float(np.max(np.abs(e))),
                "bias_pct": float(np.mean(e)),
                "n_novel": len(errs) - len(reach), "novel_features": sorted(novel_feats),
                "mean_abs_pct_reachable": float(np.mean(np.abs(reach))) if reach else None,
                "runs": runs,
            }
            all_errs.extend(errs)
            reach_errs.extend(reach)
        scored = [d for d in per_version.values() if d.get("mean_abs_pct") is not None]
        reach_scored = [d for d in scored if d.get("mean_abs_pct_reachable") is not None]
        # Two conventions, both reported. by_lineage: the mean of the per-lineage means (a 3-run
        # lineage weighs as much as a 48-run one). by_run: every held-out run counts once, the
        # convention rmse_pct uses. They part when the big lineages are the hard ones (M9, M12).
        mean_abs_lineage = float(np.mean([d["mean_abs_pct"] for d in scored])) if scored else None
        mean_abs_reach_lineage = (float(np.mean([d["mean_abs_pct_reachable"] for d in reach_scored]))
                                  if reach_scored else None)
        mean_abs_run = float(np.mean(np.abs(all_errs))) if all_errs else None
        mean_abs_reach_run = float(np.mean(np.abs(reach_errs))) if reach_errs else None
        n_reach_above = int(np.sum(np.abs(reach_errs) > target_pct)) if reach_errs else 0
        max_abs = float(max(d["max_abs_pct"] for d in scored)) if scored else None
        rmse_pct = float(math.sqrt(np.mean(np.square(all_errs)))) if all_errs else None
        rmse_reach = float(math.sqrt(np.mean(np.square(reach_errs)))) if reach_errs else None
        if rmse_reach is not None:
            self._fallback_sd_log = rmse_reach / 100.0
        elif rmse_pct is not None:
            self._fallback_sd_log = rmse_pct / 100.0
        judged = mean_abs_reach_run if mean_abs_reach_run is not None else mean_abs_run
        return {"phase": phase, "n_versions": len(f.versions), "n_scored": len(scored), "per_version": per_version,
                "mean_abs_pct": mean_abs_run, "mean_abs_pct_by_run": mean_abs_run,
                "mean_abs_pct_by_lineage": mean_abs_lineage,
                "mean_abs_pct_reachable": mean_abs_reach_run, "mean_abs_pct_reachable_by_run": mean_abs_reach_run,
                "mean_abs_pct_reachable_by_lineage": mean_abs_reach_lineage,
                "max_abs_pct": max_abs, "rmse_pct": rmse_pct, "rmse_pct_reachable": rmse_reach,
                "n_runs": len(all_errs), "n_novel": len(all_errs) - len(reach_errs),
                "n_reachable": len(reach_errs), "n_reachable_above_target": n_reach_above,
                "target_pct": target_pct, "judged_on": "mean_abs_pct_reachable_by_run",
                "passes": bool(judged is not None and judged <= target_pct)}

    def loro_report(self, phase: str = "k4") -> Dict[str, Any]:
        """Leave-one-run-out within known lineages: the within-lineage prediction noise floor."""
        idx = [i for i, r in enumerate(self._records) if self._usable(r, phase)]
        errs = []
        for i in idx:
            r = self._records[i]
            sub = self._clone().fit([x for j, x in enumerate(self._records) if j != i])
            if phase not in sub._fits:
                continue
            v = nearest_version(r.code_version, sub._fits[phase].versions)
            if v is None:
                continue
            pred = math.exp(sub._predict_log(phase, v, r.knobs, r.chip, bool(r.is_screen), r.code_version))
            errs.append(100.0 * (pred / getattr(r.step_time, phase) - 1.0))
        e = np.array(errs) if errs else np.zeros(0)
        return {"phase": phase, "n": len(errs),
                "mean_abs_pct": float(np.mean(np.abs(e))) if errs else None,
                "rmse_pct": float(math.sqrt(np.mean(e ** 2))) if errs else None,
                "max_abs_pct": float(np.max(np.abs(e))) if errs else None}


# ----------------------------------------------------------------------------------------------
# Steps from step time: the campaign's FF_ACCUM_SCHED phase logic
# ----------------------------------------------------------------------------------------------
def parse_accum_sched(accum_sched: str) -> List[Tuple[int, Optional[float]]]:
    """'1:0.06,2:0.2,4' -> [(1, 0.06), (2, 0.2), (4, None)]; '2:0.12,4' -> [(2, 0.12), (4, None)];
    '4' -> [(4, None)]. The last entry is the final k (K); each earlier entry runs until
    charged / time_target reaches its fraction (train_ff.py accum_phase_of)."""
    parts = [p.strip() for p in (accum_sched or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("FF_ACCUM_SCHED is empty")
    sched: List[Tuple[int, Optional[float]]] = []
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        if ":" in p:
            if last:
                raise ValueError(f"FF_ACCUM_SCHED={accum_sched!r}: the final entry must be a bare K")
            k_s, u_s = p.split(":", 1)
            k, until = int(k_s), float(u_s)
            if not (0.0 < until <= 1.0):
                raise ValueError(f"FF_ACCUM_SCHED={accum_sched!r}: until={until} must be in (0, 1]")
            sched.append((k, until))
        else:
            if not last:
                raise ValueError(f"FF_ACCUM_SCHED={accum_sched!r}: only the final entry may lack ':until'")
            sched.append((int(p), None))
    k_final = sched[-1][0]
    if k_final < 1:
        raise ValueError(f"FF_ACCUM_SCHED={accum_sched!r}: K must be >= 1")
    fracs = [u for _, u in sched[:-1]]
    if any(b <= a for a, b in zip(fracs, fracs[1:])):
        raise ValueError(f"FF_ACCUM_SCHED={accum_sched!r}: phase fractions must increase")
    for k, _ in sched[:-1]:
        if not (1 <= k < k_final) or k_final % k:
            raise ValueError(f"FF_ACCUM_SCHED={accum_sched!r}: k={k} must be in [1, K) and divide K={k_final}")
    return sched


def _phase_seconds(step_time: Dict[str, Any], k: int) -> float:
    key = f"k{k}"
    v = step_time.get(key) if isinstance(step_time, dict) else getattr(step_time, key, None)
    if v is None or not math.isfinite(float(v)) or float(v) <= 0:
        raise ValueError(f"step_time[{key!r}] is needed by the schedule but is {v!r}")
    return float(v)


def simulate_phases(step_time: Dict[str, Any], charged_seconds: float, accum_sched: str = DEFAULT_ACCUM_SCHED,
                    warmup_steps: int = DEFAULT_WARMUP_STEPS, time_target: Optional[float] = None,
                    startup_steps_excluded: int = STARTUP_STEPS_EXCLUDED, overhead_frac: float = 0.0) -> Dict[str, Any]:
    """Walk the run step by step the way train_ff.py does.

    - steps 0..startup_steps_excluded-1 (step 0 = compile) run before the charged clock starts;
    - the warm-up (step < warmup_steps) runs at k = K (accum_phase_of: phase 0 is "k = K");
    - after the warm-up the phase for the NEXT step is chosen from progress = charged / time_target
      (FF_TIME_TARGET; defaults to charged_seconds, K59: 1793 vs 1790 charged), k0 while
      progress < u0, k1 while progress < u1, K after;
    - the run stops before the step that would not fit inside charged_seconds;
    - overhead_frac scales every phase's median by (1 + overhead_frac) (see MEDIAN_TO_MEAN_OVERHEAD).

    Returns {"steps": last step index + 1 (how the logs count), "switches": [(step, k), ...],
    "charged_used": seconds, "steps_by_k": {k: n}}.
    """
    sched = parse_accum_sched(accum_sched)
    k_final = sched[-1][0]
    if charged_seconds <= 0:
        raise ValueError("charged_seconds must be > 0")
    target = float(time_target) if time_target else float(charged_seconds)
    dt = {k: _phase_seconds(step_time, k) * (1.0 + float(overhead_frac)) for k, _ in sched}
    charged = 0.0
    step = 0
    switches: List[Tuple[int, int]] = []
    steps_by_k: Dict[int, int] = {}
    k_prev: Optional[int] = None
    while True:
        if step < warmup_steps:
            k = k_final
        else:
            progress = charged / target
            k = k_final
            for kk, until in sched[:-1]:
                if progress < until:
                    k = kk
                    break
        if step >= startup_steps_excluded:
            if charged + dt[k] > charged_seconds:
                break
            charged += dt[k]
        if k != k_prev:
            switches.append((step, k))
            k_prev = k
        steps_by_k[k] = steps_by_k.get(k, 0) + 1
        step += 1
        if step > 10_000_000:
            raise RuntimeError("simulate_phases did not terminate")
    return {"steps": step, "switches": switches, "charged_used": charged, "steps_by_k": steps_by_k}


def steps_from_step_time(step_time: Dict[str, Any], charged_seconds: float, accum_sched: str = DEFAULT_ACCUM_SCHED,
                         warmup_steps: int = DEFAULT_WARMUP_STEPS, time_target: Optional[float] = None,
                         overhead_frac: float = 0.0) -> int:
    """Optimizer steps a run completes in charged_seconds (last step index + 1, as train.log counts).
    Anchor: chip C k4 0.928 / k2 ~0.51 / k1 ~0.29 s, charged 1790 s -> ~2357 (K59 C40) within 2%
    (C40 from its own medians: +0.47% raw, -0.13% with overhead_frac=MEDIAN_TO_MEAN_OVERHEAD; over
    the 129 chip C/D full runs the raw walk is +0.78% +- 0.33 and the corrected one +0.20% +- 0.33,
    see the table at MEDIAN_TO_MEAN_OVERHEAD)."""
    return int(simulate_phases(step_time, charged_seconds, accum_sched, warmup_steps, time_target,
                               overhead_frac=overhead_frac)["steps"])


# ----------------------------------------------------------------------------------------------
# Hook for a later first-principles / op-count feature extractor (NOT implemented on purpose)
# ----------------------------------------------------------------------------------------------
def features_from_code(train_py_path: str) -> Dict[str, float]:
    """TODO: derive step-time features from a train.py (op counts per block from the torch.compile
    graph, elementwise passes over the 4C MLP tensor, collective count/bytes, eager-optimizer op
    count). The spec says the compiler is not linear in op count, so this stays a stub until the
    regression on measured runs has enough lineages to calibrate such features. Returns {}."""
    return {}


__all__ = [
    "KNOB_DEFAULTS", "KNOB_FEATURES", "FEATURE_SETS", "STRUCT_FEATURES", "FEATURE_DOC", "CODE_DIR_KNOBS",
    "LINEAGE_ALIASES", "EXCLUDE_VERSIONS", "PHASE_RATIO_FALLBACK", "MIN_N", "N_HALF", "NOVEL_PRIOR_LOG",
    "MEDIAN_TO_MEAN_OVERHEAD", "STARTUP_STEPS_EXCLUDED", "DEFAULT_ACCUM_SCHED", "DEFAULT_WARMUP_STEPS",
    "StepTimeModel", "effective_env", "knob_features", "structural_features", "version_key", "lineage_label",
    "version_similarity", "nearest_version", "record_weight", "parse_accum_sched", "simulate_phases",
    "steps_from_step_time", "features_from_code",
]
