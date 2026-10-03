"""FrontierForge Trainium simulator: a Gradio Space over `ffsim` route 1 (numpy surrogate).

The three models (step time, quality, rehearsal -> official offset) are fitted from the shipped
`research/sim-data` tables when the app starts (well under a second; no pickle is shipped or loaded).
Every function wired to a button is also a Gradio API endpoint (see the "Use via API" link).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from ffsim.contrib import (APPROVAL_LABEL, BASE_RECIPES, CONSENT_TEXT, HARDWARE, build_issue_url,  # noqa: E402
                           build_plain_issue_url, contrib_fit_records, contributor_counts, load_contrib,
                           validate_record)
from ffsim.dataset import load_runs  # noqa: E402
from ffsim.offset import OffsetModel  # noqa: E402
from ffsim.quality import FEATURES, QualityModel, knob_features  # noqa: E402
from ffsim.schema import Recipe  # noqa: E402
from ffsim.search import make_candidate, same_value, space_from_dict, support_check, validate_knobs  # noqa: E402
from ffsim.simulate import (DEFAULT_ACCUM_SCHED, MEDIAN_TO_MEAN_OVERHEAD, SAVE_RESERVE_S, Models,  # noqa: E402
                            fill_phases, load_recipe, parse_accum_sched, predict_step_time, recipe_time_target,
                            simulate_pair, steps_scalar, walk_step_times)
from ffsim.steptime import StepTimeModel  # noqa: E402

SIM_DATA = HERE / "research" / "sim-data"
RUNS = SIM_DATA / "runs.jsonl"
PAIRS = SIM_DATA / "validation-pairs.json"
UPLOADS = SIM_DATA / "official-uploads.csv"
CONTRIB = SIM_DATA / "contrib" / "runs-contrib.jsonl"   # merged community runs (synced by the refit workflow)

# Fill these two in before publishing (HF sets SPACE_ID itself inside a running Space).
GITHUB_REPO_URL = os.environ.get("FF_GITHUB_REPO_URL", "https://github.com/Marker2601/trainium-simulator")
SPACE_ID = os.environ.get("SPACE_ID", "<owner>/<space-name>")

BEST_OFFICIAL = 0.9617          # K77a: our best scored upload as of 30 Sep 2026 (K82s4 had not been scored then)
BEST_OFFICIAL_ASOF = "30 Sep 2026"
TOP10_SCORE = 0.9555            # leaderboard #10 at about 11 PM CDT on 30 Sep 2026
COMPUTE_SLOPE = 0.063           # delta_bpb ~= 0.063 * ln(compute ratio) (24 Sep +28% compute run)
STEP_SLOPE_PER_PCT = 0.00057    # campaign rule of thumb: ~0.00057 bpb per 1% more optimizer steps (not the fitted slope)
CONFIRM_FLOOR = 0.0003          # smallest effect a three-seed chip pair resolves
OOS_ERROR = 0.001               # measured out-of-sample error on genuinely new changes
MAX_DRAWS = 20000
CHIPS = ("C", "D")              # the chips whose step time and quality the models are fitted for
TT_HARD = (900.0, 2700.0)       # FF_TIME_TARGET accepted range (s)
TT_FITTED = (1760.0, 1793.0)    # FF_TIME_TARGET values of the fitted full runs
TT_SOFT = (1700.0, 1900.0)      # beyond this the ln(steps) curve is an extrapolation
FAR_LIMIT = 3.0                 # refuse a fitted quality feature more than this many seen-range widths outside
MAX_SD = 0.01                   # refuse a prediction whose own local-bpb sd exceeds this (10x the measured error)


# --------------------------------------------------------------------------- model fit at start-up
CONTRIB_SKIPPED: List[str] = []
try:            # load_contrib already skips bad lines; this guard only keeps a broken file from stopping the app
    CONTRIB_ROWS = load_contrib(CONTRIB, skipped=CONTRIB_SKIPPED)
except Exception as _e:  # noqa: BLE001
    CONTRIB_ROWS, CONTRIB_SKIPPED = [], [f"contributed runs not loaded ({type(_e).__name__})"]
CONTRIB_HASHES = {m.get("content_hash") for _, m in CONTRIB_ROWS if m.get("content_hash")}
CONTRIB_COUNTS = contributor_counts(CONTRIB_ROWS)
MAX_CONTRIB_TEXT = 20000        # characters of any free-text input to /contrib_record, checked before parsing


def fit_models() -> Tuple[Models, Dict[str, float]]:
    """Fit the three route-1 models from the shipped tables (same inputs as `python -m ffsim fit`)."""
    t0 = time.perf_counter()
    builtin = list(load_runs(str(RUNS)))
    contrib = contrib_fit_records(CONTRIB_ROWS)
    records = builtin + contrib
    with open(PAIRS, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs", pairs)
    t1 = time.perf_counter()
    st = StepTimeModel()
    st = st.fit(records) or st      # the step-time model never reads contributed runs (no per-phase step times)
    t2 = time.perf_counter()
    try:        # one bad contributed record must never stop the app: fall back to the built-in data
        q = QualityModel()
        q = q.fit(records, pairs) or q
    except Exception as e:  # noqa: BLE001
        CONTRIB_SKIPPED.append(f"fit with contributed runs failed ({type(e).__name__}): built-in data only")
        print("app: " + CONTRIB_SKIPPED[-1], file=sys.stderr)
        records = builtin
        q = QualityModel()
        q = q.fit(records, pairs) or q
    t3 = time.perf_counter()
    off = OffsetModel()
    off = off.fit(str(UPLOADS)) or off
    t4 = time.perf_counter()
    timings = {"load_s": t1 - t0, "steptime_fit_s": t2 - t1, "quality_fit_s": t3 - t2, "offset_fit_s": t4 - t3,
               "total_s": t4 - t0, "n_records": float(len(records)), "n_pairs": float(len(pairs))}
    return Models(st, q, off, meta={"n_records": len(records)}), timings


class ScaledStepTime:
    """A step-time model whose phase times are multiplied by `factor` (anchors a base to its measured steps)."""

    def __init__(self, inner: Any, factor: float):
        self.inner = inner
        self.factor = float(factor)

    def predict(self, code_version: str, knobs: Dict[str, str], chip: str = "C") -> Any:
        out = self.inner.predict(code_version, knobs, chip)
        if isinstance(out, dict):
            out = dict(out)
            for k, v in list(out.items()):
                if v is not None and isinstance(k, str) and (re.fullmatch(r"k\d+", k) or k == "sd_k4"):
                    out[k] = float(v) * self.factor
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["inner"], name)


class ShiftedQuality:
    """A quality model whose mean is shifted by `shift` (anchors a base to its measured rehearsal bpb).
    Everything else (design rows, coefficient covariance, residual sigma, slope, support) is the fitted model's."""

    def __init__(self, inner: Any, shift: float):
        self.inner = inner
        self.shift = float(shift)

    def predict(self, knobs: Dict[str, str], steps: float, code_version: str, chip: str = "C",
                seed: Optional[int] = None) -> Tuple[float, float]:
        mean, sd = self.inner.predict(knobs, steps, code_version, chip, seed)
        return float(mean) + self.shift, float(sd)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["inner"], name)


def point_steps(steptime: Any, recipe: Recipe) -> float:
    """Steps at the model's step times (no jitter): the simulator's own step walk."""
    st_pred, _sd, _notes = predict_step_time(steptime, recipe)
    sched = recipe.knobs.get("FF_ACCUM_SCHED", DEFAULT_ACCUM_SCHED)
    warmup = int(float(recipe.knobs.get("FF_WARMUP", 20) or 20))
    filled, _ = fill_phases(st_pred, [p for p, _ in parse_accum_sched(sched)])
    walk = walk_step_times(filled, MEDIAN_TO_MEAN_OVERHEAD)
    tt = recipe_time_target(recipe)
    return float(steps_scalar(walk, tt - SAVE_RESERVE_S, sched, warmup, time_target=tt))


@dataclass
class Base:
    key: str
    label: str
    recipe: Recipe
    models: Models
    measured_bpb: float
    measured_steps: int
    measured_official: Optional[float]
    seed: int
    anchored: bool
    step_factor: float = 1.0
    quality_shift: float = 0.0
    raw_steps: float = float("nan")
    raw_bpb: float = float("nan")
    note: str = ""


def build_bases(models: Models) -> Dict[str, Base]:
    k82 = load_recipe(HERE / "recipes" / "recipe-K82s4.json")
    k60 = load_recipe(HERE / "ffsim" / "examples" / "recipe-K60.json")
    st, q, off = models.as_tuple()
    bases: Dict[str, Base] = {}

    # K60: inside the fitted data; used as is (its own anchor check: predicted 0.95888 @ 2365 vs measured 0.95910 @ 2361)
    raw_steps60 = point_steps(st, k60)
    raw_bpb60 = q.predict(k60.knobs, 2361.0, k60.code_version, k60.chip, 73)[0]
    bases["K60"] = Base("K60", "K60 (29 Sep; the simulator's calibration recipe)", k60, models, 0.95910, 2361, 0.9655,
                        73, False, raw_steps=raw_steps60, raw_bpb=raw_bpb60,
                        note="K60 is inside the fitted data: the models are used unmodified.")

    # K82s4: anchored to its measured rehearsal (seed 73: 0.954472 @ 2,388 steps). Its newest levers (row pool,
    # EMA blend, fused Newton-Schulz, XU queue 48, c_proj LR) were never varied in the fitted runs.
    target_steps, target_bpb = 2388.0, 0.954472
    raw_steps82 = point_steps(st, k82)
    factor = 1.0
    for _ in range(8):
        s = point_steps(ScaledStepTime(st, factor), k82)
        factor *= s / target_steps
    raw_bpb82 = q.predict(k82.knobs, target_steps, k82.code_version, k82.chip, 73)[0]
    shift = target_bpb - raw_bpb82
    anchored = Models(ScaledStepTime(st, factor), ShiftedQuality(q, shift), off, meta=dict(models.meta))
    bases["K82s4"] = Base("K82s4", "K82s4 (best published recipe; anchored to its rehearsal)", k82, anchored, target_bpb,
                          2388, None, 73, True, step_factor=factor, quality_shift=shift, raw_steps=raw_steps82,
                          raw_bpb=raw_bpb82,
                          note=(f"K82s4's newest levers are outside the fitted data, so this base is anchored to its measured "
                                f"rehearsal (seed 73, measured on chip H; the app runs it as chip C): step times x {factor:.4f} "
                                f"(the raw model walks {raw_steps82:.0f} steps, measured 2,388) and quality {shift:+.5f} (the raw "
                                f"model says {raw_bpb82:.5f} at 2,388 steps, measured 0.954472). The step factor absorbs both the "
                                f"new levers and any step-time difference between chips H and C. Knob changes on top of it move "
                                f"the prediction by the fitted model's own effects."))
    return bases


T_START = time.perf_counter()
MODELS, FIT_TIMINGS = fit_models()
BUILTIN_RECORDS = list(load_runs(str(RUNS)))
BASES = build_bases(MODELS)
BASE_CHOICES = [(b.label, b.key) for b in BASES.values()]
DEFAULT_BASE = "K82s4"
KNOWN_SEEDS = sorted(int(s) for s in MODELS.quality.seeds)
AVG_SEED = "average over seeds"
SEED_CHOICES = [AVG_SEED] + [str(s) for s in KNOWN_SEEDS]
FITTED_STEP_SLOPE = float(MODELS.quality.steps_slope(2388.0))   # d bpb / d ln(steps) of the fitted quality model
STARTUP_S = time.perf_counter() - T_START


# --------------------------------------------------------------------------- knobs exposed in the UI
# (knob, label, kind, info). kind: "num" (gr.Number), "text", or ("choice", [values]).
KNOB_FIELDS: List[Tuple[str, str, Any, str]] = [
    ("FF_TIME_TARGET", "Time target FF_TIME_TARGET (charged s)", "num",
     f"fitted runs: 1760-1793 (K82s4 uses 1795); accepted {TT_HARD[0]:.0f}-{TT_HARD[1]:.0f}, outside "
     f"{TT_SOFT[0]:.0f}-{TT_SOFT[1]:.0f} is extrapolation"),
    ("FF_COOLDOWN_FRAC", "Cooldown fraction FF_COOLDOWN_FRAC", "num", "seen: 0.5, 0.6, 0.7"),
    ("FF_ACCUM_SCHED", "Batch / accumulation schedule FF_ACCUM_SCHED", ("choice", [
        "1:0.06,2:0.2,4", "1:0.06,2:0.3,4", "1:0.08,2:0.2,4", "1:0.06,4", "2:0.2,4", "2:0.3,4", "2:0.12,4", "4"]),
     "k0:until,k1:until,K micro-batches (only 1, 2, 4: the phases the step-time model has); "
     "until = fraction of the time target"),
    ("FF_ACCUM_LR", "Small-batch LR scales FF_ACCUM_LR", "text", "lr in phase k0, k1 (seen: 0.5,0.7071 and 0.6,0.8)"),
    ("FF_WARMUP", "Warm-up steps FF_WARMUP", "num", "seen: 20, 40"),
    ("FF_TOTAL_BATCH", "Tokens per optimizer step FF_TOTAL_BATCH", ("choice", ["131072", "262144", "327680", "393216", "524288"]),
     "quality seen at 262144 and 327680; any change from 262144 gets the flat +3% 'cost unknown' step-time charge "
     "(its step-time cost was never measured)"),
    ("FF_MATRIX_LR_SCALE", "Muon (matrix) LR scale FF_MATRIX_LR_SCALE", "num", "seen: 1.75, 2.0, 2.3"),
    ("FF_ADAMW_LR_SCALE", "AdamW LR scale FF_ADAMW_LR_SCALE", "num", "seen: 1.4142, 1.7"),
    ("FF_EMB_LR", "Embedding LR FF_EMB_LR", "num", "seen: 0.30, 0.36"),
    ("FF_UNEMB_LR", "Unembedding LR FF_UNEMB_LR", "num", "seen: 0.006, 0.0072"),
    ("FF_SCALAR_LR", "Scalar LR FF_SCALAR_LR", "num", "seen: 0.1664, 0.20"),
    ("FF_WD", "Weight decay FF_WD", "num", "seen: 0.012, 0.02, 0.03"),
    ("FF_WD_SCHED", "WD schedule FF_WD_SCHED", ("choice", ["0", "1", "2"]), "seen: 0, 1, 2"),
    ("FF_MOM_PEAK", "Muon momentum peak FF_MOM_PEAK", "num", "seen: 0.93, 0.95, 0.97"),
    ("FF_MUON_BETA2", "Muon beta2 FF_MUON_BETA2", "num", "seen: 0.8, 0.9"),
    ("FF_LEAKY_RELU2", "LeakyReLU^2 slope FF_LEAKY_RELU2", "num", "seen: 0, 0.25, 0.35, 0.5"),
    ("FF_SOFTCAP", "Logit softcap FF_SOFTCAP", "num", "seen: 15, 18"),
    ("FF_QK_GAIN", "QK gain FF_QK_GAIN", "num", "seen: 1.44, 1.6"),
    ("FF_ROPE_BASE", "RoPE base FF_ROPE_BASE", "num", "seen: 100000, 300000"),
    ("FF_KEY_OFFSET_RAD", "Key offset radius FF_KEY_OFFSET_RAD", "num", "seen: 0.03 - 3.0"),
    ("FF_ATTN_SRC", "Attention-source reuse FF_ATTN_SRC", ("choice", ["5:6,7,8", ""]), "'' = off; 5:6,7,8 = K60's AS-a"),
    # Model size: no fitted run varied these. Any change gets the same flat +3% 'cost unknown' step-time charge,
    # whatever its direction or size, and its quality effect is an unfitted prior. (FF_HEAD_DIM, FF_KV_HEADS and
    # FF_MLP_MULT are not offered at all: the model has no fitted effect for them; they still work as overrides.)
    ("FF_DEPTH", "Depth FF_DEPTH (layers)", "num",
     "NOT FITTED (all runs at 9): step time = flat +3% for any change, even smaller; quality = unfitted prior"),
    ("FF_ASPECT_RATIO", "Aspect ratio FF_ASPECT_RATIO (width ~ depth x aspect)", "num",
     "NOT FITTED (all runs at 113, width 1024): step time = flat +3% for any change; quality = unfitted prior"),
    ("FF_MTP", "Multi-token prediction FF_MTP", ("choice", ["1", "0"]),
     "1 = on (every fitted run); 0 was never in the fitted data: flat +3% step time, unfitted quality prior"),
]
KNOB_NAMES = [k for k, *_ in KNOB_FIELDS]
KNOB_GROUPS = [
    ("Schedule and batch", ["FF_TIME_TARGET", "FF_COOLDOWN_FRAC", "FF_ACCUM_SCHED", "FF_ACCUM_LR", "FF_WARMUP",
                            "FF_TOTAL_BATCH"], None),
    ("Learning rates and optimizer", ["FF_MATRIX_LR_SCALE", "FF_ADAMW_LR_SCALE", "FF_EMB_LR", "FF_UNEMB_LR",
                                      "FF_SCALAR_LR", "FF_WD", "FF_WD_SCHED", "FF_MOM_PEAK", "FF_MUON_BETA2"], None),
    ("Model mechanisms", ["FF_LEAKY_RELU2", "FF_SOFTCAP", "FF_QK_GAIN", "FF_ROPE_BASE", "FF_KEY_OFFSET_RAD",
                          "FF_ATTN_SRC"], None),
    ("Model size (not fitted: guesses only)", ["FF_DEPTH", "FF_ASPECT_RATIO", "FF_MTP"],
     "**No fitted run varied these.** Any change, in either direction, is charged the same flat +3% step-time "
     "penalty ('cost unknown'), so a *smaller* model is also predicted slower. The quality effect is an unfitted "
     "prior. Treat any number from this group as a placeholder, not a prediction."),
]
assert sorted(n for _, ns, _ in KNOB_GROUPS for n in ns) == sorted(KNOB_NAMES)

# Input sanity rules (checked before anything is simulated). Positive: > 0; nonneg: >= 0; unit: 0 < x < 1.
POSITIVE = {"FF_MATRIX_LR_SCALE", "FF_ADAMW_LR_SCALE", "FF_EMB_LR", "FF_UNEMB_LR", "FF_SCALAR_LR", "FF_SOFTCAP",
            "FF_QK_GAIN", "FF_ROPE_BASE", "FF_ASPECT_RATIO", "FF_MLP_MULT", "FF_DEPTH", "FF_HEAD_DIM", "FF_TOTAL_BATCH",
            "FF_TIME_TARGET"}
NONNEG = {"FF_WD", "FF_LEAKY_RELU2", "FF_KEY_OFFSET_RAD", "FF_WARMUP", "FF_KV_HEADS", "FF_COOLDOWN_FRAC"}
UNIT = {"FF_MOM_PEAK", "FF_MUON_BETA2"}
INTEGER = {"FF_DEPTH", "FF_HEAD_DIM", "FF_KV_HEADS", "FF_WARMUP", "FF_TOTAL_BATCH", "FF_WD_SCHED", "FF_MTP"}
RANGES = {"FF_TIME_TARGET": TT_HARD, "FF_COOLDOWN_FRAC": (0.0, 0.95), "FF_DEPTH": (4, 16), "FF_ASPECT_RATIO": (32, 256),
          "FF_HEAD_DIM": (32, 512), "FF_KV_HEADS": (0, 16), "FF_MLP_MULT": (1, 8), "FF_WARMUP": (0, 1000),
          "FF_TOTAL_BATCH": (65536, 1048576)}
CHOICES = {"FF_WD_SCHED": {"0", "1", "2"}, "FF_MTP": {"0", "1", "2"}}
NUMERIC = POSITIVE | NONNEG | UNIT | INTEGER | {"FF_MOM_PEAK", "FF_MUON_BETA2"} | {
    k for k, _l, kind, _i in KNOB_FIELDS if kind == "num"}


def _fmt_knob(v: Any) -> str:
    """A UI value the way train.py's _env_* readers see it (15.0 -> '15')."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        return repr(f) if abs(f) < 1e-4 else f"{f:.10g}"
    return str(v).strip()


def parse_overrides(text: Optional[str]) -> Dict[str, str]:
    """`KEY=VALUE` lines / space-separated pairs, or a JSON object -> {FF_*: str}."""
    if not text or not str(text).strip():
        return {}
    s = str(text).strip()
    if s.startswith("{"):
        d = json.loads(s)
        if not isinstance(d, dict):
            raise ValueError("a JSON override must be an object {\"FF_KNOB\": value, ...}")
        return {str(k).strip(): _norm_value(_fmt_knob(v)) for k, v in d.items()}
    out: Dict[str, str] = {}
    for tok in re.split(r"[\n;]+|\s+(?=[A-Za-z_][A-Za-z0-9_]*=)", s):
        tok = tok.strip()
        if not tok or tok.startswith("#"):
            continue
        if "=" not in tok:
            raise ValueError(f"override {tok!r} is not KEY=VALUE")
        k, v = tok.split("=", 1)
        out[k.strip()] = _norm_value(v.strip().strip("'\""))
    return out


def _norm_value(v: str) -> str:
    """A numeric override spelled like a UI value ('9.0' -> '9', '0.30' -> '0.3'); anything else unchanged."""
    s = str(v).strip()
    if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s):
        f = float(s)
        if math.isfinite(f):
            return _fmt_knob(f)
    return s


def _num_or_none(v: str) -> Optional[float]:
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def check_inputs(knobs: Dict[str, str], changed: Dict[str, str]) -> List[str]:
    """Input sanity (not train.py's rules): values the simulator cannot meaningfully take. Only changed knobs are
    judged, so a base's own spelling never trips a rule."""
    errs: List[str] = []
    for k, v in sorted(changed.items()):
        s = str(v).strip()
        if re.fullmatch(r"[+-]?(nan|inf|infinity)", s, flags=re.I):
            errs.append(f"{k}={s!r} is not a finite number")
            continue
        if k not in NUMERIC:
            continue
        if s == "":
            errs.append(f"{k} is empty: give a number")
            continue
        f = _num_or_none(s)
        if f is None or not math.isfinite(f):
            errs.append(f"{k}={s!r} is not a number")
            continue
        if k in INTEGER and not float(f).is_integer():
            errs.append(f"{k}={s} must be a whole number")
        if k in POSITIVE and f <= 0:
            errs.append(f"{k}={s} must be > 0")
        if k in NONNEG and f < 0:
            errs.append(f"{k}={s} must be >= 0")
        if k in UNIT and not (0.0 < f < 1.0):
            errs.append(f"{k}={s} must be between 0 and 1 (exclusive)")
        if k in RANGES:
            lo, hi = RANGES[k]
            if not (lo <= f <= hi):
                errs.append(f"{k}={s} is outside the range this app accepts ({lo:g} to {hi:g})")
        if k in CHOICES and _fmt_knob(f) not in CHOICES[k]:
            errs.append(f"{k}={s} must be one of {sorted(CHOICES[k])}")
    if "FF_ACCUM_LR" in changed:
        lrs = [x.strip() for x in str(changed["FF_ACCUM_LR"]).split(",") if x.strip()]
        vals = [_num_or_none(x) for x in lrs]
        if not lrs or any(x is None or not math.isfinite(x) or x <= 0 for x in vals):
            errs.append(f"FF_ACCUM_LR={changed['FF_ACCUM_LR']!r} must be one or two positive numbers 'lr0,lr1'")
    sched = (knobs.get("FF_ACCUM_SCHED") or "").strip()
    if not sched:
        errs.append("FF_ACCUM_SCHED is empty: use '4' for a constant batch")
    else:
        try:
            parts = [p.strip() for p in sched.split(",")]
            ks = [int(p.split(":")[0]) for p in parts[:-1]] + [int(parts[-1])]
            for p in parts[:-1]:
                float(p.split(":")[1])
            if any(len(p.split(":")) != 2 for p in parts[:-1]):
                raise ValueError(sched)
        except (ValueError, IndexError):
            errs.append(f"FF_ACCUM_SCHED={sched!r} must look like 'K', 'k0:until,K' or 'k0:u0,k1:u1,K' "
                        "(e.g. '1:0.06,2:0.2,4')")
        else:
            bad = sorted({k for k in ks if k not in (1, 2, 4)})
            if bad:
                errs.append(f"FF_ACCUM_SCHED={sched!r}: micro-batch counts {bad} are not modelled; the step-time model "
                            "only has phases of 1, 2 and 4 micro-batches")
    return errs


def far_extrapolations(knobs: Dict[str, str], base_knobs: Dict[str, str]) -> Dict[str, float]:
    """Fitted quality features that a change pushes more than FAR_LIMIT seen-range widths outside the fitted runs,
    keyed 'feature (knobs)'. Judged on the linear features only (a bowl's square exaggerates the distance); features
    no fitted run varied are left to RANGES (their range width is zero), and the time target to TT_HARD / TT_SOFT
    (its effect runs through ln(steps), which extrapolates smoothly)."""
    q = MODELS.quality
    try:
        X = q._X[:, q._i_knob0:q._i_seed0]
    except Exception:  # noqa: BLE001
        return {}
    x, xb = knob_features(knobs), knob_features(base_knobs)
    lo, hi = X.min(axis=0), X.max(axis=0)
    out: Dict[str, float] = {}
    for j, f in enumerate(FEATURES):
        if f.bowl or f.name.endswith("_sq") or "FF_TIME_TARGET" in f.knobs:
            continue
        width = float(hi[j] - lo[j])
        if width <= 0 or abs(float(x[j]) - float(xb[j])) < 1e-12:
            continue
        key = f"{f.name} ({', '.join(f.knobs)})"
        if not math.isfinite(float(x[j])):
            out[key] = float("inf")
            continue
        far = max(float(lo[j] - x[j]), float(x[j] - hi[j]), 0.0) / width
        if far > FAR_LIMIT:
            out[key] = far
    return out


def _bpb(x: float) -> str:
    return f"{x:.5f}"


def _check_schedule(knobs: Dict[str, str]) -> List[str]:
    """train.py's accumulation-schedule asserts that ffsim.search.validate_knobs does not cover."""
    out: List[str] = []
    raw = (knobs.get("FF_ACCUM_SCHED") or "").strip()
    if not raw:
        return out
    try:
        parts = [p.strip() for p in raw.split(",")]
        final_k = int(parts[-1])
        stages = []
        for p in parts[:-1]:
            k, u = p.split(":")
            stages.append((int(k), float(u)))
    except ValueError:
        return [f"FF_ACCUM_SCHED={raw!r} must look like 'K', 'k0:until,K' or 'k0:u0,k1:u1,K'"]
    cd = float(knobs.get("FF_COOLDOWN_FRAC", "0.6") or 0.6)
    ks = [k for k, _ in stages] + [final_k]
    if any(a >= b for a, b in zip(ks, ks[1:])) or any(final_k % k for k, _ in stages):
        out.append(f"FF_ACCUM_SCHED={raw!r}: needs k0 < k1 < K, each dividing K")
    us = [u for _, u in stages]
    if any(a >= b for a, b in zip(us, us[1:])) or any(not (0.0 < u <= 1.0 - cd + 1e-9) for u in us):
        out.append(f"FF_ACCUM_SCHED={raw!r}: every `until` must be increasing and <= 1 - FF_COOLDOWN_FRAC = {1.0 - cd:g}")
    lrs = [x for x in (knobs.get("FF_ACCUM_LR") or "").split(",") if x.strip()]
    if len(stages) == 2 and len(lrs) != 2:
        out.append("FF_ACCUM_LR: a two-stage schedule 'k0:u0,k1:u1,K' needs two values 'lr0,lr1'")
    return out


def outside_fitted_range(knobs: Dict[str, str], tol: float = 0.1) -> Dict[str, float]:
    """Quality features whose value lies outside the fitted runs' range by more than `tol` of that range's width
    (feature -> how far outside, in range widths). A small overshoot (K82s4: 1795 s vs the 1760-1793 s seen) is ignored."""
    q = MODELS.quality
    try:
        X = q._X[:, q._i_knob0:q._i_seed0]
    except Exception:  # noqa: BLE001
        return {}
    x = knob_features(knobs)
    lo, hi = X.min(axis=0), X.max(axis=0)
    out: Dict[str, float] = {}
    for j, f in enumerate(FEATURES):
        width = float(hi[j] - lo[j]) or 1.0
        far = max(float(lo[j] - x[j]), float(x[j] - hi[j]), 0.0) / width
        if far > tol:
            out[f.name] = far
    return out


REASONS = {"weak": "a changed value was seen in too few fitted runs (or none in the current era)",
           "interaction": "two changed knobs were never combined in one fitted run",
           "extrap": "a changed value lies outside the values the fitted runs used",
           "never": "a changed knob never varied in the fitted runs (the model cannot see it)"}


def _verdict(level: str, in_support: bool, delta: float, novel_cost: bool, n_changes: int,
             moved_base_extrap: List[str], violations: Optional[List[str]] = None) -> str:
    if violations:
        return ("**Invalid recipe: train.py would refuse it** (" + "; ".join(violations) + "). "
                "Nothing was simulated: a prediction for a recipe that cannot run is meaningless.")
    if n_changes == 0:
        return "This is the default recipe itself (no knob changed)."
    if not in_support or novel_cost:
        reasons = []
        if not in_support:
            reasons.append(REASONS.get(level, "a change is outside what the fitted runs cover"))
        if novel_cost:
            reasons.append("the step-time cost of a changed mechanism was never measured (a +3% prior is charged)")
        why = "; ".join(reasons)
        return f"**Screen first.** {why[0].upper() + why[1:]}. Treat the number as a guess."
    if moved_base_extrap:
        return ("**Screen first.** The default itself sits outside the fitted runs on "
                + ", ".join(f"`{f}`" for f in moved_base_extrap)
                + ", and this change moves that feature: the difference rests on the model's extrapolation there, "
                "not on data. (For K82s4 this is the small-batch LR 0.35/0.6, chosen after the fit.)")
    if delta <= -CONFIRM_FLOOR:
        return (f"**Worth a chip confirmation** (seeds 73, 58, 67): in support and {delta:+.5f} clears the "
                f"{CONFIRM_FLOOR} floor. Held out, the model's sign was right ~95% of the time at this size, "
                "but genuinely new mechanisms scored ~coin-flip after the fit.")
    if delta < 0:
        return (f"**A tie.** {delta:+.5f} is better on paper but inside the {CONFIRM_FLOOR} noise floor: the model "
                "cannot order changes this small (held-out MAE ~0.0004, out-of-sample ~0.001).")
    return f"**Predicted worse or equal** ({delta:+.5f} vs the default)."


# --------------------------------------------------------------------------- prediction core
def resolve_seed(seed: Any) -> Tuple[Optional[int], Optional[str]]:
    """UI/API seed -> (fitted seed or None for the average over seeds, a note when the request was changed)."""
    if seed is None:
        return None, None
    s = str(seed).strip()
    if s == "" or s.lower().startswith("average") or s.lower() in ("none", "marginal", "avg"):
        return None, None
    try:
        f = float(s)
    except ValueError:
        raise ValueError(f"seed {s!r} is not a number; use one of {KNOWN_SEEDS} or '{AVG_SEED}'") from None
    if not math.isfinite(f) or f < 0:
        return None, None
    if not f.is_integer() or int(f) not in MODELS.quality.seeds:
        shown = int(f) if f.is_integer() and abs(f) < 1e15 else s
        return None, (f"seed {shown} is not in the fitted data (known seeds: {', '.join(map(str, KNOWN_SEEDS))}): "
                      "showing the average over seeds")
    return int(f), None


def resolve_chip(chip: Any) -> Tuple[str, Optional[str]]:
    c = str(chip or "C").strip().upper()
    if c in CHIPS:
        return c, None
    if c == "H":
        return "C", ("chip H is not in the fitted data; it is treated as chip C (as for K82s4's own chip-H "
                     "rehearsal, which the K82s4 anchor absorbs)")
    raise ValueError(f"chip {chip!r} is not modelled: use 'C' (the default) or 'D' (slower runtime "
                     "2.33.10)")


def run_prediction(base_key: str, changes: Dict[str, str], seed: Any, chip: str = "C",
                   n_draws: int = 2000, rng: int = 0) -> Dict[str, Any]:
    if base_key not in BASES:
        raise ValueError(f"unknown base {base_key!r}; choose one of {list(BASES)}")
    base = BASES[base_key]
    n = int(max(200, min(MAX_DRAWS, int(n_draws or 2000))))
    b = base.recipe
    input_notes: List[str] = []
    seed_req = seed
    seed, seed_note = resolve_seed(seed)
    if seed_note:
        input_notes.append(seed_note)
    chip, chip_note = resolve_chip(chip)
    if chip_note:
        input_notes.append(chip_note)
    knobs = dict(b.knobs)
    clean = {str(k).strip(): _norm_value(_fmt_knob(v)) for k, v in (changes or {}).items() if str(k).strip()}
    for k in clean:
        if not re.fullmatch(r"(FF|NEURON)_[A-Z0-9_]+", k):
            raise ValueError(f"{k!r} is not an FF_* knob")
    if "FF_SEED" in clean:
        if not same_value(clean.pop("FF_SEED"), b.knobs.get("FF_SEED", "")):
            input_notes.append("FF_SEED in the overrides is ignored: use the seed input")
    # a value equal to the base's keeps the base's spelling, so both sides share their random draws
    for k, v in clean.items():
        knobs[k] = b.knobs[k] if k in b.knobs and same_value(v, b.knobs[k]) else v
    diff = {k: v for k, v in knobs.items() if not same_value(v, b.knobs.get(k, ""))}
    errs = check_inputs(knobs, diff)
    if errs:
        raise ValueError("invalid input: " + "; ".join(errs))
    far = far_extrapolations(knobs, b.knobs)
    if far:
        raise ValueError("too far outside the fitted runs for the model to say anything: " + ", ".join(
            f"{f} ({'unbounded' if not math.isfinite(w) else f'{w:.1f}x the seen range'} beyond it)"
            for f, w in sorted(far.items())) + f" (limit {FAR_LIMIT:g}x). Move the value closer to what was run.")
    tt = float(knobs.get("FF_TIME_TARGET") or b.time_target)
    if not (TT_SOFT[0] <= tt <= TT_SOFT[1]):
        input_notes.append(f"FF_TIME_TARGET={tt:g} s is outside the step-count range the model was fitted on "
                           f"(full runs at {TT_FITTED[0]:.0f}-{TT_FITTED[1]:.0f} s): the steps -> bpb curve is "
                           "extrapolated")
    sup_changes: Dict[str, Any] = dict(diff)
    if chip != b.chip:
        sup_changes["chip"] = chip
    violations = validate_knobs(knobs) + _check_schedule(knobs)
    if violations:
        return {"base": base.key, "seed": seed, "seed_requested": seed_req, "chip": chip, "n_draws": n,
                "invalid": True, "changes": diff | ({"chip": chip} if chip != b.chip else {}),
                "train_py_violations": violations, "input_notes": input_notes,
                "verdict": _verdict("", True, 0.0, False, len(sup_changes), [], violations)}

    cand = Recipe(name="candidate", code_version=b.code_version, knobs=knobs, chip=chip, time_target=tt)
    base_r = Recipe(name=base.key, code_version=b.code_version, knobs=dict(b.knobs), chip=b.chip, time_target=b.time_target)
    space = space_from_dict({"name": base.key, "base": {"name": base.key, "code_version": b.code_version, "chip": b.chip,
                                                         "time_target": b.time_target, "knobs": dict(b.knobs)}, "dims": []})
    in_support, support_note, level = support_check(base.models, make_candidate(sup_changes), space=space)

    pr = simulate_pair(cand, base_r, *base.models.as_tuple(), n=n, rng=rng, best_official=BEST_OFFICIAL, seed=seed)
    a, c = pr.a, pr.b
    offset = float(getattr(base.models.offset, "mean", float("nan")))
    off_sd = float(getattr(base.models.offset, "predictive_sd", getattr(base.models.offset, "sd", float("nan"))))
    notes = [x for x in (a.notes or "").split("; ") if x]
    novel = any("cost unknown" in x for x in notes)
    nums = [a.steps_mean, a.bpb_2m_mean, a.bpb_2m_sd, a.official_mean, pr.diff_mean, pr.diff_sd]
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in nums):
        raise ValueError("the simulator returned a non-finite result for this recipe; it is outside what the model "
                         "can evaluate")
    if a.bpb_2m_sd > MAX_SD:
        raise ValueError(f"the model's own spread for this recipe is +-{a.bpb_2m_sd:.3f} bpb (over {MAX_SD}, ten times "
                         "its measured error): it has no information this far from the fitted runs. Move the changed "
                         "values closer to what was run.")
    base_far = outside_fitted_range(b.knobs)
    cand_far = outside_fitted_range(knobs)
    xb, xc = knob_features(b.knobs), knob_features(knobs)
    moved = [f.name for j, f in enumerate(FEATURES) if f.name in base_far and abs(xb[j] - xc[j]) > 1e-12]

    def side(r: Any) -> Dict[str, Any]:
        st = dict(r.step_time)
        mean_step = (tt if r is a else b.time_target) - SAVE_RESERVE_S
        return {
            "steps_mean": r.steps_mean, "steps_sd": r.steps_sd,
            "step_time_median_s": {k: st[k] for k in sorted(st)},
            "avg_step_s": mean_step / r.steps_mean if r.steps_mean else None,
            "local_bpb_mean": r.bpb_2m_mean, "local_bpb_sd": r.bpb_2m_sd,
            "official_mean": r.official_mean, "official_sd": r.official_sd,
            "official_p10": r.official_p10, "official_p50": r.official_p50, "official_p90": r.official_p90,
            "p_beat_best_official": r.p_beat_best,
        }

    # local-chip band: the official band minus the (shared) offset mean is not exact; use bpb_2m mean +- 1.2816 sd
    res = {
        "base": base.key, "seed": seed, "seed_requested": seed_req, "chip": chip, "n_draws": n, "invalid": False,
        "changes": diff | ({"chip": chip} if chip != b.chip else {}),
        "input_notes": input_notes,
        "uncertainty_note": (f"Bands are the simulator's own spread. On genuinely new changes its measured error was "
                             f"about {OOS_ERROR} bpb, so read |difference| < {OOS_ERROR} as uncertain."),
        "candidate": side(a), "default": side(c),
        "paired_delta": {"mean": pr.diff_mean, "sd": pr.diff_sd, "p10": pr.diff_p10, "p90": pr.diff_p90,
                         "p_candidate_better": pr.p_a_better},
        "offset_used": {"mean": offset, "predictive_sd": off_sd},
        "best_official_reference": BEST_OFFICIAL,
        "support": {"in_support": bool(in_support), "level": level, "detail": support_note},
        "quality_features_outside_fitted_range": sorted(cand_far),
        "default_features_outside_fitted_range": sorted(base_far),
        "moved_default_extrapolated_features": moved,
        "train_py_violations": violations,
        "notes": notes,
        "anchor": {"anchored": base.anchored, "step_factor": base.step_factor, "quality_shift": base.quality_shift,
                   "measured_rehearsal_bpb": base.measured_bpb, "measured_steps": base.measured_steps,
                   "measured_official": base.measured_official, "note": base.note},
    }
    res["verdict"] = _verdict(level, bool(in_support), pr.diff_mean, novel, len(sup_changes), moved)
    return res


def _seed_txt(res: Dict[str, Any]) -> str:
    if res["seed"] is not None:
        return f"seed {res['seed']}"
    req = res.get("seed_requested")
    try:
        if req is not None and str(req).strip() and float(req) >= 0 and math.isfinite(float(req)):
            return f"average over seeds (seed {req} is not in the fitted data)"
    except ValueError:
        pass
    return "average over seeds"


def format_result(res: Dict[str, Any]) -> str:
    if res.get("invalid"):
        lines = [f"### Invalid recipe ({res['base']} + your changes)", "", res["verdict"], "",
                 "**train.py would refuse this recipe:**"] + [f"- {v}" for v in res["train_py_violations"]]
        if res["changes"]:
            lines += ["", "**Changed vs default:** " + ", ".join(f"`{k}={v}`" for k, v in sorted(res["changes"].items()))]
        if res.get("input_notes"):
            lines += [""] + [f"- {x}" for x in res["input_notes"]]
        return "\n".join(lines)
    a, c, d = res["candidate"], res["default"], res["paired_delta"]
    base = BASES[res["base"]]
    z = 1.2815516   # p10/p90 of a normal
    seed_txt = _seed_txt(res)

    def band(m: float, s: float) -> str:
        return f"{_bpb(m)} +- {s:.5f} (p10-p90 {_bpb(m - z * s)} - {_bpb(m + z * s)})"

    def st(x: Dict[str, Any]) -> str:
        t = x["step_time_median_s"]
        ph = " / ".join(f"{k} {t[k]:.3f}" for k in ("k1", "k2", "k4") if k in t)
        return f"{ph} s (avg {x['avg_step_s']:.4f} s/step)"

    lines = [
        f"### Prediction ({seed_txt}, chip {res['chip']}, {res['n_draws']:,} Monte Carlo draws)",
        "",
        f"| | your recipe | default {base.key} |",
        "|---|---|---|",
        f"| predicted steps | {a['steps_mean']:,.0f} +- {a['steps_sd']:.0f} | {c['steps_mean']:,.0f} +- {c['steps_sd']:.0f} |",
        f"| step time, median per phase | {st(a)} | {st(c)} |",
        f"| **local chip val_bpb** (2M-token rehearsal) | **{band(a['local_bpb_mean'], a['local_bpb_sd'])}** | "
        f"{band(c['local_bpb_mean'], c['local_bpb_sd'])} |",
        f"| **official-equivalent** (local + offset {res['offset_used']['mean']:+.5f}) | **{_bpb(a['official_mean'])} +- "
        f"{a['official_sd']:.5f}** (p10-p90 {_bpb(a['official_p10'])} - {_bpb(a['official_p90'])}) | "
        f"{_bpb(c['official_mean'])} +- {c['official_sd']:.5f} (p10-p90 {_bpb(c['official_p10'])} - {_bpb(c['official_p90'])}) |",
        f"| P(official < {BEST_OFFICIAL}, our best scored as of {BEST_OFFICIAL_ASOF}) | {a['p_beat_best_official']:.2f} | "
        f"{c['p_beat_best_official']:.2f} |",
        "",
        f"**Difference vs the default (paired, common random numbers): {d['mean']:+.5f} +- {d['sd']:.5f}** "
        f"(p10 {d['p10']:+.5f}, p90 {d['p90']:+.5f}); P(your recipe is better) = {d['p_candidate_better']:.2f}.",
        "",
        f"<sub>{res['uncertainty_note']}</sub>",
        "",
        res["verdict"],
        "",
    ]
    if res.get("input_notes"):
        lines.append("**Note:** " + " ".join(f"{x[0].upper() + x[1:]}." for x in res["input_notes"]))
        lines.append("")
    if res["changes"]:
        lines.append("**Changed vs default:** " + ", ".join(f"`{k}={v}`" for k, v in sorted(res["changes"].items())))
        lines.append("")
        lines.append(f"**Support** ({res['support']['level']}): `{res['support']['detail']}`")
        lines.append("")
    if res["quality_features_outside_fitted_range"]:
        own = [f for f in res["quality_features_outside_fitted_range"] if f not in res["default_features_outside_fitted_range"]]
        if own:
            lines.append("**Outside the fitted range** (quality features): " + ", ".join(f"`{f}`" for f in own))
            lines.append("")
    if res["default_features_outside_fitted_range"]:
        lines.append(f"<sub>The default {base.key} itself lies outside the fitted runs on "
                     + ", ".join(f"`{f}`" for f in res["default_features_outside_fitted_range"])
                          + (" (absorbed by the anchor; " if base.anchored else " (")
                     + "changes that move these features are extrapolations).</sub>")
        lines.append("")
    if res["notes"]:
        lines.append("**Model notes:**")
        lines += [f"- {n}" for n in res["notes"]]
        lines.append("")
    ref = (f"Measured for the default: rehearsal {base.measured_bpb:.6f} @ {base.measured_steps:,} steps (seed {base.seed})"
           + (f", official {base.measured_official}" if base.measured_official else ", not scored officially when released")
           + ".")
    lines.append(f"<sub>{ref} {base.note}</sub>")
    return "\n".join(lines)


def _err(e: Exception) -> str:
    return str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"


def predict_recipe(base_key: str, *args: Any) -> Tuple[str, Dict[str, Any]]:
    """UI handler: base, every knob field, then seed, chip, n_draws, extra overrides."""
    nk = len(KNOB_NAMES)
    vals, (seed, chip, n_draws, extra) = args[:nk], args[nk:nk + 4]
    try:
        changes = {k: _fmt_knob(v) for k, v in zip(KNOB_NAMES, vals) if v is not None}
        changes.update(parse_overrides(extra))
        res = run_prediction(base_key, changes, seed, chip or "C", int(n_draws or 2000))
        return format_result(res), res
    except Exception as e:  # noqa: BLE001 - show the error in the page instead of a stack trace
        return f"**Error:** {_err(e)}", {"error": _err(e)}


def predict_overrides(base: str = DEFAULT_BASE, overrides: str = "", seed: float = 73, chip: str = "C",
                      n_draws: float = 2000) -> Dict[str, Any]:
    """API: base recipe ('K82s4' or 'K60') + overrides ('FF_COOLDOWN_FRAC=0.6 FF_WD=0.03' or a JSON object);
    seed: one of the fitted seeds, or < 0 for the average over seeds (an unknown seed also gives the average,
    with a note); chip 'C' or 'D'. Returns the full prediction as JSON."""
    try:
        return run_prediction(base, parse_overrides(overrides), seed, chip or "C", int(n_draws or 2000))
    except Exception as e:  # noqa: BLE001
        return {"error": _err(e)}


def base_defaults(base_key: str) -> List[Any]:
    """The UI field values of a base recipe."""
    b = BASES.get(base_key, BASES[DEFAULT_BASE]).recipe
    out: List[Any] = []
    for k, _label, kind, _info in KNOB_FIELDS:
        v = b.knobs.get(k, "")
        if kind == "num":
            try:
                out.append(float(v))
            except ValueError:
                out.append(None)
        else:
            out.append(v)
    return out


# --------------------------------------------------------------------------- speed <-> score
def speed_to_score(base_score: float = BEST_OFFICIAL, throughput_gain_pct: float = 10.0) -> Tuple[str, Dict[str, Any]]:
    """+X% throughput (more optimizer steps in the same time budget) -> predicted score."""
    try:
        x = float(throughput_gain_pct)
        ratio = 1.0 + x / 100.0
        if not ratio > 0:
            raise ValueError("throughput change must be above -100%")
        if not (math.isfinite(x) and math.isfinite(float(base_score))):
            raise ValueError("inputs must be finite numbers")
        d = -COMPUTE_SLOPE * math.log(ratio)
        d_sim = FITTED_STEP_SLOPE * math.log(ratio)
        d_lin = -STEP_SLOPE_PER_PCT * x
        new = float(base_score) + d
        res = {"base_score": float(base_score), "throughput_gain_pct": x, "compute_ratio": ratio,
               "delta_bpb": d, "predicted_score": new, "fitted_step_slope_per_ln_steps": FITTED_STEP_SLOPE,
               "delta_bpb_fitted_quality_model": d_sim, "delta_bpb_linear_rule_of_thumb": d_lin}
        md = (f"### {x:+.2f}% throughput -> **{new:.4f}** ({d:+.5f} bpb)\n\n"
              f"- compute ratio {ratio:.4f}; `delta_bpb = -0.063 x ln({ratio:.4f}) = {d:+.5f}` (the campaign's measured "
              "exchange rate)\n"
              f"- the fitted quality model used by the Predict tab ({FITTED_STEP_SLOPE:+.4f} per ln steps at 2,388 "
              f"steps; steps only, at a fixed recipe) gives {d_sim:+.5f}\n"
              f"- campaign rule of thumb (0.00057 bpb per 1% steps; small changes only, not the fitted slope) gives "
              f"{d_lin:+.5f}\n\n"
              "<sub>Throughput here means more optimizer steps in the same charged time, at an unchanged recipe. "
              "A change that also alters what the model learns per step needs the Predict tab (or a chip run).</sub>")
        return md, res
    except Exception as e:  # noqa: BLE001
        return f"**Error:** {e}", {"error": str(e)}


def score_to_speed(base_score: float = BEST_OFFICIAL, target_score: float = TOP10_SCORE) -> Tuple[str, Dict[str, Any]]:
    """Target score -> the throughput (compute) gain needed at an unchanged recipe."""
    try:
        gap = float(base_score) - float(target_score)
        if not math.isfinite(gap):
            raise ValueError("inputs must be finite numbers")
        if abs(gap) > 0.2:
            raise ValueError("the gap is over 0.2 bpb: far outside where this exchange rate means anything")
        ratio = math.exp(gap / COMPUTE_SLOPE)
        pct = (ratio - 1.0) * 100.0
        res = {"base_score": float(base_score), "target_score": float(target_score), "gap_bpb": gap,
               "compute_ratio": ratio, "throughput_gain_pct": pct,
               "steps_at_2388_base": 2388.0 * ratio}
        verb = "faster" if pct >= 0 else "slower (you have headroom)"
        md = (f"### {float(target_score):.4f} needs **{pct:+.1f}% throughput** ({verb})\n\n"
              f"- gap {gap:+.5f} bpb; `ratio = exp({gap:+.5f} / 0.063) = {ratio:.4f}`\n"
              f"- at K82s4's 2,388 steps that is about {2388.0 * ratio:,.0f} optimizer steps in the same time budget\n\n"
              "<sub>Fitted on one 24 Sep run trained 28% longer (-0.0155 bpb); the per-step rate was confirmed on two chips. "
              "Far from +-30% it is an extrapolation.</sub>")
        return md, res
    except Exception as e:  # noqa: BLE001
        return f"**Error:** {e}", {"error": str(e)}


# --------------------------------------------------------------------------- add my runs (community contributions)
INT_FIELDS = ("seed", "chip_eval_tokens", "steps")
NUM_FIELDS = ("time_budget_s", "chip_val_bpb", "official_val_bpb", "step_seconds")
CONTRIB_SHARED_MD = f"""
**Help the simulator learn from your runs.** Every merged submission is added to the training data and the models
in the repository are refitted on it (the refit reports the error on contributed runs before and after,
leave-one-out, and refuses a change that makes the model worse on the team's own runs).

**What is shared:** exactly the JSON shown below, nothing else: hardware type, time budget, the `FF_*` knob values
(only the ones you changed when you start from a published recipe), seed, steps, your val_bpb score(s), optional
step time, date, notes and a public handle. It becomes a public GitHub issue and, after review, a line in
`research/sim-data/contrib/runs-contrib.jsonl` (data under CC-BY-4.0). This page sends nothing anywhere: the
button only builds a link; you open it, check it, and submit the issue yourself.

**Never included, and rejected if present:** AWS account ids, instance ids, ARNs, keys or tokens, e-mail or IP
addresses. Use a public handle, not your name, if you want credit.

**What happens next:** the issue is public as soon as you submit it. A GitHub Action re-checks the record and
comments the result; a maintainer reviews it and adds the `{APPROVAL_LABEL}` label, which opens a pull request; the
merge triggers a refit of the repository's model. Unusual results are flagged for the reviewer, never rejected for
being surprising. A hosted copy of this app picks up merged runs when its maintainers redeploy it. Details:
`CONTRIBUTING.md` in the repository.

<sub>Consent text: {CONSENT_TEXT}</sub>
"""


def _blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, float):
        return not math.isfinite(v)
    return False


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on", "x")
    return bool(v)


def contrib_record(hardware: str = "trn2.3xlarge", hardware_other: str = "", time_budget_s: float = 1800,
                   base_recipe: str = "K82s4", code_version: str = "", recipe: str = "", seed: Optional[float] = None,
                   chip_val_bpb: Optional[float] = None, chip_eval_tokens: Optional[float] = 2097152,
                   official_val_bpb: Optional[float] = None, steps: Optional[float] = None,
                   step_seconds: Optional[float] = None, date: str = "", framework_notes: str = "",
                   contributor: str = "", consent: Any = False) -> Dict[str, Any]:
    """API: one run's fields -> {"ok", "errors", "warnings", "flags", "record", "content_hash", "prediction",
    "usable_for_fit", "issue_url", "issue_url_plain"}. recipe: 'FF_KNOB=value ...' or a JSON object (only the knobs
    you changed when base_recipe is K82s4 / K60 / K59). Nothing is sent anywhere: open issue_url to submit."""
    try:
        for name, v in (("recipe", recipe), ("framework_notes", framework_notes), ("hardware_other", hardware_other),
                        ("code_version", code_version), ("date", date), ("contributor", contributor)):
            if isinstance(v, str) and len(v) > MAX_CONTRIB_TEXT:
                return {"ok": False, "errors": [f"{name} is longer than {MAX_CONTRIB_TEXT:,} characters"],
                        "warnings": [], "flags": [], "record": None, "issue_url": None, "issue_url_plain": None}
        vals = {"hardware": hardware, "hardware_other": hardware_other, "time_budget_s": time_budget_s,
                "base_recipe": base_recipe, "code_version": code_version, "seed": seed, "chip_val_bpb": chip_val_bpb,
                "chip_eval_tokens": chip_eval_tokens, "official_val_bpb": official_val_bpb, "steps": steps,
                "step_seconds": step_seconds, "date": date, "framework_notes": framework_notes,
                "contributor": contributor}
        rec: Dict[str, Any] = {"schema_version": "1"}
        for k, v in vals.items():
            if _blank(v):
                continue
            if k in INT_FIELDS:
                f = float(v)
                rec[k] = int(f) if f.is_integer() else f
            elif k in NUM_FIELDS:
                rec[k] = float(v)
            else:
                rec[k] = str(v).strip()
        if _blank(chip_val_bpb):
            rec.pop("chip_eval_tokens", None)
        rec["recipe"] = parse_overrides(recipe)
        rec["consent"] = _truthy(consent)
        res = validate_record(rec, quality=MODELS.quality, existing_hashes=CONTRIB_HASHES, builtin=BUILTIN_RECORDS,
                              offset_mean=float(MODELS.offset.mean), contributor_counts=CONTRIB_COUNTS)
        out = res.to_dict()
        out.pop("run_record", None)
        out["issue_url"] = out["issue_url_plain"] = None
        if res.ok:
            try:
                out["issue_url"] = build_issue_url(res.record)
                out["issue_url_plain"] = build_plain_issue_url(res.record)
            except ValueError as e:
                out["warnings"].append(str(e))
        return out
    except Exception as e:  # noqa: BLE001 - show the error instead of a stack trace
        return {"ok": False, "errors": [_err(e)], "warnings": [], "flags": [], "record": None,
                "issue_url": None, "issue_url_plain": None}


def format_contrib(res: Dict[str, Any]) -> str:
    if not res.get("ok"):
        return "### Not ready to submit\n\n" + "\n".join(f"- {e}" for e in res.get("errors") or ["unknown error"])
    lines = ["### Valid: ready to submit", ""]
    if res.get("issue_url"):
        lines.append(f"**[Open the pre-filled GitHub issue]({res['issue_url']})** (review it, tick the consent box, "
                     f"submit). If the form does not open pre-filled, use the [plain issue]({res['issue_url_plain']}) "
                     "or paste the JSON below into the form.")
    else:
        lines.append("Copy the JSON below into the run-submission issue form on GitHub.")
    p = res.get("prediction")
    if p:
        lines += ["", f"The current model predicted {p['predicted_bpb_2m']:.5f} +- {p['sd']:.5f} for this run; you "
                      f"measured {p['observed_bpb_2m']:.5f} (z = {p['z']:+.1f})."]
    lines += ["", f"Used for fitting: {'yes' if res.get('usable_for_fit') else 'no (stored only)'}; content hash "
                  f"`{(res.get('content_hash') or '')[:12]}`."]
    if res.get("warnings"):
        lines += ["", "**Notes:**"] + [f"- {w}" for w in res["warnings"]]
    if res.get("flags"):
        lines += ["", "Flags for the reviewer: " + ", ".join(f"`{f}`" for f in res["flags"])]
    return "\n".join(lines)


def contrib_ui(*args: Any) -> Tuple[str, str]:
    res = contrib_record(*args)
    rec = res.get("record")
    return format_contrib(res), (json.dumps(rec, indent=1) if rec else "")



# --------------------------------------------------------------------------- about text
def about_md() -> str:
    k82 = BASES["K82s4"]
    return f"""
## What this is

A browser front end for **ffsim**, the run simulator team FrontierForge built during Phase 1 of the AWS Trainium
Frontier challenge (15 Sep - 1 Oct 2026). It predicts what a `train.py` recipe would score on Trainium without
spending chip time:

```
official_bpb = quality(recipe, steps, seed) + offset(eval shard)
steps        = the FF_ACCUM_SCHED walk over (FF_TIME_TARGET - 5.5 s) at the predicted step time per batch phase
```

Three small numpy models, fitted when this Space starts ({FIT_TIMINGS['total_s']:.2f} s on
{int(FIT_TIMINGS['n_records']):,} run records from four Trainium chips, {len(CONTRIB_ROWS)} of them contributed by the community; no pickle is shipped):

1. **Step time**: median seconds per step for each batch phase (k1/k2/k4 micro-batches), per code lineage, with
   knob features for the mechanisms the fitted runs varied (fused paths, attention variants, MTP layout, ...).
   A feature no fitted run carries is charged a flat +3% prior and flagged "cost unknown", whatever the direction
   and size of the change. That includes every model-size knob (depth, width, head dim, KV heads) and a token batch
   other than 262,144: no fitted run changed them, so a *smaller* model is also charged +3%.
2. **Quality**: a ridge regression of the 2M-token rehearsal bpb on knob features, ln(steps), seed, chip and era,
   fitted on 131 full chip runs (residual sd 0.0003).
3. **Offset**: rehearsal -> official, from seven scored chip-C uploads: mean +0.00653, sd 0.00042.

The Monte Carlo draws chip jitter, seed luck, coefficient uncertainty, per-run residuals and the offset. Your
recipe and the default share the draws (common random numbers), so the paired difference is a real confidence
statement, not two independent bands overlapping.

## How accurate is it (honestly)

- **Measured error is about 0.001 bpb** on genuinely new changes: on the 15 same-seed pairs that finished after the
  fit, mean absolute error 0.00104 and sign agreement 53%.
- Held out on the 99 historical pairs it scores 85% sign agreement and MAE 0.00039, close to the pair noise floor
  (~0.00034). On decisive pairs (|effect| >= 0.0003) the sign was right 97% of the time.
- **For effects at the +-0.0003 level the sign is about a coin flip.** Read anything smaller than 0.0003 as a tie.
- Anchors: K60 predicted 0.95888 @ 2,365 steps vs measured 0.95910 @ 2,361 (official 0.9655 predicted 0.9654).
- Step time: ~0.5% leave-one-lineage-out on known mechanisms, 1-14% off for mechanisms the data never saw.
- **The bands are the simulator's own spread**, not its track record. A +-0.0005 band on a new change looks
  tighter than the model has earned: on genuinely new changes its measured error was about 0.001 bpb, so read any
  |difference| below 0.001 as uncertain.
- **Calibrated to one hardware setup and one recipe family**: single-chip Trainium (trn2) runs of this `train.py`
  lineage (depth 9, width 1024, 262k-token batches, ~1,760-1,795 s budget). **Model size is not modelled.** Depth,
  width, head dim, KV heads and MLP multiplier never varied in the fitted runs: their step-time cost is the flat +3%
  "cost unknown" charge for any change (in either direction), and their quality effect is an unfitted prior. The
  app therefore groups depth / aspect ratio / MTP under "not fitted" and does not offer head dim, KV heads or MLP
  multiplier as inputs (they still work as overrides, flagged).
- Inputs are checked first: non-numbers, non-finite values, non-positive LRs, time targets outside
  {TT_HARD[0]:.0f}-{TT_HARD[1]:.0f} s, schedules with micro-batch counts other than 1/2/4, and changes that push a
  fitted feature more than {FAR_LIMIT:g}x its seen range outside the data (or whose own spread exceeds +-{MAX_SD}
  bpb) are refused rather than predicted.
  Time targets outside {TT_SOFT[0]:.0f}-{TT_SOFT[1]:.0f} s are predicted but flagged (the fitted full runs used
  {TT_FITTED[0]:.0f}-{TT_FITTED[1]:.0f} s). A recipe that train.py's own start-up checks would refuse is not simulated.
- **Seeds**: the quality model knows the {len(KNOWN_SEEDS)} seeds the fitted runs used; any other seed is shown as the
  average over seeds, and labelled so. Chips: C and D (chip H, K82s4's rehearsal chip, is treated as C).
- **K82s4 is anchored.** Its newest levers (row pool `FF_ROW_SHUFFLE`, EMA blend, fused Newton-Schulz, XU queue,
  c_proj LR) postdate the fit. The raw model puts K82s4 at {k82.raw_bpb:.5f} @ 2,388 steps; it measured 0.954472
  (seed 73, on chip H). The app shifts quality by {k82.quality_shift:+.5f} and step time by x{k82.step_factor:.4f}
  so the default reproduces the measurement; the step factor absorbs both the new levers and any step-time
  difference between chips H and C. Changes on top of it use the fitted effects, which were learned around K59/K60.
- **Official-equivalent = local + the simulator's offset** (+0.00653, the seven K51-K60 chip-C uploads). The five
  final-day uploads (K63-K77a, other chips) averaged +0.00686, so for K82s4-family recipes the official score is
  likely ~0.0003 above the official-equivalent shown (K82s4: 0.9610 shown, projected 0.9611-0.9619).
- No data-sampling knob is modelled: the row pool is the only sampling change that won, and it came after the fit.
- In the campaign the surrogate's main use was to stop us spending chip time on ties. Every late win came from
  mechanisms the data had never seen.

## The speed -> score exchange rate

`delta_bpb ~= 0.063 x ln(compute ratio)`, measured on a run trained 28% longer (-0.0155 bpb). The campaign's rule of
thumb is that near our operating point 1% more steps is worth about 0.00057 bpb. Neither is the fitted quality
model's own step slope, which is {FITTED_STEP_SLOPE:+.4f} per ln(steps) at 2,388 steps (about
{-FITTED_STEP_SLOPE * math.log(1.01):.5f} bpb per 1% steps); the calculator shows all three. Our best official score
as of {BEST_OFFICIAL_ASOF} was {BEST_OFFICIAL} (K77a); K82s4 had been uploaded but not yet scored. Leaderboard #10 was
{TOP10_SCORE} at about 11 PM CDT on 30 Sep 2026.

## API

Every button is an endpoint: `predict` (all fields), `predict_overrides` (base + `KEY=VALUE` overrides, returns
JSON), `speed_to_score`, `score_to_speed`, and `contrib_record` (checks one of your runs and returns the record and
a pre-filled GitHub issue link; nothing is sent). Example with `gradio_client`:

```python
from gradio_client import Client
c = Client("{SPACE_ID}")
c.predict("K82s4", "FF_COOLDOWN_FRAC=0.6 FF_MATRIX_LR_SCALE=2.3", 73, "C", 2000, api_name="/predict_overrides")
c.predict({BEST_OFFICIAL}, 10, api_name="/speed_to_score")
c.predict({BEST_OFFICIAL}, {TOP10_SCORE}, api_name="/score_to_speed")
c.predict("trn2.3xlarge", "", 1800, "K82s4", "", "FF_COOLDOWN_FRAC=0.65", 58, 0.9551, 2097152, None, 2390, None,
          "", "", "", True, api_name="/contrib_record")
```

`predict_overrides(base, overrides, seed, chip, n_draws)`: seed is one of {', '.join(map(str, KNOWN_SEEDS))}, or
< 0 for the average over seeds; chip `C` or `D`. Invalid input returns `{{"error": ...}}`; a recipe train.py would
refuse returns `"invalid": true` and no numbers.

## Source, credits, licence

- Code and data: {GITHUB_REPO_URL} (the `ffsim` package, `research/sim-data`, recipes and write-ups).
- Built by the **FrontierForge team**.
- The simulator code, data and this app are MIT-licensed (`LICENSE`). The recipe `train.py` files in the full
  release are modified versions of the organiser's Apache-2.0 baseline and stay under Apache-2.0; they are not part
  of this Space (only K82s4's knob values are, in `recipes/recipe-K82s4.json`).
- The challenge, its kit and data are the organiser's
  ([github.com/aws-neuron/trainium-frontier](https://github.com/aws-neuron/trainium-frontier)); none of it is
  redistributed here.

<sub>Start-up: models fitted in {FIT_TIMINGS['total_s']:.2f} s, app ready in {STARTUP_S:.2f} s.</sub>
"""


# --------------------------------------------------------------------------- UI
def build_ui() -> Any:
    import gradio as gr

    defaults = base_defaults(DEFAULT_BASE)
    with gr.Blocks(title="FrontierForge Trainium simulator") as demo:
        gr.Markdown("# FrontierForge Trainium simulator\n"
                    "Predict what a training recipe scores on AWS Trainium (val_bpb) without a chip run. "
                    "Start from our best published recipe, change knobs, and compare. "
                    "Read the **About** tab before trusting small differences.")
        with gr.Tabs():
            with gr.Tab("Predict a recipe"):
                with gr.Row():
                    base_dd = gr.Dropdown(choices=BASE_CHOICES, value=DEFAULT_BASE, label="Default recipe to start from",
                                          info="your changes are compared against it", scale=3)
                    seed_in = gr.Dropdown(choices=SEED_CHOICES, value="73", label="Training seed",
                                          info="the seeds the fitted runs used (73 = our upload seed)", scale=1)
                    chip_in = gr.Dropdown(choices=list(CHIPS), value="C", label="Chip",
                                          info="D = slower runtime 2.33.10", scale=1)
                comps: List[Any] = []
                by_name = {k: (label, kind, info) for k, label, kind, info in KNOB_FIELDS}
                made: Dict[str, Any] = {}
                for title, names, note in KNOB_GROUPS:
                    with gr.Accordion(title, open=(title == "Schedule and batch")):
                        if note:
                            gr.Markdown(note)
                        for i in range(0, len(names), 3):
                            with gr.Row():
                                for k in names[i:i + 3]:
                                    label, kind, info = by_name[k]
                                    v = defaults[KNOB_NAMES.index(k)]
                                    if kind == "num":
                                        made[k] = gr.Number(value=v, label=label, info=info)
                                    elif kind == "text":
                                        made[k] = gr.Textbox(value=v, label=label, info=info)
                                    else:
                                        made[k] = gr.Dropdown(choices=kind[1], value=v, label=label, info=info,
                                                              allow_custom_value=True)
                comps = [made[k] for k in KNOB_NAMES]
                with gr.Accordion("Advanced", open=False):
                    extra_in = gr.Textbox(label="Extra FF_* overrides", lines=2,
                                          placeholder="FF_SOFTCAP_A=16.5 FF_XSA_LAYERS=all   (or a JSON object)",
                                          info="applied last; any knob train.py reads (unmodelled knobs are flagged); "
                                               "FF_SEED here is ignored (use the seed input)")
                    n_in = gr.Slider(500, MAX_DRAWS, value=2000, step=500, label="Monte Carlo draws")
                with gr.Row():
                    go = gr.Button("Predict", variant="primary")
                    reset = gr.Button("Reset to default recipe")
                out_md = gr.Markdown()
                with gr.Accordion("Raw result (JSON)", open=False):
                    out_json = gr.JSON()
                inputs = [base_dd] + comps + [seed_in, chip_in, n_in, extra_in]
                go.click(predict_recipe, inputs, [out_md, out_json], api_name="predict")
                base_dd.change(base_defaults, [base_dd], comps, api_visibility="private")
                reset.click(base_defaults, [base_dd], comps, api_visibility="private")
                demo.load(predict_recipe, inputs, [out_md, out_json], api_visibility="private")

                # API-only endpoint: base + KEY=VALUE overrides -> JSON
                with gr.Row(visible=False):
                    api_base = gr.Textbox(value=DEFAULT_BASE)
                    api_ov = gr.Textbox()
                    api_seed = gr.Number(value=73)
                    api_chip = gr.Textbox(value="C")
                    api_n = gr.Number(value=2000)
                    api_out = gr.JSON()
                    api_btn = gr.Button()
                api_btn.click(predict_overrides, [api_base, api_ov, api_seed, api_chip, api_n], api_out,
                              api_name="predict_overrides")

            with gr.Tab("Speed -> score calculator"):
                gr.Markdown("`delta_bpb = 0.063 x ln(compute ratio)`: more optimizer steps in the same time budget, "
                            "same recipe (the campaign's measured exchange rate). Rule of thumb near our operating "
                            f"point: 1% more steps ~ 0.00057 bpb. The Predict tab's fitted quality model has its own, "
                            f"flatter slope ({FITTED_STEP_SLOPE:+.4f} per ln steps at 2,388 steps); both are shown.")
                base_score = gr.Number(value=BEST_OFFICIAL, label="Starting score (val_bpb)",
                                       info=f"{BEST_OFFICIAL} = our best official (K77a) as of {BEST_OFFICIAL_ASOF}")
                with gr.Row():
                    with gr.Column():
                        pct_in = gr.Number(value=10.0, label="Throughput gain (%)", info="e.g. a faster kernel")
                        b1 = gr.Button("Throughput -> score", variant="primary")
                        o1 = gr.Markdown()
                        j1 = gr.JSON(visible=False)
                    with gr.Column():
                        tgt_in = gr.Number(value=TOP10_SCORE, label="Target score (val_bpb)",
                                           info=f"{TOP10_SCORE} = leaderboard #10 at ~11 PM CDT, 30 Sep 2026")
                        b2 = gr.Button("Score -> throughput needed", variant="primary")
                        o2 = gr.Markdown()
                        j2 = gr.JSON(visible=False)
                b1.click(speed_to_score, [base_score, pct_in], [o1, j1], api_name="speed_to_score")
                b2.click(score_to_speed, [base_score, tgt_in], [o2, j2], api_name="score_to_speed")

            with gr.Tab("Add my runs"):
                gr.Markdown(CONTRIB_SHARED_MD)
                with gr.Row():
                    c_hw = gr.Dropdown(choices=list(HARDWARE), value="trn2.3xlarge", label="Hardware")
                    c_hw_other = gr.Textbox(label="Hardware (if other)", placeholder="e.g. trn2u.48xlarge")
                    c_tb = gr.Number(value=1800, label="Time budget (s)", info="the challenge: 1800")
                with gr.Row():
                    c_base = gr.Dropdown(choices=list(BASE_RECIPES), value="K82s4", label="Started from recipe",
                                         info="custom = list your full FF_* environment below")
                    c_cv = gr.Textbox(label="Code version (optional)", placeholder="e.g. M14")
                    c_seed = gr.Number(value=None, label="Seed", info="FF_SEED", precision=0)
                c_recipe = gr.Textbox(label="Recipe: FF_* knobs you changed", lines=3,
                                      placeholder="FF_COOLDOWN_FRAC=0.65 FF_WD=0.03   (or a JSON object)")
                with gr.Row():
                    c_chip = gr.Number(value=None, label="Local val_bpb (rehearsal)", info="trained weights")
                    c_tok = gr.Number(value=2097152, label="Local eval tokens", precision=0,
                                      info="2097152 = the first 2M public tokens (what the model fits)")
                    c_off = gr.Number(value=None, label="Official val_bpb (if scored)")
                with gr.Row():
                    c_steps = gr.Number(value=None, label="Optimizer steps", precision=0)
                    c_sps = gr.Number(value=None, label="Seconds per step (optional)")
                    c_date = gr.Textbox(label="Date (optional)", placeholder="YYYY-MM-DD")
                with gr.Row():
                    c_notes = gr.Textbox(label="Framework notes (optional)", placeholder="Neuron SDK / runtime versions")
                    c_who = gr.Textbox(label="Public handle (optional)", placeholder="GitHub-style handle for credit")
                c_consent = gr.Checkbox(value=False, label="I agree that this record becomes public (see the text above)")
                c_btn = gr.Button("Check and build my submission", variant="primary")
                c_md = gr.Markdown()
                c_json = gr.Code(language="json", label="The record that will be shared")
                c_inputs = [c_hw, c_hw_other, c_tb, c_base, c_cv, c_recipe, c_seed, c_chip, c_tok, c_off, c_steps,
                            c_sps, c_date, c_notes, c_who, c_consent]
                c_btn.click(contrib_ui, c_inputs, [c_md, c_json], api_visibility="private")
                with gr.Row(visible=False):
                    a_in = [gr.Textbox(value="trn2.3xlarge"), gr.Textbox(), gr.Number(value=1800),
                            gr.Textbox(value="K82s4"), gr.Textbox(), gr.Textbox(), gr.Number(), gr.Number(),
                            gr.Number(value=2097152), gr.Number(), gr.Number(), gr.Number(), gr.Textbox(),
                            gr.Textbox(), gr.Textbox(), gr.Checkbox()]
                    a_out = gr.JSON()
                    a_btn = gr.Button()
                a_btn.click(contrib_record, a_in, a_out, api_name="contrib_record")

            with gr.Tab("About"):
                gr.Markdown(about_md())
    return demo


demo = build_ui()

demo.queue(default_concurrency_limit=4, max_size=64)    # bounded: a public endpoint must not queue without limit

if __name__ == "__main__":
    demo.launch()
