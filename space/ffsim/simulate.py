"""ffsim.simulate: Monte Carlo over step-time jitter, seed and eval-shard offset -> SimResult.

    official_bpb = quality(recipe, steps, seed) + offset(eval shard)
    steps        = (time_target - step-clock reserve) split over the FF_ACCUM_SCHED phases at the
                   per-phase step times the step-time model predicts, each median scaled by
                   (1 + MEDIAN_TO_MEAN_OVERHEAD) because logged medians under-count the mean step

The core is `simulate_batch`: a list of recipes is scored as one (recipes x draws) numpy matrix
under one set of common random numbers, so a batch of hundreds of candidates costs a few dozen
numpy calls in total plus two model calls per recipe (well under a millisecond per candidate).
`simulate` and `simulate_pair` are batches of one and two.

Quality noise is split into three components with the right correlation between recipes:

    shared     seed luck (and the unseen seed/era/chip prior): one draw per Monte Carlo draw,
               scaled by each recipe's own sd, so it cancels between recipes in a paired run;
    parameter  the quality model's coefficient uncertainty: one beta ~ N(beta, cov) per draw,
               evaluated on each recipe's design row, so it cancels only where the rows agree;
    residual   the model's per-run residual sigma, drawn independently per recipe and draw
               (keyed on the recipe, so an identical recipe reproduces the base's own draw).

With `ffsim.quality.QualityModel` the paired difference then has the sd its `predict_delta`
reports, sqrt(d cov d + 2 sigma^2); P(< base) is a probability, not a sign indicator.

The three models are duck-typed (see `ffsim/CONTRACT.md`):

    steptime_model.predict(code_version, knobs, chip) -> {"k1": s, "k2": s, "k4": s}
        (a phase may be None; optional "sd" key or a `jitter_sd` attribute = relative per-run jitter)
    quality_model.predict(knobs, steps, code_version, chip, seed) -> (mean_bpb_2m, sd)
        (seed None = marginal over seeds; seed given = that seed's effect)
        optional `steps_slope` (float or callable) = d bpb / d ln(steps); else a finite difference
        optional `design_row(knobs, steps, code_version, chip, seed) -> (x, extra_var, notes)` with
        `beta`, `cov`, `sigma` = the parameter/residual split above; a model with only `sigma`
        gets an independent residual of that size; a model with neither keeps its sd shared (noted)
    offset_model.sample(rng, n) -> array of rehearsal->official offsets; .mean / .sd

`ffsim.steptime.steps_from_step_time` is the contract's scalar step counter. `steps_scalar` /
`steps_from_step_times` below implement the same train.py semantics; `simulate()` cross-checks
against the contract function when that module is importable and records a disagreement in
`SimResult.notes`.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ffsim.schema import K_PHASES, Recipe, SimResult

try:  # one source of truth for the median -> mean step overhead; the fallback keeps this module standalone
    from ffsim.steptime import MEDIAN_TO_MEAN_OVERHEAD
except Exception:  # noqa: BLE001 - the module is written by another agent
    MEDIAN_TO_MEAN_OVERHEAD = 0.0057

# --- hard facts (CONTRACT.md, runs.jsonl) ---------------------------------------------------
# The step clock stops short of FF_TIME_TARGET: over the 136 completed chip C/D full runs in
# runs.jsonl, time_target - charged_seconds (the log clock at the last step) is median 5.5 s
# (p10 5.1, p90 5.9, mean 5.51): C40 1793 -> 1787.1, C36 1790.5 -> 1784.9, the K59 queue seeds
# 1760 -> 1754.3. CONTRACT's "charged ~1790" for K59 is the harness clock (timing.txt 1790.1).
SAVE_RESERVE_S = 5.5
STARTUP_STEPS_EXCLUDED = 5      # steps 0-4 are not on the charged clock
DEFAULT_WARMUP = 20             # FF_WARMUP; the accumulation phases start at step FF_WARMUP
DEFAULT_ACCUM_SCHED = "1:0.06,2:0.2,4"
DEFAULT_JITTER_SD = 0.004       # chip jitter: per-run relative sd of the step time
BEST_OFFICIAL = 0.9655          # K60 (29 Sep 19:50 UTC; K59 scored 0.9671)
DEFAULT_STEPS_SLOPE = -0.057    # d bpb / d ln(steps) at ~2300 steps (1% of steps ~ 0.00057 bpb)
# k1/k2 as a fraction of k4 when the step-time model gives only k4: the measured full-run medians
# on chip C (C32..C40: k1/k4 0.306-0.318, k2/k4 0.544-0.547; K59 C40 0.3154 / 0.5444; the fitted
# K59 prediction 0.3155 / 0.5441). The earlier 0.325 / 0.56 were a calibration that absorbed the
# median -> mean overhead and the 3 s reserve, both of which the step walk now applies explicitly.
PHASE_RATIO_TO_K4 = {"k1": 0.316, "k2": 0.545}


# --- accumulation schedule ------------------------------------------------------------------
def parse_accum_sched(sched: Optional[str]) -> List[Tuple[str, float]]:
    """'1:0.06,2:0.2,4' -> [('k1', 0.06), ('k2', 0.2), ('k4', 1.0)]; '2:0.12,4' -> [('k2', 0.12), ('k4', 1.0)];
    '4' or '' -> [('k4', 1.0)]. The `until` values are fractions of FF_TIME_TARGET on the charged clock."""
    parts = [p.strip() for p in (sched or "").strip().split(",") if p.strip()]
    if not parts:
        return [("k4", 1.0)]
    phases: List[Tuple[str, float]] = []
    for p in parts[:-1]:
        if p.count(":") != 1:
            raise ValueError(f"FF_ACCUM_SCHED={sched!r}: expected 'k:until' parts before the final K")
        k, until = p.split(":")
        phases.append((f"k{int(k)}", float(until)))
    if ":" in parts[-1]:
        raise ValueError(f"FF_ACCUM_SCHED={sched!r}: the final entry is the plain K")
    phases.append((f"k{int(parts[-1])}", 1.0))
    last = 0.0
    for _name, until in phases:
        if not (last < until <= 1.0):
            raise ValueError(f"FF_ACCUM_SCHED={sched!r}: phase boundaries must increase in (0, 1]")
        last = until
    return phases


def fill_phases(step_times: Dict[str, Any], needed: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
    """Return a float per needed phase; phases the model did not give are filled from k4 by the
    full-run ratios (noted). Raises when the final phase is missing and cannot be derived."""
    have = {k: float(v) for k, v in step_times.items() if v is not None}
    notes: List[str] = []
    if "k4" not in have:
        for k, ratio in PHASE_RATIO_TO_K4.items():
            if k in have:
                have["k4"] = have[k] / ratio
                notes.append(f"k4 derived from {k}/{ratio}")
                break
    for k in needed:
        if k not in have:
            if "k4" in have and k in PHASE_RATIO_TO_K4:
                have[k] = have["k4"] * PHASE_RATIO_TO_K4[k]
                notes.append(f"{k} filled as k4*{PHASE_RATIO_TO_K4[k]}")
            else:
                raise ValueError(f"step-time model gave no '{k}' phase and it cannot be derived")
    for k, v in have.items():
        if not (v > 0.0 and math.isfinite(v)):
            raise ValueError(f"step time {k}={v} is not a positive number")
    return have, notes


def steps_scalar(step_times: Dict[str, float], charged_seconds: float, accum_sched: str = DEFAULT_ACCUM_SCHED,
                 warmup_steps: int = DEFAULT_WARMUP, time_target: Optional[float] = None,
                 startup_steps: int = STARTUP_STEPS_EXCLUDED) -> float:
    """Pure-Python optimizer-step count with train.py semantics (see accum_phase_of in train.py):

    - steps 0..startup_steps-1 run before the charged clock starts (counted, not charged);
    - steps startup_steps..warmup_steps-1 are the warm-up at K (the final phase's step time);
    - each earlier phase runs while charged / time_target < until (the crossing step still belongs to it);
    - the final phase runs until the next step would cross charged_seconds."""
    phases = parse_accum_sched(accum_sched)
    st, _ = fill_phases(step_times, [p for p, _ in phases])
    t_final = st[phases[-1][0]]
    tt = float(charged_seconds if time_target is None else time_target)
    n_warm = max(int(warmup_steps) - int(startup_steps), 0)
    clock = n_warm * t_final
    steps = float(startup_steps + n_warm)
    for phase, until in phases[:-1]:
        remaining = max(until * tt - clock, 0.0)
        k = math.ceil(remaining / st[phase])
        clock += k * st[phase]
        steps += k
    steps += math.floor(max(float(charged_seconds) - clock, 0.0) / t_final)
    return steps


def steps_from_step_times(step_times: Dict[str, Any], charged_seconds: float,
                          accum_sched: str = DEFAULT_ACCUM_SCHED, warmup_steps: int = DEFAULT_WARMUP,
                          time_target: Optional[float] = None,
                          startup_steps: int = STARTUP_STEPS_EXCLUDED) -> np.ndarray:
    """Vectorised `steps_scalar`: step_times values may be floats or arrays of any broadcastable shape
    (e.g. (m, 1) recipes x (1, n) jitter draws). Missing phases are filled like `fill_phases`."""
    phases = parse_accum_sched(accum_sched)
    needed = [p for p, _ in phases]
    arrs = {k: np.asarray(v, dtype=float) for k, v in step_times.items() if v is not None}
    if not all(k in arrs for k in needed):
        # fill missing phases from k4 (or derive k4), keeping the array shapes
        if "k4" not in arrs:
            for k, ratio in PHASE_RATIO_TO_K4.items():
                if k in arrs:
                    arrs["k4"] = arrs[k] / ratio
                    break
        for k in needed:
            if k not in arrs:
                if "k4" in arrs and k in PHASE_RATIO_TO_K4:
                    arrs[k] = arrs["k4"] * PHASE_RATIO_TO_K4[k]
                else:
                    raise ValueError(f"step-time model gave no '{k}' phase and it cannot be derived")
    shape = np.broadcast(*(arrs[k] for k in needed)).shape
    t_final = np.broadcast_to(arrs[phases[-1][0]], shape)
    tt = float(charged_seconds if time_target is None else time_target)
    n_warm = max(int(warmup_steps) - int(startup_steps), 0)
    clock = n_warm * t_final
    steps = float(startup_steps + n_warm)
    for phase, until in phases[:-1]:
        tp = arrs[phase]
        k = np.ceil(np.maximum(until * tt - clock, 0.0) / tp)
        clock = clock + k * tp
        steps = steps + k
    steps = steps + np.floor(np.maximum(float(charged_seconds) - clock, 0.0) / t_final)
    return np.asarray(steps, dtype=float)


# --- model adapters (duck-typed, defensive) -------------------------------------------------
def _call_predict_quality(model: Any, knobs: Dict[str, str], steps: float, code_version: str, chip: str,
                          seed: Optional[int]) -> Tuple[float, float]:
    attempts = (
        lambda: model.predict(knobs, steps, code_version, chip, seed),
        lambda: model.predict(knobs=knobs, steps=steps, code_version=code_version, chip=chip, seed=seed),
        lambda: model.predict(knobs, steps, code_version, seed),
        lambda: model.predict(knobs, steps, code_version),
    )
    err: Optional[Exception] = None
    for call in attempts:
        try:
            out = call()
            break
        except TypeError as e:  # signature mismatch: try the next form
            err = e
    else:
        raise TypeError(f"quality model predict() signature not understood: {err}")
    if isinstance(out, dict):
        return float(out["mean"]), float(out.get("sd", 0.0))
    mean, sd = out
    return float(mean), float(sd)


def predict_step_time(model: Any, recipe: Recipe) -> Tuple[Dict[str, Optional[float]], float, List[str]]:
    """-> ({phase: seconds or None}, relative jitter sd, notes)."""
    out = model.predict(recipe.code_version, recipe.knobs, recipe.chip)
    if out is None:
        raise ValueError(f"step-time model has no prediction for code_version={recipe.code_version!r} chip={recipe.chip!r}")
    notes: List[str] = []
    sd: Optional[float] = None
    st: Dict[str, Optional[float]] = {}
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(k, str) and k.startswith("k") and k[1:].isdigit():
                st[k] = None if v is None else float(v)
        for key in ("sd", "jitter_sd", "rel_sd"):
            if out.get(key) is not None:
                sd = float(out[key])
                break
        if sd is None and out.get("sd_k4") is not None and st.get("k4"):
            sd = float(out["sd_k4"]) / float(st["k4"])          # ffsim.steptime: absolute seconds on k4
        n = out.get("notes")
        if isinstance(n, str) and n:
            notes.append(n)
        elif isinstance(n, (list, tuple)):
            notes += [str(x) for x in n if x]
        if out.get("fallback"):
            notes.append(f"step-time fallback: {out['fallback']}")
        if out.get("version_used") not in (None, recipe.code_version):
            notes.append(f"step-time lineage {out['version_used']} used for {recipe.code_version}")
    else:  # a StepTime dataclass or anything with k1/k2/k4 attributes
        for k in K_PHASES:
            v = getattr(out, k, None)
            st[k] = None if v is None else float(v)
    if sd is None:
        sd = getattr(model, "jitter_sd", None)
    if sd is None:
        sd = DEFAULT_JITTER_SD
    return st, float(sd), notes


def predict_quality(model: Any, recipe: Recipe, steps_mean: float, seed: Optional[int]) -> Tuple[float, float, float, List[str]]:
    """-> (mean bpb_2m at steps_mean, sd, d bpb / d ln steps, notes)."""
    notes: List[str] = []
    mean, sd = _call_predict_quality(model, recipe.knobs, steps_mean, recipe.code_version, recipe.chip, seed)
    slope_attr = getattr(model, "steps_slope", None)
    slope: Optional[float] = None
    if callable(slope_attr):
        # ffsim.quality.QualityModel.steps_slope(steps) first, then a knob-aware 4-argument form,
        # then a constant callable: the slope at the recipe's own step count, not at STEPS_REF
        for call in (lambda: slope_attr(steps_mean),
                     lambda: slope_attr(recipe.knobs, steps_mean, recipe.code_version, recipe.chip),
                     lambda: slope_attr()):
            try:
                slope = float(call())
                break
            except TypeError:
                continue
        if slope is None:
            notes.append("steps_slope signature not understood: finite difference used")
    elif slope_attr is not None:
        slope = float(slope_attr)
    if slope is None:
        h = 0.02
        try:
            m1, _ = _call_predict_quality(model, recipe.knobs, steps_mean * (1.0 + h), recipe.code_version, recipe.chip, seed)
            slope = (m1 - mean) / math.log1p(h)
        except Exception as e:  # noqa: BLE001 - the model is foreign code
            slope = DEFAULT_STEPS_SLOPE
            notes.append(f"steps slope defaulted to {slope} ({type(e).__name__})")
    return float(mean), float(sd), float(slope), notes


@dataclass
class QualitySplit:
    """How one recipe's quality sd decomposes (see the module docstring); variances add up to sd^2."""
    resid_sd: float                     # independent per recipe and draw
    shared_sd: float                    # one shared draw, scaled per recipe (seed luck, unseen priors)
    x: Optional[np.ndarray] = None      # design row for the shared parameter draw (None: no cov)
    notes: List[str] = field(default_factory=list)


def quality_noise_split(model: Any, recipe: Recipe, steps: float, seed: Optional[int], sd_total: float) -> QualitySplit:
    """Split the model's sd at `steps` into residual / shared / parameter parts.

    With `design_row` + `beta` + `cov` + `sigma` (ffsim.quality.QualityModel): parameter variance
    x cov x from the row, residual sigma, and whatever predict() added on top (seed spread, unseen
    era/chip prior) as the shared part, so the three add up to exactly predict()'s sd. With only
    `sigma`: residual sigma, the rest shared. With neither: all shared (the old behaviour), noted,
    because P(< base) then degenerates to a sign indicator of the mean difference."""
    sd_total = float(sd_total)
    sigma = getattr(model, "sigma", None)
    cov = getattr(model, "cov", None)
    design_row = getattr(model, "design_row", None)
    x: Optional[np.ndarray] = None
    if callable(design_row) and cov is not None and sigma is not None:
        try:
            out = design_row(recipe.knobs, steps, recipe.code_version, recipe.chip, seed)
            cand = np.asarray(out[0] if isinstance(out, (tuple, list)) else out, dtype=float).reshape(-1)
            cov_a = np.asarray(cov, dtype=float)
            if cov_a.shape == (cand.shape[0], cand.shape[0]) and np.all(np.isfinite(cand)):
                x = cand
        except Exception:  # noqa: BLE001 - foreign model code: fall through to the sigma-only split
            x = None
    if x is not None:
        param_var = max(float(x @ np.asarray(cov, dtype=float) @ x), 0.0)
        resid = min(max(float(sigma), 0.0), sd_total)
        shared_var = max(sd_total ** 2 - resid ** 2 - param_var, 0.0)
        return QualitySplit(resid_sd=resid, shared_sd=math.sqrt(shared_var), x=x)
    if sigma is not None:
        try:
            resid = min(max(float(sigma), 0.0), sd_total)
        except (TypeError, ValueError):
            resid = 0.0
        return QualitySplit(resid_sd=resid, shared_sd=math.sqrt(max(sd_total ** 2 - resid ** 2, 0.0)))
    return QualitySplit(resid_sd=0.0, shared_sd=sd_total,
                        notes=["quality noise all shared (model exposes no sigma/cov): P(< base) is a sign indicator"])


def cov_factor(cov: np.ndarray) -> np.ndarray:
    """L with L L^T = cov for a symmetric positive semi-definite cov (eigen form: never fails on a
    ridge covariance whose smallest eigenvalues are ~1e-11)."""
    c = np.asarray(cov, dtype=float)
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    return v * np.sqrt(np.clip(w, 0.0, None))[None, :]


def recipe_key(recipe: Recipe, seed: Optional[int]) -> str:
    """What makes two recipes the same run: code version, chip, training seed, every knob and the
    time target (never the name)."""
    return json.dumps([recipe.code_version, recipe.chip, seed, recipe_time_target(recipe),
                       sorted((str(k), str(v)) for k, v in recipe.knobs.items())], separators=(",", ":"))


def residual_draw(resid_seed: int, key: str, n: int) -> np.ndarray:
    """Standard normals for one recipe's per-run residual: deterministic in (resid_seed, recipe),
    independent between different recipes, identical for identical recipes across batches, chunks
    and worker processes (a stable hash, not Python's)."""
    h = hashlib.blake2b(f"{int(resid_seed)}|{key}".encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big")).standard_normal(n)


def sample_offset(model: Any, rng: np.random.Generator, n: int) -> np.ndarray:
    sample = getattr(model, "sample", None)
    if callable(sample):
        return np.asarray(sample(rng, n), dtype=float).reshape(n)
    mean = float(getattr(model, "mean", 0.0))
    sd = float(getattr(model, "sd", 0.0))
    return rng.normal(mean, sd, n)


# --- noise ----------------------------------------------------------------------------------
@dataclass
class Noise:
    """Common random numbers for one Monte Carlo batch: reuse the same Noise for every recipe you
    want to compare, so chip jitter, seed luck, the shared coefficient draw and the upload offset
    cancel in the differences. Only each recipe's own residual (sigma) stays independent."""
    z_time: np.ndarray      # (n,) standard normal: per-run chip jitter (all phases move together)
    z_quality: np.ndarray   # (n,) standard normal: seed luck / unseen-prior part of the quality sd
    offset: np.ndarray      # (n,) rehearsal -> official offset draws
    param_seed: int = 0     # seeds the (n, p) coefficient draw beta ~ N(beta, cov), shared by all recipes
    resid_seed: int = 0     # seeds each recipe's own residual draw (hashed with the recipe key)

    @property
    def n(self) -> int:
        return int(self.z_time.shape[0])

    def param_draw(self, p: int) -> np.ndarray:
        """(n, p) standard normals for the coefficient draw: the same for every batch under this Noise."""
        return np.random.default_rng(int(self.param_seed)).standard_normal((self.n, int(p)))


def make_rng(rng: Union[None, int, np.random.Generator]) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def draw_noise(rng: Union[None, int, np.random.Generator], n: int, offset_model: Any) -> Noise:
    g = make_rng(rng)
    z_time, z_quality, offset = g.standard_normal(n), g.standard_normal(n), sample_offset(offset_model, g, n)
    return Noise(z_time=z_time, z_quality=z_quality, offset=offset,
                 param_seed=int(g.integers(0, 2 ** 62)), resid_seed=int(g.integers(0, 2 ** 62)))


def recipe_time_target(recipe: Recipe) -> float:
    """FF_TIME_TARGET in the knobs wins (it is what train.py reads); otherwise Recipe.time_target."""
    v = recipe.knobs.get("FF_TIME_TARGET")
    if v not in (None, ""):
        return float(v)
    return float(recipe.time_target)


# --- the batched core -----------------------------------------------------------------------
@dataclass
class BatchDraws:
    """(m recipes x n draws) arrays under one Noise. Failed recipes have NaN rows and an entry in errors."""
    recipes: List[Recipe]
    steps: np.ndarray               # (m, n)
    bpb_2m: np.ndarray              # (m, n)
    official: np.ndarray            # (m, n)
    step_time: List[Dict[str, float]]   # the model's phase medians (what train.log medians show)
    steps_point: np.ndarray         # (m,) steps at the model's mean step time
    quality_mean: np.ndarray        # (m,)
    quality_sd: np.ndarray          # (m,) the model's total sd (shared^2 + parameter + residual^2)
    steps_slope: np.ndarray         # (m,)
    notes: List[List[str]]
    errors: Dict[int, str] = field(default_factory=dict)
    quality_resid_sd: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (m,) independent part
    quality_shared_sd: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (m,) shared-draw part
    quality_param_sd: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (m,) sqrt(x cov x)

    @property
    def m(self) -> int:
        return len(self.recipes)

    def row(self, i: int) -> "Draws":
        def at(a: np.ndarray) -> float:
            return float(a[i]) if a.shape[0] > i else 0.0
        return Draws(steps=self.steps[i], bpb_2m=self.bpb_2m[i], official=self.official[i],
                     step_time=dict(self.step_time[i]), quality_mean=float(self.quality_mean[i]),
                     quality_sd=float(self.quality_sd[i]), steps_slope=float(self.steps_slope[i]),
                     notes=list(self.notes[i]), quality_resid_sd=at(self.quality_resid_sd),
                     quality_shared_sd=at(self.quality_shared_sd), quality_param_sd=at(self.quality_param_sd))


@dataclass
class Draws:
    steps: np.ndarray
    bpb_2m: np.ndarray
    official: np.ndarray
    step_time: Dict[str, float]
    quality_mean: float
    quality_sd: float
    steps_slope: float
    notes: List[str] = field(default_factory=list)
    quality_resid_sd: float = 0.0
    quality_shared_sd: float = 0.0
    quality_param_sd: float = 0.0


def walk_step_times(step_time: Dict[str, float], overhead_frac: float) -> Dict[str, float]:
    """Phase medians -> the per-step means the charged clock actually advances by."""
    return {k: float(v) * (1.0 + float(overhead_frac)) for k, v in step_time.items()}


def simulate_batch(recipes: Sequence[Recipe], steptime_model: Any, quality_model: Any, noise: Noise,
                   seed: Optional[int] = None, save_reserve: float = SAVE_RESERVE_S,
                   jitter_sd: Optional[float] = None, on_error: str = "raise",
                   overhead_frac: float = MEDIAN_TO_MEAN_OVERHEAD) -> BatchDraws:
    """Score every recipe under the same Noise. Per recipe: one step-time and one quality model call
    plus pure-Python scalar work; then the (m, n) matrices are built in a handful of numpy calls.

    `save_reserve` is the step-clock reserve (charged = FF_TIME_TARGET - save_reserve) and
    `overhead_frac` the median -> mean step overhead applied to every phase before the step walk."""
    m, n = len(recipes), noise.n
    step_time: List[Dict[str, float]] = [{} for _ in range(m)]
    walk: List[Dict[str, float]] = [{} for _ in range(m)]
    notes: List[List[str]] = [[] for _ in range(m)]
    errors: Dict[int, str] = {}
    sd_rel = np.zeros(m)
    q_mean = np.full(m, np.nan)
    q_sd = np.zeros(m)
    resid_sd = np.zeros(m)
    shared_sd = np.zeros(m)
    param_sd = np.zeros(m)
    slope = np.zeros(m)
    steps_point = np.full(m, np.nan)
    rows: Dict[int, np.ndarray] = {}
    z_resid = np.zeros((m, n))
    groups: Dict[Tuple[str, int, float, float], List[int]] = {}
    for i, r in enumerate(recipes):
        try:
            st_pred, model_sd, st_notes = predict_step_time(steptime_model, r)
            sched = r.knobs.get("FF_ACCUM_SCHED", DEFAULT_ACCUM_SCHED)
            warmup = int(float(r.knobs.get("FF_WARMUP", DEFAULT_WARMUP) or DEFAULT_WARMUP))
            phases = parse_accum_sched(sched)
            st_filled, fill_notes = fill_phases(st_pred, [p for p, _ in phases])
            st_walk = walk_step_times(st_filled, overhead_frac)
            tt = recipe_time_target(r)
            charged = tt - float(save_reserve)
            sp = steps_scalar(st_walk, charged, sched, warmup, time_target=tt)
            qm, qs, sl, q_notes = predict_quality(quality_model, r, sp, seed)
            split = quality_noise_split(quality_model, r, sp, seed, qs)
            z_resid[i] = residual_draw(noise.resid_seed, recipe_key(r, seed), n)
        except Exception as e:  # noqa: BLE001 - foreign model code
            if on_error == "raise":
                raise
            errors[i] = f"{type(e).__name__}: {e}"
            continue
        step_time[i] = st_filled
        walk[i] = st_walk
        notes[i] = st_notes + fill_notes + q_notes + split.notes
        sd_rel[i] = float(jitter_sd if jitter_sd is not None else model_sd)
        q_mean[i], q_sd[i], slope[i], steps_point[i] = qm, qs, sl, sp
        resid_sd[i], shared_sd[i] = split.resid_sd, split.shared_sd
        if split.x is not None:
            rows[i] = split.x
        groups.setdefault((sched, warmup, tt, charged), []).append(i)
    factor = 1.0 + np.outer(sd_rel, noise.z_time)                 # (m, n): one jitter factor per run
    steps = np.full((m, n), np.nan)
    for (sched, warmup, tt, charged), idx in groups.items():
        phases = [p for p, _ in parse_accum_sched(sched)]
        ia = np.asarray(idx)
        st = {p: np.array([walk[i][p] for i in idx])[:, None] * factor[ia] for p in phases}
        steps[ia] = steps_from_step_times(st, charged, sched, warmup, time_target=tt)
    # coefficient uncertainty: one beta ~ N(beta, cov) per draw, shared by every recipe, evaluated on
    # each recipe's design row (m, p) @ (p, p) @ (p, n): cancels between recipes exactly where the
    # rows agree, which is what the quality model's own predict_delta assumes
    param = np.zeros((m, n))
    if rows:
        cov = np.asarray(getattr(quality_model, "cov"), dtype=float)
        p = cov.shape[0]
        X = np.zeros((m, p))
        for i, x in rows.items():
            X[i] = x
        XL = X @ cov_factor(cov)                                   # (m, p): rows of x L
        param = XL @ noise.param_draw(p).T                         # (m, n)
        param_sd = np.sqrt(np.maximum(np.einsum("ij,ij->i", XL, XL), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        ln_ratio = np.log(np.maximum(steps, 1.0)) - np.log(np.maximum(steps_point, 1.0))[:, None]
        bpb = (q_mean[:, None] + slope[:, None] * ln_ratio + param
               + np.outer(shared_sd, noise.z_quality) + resid_sd[:, None] * z_resid)
    official = bpb + noise.offset[None, :]
    return BatchDraws(recipes=list(recipes), steps=steps, bpb_2m=bpb, official=official, step_time=step_time,
                      steps_point=steps_point, quality_mean=q_mean, quality_sd=q_sd, steps_slope=slope,
                      notes=notes, errors=errors, quality_resid_sd=resid_sd, quality_shared_sd=shared_sd,
                      quality_param_sd=param_sd)


def simulate_draws(recipe: Recipe, steptime_model: Any, quality_model: Any, noise: Noise,
                   seed: Optional[int] = None, save_reserve: float = SAVE_RESERVE_S,
                   jitter_sd: Optional[float] = None, overhead_frac: float = MEDIAN_TO_MEAN_OVERHEAD) -> Draws:
    """One recipe under a fixed Noise -> per-draw steps, bpb_2m and official (a batch of one)."""
    return simulate_batch([recipe], steptime_model, quality_model, noise, seed=seed, save_reserve=save_reserve,
                          jitter_sd=jitter_sd, overhead_frac=overhead_frac).row(0)


def p_less(a: np.ndarray, b: np.ndarray) -> float:
    """P(a < b) over paired draws, ties counted half (identical recipes -> 0.5)."""
    return float(np.mean(a < b) + 0.5 * np.mean(a == b))


def summarise_batch(bd: BatchDraws, best_official: float = BEST_OFFICIAL,
                    base_official: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """SimResult fields for every recipe as dicts (axis=1 reductions: one numpy call per statistic
    for the whole batch), plus delta_vs_base / p_beat_base when the base's official draws are given.
    Failed recipes get an 'error' entry instead."""
    m = bd.m
    steps_mean, steps_sd = bd.steps.mean(1), bd.steps.std(1)
    bpb_mean, bpb_sd = bd.bpb_2m.mean(1), bd.bpb_2m.std(1)
    off_mean, off_sd = bd.official.mean(1), bd.official.std(1)
    pct = np.percentile(bd.official, [10, 50, 90], axis=1) if m else np.zeros((3, 0))
    p_best = (bd.official < best_official).mean(1)
    if base_official is not None:
        b = base_official[None, :]
        p_base = (bd.official < b).mean(1) + 0.5 * (bd.official == b).mean(1)
        delta = off_mean - float(base_official.mean())
        delta_sd = (bd.official - b).std(1)          # the paired sd: what a confirmation run has to beat
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(bd.recipes):
        if i in bd.errors:
            rows.append({"recipe": r.name, "error": bd.errors[i]})
            continue
        row: Dict[str, Any] = {
            "recipe": r.name, "n": int(bd.steps.shape[1]),
            "steps_mean": float(steps_mean[i]), "steps_sd": float(steps_sd[i]),
            "bpb_2m_mean": float(bpb_mean[i]), "bpb_2m_sd": float(bpb_sd[i]),
            "official_mean": float(off_mean[i]), "official_sd": float(off_sd[i]),
            "official_p10": float(pct[0, i]), "official_p50": float(pct[1, i]), "official_p90": float(pct[2, i]),
            "p_beat_best": float(p_best[i]), "step_time": dict(bd.step_time[i]),
            "notes": "; ".join(x for x in bd.notes[i] if x),
        }
        if base_official is not None:
            row["delta_vs_base"] = float(delta[i])
            row["delta_sd_vs_base"] = float(delta_sd[i])
            row["p_beat_base"] = float(p_base[i])
        rows.append(row)
    return rows


def summarise(recipe: Recipe, d: Draws, best_official: float = BEST_OFFICIAL, extra_notes: Sequence[str] = ()) -> SimResult:
    p10, p50, p90 = np.percentile(d.official, [10, 50, 90])
    notes = list(d.notes) + list(extra_notes)
    return SimResult(
        recipe=recipe.name, n=int(d.steps.shape[0]),
        steps_mean=float(d.steps.mean()), steps_sd=float(d.steps.std()),
        bpb_2m_mean=float(d.bpb_2m.mean()), bpb_2m_sd=float(d.bpb_2m.std()),
        official_mean=float(d.official.mean()), official_sd=float(d.official.std()),
        official_p10=float(p10), official_p50=float(p50), official_p90=float(p90),
        p_beat_best=float(np.mean(d.official < best_official)),
        step_time=dict(d.step_time), notes="; ".join(n for n in notes if n),
    )


def _crosscheck_contract_steps(recipe: Recipe, d: Draws, save_reserve: float,
                               overhead_frac: float = MEDIAN_TO_MEAN_OVERHEAD) -> List[str]:
    """Compare the mean-step-time step count with ffsim.steptime.steps_from_step_time when that
    module exists (both walks get the overhead-scaled medians). A disagreement is reported, never hidden."""
    try:
        from ffsim.steptime import steps_from_step_time  # lazy: the module is written by another agent
    except Exception:  # noqa: BLE001 - missing module or import error
        return []
    sched = recipe.knobs.get("FF_ACCUM_SCHED", DEFAULT_ACCUM_SCHED)
    warmup = int(float(recipe.knobs.get("FF_WARMUP", DEFAULT_WARMUP) or DEFAULT_WARMUP))
    tt = recipe_time_target(recipe)
    charged = tt - save_reserve
    st_walk = walk_step_times(d.step_time, overhead_frac)
    try:
        try:
            ref = steps_from_step_time(st_walk, charged, sched, warmup, time_target=tt)
        except TypeError:
            ref = steps_from_step_time(st_walk, charged, sched, warmup)
        ref = float(ref if not isinstance(ref, dict) else ref.get("steps"))
    except Exception as e:  # noqa: BLE001
        return [f"contract steps_from_step_time not comparable ({type(e).__name__})"]
    mine = steps_scalar(st_walk, charged, sched, warmup, time_target=tt)
    if abs(ref - mine) > max(2.0, 0.003 * mine):
        return [f"steps cross-check: ffsim.steptime {ref:.0f} vs simulate {mine:.0f}"]
    return []


def simulate(recipe: Recipe, steptime_model: Any, quality_model: Any, offset_model: Any, n: int = 1000,
             rng: Union[None, int, np.random.Generator] = None, best_official: float = BEST_OFFICIAL,
             seed: Optional[int] = None, save_reserve: float = SAVE_RESERVE_S,
             jitter_sd: Optional[float] = None, noise: Optional[Noise] = None,
             overhead_frac: float = MEDIAN_TO_MEAN_OVERHEAD) -> SimResult:
    """Monte Carlo prediction of one recipe. `seed` is the training seed handed to the quality model
    (None = marginal over seeds). `rng` seeds the Monte Carlo draws; pass `noise` to reuse draws."""
    nz = noise if noise is not None else draw_noise(rng, n, offset_model)
    d = simulate_draws(recipe, steptime_model, quality_model, nz, seed=seed, save_reserve=save_reserve,
                       jitter_sd=jitter_sd, overhead_frac=overhead_frac)
    return summarise(recipe, d, best_official, _crosscheck_contract_steps(recipe, d, save_reserve, overhead_frac))


@dataclass
class PairResult:
    a: SimResult
    b: SimResult
    p_a_better: float       # P(official_a < official_b) under common random numbers
    diff_mean: float        # mean(official_a - official_b): negative = a is better
    diff_sd: float
    diff_p10: float
    diff_p50: float
    diff_p90: float
    p_a_better_2m: float    # same on the rehearsal (bpb_2m) scale, offset cancels exactly


def simulate_pair(recipe_a: Recipe, recipe_b: Recipe, steptime_model: Any, quality_model: Any, offset_model: Any,
                  n: int = 1000, rng: Union[None, int, np.random.Generator] = None,
                  best_official: float = BEST_OFFICIAL, seed: Optional[int] = None,
                  save_reserve: float = SAVE_RESERVE_S, jitter_sd: Optional[float] = None,
                  noise: Optional[Noise] = None, overhead_frac: float = MEDIAN_TO_MEAN_OVERHEAD) -> PairResult:
    """P(a beats b) with common random numbers: the same chip-jitter, seed-luck, coefficient and
    offset draws are used for both recipes, so shared noise cancels; each recipe keeps its own
    per-run residual, so diff_sd is the pair noise floor sqrt(d cov d + 2 sigma^2), not ~0."""
    nz = noise if noise is not None else draw_noise(rng, n, offset_model)
    bd = simulate_batch([recipe_a, recipe_b], steptime_model, quality_model, nz, seed=seed,
                        save_reserve=save_reserve, jitter_sd=jitter_sd, overhead_frac=overhead_frac)
    da, db = bd.row(0), bd.row(1)
    diff = da.official - db.official
    p10, p50, p90 = np.percentile(diff, [10, 50, 90])
    return PairResult(
        a=summarise(recipe_a, da, best_official), b=summarise(recipe_b, db, best_official),
        p_a_better=p_less(da.official, db.official), diff_mean=float(diff.mean()), diff_sd=float(diff.std()),
        diff_p10=float(p10), diff_p50=float(p50), diff_p90=float(p90),
        p_a_better_2m=p_less(da.bpb_2m, db.bpb_2m),
    )


# --- recipes and model bundles --------------------------------------------------------------
def recipe_from_dict(d: Dict[str, Any], name: Optional[str] = None) -> Recipe:
    knobs = {str(k): ("" if v is None else str(v)) for k, v in (d.get("knobs") or {}).items()}
    return Recipe(name=str(name or d.get("name") or "recipe"), code_version=str(d.get("code_version", "M12")),
                  knobs=knobs, chip=str(d.get("chip", "C")), time_target=float(d.get("time_target", 1790.0)))


def load_recipe(path: Union[str, Path]) -> Recipe:
    with open(path, "r", encoding="utf-8") as f:
        return recipe_from_dict(json.load(f))


@dataclass
class Models:
    steptime: Any
    quality: Any
    offset: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[Any, Any, Any]:
        return self.steptime, self.quality, self.offset


def save_models(models: Models, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump({"steptime": models.steptime, "quality": models.quality, "offset": models.offset,
                     "meta": dict(models.meta)}, f)


def load_models(path: Union[str, Path]) -> Models:
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, Models):
        return d
    return Models(steptime=d["steptime"], quality=d["quality"], offset=d["offset"], meta=dict(d.get("meta", {})))


# --- anchor stubs: knob-blind models pinned to the K59 facts (pipeline smoke tests only) ------
class AnchorStepTimeModel:
    """Knob-blind: chip C k4 median 0.928 s (LK0.35 s73 full runs), k1/k2 by the measured full-run ratios;
    chip D k4 1.036 s. With the overhead and the 5.5 s reserve this walks to 2362 steps at FF_TIME_TARGET
    1793 (C40 itself: 2357 with its own slower 0.9312 s median). Exists so the CLI runs end to end before
    a fitted models.pkl exists. Not a surrogate."""
    jitter_sd = DEFAULT_JITTER_SD
    support: Dict[str, Any] = {}
    K4 = {"A": 0.928, "B": 0.928, "C": 0.928, "D": 1.036}

    def predict(self, code_version: str, knobs: Dict[str, str], chip: str = "C") -> Dict[str, Any]:
        k4 = self.K4.get(chip, 0.928)
        return {"k1": k4 * PHASE_RATIO_TO_K4["k1"], "k2": k4 * PHASE_RATIO_TO_K4["k2"], "k4": k4,
                "sd": self.jitter_sd, "notes": "anchor stub (knob-blind)"}


class AnchorQualityModel:
    """Knob-blind: C40 (K59, seed 73) 0.96092 @ 2357 steps, slope -0.057 per ln(steps), seed sd 0.0006.
    seed=None -> marginal over seeds; seed 73 -> the anchor itself; another seed -> a deterministic
    pseudo seed effect (sd 0.0002 residual). `sigma` = the residual the simulator draws per recipe."""
    support: Dict[str, Any] = {}
    steps_slope = DEFAULT_STEPS_SLOPE
    anchor_bpb = 0.96092
    anchor_steps = 2357.0
    anchor_seed = 73
    seed_sd = 0.0006
    fit_sd = 0.0002
    sigma = fit_sd

    def predict(self, knobs: Dict[str, str], steps: float, code_version: str = "M12", chip: str = "C",
                seed: Optional[int] = None) -> Tuple[float, float]:
        mean = self.anchor_bpb + self.steps_slope * (math.log(max(float(steps), 1.0)) - math.log(self.anchor_steps))
        if seed is None:
            return mean, math.hypot(self.seed_sd, self.fit_sd)
        z = 0.0 if int(seed) == self.anchor_seed else float(np.random.default_rng(int(seed)).standard_normal())
        return mean + self.seed_sd * z, self.fit_sd


class AnchorOffsetModel:
    """Rehearsal -> official: +0.0061..+0.0072 over K51..K59, mean +0.0066, sd 0.0004."""
    mean = 0.0066
    sd = 0.0004

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(self.mean, self.sd, n)


def anchor_models() -> Models:
    return Models(AnchorStepTimeModel(), AnchorQualityModel(), AnchorOffsetModel(),
                  meta={"kind": "anchor-stub", "warning": "knob-blind: exercises the pipeline only"})


def simresult_to_dict(r: SimResult) -> Dict[str, Any]:
    return asdict(r)
