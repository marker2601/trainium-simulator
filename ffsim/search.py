"""ffsim.search: candidate spaces (JSON), generators, validation and batch evaluation over the surrogate.

Space format (mixed categorical / numeric dims; values are written the way train.py reads them):

    {"name": "k59-local",
     "base": {"name": "K59", "code_version": "M12", "chip": "C", "time_target": 1793, "knobs": {...}},
     "dims": [{"knob": "FF_LEAKY_RELU2", "values": [0.25, 0.3, 0.35, 0.4, 0.45]},
              {"knob": "FF_SOFTCAP", "range": [12, 20], "step": 1},
              {"knob": "code_version", "values": ["M12", "M14"]}],      # special dims: code_version, chip, time_target
     "max_changes": 2,
     "queue": {"reference_run": "C:0816_R_g7lk35_s73",                   # optional: how the base runs on the chip queue
               "job_env": "FF_COOLDOWN_FRAC=0.60 ... FF_LEAKY_RELU2=0.35",   # or "runner_log": path + reference_run
               "code_dirs": {"M12": "/root/ff-claude/code_k12off"}}}

`base` may instead carry {"recipe": "recipe-K59.json"} (a path relative to the space file).

Generators: grid (full product, then the max_changes filter), random (n draws), oat (one-at-a-time)
and local (every single and double change from the base).

Validation: every candidate's effective env is checked against the startup asserts of the K59 train.py
(TRAIN_PY_RULES: mutual exclusions such as FF_RELU2_FN x FF_LEAKY_RELU2, prerequisites, ranges). A candidate
train.py would refuse is not simulated; it is listed in results["invalid"] with the rule that fires. A rule
whose knobs the effective env does not state is skipped (train.py's default for that code dir is not known here).

Scoring: the ranking keys are ANALYTIC, not Monte Carlo means. `official_mean` = the quality model's mean at the
step count the model's own step times give + the offset model's mean; `bpb_2m_mean` and `delta_vs_base` likewise.
They are therefore identical for every `--rng` and draw count (before this a candidate's Monte Carlo mean carried
sd_i x mean(z), so wide-sd EXTRAP candidates moved together by up to +-0.0003 with the rng seed and draw count).
The draws give only the sd, percentiles and probabilities (`official_sd`, `official_p10/50/90`, `p_beat_best`,
`p_beat_base`); the raw Monte Carlo mean is kept as `official_mc_mean` so the residual sampling noise is visible.
Every candidate is scored with common random numbers (the same Noise, regenerated from `rng_seed` in each worker)
that are STANDARDISED (`StandardisedNoise`): the jitter and shared-quality z vectors and each column of the
coefficient draw are centred to mean 0 and scaled to sd 1, and the offset draws are re-centred on the offset
model's mean and sd, so the percentiles and probabilities carry no sample-mean bias either.

Support: `support_check` tags every changed dim. A knob value seen in the fitted runs is `ok(n=..)` when at least
`min_value_runs` fitted runs carry it (the quality model's own threshold, `QualityModel.config.min_value_runs`;
WEAK_MIN_RUNS when the model has none) and one of them is in the base's era, `weak(..)` otherwise (not in
support); a value between seen values is `interp`; outside them `EXTRAP`; a knob the fitted data never varied is
`NEVER-VARIED`. `code_version` is judged by the quality model's eras (`fitted(n runs, era ..)` or
`fallback->era (distance d)`) and `chip` by the fitted chips. Pairs of changed dims that no fitted run combined
are `interaction EXTRAP` (the surrogate is additive: it has no data on the pair). `/steptime-extrap` is appended
only for knobs the step-time model models, when the value is outside its fitted range.

Launch lines: each row carries `job_env`, the queue job env of the base's reference run with the candidate's
changes substituted (FF_CODE_DIR included), so `env_overrides(row, seed)` is what to put in a queue job on top
of the chip's base.env. Built in for the K59 base (reference run C:0816_R_g7lk35_s73, checked against the base
recipe's knobs); other bases state theirs in the space's `queue` block or get the bare change list plus a note
in meta["queue"].
"""
from __future__ import annotations

import itertools
import json
import math
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ffsim.schema import Recipe
from ffsim.simulate import (BEST_OFFICIAL, SAVE_RESERVE_S, Models, Noise, draw_noise, load_recipe, recipe_from_dict,
                            simulate_batch, simulate_draws, summarise_batch)

try:  # lineage spelling ('code_k12off' == 'M12'); optional so the module works with stub models alone
    from ffsim.steptime import version_key as _steptime_version_key
except Exception:  # noqa: BLE001 - the module is another owner's
    _steptime_version_key = None

SPECIAL_DIMS = ("code_version", "chip", "time_target")
GENERATORS = ("local", "oat", "random", "grid")
WEAK_MIN_RUNS = 2          # default for a quality model without config.min_value_runs: a value carried by fewer
                           # fitted runs, or by none in the base's era, is weak support
SD_BALLOON_RATIO = 3.0     # a quality sd above this multiple of the base's is noted as an extrapolation


# --- values -----------------------------------------------------------------------------------
def fmt_value(v: Any) -> str:
    """Format a candidate value the way train.py's _env_* readers expect it."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:.10g}"
    return str(v)


def same_value(a: Any, b: Any) -> bool:
    sa, sb = fmt_value(a), fmt_value(b)
    if sa == sb:
        return True
    try:
        return math.isclose(float(sa), float(sb), rel_tol=1e-9, abs_tol=1e-12)
    except ValueError:
        return False


def version_key(v: Any) -> str:
    """Lineage key of a code-version spelling: 'code_k12off' -> 'm12', 'M12' -> 'm12' (ffsim.steptime's table
    when importable, else a plain lower-case spelling without the code_ prefix)."""
    s = str(v).strip()
    if _steptime_version_key is not None:
        try:
            return str(_steptime_version_key(s))
        except Exception:  # noqa: BLE001
            pass
    s = s.lower()
    return s[len("code_"):] if s.startswith("code_") else s


def same_version(a: Any, b: Any) -> bool:
    return version_key(a) == version_key(b)


def dim_values(dim: Dict[str, Any]) -> List[Any]:
    if "values" in dim:
        return list(dim["values"])
    if "range" in dim:
        lo, hi = dim["range"]
        if dim.get("step") is not None:
            step = dim["step"]
            if step <= 0:
                raise ValueError(f"dim {dim.get('knob')}: step must be > 0")
            count = int(math.floor((hi - lo) / step + 1e-9)) + 1
            vals = [lo + i * step for i in range(count)]
        elif dim.get("n") is not None:
            vals = [float(x) for x in np.linspace(lo, hi, int(dim["n"]))]
        else:
            raise ValueError(f"dim {dim.get('knob')}: a range needs 'step' or 'n'")
        int_ends = all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, hi))
        int_step = "step" in dim and isinstance(dim["step"], int) and not isinstance(dim["step"], bool)
        if int_ends and (int_step or ("step" not in dim and all(float(v).is_integer() for v in vals))):
            return [int(round(v)) for v in vals]
        return [float(round(v, 10)) for v in vals]
    raise ValueError(f"dim {dim.get('knob')}: needs 'values' or 'range'")


# --- space --------------------------------------------------------------------------------------
@dataclass
class Dim:
    knob: str
    values: List[Any]           # values that differ from the base (base value removed)
    base_value: Any


@dataclass
class Space:
    name: str
    base: Recipe
    dims: List[Dim]
    max_changes: Optional[int] = 2
    description: str = ""
    queue: Optional[Dict[str, Any]] = None    # how the base runs on the chip queue (module docstring)

    def n_single(self) -> int:
        return sum(len(d.values) for d in self.dims)


def base_value_of(base: Recipe, knob: str) -> Any:
    if knob in SPECIAL_DIMS:
        return getattr(base, knob)
    return base.knobs.get(knob)


def load_space(path: Union[str, Path]) -> Space:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return space_from_dict(d, base_dir=p.parent, name=p.stem)


def space_from_dict(d: Dict[str, Any], base_dir: Optional[Path] = None, name: Optional[str] = None) -> Space:
    b = d.get("base") or {}
    if "recipe" in b:
        rp = Path(b["recipe"])
        if not rp.is_absolute() and base_dir is not None:
            rp = base_dir / rp
        base = load_recipe(rp)
        for k in ("code_version", "chip", "time_target", "name"):
            if k in b:
                setattr(base, k, type(getattr(base, k))(b[k]))
        base.knobs.update({str(k): str(v) for k, v in (b.get("knobs") or {}).items()})
    else:
        base = recipe_from_dict(b, name=b.get("name", "base"))
    dims: List[Dim] = []
    for raw in d.get("dims", []):
        knob = str(raw["knob"])
        bv = base_value_of(base, knob)
        vals = [v for v in dim_values(raw) if bv is None or not same_value(v, bv)]
        # keep a stable order, drop duplicates
        seen: List[Any] = []
        for v in vals:
            if not any(same_value(v, s) for s in seen):
                seen.append(v)
        dims.append(Dim(knob=knob, values=seen, base_value=bv))
    mc = d.get("max_changes", 2)
    queue = d.get("queue")
    if isinstance(queue, dict):
        queue = dict(queue)
        if not queue.get("job_env") and queue.get("runner_log") and queue.get("reference_run"):
            lp = Path(str(queue["runner_log"]))
            if not lp.is_absolute() and base_dir is not None:
                lp = base_dir / lp
            env = job_env_from_runner_log(lp, str(queue["reference_run"]))
            if env:
                queue["job_env"] = env
            else:
                queue["note"] = f"reference run {queue['reference_run']!r} has no START line in {lp}"
    else:
        queue = None
    return Space(name=str(d.get("name") or name or "space"), base=base, dims=dims,
                 max_changes=None if mc is None else int(mc), description=str(d.get("description", "")), queue=queue)


# --- candidates ---------------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    changes: Dict[str, Any] = field(default_factory=dict)   # knob -> value (special dims included)

    def key(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted((k, fmt_value(v)) for k, v in self.changes.items()))


def candidate_name(changes: Dict[str, Any]) -> str:
    if not changes:
        return "base"
    return "+".join(f"{k}={fmt_value(v)}" for k, v in sorted(changes.items()))


def make_candidate(changes: Dict[str, Any]) -> Candidate:
    return Candidate(name=candidate_name(changes), changes=dict(changes))


def to_recipe(space: Space, cand: Candidate) -> Recipe:
    b = space.base
    knobs = dict(b.knobs)
    code_version, chip, time_target = b.code_version, b.chip, b.time_target
    for k, v in cand.changes.items():
        if k == "code_version":
            code_version = str(v)
        elif k == "chip":
            chip = str(v)
        elif k == "time_target":
            time_target = float(v)
            if "FF_TIME_TARGET" in knobs:        # the knob wins in simulate.recipe_time_target: move both
                knobs["FF_TIME_TARGET"] = fmt_value(v)
        else:
            knobs[k] = fmt_value(v)
    return Recipe(name=cand.name, code_version=code_version, knobs=knobs, chip=chip, time_target=time_target)


def _dedupe(cands: Iterable[Candidate]) -> List[Candidate]:
    out: List[Candidate] = []
    seen = set()
    for c in cands:
        k = c.key()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def gen_combinations(space: Space, max_changes: Optional[int]) -> List[Candidate]:
    """Base plus every combination of up to max_changes dims (all their values)."""
    dims = [d for d in space.dims if d.values]
    depth = len(dims) if max_changes is None else min(max_changes, len(dims))
    cands = [make_candidate({})]
    for m in range(1, depth + 1):
        for combo in itertools.combinations(dims, m):
            for values in itertools.product(*(d.values for d in combo)):
                cands.append(make_candidate({d.knob: v for d, v in zip(combo, values)}))
    return _dedupe(cands)


def gen_oat(space: Space) -> List[Candidate]:
    return gen_combinations(space, 1)


def gen_local(space: Space) -> List[Candidate]:
    depth = 2 if space.max_changes is None else min(2, space.max_changes)
    return gen_combinations(space, depth)


def gen_grid(space: Space) -> List[Candidate]:
    """Full cartesian product of every dim (base value included), then the max_changes filter."""
    dims = [d for d in space.dims if d.values]
    cands = [make_candidate({})]
    axes = [[None] + list(d.values) for d in dims]
    for values in itertools.product(*axes):
        changes = {d.knob: v for d, v in zip(dims, values) if v is not None}
        if space.max_changes is not None and len(changes) > space.max_changes:
            continue
        cands.append(make_candidate(changes))
    return _dedupe(cands)


def gen_random(space: Space, n: int, rng: Union[None, int, np.random.Generator] = None) -> List[Candidate]:
    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    dims = [d for d in space.dims if d.values]
    if not dims:
        return [make_candidate({})]
    max_m = len(dims) if space.max_changes is None else max(1, min(space.max_changes, len(dims)))
    cands = [make_candidate({})]
    seen = {cands[0].key()}
    attempts = 0
    while len(cands) < n + 1 and attempts < 50 * (n + 1):
        attempts += 1
        m = int(g.integers(1, max_m + 1))
        idx = g.choice(len(dims), size=m, replace=False)
        changes = {dims[i].knob: dims[i].values[int(g.integers(len(dims[i].values)))] for i in idx}
        c = make_candidate(changes)
        if c.key() not in seen:
            seen.add(c.key())
            cands.append(c)
    return cands


def generate(space: Space, gen: str = "local", n_random: int = 200,
             rng: Union[None, int, np.random.Generator] = None) -> List[Candidate]:
    if gen == "local":
        return gen_local(space)
    if gen == "oat":
        return gen_oat(space)
    if gen == "grid":
        return gen_grid(space)
    if gen == "random":
        return gen_random(space, n_random, rng)
    raise ValueError(f"unknown generator {gen!r}: choose from {GENERATORS}")


# --- train.py startup rules ---------------------------------------------------------------------
# The asserts train.py runs at import (the K59 upload train.py, M12 code, not shipped; line numbers below):
# a candidate that trips one is refused on the chip before step 0, whatever the surrogate predicts. Values are
# read exactly like train.py's _env_int / _env_float / _env_flag ('' = the default, unknown here -> rule skipped;
# a flag is on unless it is '0', 'false' or 'False'). Not importable from train.py itself (torch), so kept here.
TRAIN_PY_SOURCE = "K59 upload train.py (M12 code; not shipped, see recipes/K60/train.py)"
_STARTUP_STEPS_EXCLUDED = 5      # TRAIN_STARTUP_STEPS_EXCLUDED: steps 0-4 are off the charged clock (CONTRACT.md)
_SEQ_LEN = 1024


class _Unknown(Exception):
    """A rule needs a knob the effective env does not state."""


class _Env:
    """train.py's _env_* readers over a candidate's effective knobs."""

    def __init__(self, knobs: Dict[str, Any]):
        self.knobs = knobs

    def raw(self, name: str) -> str:
        v = self.knobs.get(name)
        if v is None:
            raise _Unknown(name)
        return str(v)

    def _set(self, name: str) -> str:
        s = self.raw(name)
        if s == "":                       # train.py: '' -> the default, which this code dir's train.py defines
            raise _Unknown(name)
        return s

    def i(self, name: str) -> int:
        s = self._set(name).strip()
        try:
            return int(s)
        except ValueError:
            f = float(s)                  # ValueError propagates as a violation
            if f.is_integer():
                return int(f)
            raise ValueError(f"{name}={s!r} is not an int")

    def f(self, name: str) -> float:
        return float(self._set(name).strip())

    def b(self, name: str) -> bool:
        return self._set(name) not in ("0", "false", "False")

    def s(self, name: str) -> str:
        return self.raw(name)


def _xsa_ok(raw: str) -> bool:
    raw = raw.replace(" ", "").lower()
    return raw in ("", "all") or all(v.isdigit() for v in raw.split(","))


# (train.py line, message, violated(env))
TRAIN_PY_RULES: Tuple[Tuple[int, str, Callable[[_Env], bool]], ...] = (
    (37, "FF_XU_QUEUE must be 0 or 1..63", lambda e: not (0 <= e.i("FF_XU_QUEUE") <= 63)),
    (111, f"FF_EMA_PREWARM needs FF_EMA=1 and FF_EMA_EVERY > {_STARTUP_STEPS_EXCLUDED}",
     lambda e: e.b("FF_EMA_PREWARM") and not (e.b("FF_EMA") and max(1, e.i("FF_EMA_EVERY")) > _STARTUP_STEPS_EXCLUDED)),
    (119, "FF_EMA_BLEND must be in [0, 1]", lambda e: not (0.0 <= e.f("FF_EMA_BLEND") <= 1.0)),
    (120, "FF_EMA_BLEND needs FF_EMA=1", lambda e: bool(e.f("FF_EMA_BLEND")) and not e.b("FF_EMA")),
    (121, "FF_EMA_BLEND_SKIP_EMB=1 needs FF_EMA_BLEND > 0", lambda e: e.b("FF_EMA_BLEND_SKIP_EMB") and not e.f("FF_EMA_BLEND")),
    (136, "FF_SCALAR_BETA2 must be in [0.9, 1)", lambda e: not (0.9 <= e.f("FF_SCALAR_BETA2") < 1.0)),
    (138, "FF_MLP_PROJ_LR must be in [0.25, 8]", lambda e: not (0.25 <= e.f("FF_MLP_PROJ_LR") <= 8.0)),
    (547, "FF_XSA_LAYERS must be '', 'all' or comma-separated block indices", lambda e: not _xsa_ok(e.s("FF_XSA_LAYERS"))),
    (550, "FF_XSA_LAYERS is not wired for FF_NKI_ATTN", lambda e: bool(e.s("FF_XSA_LAYERS").strip()) and e.b("FF_NKI_ATTN")),
    (553, "FF_PARALLEL_BLOCK cannot be combined with FF_ATTN_LAYERS",
     lambda e: e.b("FF_PARALLEL_BLOCK") and bool(e.s("FF_ATTN_LAYERS").strip(" ,"))),
    (556, "FF_SMEAR cannot be combined with FF_DOC_MASK", lambda e: e.b("FF_SMEAR") and e.b("FF_DOC_MASK")),
    (561, "FF_LEAKY_FORM must be abs, max, fn or fnm", lambda e: e.s("FF_LEAKY_FORM") not in ("abs", "max", "fn", "fnm")),
    (562, "FF_LEAKY_FORM fn/fnm needs FF_LEAKY_RELU2 > 0", lambda e: e.s("FF_LEAKY_FORM") in ("fn", "fnm") and not e.f("FF_LEAKY_RELU2") > 0.0),
    (565, "FF_RELU2_FN changes relu(h)**2, which FF_LEAKY_RELU2 replaces: set one or neither",
     lambda e: e.b("FF_RELU2_FN") and bool(e.f("FF_LEAKY_RELU2"))),
    (566, "FF_LEAKY_RELU2 must be in [0, 1]", lambda e: not (0.0 <= e.f("FF_LEAKY_RELU2") <= 1.0)),
    (567, "FF_LEAKY_INPLACE=0 needs FF_LEAKY_RELU2 > 0", lambda e: not e.b("FF_LEAKY_INPLACE") and not e.f("FF_LEAKY_RELU2")),
    (569, "FF_COOLDOWN_SHAPE must be linear or sqrt", lambda e: e.s("FF_COOLDOWN_SHAPE") not in ("linear", "sqrt")),
    (572, "FF_MUON_VIEWS is eager-only: not with FF_COMPILE_OPT", lambda e: e.b("FF_MUON_VIEWS") and e.b("FF_COMPILE_OPT")),
    (583, "FF_MUON_BETA2 must be in [0, 1)", lambda e: not (0.0 <= e.f("FF_MUON_BETA2") < 1.0)),
    (584, "FF_NS_STEPS must be in 1..5", lambda e: not (1 <= e.i("FF_NS_STEPS") <= 5)),
    (593, "FF_ZLOSS must be >= 0", lambda e: e.f("FF_ZLOSS") < 0.0),
    (594, "FF_CE_BF16 needs FF_FUSED_CE > 0", lambda e: e.b("FF_CE_BF16") and not e.i("FF_FUSED_CE")),
    (595, "FF_CE_BF16 needs FF_ZLOSS=0", lambda e: e.b("FF_CE_BF16") and bool(e.f("FF_ZLOSS"))),
    (596, "FF_SOFTCAP must be >= 0", lambda e: e.f("FF_SOFTCAP") < 0.0),
    (599, "FF_KEY_OFFSET_RAD must be in (0, 1000)", lambda e: not (0.0 < e.f("FF_KEY_OFFSET_RAD") < 1000.0)),
    (600, "FF_KEY_OFFSET_RAD needs FF_KEY_OFFSET=1", lambda e: e.f("FF_KEY_OFFSET_RAD") != 0.1 and not e.b("FF_KEY_OFFSET")),
    (605, "FF_SOFTCAP_A must be >= 0", lambda e: e.f("FF_SOFTCAP_A") < 0.0),
    (606, "FF_SOFTCAP_B must be in [-100, 100]", lambda e: abs(e.f("FF_SOFTCAP_B")) > 100.0),
    (607, "FF_SOFTCAP_A / FF_SOFTCAP_B need FF_SOFTCAP > 0",
     lambda e: bool(e.f("FF_SOFTCAP_A") or e.f("FF_SOFTCAP_B")) and not e.f("FF_SOFTCAP") > 0.0),
    (609, "FF_ATTN_PROJ_STD must be in [0, 0.5]", lambda e: not (0.0 <= e.f("FF_ATTN_PROJ_STD") <= 0.5)),
    (610, "FF_KEY_OFFSET cannot be combined with FF_DOC_MASK", lambda e: e.b("FF_KEY_OFFSET") and e.b("FF_DOC_MASK")),
    (612, "FF_ZLOSS is not implemented in the FF_LOSS_CHUNK head", lambda e: bool(e.f("FF_ZLOSS")) and bool(e.i("FF_LOSS_CHUNK"))),
    (615, "FF_ADAMW_LR_POW must be in 0.1..5.0", lambda e: not (0.1 <= e.f("FF_ADAMW_LR_POW") <= 5.0)),
    (619, "FF_MOM_CD_LEVELS must be in 0..19", lambda e: not (0 <= e.i("FF_MOM_CD_LEVELS") <= 19)),
    (620, "FF_MOM_CD_FLOOR must be in 0.5..0.99", lambda e: not (0.5 <= e.f("FF_MOM_CD_FLOOR") <= 0.99)),
    (621, "FF_ASYNC_LOOP must be in 0..64", lambda e: not (0 <= e.i("FF_ASYNC_LOOP") <= 64)),
    (623, "FF_SCALAR_CONSTS must be 0, 1, 2 or 3", lambda e: e.i("FF_SCALAR_CONSTS") not in (0, 1, 2, 3)),
    (624, "FF_SCALAR_CONSTS is not for FF_COMPILE_OPT's traced step", lambda e: bool(e.i("FF_SCALAR_CONSTS")) and e.b("FF_COMPILE_OPT")),
    (627, "FF_ASYNC_OPT must be 0 or 1", lambda e: e.i("FF_ASYNC_OPT") not in (0, 1)),
    (629, "FF_ASYNC_OPT=1 needs FF_SCALAR_CONSTS=2 or 3", lambda e: e.i("FF_ASYNC_OPT") == 1 and e.i("FF_SCALAR_CONSTS") not in (2, 3)),
    (632, "FF_ASYNC_OPT=1 needs FF_TENSOR_LR=1", lambda e: e.i("FF_ASYNC_OPT") == 1 and not e.b("FF_TENSOR_LR")),
    (633, "FF_ASYNC_OPT=1 is not with FF_FOREACH_ADAMW, FF_NS_FUSED or FF_EMA_FOREACH",
     lambda e: e.i("FF_ASYNC_OPT") == 1 and (e.b("FF_FOREACH_ADAMW") or e.b("FF_NS_FUSED") or e.b("FF_EMA_FOREACH"))),
    (638, "FF_OPT_FUSE must be 0..3", lambda e: e.i("FF_OPT_FUSE") not in (0, 1, 2, 3)),
    (639, "FF_OPT_FUSE needs FF_ASYNC_OPT=1", lambda e: bool(e.i("FF_OPT_FUSE")) and not e.i("FF_ASYNC_OPT")),
    (643, "FF_MUON_SCHED=1 needs FF_MUON_SHARD=1", lambda e: e.b("FF_MUON_SCHED") and not e.b("FF_MUON_SHARD")),
    (644, "FF_MUON_SCHED=1 is eager-only: not with FF_COMPILE_OPT", lambda e: e.b("FF_MUON_SCHED") and e.b("FF_COMPILE_OPT")),
    (645, "FF_MUON_SCHED_OP_US must be in 0..100000", lambda e: not (0 <= e.i("FF_MUON_SCHED_OP_US") <= 100000)),
    (646, "FF_MUON_SCHED_TFLOPS must be in 1..10000", lambda e: not (1 <= e.i("FF_MUON_SCHED_TFLOPS") <= 10000)),
    (845, "FF_STAGE_LOADER must be 0..6", lambda e: e.i("FF_STAGE_LOADER") not in (0, 1, 2, 3, 4, 5, 6)),
    (849, "FF_STAGE_LOADER needs FF_STREAM_ROWS=1", lambda e: bool(e.i("FF_STAGE_LOADER")) and not e.b("FF_STREAM_ROWS")),
    (850, "FF_STAGE_LOADER 4/5/6 is not with FF_ASYNC_LOOP", lambda e: e.i("FF_STAGE_LOADER") in (4, 5, 6) and bool(e.i("FF_ASYNC_LOOP"))),
    (854, "FF_COOLDOWN_FLOOR must be in 0..20", lambda e: not (0 <= e.i("FF_COOLDOWN_FLOOR") <= 20)),
    (855, "FF_EMB_WD must be >= 0", lambda e: e.f("FF_EMB_WD") < 0.0),
    (859, "FF_POS_WEIGHT must be in 0..4", lambda e: not (0.0 <= e.f("FF_POS_WEIGHT") <= 4.0)),
    (860, f"FF_POS_WEIGHT_N must be in 1..{_SEQ_LEN - 1}", lambda e: not (1 <= e.i("FF_POS_WEIGHT_N") < _SEQ_LEN)),
    (861, "FF_POS_WEIGHT is not implemented for the FF_LOSS_CHUNK head", lambda e: e.f("FF_POS_WEIGHT") != 1.0 and bool(e.i("FF_LOSS_CHUNK"))),
    (1107, "FF_NULL_ATTN must be 0, 1 or 2", lambda e: e.i("FF_NULL_ATTN") not in (0, 1, 2)),
    (1108, "FF_NULL_MLP must be 0, 1 or 2", lambda e: e.i("FF_NULL_MLP") not in (0, 1, 2)),
    (1109, "FF_NULL_MLP skips the MLP activation that FF_LEAKY_RELU2 changes: set one or neither",
     lambda e: bool(e.i("FF_NULL_MLP")) and bool(e.f("FF_LEAKY_RELU2"))),
    (1111, "FF_NULL_MLP skips the MLP activation that FF_RELU2_FN changes: set one or neither",
     lambda e: bool(e.i("FF_NULL_MLP")) and e.b("FF_RELU2_FN")),
    (1112, "FF_MUON_ZERO2=1 is not with FF_NULL_MLP=2", lambda e: e.b("FF_MUON_ZERO2") and e.i("FF_NULL_MLP") == 2),
    (1115, f"FF_TRAIN_SEQ must be in 1..{_SEQ_LEN}", lambda e: not (0 < e.i("FF_TRAIN_SEQ") <= _SEQ_LEN)),
    (1116, "FF_NO_CAUSAL applies to the SDPA path only (drop FF_NKI_ATTN / FF_MANUAL_ATTN)",
     lambda e: e.b("FF_NO_CAUSAL") and (e.b("FF_NKI_ATTN") or e.b("FF_MANUAL_ATTN"))),
    (1118, "FF_KV_SDPA applies to the SDPA path only (drop FF_NKI_ATTN / FF_MANUAL_ATTN / FF_NULL_ATTN)",
     lambda e: e.b("FF_KV_SDPA") and (e.b("FF_NKI_ATTN") or e.b("FF_MANUAL_ATTN") or bool(e.i("FF_NULL_ATTN")))),
)


def validate_knobs(knobs: Dict[str, Any]) -> List[str]:
    """The train.py startup asserts this effective env trips (empty = train.py would start). Rules whose knobs
    the env does not state are skipped; a value train.py could not parse is reported as a violation."""
    env = _Env(knobs)
    out: List[str] = []
    for line, msg, violated in TRAIN_PY_RULES:
        try:
            bad = bool(violated(env))
        except _Unknown:
            continue
        except (ValueError, TypeError) as e:
            out.append(f"{msg} (train.py:{line}; {e})")
            continue
        if bad:
            out.append(f"{msg} (train.py:{line})")
    return out


def validate_candidate(space: Space, cand: Candidate) -> List[str]:
    """Violations of the candidate's effective env (base knobs + changes)."""
    return validate_knobs(to_recipe(space, cand).knobs)


# --- support -----------------------------------------------------------------------------------
def _support_of(model: Any) -> Dict[str, Any]:
    s = getattr(model, "support", None)
    if callable(s):
        try:
            s = s()
        except TypeError:
            s = None
    return dict(s) if isinstance(s, dict) else {}


def support_status(support: Dict[str, Any], knob: str, value: Any) -> str:
    """'seen' (exact value in the fitted data), 'interp' (numeric, inside the seen range),
    'unseen' (outside the range or a category never seen) or 'never' (the knob never varied)."""
    if knob not in support:
        return "never"
    seen = support[knob]
    if isinstance(seen, dict) and ("min" in seen or "max" in seen):
        try:
            fv = float(fmt_value(value))
        except ValueError:
            return "unseen"
        lo = float(seen.get("min", -math.inf))
        hi = float(seen.get("max", math.inf))
        vals = seen.get("values") or []
        if any(same_value(fv, s) for s in vals):
            return "seen"
        return "interp" if lo <= fv <= hi else "unseen"
    try:
        items = list(seen)
    except TypeError:
        items = [seen]
    if any(same_value(value, s) for s in items):
        return "seen"
    try:
        fv = float(fmt_value(value))
        nums = [float(fmt_value(s)) for s in items]
        nums = [x for x in nums if not math.isnan(x)]
    except ValueError:
        return "unseen"
    if nums and min(nums) <= fv <= max(nums):
        return "interp"
    return "unseen"


def quality_support(models: Models) -> Dict[str, Any]:
    """The quality model's own `support` when it exposes one, else the support `ffsim fit` derived
    from the fitted records (Models.meta['support']: knob -> values seen in full runs)."""
    qs = _support_of(models.quality)
    if not qs:
        meta = getattr(models, "meta", None) or {}
        qs = dict(meta.get("support") or {})
    return qs


class FittedRows:
    """The runs the quality model was fitted on: effective knobs, lineage version, era and chip per run
    (QualityModel._rows, else fit_records; empty for a model that exposes neither). Answers how many fitted
    runs carry a knob value and whether two values were ever combined in one run."""

    def __init__(self, quality: Any):
        self.rows: List[Tuple[Dict[str, str], str, str, str]] = []
        rows = getattr(quality, "_rows", None)
        aliases = dict(getattr(quality, "version_aliases", None) or {})
        v2e = dict(getattr(quality, "version_to_era", None) or {})
        if rows:
            for r in rows:
                knobs = getattr(r, "effective", None) or getattr(r, "knobs", None) or {}
                self.rows.append((dict(knobs), str(getattr(r, "version", "") or ""), str(getattr(r, "era", "") or ""),
                                  str(getattr(r, "chip", "") or "")))
        else:
            for rec in getattr(quality, "fit_records", None) or []:
                cv = str(getattr(rec, "code_version", "") or "")
                cv = aliases.get(cv, cv)
                self.rows.append((dict(getattr(rec, "knobs", None) or {}), cv, str(v2e.get(cv, "")),
                                  str(getattr(rec, "chip", "") or "")))
        self._idx: Dict[Tuple[str, str], frozenset] = {}

    @property
    def empty(self) -> bool:
        return not self.rows

    @staticmethod
    def _value(row: Tuple[Dict[str, str], str, str, str], k: str) -> Optional[str]:
        knobs, version, _era, chip = row
        if k == "code_version":
            return version or None
        if k == "chip":
            return chip or None
        if k == "time_target":
            return knobs.get("FF_TIME_TARGET")
        return knobs.get(k)

    def ids(self, k: str, v: Any) -> frozenset:
        key = (k, fmt_value(v))
        if key not in self._idx:
            match = same_version if k == "code_version" else same_value
            self._idx[key] = frozenset(i for i, row in enumerate(self.rows)
                                       if self._value(row, k) is not None and match(self._value(row, k), v))
        return self._idx[key]

    def count(self, k: str, v: Any, era: Optional[str] = None) -> Tuple[int, int]:
        """(fitted runs carrying k=v, those among them in `era`)."""
        ids = self.ids(k, v)
        n_era = sum(1 for i in ids if era is not None and self.rows[i][2] == era) if era is not None else 0
        return len(ids), n_era

    def co_observed(self, k1: str, v1: Any, k2: str, v2: Any) -> bool:
        return bool(self.ids(k1, v1) & self.ids(k2, v2))


def base_era_of(models: Models, space: Space) -> Optional[str]:
    """The quality-model era the base was fitted in (None when the base itself is a fallback or unknown)."""
    resolve = getattr(models.quality, "resolve_era", None)
    if not callable(resolve):
        return None
    try:
        era, dist, _cross = resolve(space.base.code_version)
    except Exception:  # noqa: BLE001
        return None
    return str(era) if era is not None and dist == 0 else None


def special_support(models: Models, fitted: FittedRows, k: str, v: Any,
                    qs: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """(status, tag) for a special dim. code_version: the quality model's era lineage plus the fitted run count
    (`fitted(n runs, era ..)`, `fallback->era (distance d)`), with the step-time lineage fallback appended;
    chip: the fitted chips; time_target: FF_TIME_TARGET's support (steps-only when the table lacks it)."""
    q = models.quality
    if k == "code_version":
        n, _ = fitted.count("code_version", v)
        resolve = getattr(q, "resolve_era", None)
        if callable(resolve):
            try:
                era, dist, cross = resolve(str(v))
            except Exception:  # noqa: BLE001
                era, dist, cross = None, 0, False
            if era is None:
                status, tag = "never", "UNKNOWN(no fitted eras)"
            elif int(dist) == 0:
                status, tag = "seen", f"fitted({n} runs, era {era})"
            else:
                status = "unseen"
                tag = f"fallback->{era}" + (" (different lineage)" if cross else f" (distance {dist})")
        elif n > 0:
            status, tag = "seen", f"fitted({n} runs)"
        else:
            status, tag = "never", "UNKNOWN(model has no eras)"
        known_fn = getattr(models.steptime, "known_versions", None)
        if callable(known_fn):
            try:
                known = [str(x) for x in known_fn()]
            except Exception:  # noqa: BLE001
                known = []
            if known and not any(same_version(x, v) for x in known):
                near_fn = getattr(models.steptime, "nearest_version", None)
                near = None
                if callable(near_fn):
                    try:
                        near = near_fn(str(v))
                    except Exception:  # noqa: BLE001
                        near = None
                tag += f"/steptime-fallback->{near}" if near else "/steptime-fallback"
        return status, tag
    if k == "chip":
        n, _ = fitted.count("chip", v)
        chips = getattr(q, "chips", None)
        ref = getattr(getattr(q, "config", None), "reference_chip", None)
        if isinstance(chips, dict):
            fitted_chips = set(str(c) for c in chips) | ({str(ref)} if ref else set())
            if str(v) in fitted_chips:
                return "seen", f"fitted({n} runs)"
            return "unseen", f"unseen chip (treated as {ref or 'C'} + prior sd)"
        if n > 0:
            return "seen", f"fitted({n} runs)"
        return "never", "UNKNOWN(model has no chip table)"
    # time_target: the quality model only sees it through the step count (and FF_TIME_TARGET when fitted)
    st = support_status(qs if qs is not None else quality_support(models), "FF_TIME_TARGET", v)
    if st == "never":
        return "interp", "steps-only"
    n, _ = fitted.count("time_target", v)
    return st, {"seen": f"ok(n={n})", "interp": "interp", "unseen": "EXTRAP"}[st]


LEVEL_ORDER = ("ok", "interp", "weak", "interaction", "extrap", "never")


def _worse(a: str, b: str) -> str:
    return a if LEVEL_ORDER.index(a) >= LEVEL_ORDER.index(b) else b


def min_value_runs(quality: Any) -> int:
    """The quality model's own 'seen' threshold (QualityModel.config.min_value_runs), else WEAK_MIN_RUNS."""
    v = getattr(getattr(quality, "config", None), "min_value_runs", None)
    try:
        return max(1, int(v)) if v is not None else WEAK_MIN_RUNS
    except (TypeError, ValueError):
        return WEAK_MIN_RUNS


def support_check(models: Models, cand: Candidate, space: Optional[Space] = None,
                  fitted: Optional[FittedRows] = None, base_era: Optional[str] = None,
                  min_runs: Optional[int] = None, qs: Optional[Dict[str, Any]] = None,
                  ss: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, str]:
    """(in_support, note, level). in_support is True when every changed dim is seen (by at least `min_runs`
    fitted runs, one of them in the base's era) or interpolated by the quality model and every pair of changed
    dims was combined in at least one fitted run. level is the worst tag: ok < interp < weak < interaction <
    extrap < never. `qs` / `ss` are the quality / step-time support tables (each a pass over the fitted records
    for the fitted models, so a search computes them once and passes them in)."""
    if not cand.changes:
        return True, "base", "base"
    if qs is None:
        qs = quality_support(models)
    if ss is None:
        ss = _support_of(models.steptime)
    if fitted is None:
        fitted = FittedRows(models.quality)
    if base_era is None and space is not None:
        base_era = base_era_of(models, space)
    if min_runs is None:
        min_runs = min_value_runs(models.quality)
    parts: List[str] = []
    status: Dict[str, str] = {}
    level = "ok"
    for k, v in sorted(cand.changes.items()):
        if k in SPECIAL_DIMS:
            st, tag = special_support(models, fitted, k, v, qs)
        else:
            st = support_status(qs, k, v)
            tag = {"seen": "ok", "interp": "interp", "unseen": "EXTRAP", "never": "NEVER-VARIED"}[st]
            if st == "seen" and not fitted.empty:
                n, n_era = fitted.count(k, v, base_era)
                if n < min_runs or (base_era is not None and n_era == 0):
                    st = "weak"
                    tag = f"weak(n={n}" + (f", {n_era} in era {base_era})" if base_era is not None else ")")
                else:
                    tag = f"ok(n={n})"
            if ss and k in ss and support_status(ss, k, v) == "unseen":   # only knobs the step-time model models
                tag += "/steptime-extrap"
        status[k] = st
        level = _worse(level, {"seen": "ok", "interp": "interp", "weak": "weak", "unseen": "extrap", "never": "never"}[st])
        parts.append(f"{k}:{tag}")
    if not fitted.empty:
        keys = [k for k in sorted(cand.changes) if status[k] in ("seen", "weak")]
        for k1, k2 in itertools.combinations(keys, 2):
            if not fitted.co_observed(k1, cand.changes[k1], k2, cand.changes[k2]):
                parts.append(f"{k1}+{k2}:never co-observed (interaction EXTRAP)")
                level = _worse(level, "interaction")
    return level in ("ok", "interp"), " ".join(parts), level


def quality_notes(quality: Any, recipe: Recipe, steps: float, seed: Optional[int]) -> List[str]:
    """The quality model's own notes for this prediction (era / chip / seed fallbacks), when it exposes
    design_row(knobs, steps, code_version, chip, seed) -> (x, extra_variance, notes)."""
    dr = getattr(quality, "design_row", None)
    if not callable(dr) or not (steps and math.isfinite(steps)):
        return []
    try:
        out = dr(recipe.knobs, float(steps), recipe.code_version, recipe.chip, seed)
        notes = out[2] if isinstance(out, tuple) and len(out) >= 3 else []
    except Exception:  # noqa: BLE001 - the model is another owner's code
        return []
    return [f"quality: {n}" for n in notes if n]


# --- chip queue launch lines ---------------------------------------------------------------------
QUEUE_ROOT = "/root/ff-claude"
# code_version -> code dir on the chips (research/sim-data/chipC/_meta/code_dirs.txt). code_k1Xoff = the K57
# recipe baked on lineage M1X with the new flags off (the K59 arm ran on code_k12off); the others are the
# lineage dirs by name, not verified for the K59 recipe. A space's queue.code_dirs overrides.
CODE_DIRS: Dict[str, str] = {
    "M12": f"{QUEUE_ROOT}/code_k12off", "M13": f"{QUEUE_ROOT}/code_k13off",
    "M14": f"{QUEUE_ROOT}/code_k14off", "M15": f"{QUEUE_ROOT}/code_k15off",
    "M9": f"{QUEUE_ROOT}/code_m9", "M10": f"{QUEUE_ROOT}/code_m10", "M10b": f"{QUEUE_ROOT}/code_m10b",
    "M11": f"{QUEUE_ROOT}/code_m11", "M11c": f"{QUEUE_ROOT}/code_m11c",
    "M6": f"{QUEUE_ROOT}/code_m6", "M7": f"{QUEUE_ROOT}/code_m7", "M7a": f"{QUEUE_ROOT}/code_m7a",
}
# The K59 arm as the chip C queue ran it: runner.log line 224, `START 0816_R_g7lk35_s73 08:16:50 :: ...`
# (research/sim-data/chipC/_meta/queue/runner.log), FF_SEED dropped and the superseded duplicates
# (FF_ACCUM_SCHED=2:0.2,4 FF_ACCUM_LR=0.7071, overridden later on the same line) removed. It runs on top of
# /root/ff-claude/queue/base.env (harvested copy research/sim-data/chipC/_meta/queue/base.env: FF_TIME_TARGET=1760,
# FF_SEED=51, FF_COOLDOWN_FRAC=0.45, FF_EMA_EVERY=4, ...). The harvested record C:0816_R_g7lk35_s73 confirms the
# effective env (FF_ACCUM_SCHED=1:0.06,2:0.2,4, FF_ACCUM_LR=0.5,0.7071, FF_EMA_EVERY=32, FF_TIME_TARGET=1760).
K59_QUEUE: Dict[str, Any] = {
    "chip": "C",
    "reference_run": "C:0816_R_g7lk35_s73",
    "base_env": f"{QUEUE_ROOT}/queue/base.env (FF_TIME_TARGET=1760 FF_SEED=51 FF_COOLDOWN_FRAC=0.45 FF_EMA_EVERY=4 ...; "
                "harvested copy research/sim-data/chipC/_meta/queue/base.env)",
    "job_env": "FF_COOLDOWN_FRAC=0.60 FF_ASYNC_OPT=1 FF_SCALAR_CONSTS=2 FF_MUON_VIEWS=1 FF_OPT_FUSE=2 FF_MUON_SCHED=1 "
               "FF_MTP=1 FF_MASK_BOS_TARGET=1 FF_XU_QUEUE=32 FF_KEY_OFFSET=1 FF_MTP_PHASES=0.2,0.45 "
               f"FF_CODE_DIR={QUEUE_ROOT}/code_k12off FF_MUON_ZERO=1 FF_MUON_ZERO2=1 FF_ADAMW_ZERO=1 FF_EMA_EVERY=32 "
               "FF_ACCUM_SCHED=1:0.06,2:0.2,4 FF_ACCUM_LR=0.5,0.7071 FF_KEY_OFFSET_RAD=0.3 FF_ACC_IN_GRAPH=1 "
               "FF_LEAKY_FORM=fnm FF_LEAKY_RELU2=0.35",
    "source": "research/sim-data/chipC/_meta/queue/runner.log START 0816_R_g7lk35_s73 (built into ffsim.search)",
}


def parse_env_line(line: str) -> Dict[str, str]:
    """'K=V K=V ...' (shell quoting honoured) -> ordered dict; a repeated key keeps the LAST value, as the
    queue runner's job file does."""
    out: Dict[str, str] = {}
    for tok in shlex.split(line.strip()):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k:
            out[k] = v
    return out


def _quote(v: Any) -> str:
    s = fmt_value(v)
    return shlex.quote(s) if s else "''"


def format_env(env: Dict[str, Any]) -> str:
    return " ".join(f"{k}={_quote(v)}" for k, v in env.items())


def job_env_from_runner_log(path: Union[str, Path], run: str) -> Optional[Dict[str, str]]:
    """The job env of the queue runner's `START <run> HH:MM:SS :: K=V ...` line (FF_SEED kept), None when absent."""
    run = run.split(":", 1)[1] if ":" in run and not run.startswith("/") else run
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(f"START {run} ") and " :: " in line:
                return parse_env_line(line.split(" :: ", 1)[1])
    return None


def code_dir_for(version: str, code_dirs: Optional[Dict[str, str]] = None) -> str:
    table = {**CODE_DIRS, **(code_dirs or {})}
    if version in table:
        return table[version]
    for k, d in table.items():
        if same_version(k, version):
            return d
    v = str(version).strip()
    return f"{QUEUE_ROOT}/{v}" if v.lower().startswith("code_") else f"{QUEUE_ROOT}/code_{v.lower()}"


def _is_k59_base(base: Recipe) -> bool:
    return base.name.strip().upper().startswith("K59")


def resolve_queue(space: Space) -> Dict[str, Any]:
    """How the base runs on the chip queue: {"env": job env dict or None, "reference_run", "base_env", "chip",
    "code_dirs", "source", "note"}. The space's `queue` block wins; a base named K59 gets the built-in K59 arm.
    The job env is used only when it agrees with the base recipe on every knob both state (else env is None
    and the note says why)."""
    q = {k: v for k, v in (space.queue or {}).items() if v not in (None, "")}
    source = "space.queue"
    if not q.get("job_env") and _is_k59_base(space.base):
        q = {**K59_QUEUE, **q}
        source = "built-in K59 queue arm"
    env_raw = q.get("job_env")
    env = parse_env_line(env_raw) if isinstance(env_raw, str) else dict(env_raw or {})
    env = {str(k): str(v) for k, v in env.items()}
    env.pop("FF_SEED", None)
    out: Dict[str, Any] = {
        "source": source if env else None, "env": env or None,
        "reference_run": q.get("reference_run"), "base_env": q.get("base_env"),
        "chip": str(q.get("chip") or space.base.chip),
        "code_dirs": {**CODE_DIRS, **{str(k): str(v) for k, v in (q.get("code_dirs") or {}).items()}},
        "note": str(q.get("note") or ""),
    }
    if not env:
        out["note"] = (out["note"] + " " if out["note"] else "") + (
            f"no queue job env known for base {space.base.name!r}: launch lines list the changes only "
            "(add a 'queue' block with job_env or runner_log + reference_run to the space)")
        return out
    mism = [f"{k}: base {space.base.knobs[k]!r} vs queue {v!r}" for k, v in env.items()
            if k in space.base.knobs and not same_value(space.base.knobs[k], v)]
    cd = env.get("FF_CODE_DIR")
    if cd and not same_version(space.base.code_version, os.path.basename(cd.rstrip("/"))):
        mism.append(f"FF_CODE_DIR {cd} vs code_version {space.base.code_version!r}")
    if str(space.base.chip) != out["chip"]:
        mism.append(f"chip: base {space.base.chip!r} vs queue {out['chip']!r}")
    if mism:
        out["env"] = None
        out["note"] = (f"base {space.base.name!r} disagrees with the {source} job env ({'; '.join(mism)}): "
                       "launch lines list the changes only")
    return out


def job_env_for(queue: Dict[str, Any], cand: Candidate) -> Optional[str]:
    """The candidate's queue job env: the base's job env with the changes substituted (a code_version change
    swaps FF_CODE_DIR; a time_target change sets FF_TIME_TARGET). None without a job env or for another chip."""
    env = queue.get("env")
    if not env:
        return None
    out = dict(env)
    for k, v in sorted(cand.changes.items()):
        if k == "chip":
            if str(v) != str(queue.get("chip")):
                return None                    # that chip's queue base.env is not known here
        elif k == "code_version":
            out["FF_CODE_DIR"] = code_dir_for(str(v), queue.get("code_dirs"))
        elif k == "time_target":
            out["FF_TIME_TARGET"] = fmt_value(v)
        else:
            out[k] = fmt_value(v)
    return format_env(out)


def queue_meta(queue: Dict[str, Any]) -> Dict[str, Any]:
    env = queue.get("env")
    meta = {"source": queue.get("source"), "reference_run": queue.get("reference_run"),
            "base_env": queue.get("base_env"), "chip": queue.get("chip"),
            "code_dir": (env or {}).get("FF_CODE_DIR"), "job_env": format_env(env) if env else None,
            "note": queue.get("note") or ""}
    if env:
        meta["note"] = (
            f"launch lines = FF_SEED + the queue job env of {queue.get('reference_run')} with the candidate's changes "
            f"substituted; they run on chip {queue.get('chip')} on top of {queue.get('base_env')} with "
            f"FF_CODE_DIR={meta['code_dir']}; a code_version change swaps FF_CODE_DIR (versions outside the code_dirs "
            f"table get {QUEUE_ROOT}/code_<version>, unverified); a chip change gets no line (that chip's base.env is "
            "not known here); FF_TIME_TARGET stays base.env's unless the job env or a time_target change sets it"
            + (f". {queue['note']}" if queue.get("note") else ""))
    return meta


def env_overrides(row: Dict[str, Any], seed: Optional[int] = None) -> str:
    """The queue job env to run this row on `seed`: FF_SEED + the row's `job_env` (the base's queue job env with
    the changes substituted, FF_CODE_DIR included). Rows without a job env (no queue known for the base) get
    FF_SEED + the bare changes, which is NOT launchable as printed on its own."""
    parts = [f"FF_SEED={seed}"] if seed is not None else []
    job = row.get("job_env")
    if job:
        return " ".join(parts + [str(job)])
    for k, v in sorted(row.get("changes", {}).items()):
        if k in SPECIAL_DIMS:
            continue
        parts.append(f"{k}={_quote(v)}")
    return " ".join(parts)


# --- evaluation ---------------------------------------------------------------------------------
ROW_COLUMNS = ("rank", "name", "changes", "n_changes", "code_version", "chip", "steps_mean", "steps_sd",
               "step_k1", "step_k2", "step_k4", "bpb_2m_mean", "bpb_2m_sd", "official_mean", "official_sd",
               "official_p10", "official_p50", "official_p90", "p_beat_best", "delta_vs_base", "delta_sd_vs_base",
               "p_beat_base", "official_mc_mean", "in_support", "support_level", "support_note", "notes", "job_env")

_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(models: Models) -> None:
    _WORKER_STATE["models"] = models


def _spawn_safe() -> bool:
    """True when multiprocessing's spawn start method can re-import the main module
    (`python -m ffsim`, a script file, pytest); False for stdin / interactive sessions."""
    import sys
    main = sys.modules.get("__main__")
    if main is None:
        return False
    if getattr(main, "__spec__", None) is not None:
        return True
    f = getattr(main, "__file__", None)
    return bool(f) and os.path.exists(f)


BATCH = 256   # candidates per (candidates x draws) matrix: 256 x 1000 doubles = 2 MB per array


def _standardise_columns(a: np.ndarray) -> np.ndarray:
    """Centre every column to mean 0 and scale it to sd 1 (a constant column is only centred)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
        return _standardise_columns(a)[:, 0]
    if a.shape[0] < 2:
        return a
    c = a - a.mean(0, keepdims=True)
    s = c.std(0, keepdims=True)
    return c / np.where(s > 0.0, s, 1.0)


class StandardisedNoise(Noise):
    """A Noise whose coefficient draw beta ~ N(beta, cov) is standardised per column too (mean 0, sd 1 over the
    draws), so a candidate's Monte Carlo mean carries no x L x mean(z) term: for wide-sd (EXTRAP) candidates that
    term was the largest part of the rng / draw-count dependence of their rank."""

    def param_draw(self, p: int) -> np.ndarray:
        return _standardise_columns(super().param_draw(p))


def standardise_noise(noise: Noise, offset_model: Any = None) -> StandardisedNoise:
    """Common random numbers without a sample-mean bias: z_time and z_quality are centred to mean 0 and scaled to
    sd 1 (so sd_i x mean(z) is exactly 0 for every candidate, rng seed and draw count), the offset draws are
    re-centred on the offset model's mean (and scaled to its sd when it exposes one), and the coefficient draw
    is standardised per column (StandardisedNoise). param_seed / resid_seed are kept, so the draws still change
    with the rng seed. The joint distribution is otherwise the same; P(...) keeps its Monte Carlo meaning."""
    z = _standardise_columns

    off = np.asarray(noise.offset, dtype=float)
    if off.size >= 2:
        mean = getattr(offset_model, "mean", None)
        sd = getattr(offset_model, "sd", None)
        try:
            mean_f = None if mean is None else float(mean)
            sd_f = None if sd is None else float(sd)
        except (TypeError, ValueError):
            mean_f = sd_f = None
        if mean_f is not None and sd_f is not None and float(off.std()) > 0.0:
            off = mean_f + sd_f * z(off)
        elif mean_f is not None:
            off = off - off.mean() + mean_f
    return StandardisedNoise(z_time=z(noise.z_time), z_quality=z(noise.z_quality), offset=off,
                             param_seed=int(noise.param_seed), resid_seed=int(noise.resid_seed))


def offset_mean_of(offset_model: Any, noise: Noise) -> float:
    """The offset model's mean (what the analytic official mean adds), else the mean of the (re-centred) draws."""
    mean = getattr(offset_model, "mean", None)
    try:
        if mean is not None:
            return float(mean)
    except (TypeError, ValueError):
        pass
    return float(np.asarray(noise.offset, dtype=float).mean())


def _score_candidates(models: Models, space: Space, cands: Sequence[Candidate], n: int, rng_seed: int,
                      best_official: float, seed: Optional[int], save_reserve: float,
                      jitter_sd: Optional[float]) -> List[Dict[str, Any]]:
    """Common random numbers: the Noise is regenerated from rng_seed (then standardised), so every worker and
    every batch scores its candidates against the same draws (and the same base draws). The ranking keys
    (bpb_2m_mean, official_mean, delta_vs_base) are the analytic means (module docstring); the Monte Carlo mean
    is kept as official_mc_mean."""
    noise = standardise_noise(draw_noise(rng_seed, n, models.offset), models.offset)
    offset_mean = offset_mean_of(models.offset, noise)
    base_d = simulate_draws(space.base, models.steptime, models.quality, noise, seed=seed,
                            save_reserve=save_reserve, jitter_sd=jitter_sd)
    base_q_mean, base_q_sd = float(base_d.quality_mean), float(base_d.quality_sd)
    fitted = FittedRows(models.quality)
    base_era = base_era_of(models, space)
    min_runs = min_value_runs(models.quality)
    qs, ss = quality_support(models), _support_of(models.steptime)     # once per worker, not per candidate
    queue = resolve_queue(space)
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(cands), BATCH):
        chunk = list(cands[start:start + BATCH])
        recipes = [to_recipe(space, c) for c in chunk]
        bd = simulate_batch(recipes, models.steptime, models.quality, noise, seed=seed,
                            save_reserve=save_reserve, jitter_sd=jitter_sd, on_error="collect")
        for i, (c, r, row) in enumerate(zip(chunk, recipes, summarise_batch(bd, best_official, base_d.official))):
            head = {"name": c.name, "changes": {k: fmt_value(v) for k, v in c.changes.items()},
                    "n_changes": len(c.changes), "code_version": r.code_version, "chip": r.chip}
            if "error" in row:   # one bad candidate must not kill a 500-candidate batch
                head["error"] = row["error"]
                rows.append(head)
                continue
            ok, note, level = support_check(models, c, space, fitted, base_era, min_runs, qs, ss)
            st = row.pop("step_time", {}) or {}
            row.pop("recipe", None)
            row.update(head)
            q_mean, q_sd = float(bd.quality_mean[i]), float(bd.quality_sd[i])
            extra = quality_notes(models.quality, r, float(bd.steps_point[i]), seed)
            if base_q_sd > 0.0 and q_sd > SD_BALLOON_RATIO * base_q_sd:
                extra.append(f"quality sd {q_sd:.4f} is {q_sd / base_q_sd:.0f}x the base's {base_q_sd:.4f}: "
                             "the surrogate is extrapolating")
            row["notes"] = "; ".join(x for x in [row.get("notes", "")] + extra if x)
            # analytic ranking keys: the model's mean at its own step count + the offset mean (exact for every
            # rng seed and draw count); the Monte Carlo mean stays visible as official_mc_mean
            row.update({"step_k1": st.get("k1"), "step_k2": st.get("k2"), "step_k4": st.get("k4"),
                        "in_support": bool(ok), "support_note": note, "support_level": level,
                        "official_mc_mean": float(row["official_mean"]),
                        "bpb_2m_mean": q_mean, "official_mean": q_mean + offset_mean,
                        "delta_vs_base": q_mean - base_q_mean,
                        "job_env": job_env_for(queue, c)})
            rows.append(row)
    return rows


def _score_chunk(args: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    space, cands, n, rng_seed, best_official, seed, save_reserve, jitter_sd = args
    return _score_candidates(_WORKER_STATE["models"], space, cands, n, rng_seed, best_official, seed,
                             save_reserve, jitter_sd)


def evaluate(space: Space, models: Models, n_sims: int = 1000, gen: str = "local", n_random: int = 200,
             rng_seed: int = 0, workers: int = 1, best_official: float = BEST_OFFICIAL,
             seed: Optional[int] = None, save_reserve: float = SAVE_RESERVE_S,
             jitter_sd: Optional[float] = None, candidates: Optional[Sequence[Candidate]] = None,
             validate: bool = True) -> Dict[str, Any]:
    """Score every candidate. Returns {"meta": {...}, "base": row, "rows": [rows sorted by official_mean],
    "errors": [...], "invalid": [candidates train.py would refuse, not simulated]}.
    Each row also carries rank_mean and rank_pbeat."""
    t0 = time.perf_counter()
    cands = list(candidates) if candidates is not None else generate(space, gen, n_random, rng_seed)
    if not cands or cands[0].changes:
        cands = [make_candidate({})] + [c for c in cands if c.changes]
    invalid: List[Dict[str, Any]] = []
    base_violations: List[str] = []
    if validate:
        base_violations = validate_candidate(space, make_candidate({}))
        kept: List[Candidate] = []
        for c in cands:
            viol = [v for v in validate_candidate(space, c) if v not in base_violations] if c.changes else []
            if viol:
                invalid.append({"name": c.name, "changes": {k: fmt_value(v) for k, v in c.changes.items()},
                                "n_changes": len(c.changes), "invalid": "; ".join(viol)})
            else:
                kept.append(c)
        cands = kept
    workers = max(1, int(workers or 1))
    if workers > 1 and not _spawn_safe():
        # `python -` / an interactive session: the spawned children could not re-import __main__ and the
        # Pool would respawn them forever. Score in-process instead (identical results: same Noise).
        workers = 1
    if workers > 1 and len(cands) >= 2 * workers:
        import multiprocessing as mp
        n_chunks = min(len(cands), workers * 4)
        chunks = [cands[i::n_chunks] for i in range(n_chunks)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(models,)) as pool:
            parts = pool.map(_score_chunk, [(space, ch, n_sims, rng_seed, best_official, seed, save_reserve, jitter_sd)
                                            for ch in chunks])
        rows = [r for part in parts for r in part]
    else:
        rows = _score_candidates(models, space, cands, n_sims, rng_seed, best_official, seed, save_reserve, jitter_sd)
    good = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]
    good.sort(key=lambda r: (r["official_mean"], r["name"]))
    for i, r in enumerate(good, 1):
        r["rank_mean"] = i
    for i, r in enumerate(sorted(good, key=lambda r: (-r["p_beat_best"], r["official_mean"], r["name"])), 1):
        r["rank_pbeat"] = i
    for r in good:
        r["rank"] = r["rank_mean"]
    base_row = next((r for r in good if r["n_changes"] == 0), None)
    levels: Dict[str, int] = {}
    for r in good:
        lv = str(r.get("support_level", ""))
        levels[lv] = levels.get(lv, 0) + 1
    meta = {
        "space": space.name, "description": space.description, "gen": gen, "n_candidates": len(good),
        "n_errors": len(errors), "n_invalid": len(invalid), "n_sims": int(n_sims), "rng_seed": int(rng_seed),
        "seed": seed, "best_official": float(best_official), "save_reserve": float(save_reserve), "workers": workers,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "base": {"name": space.base.name, "code_version": space.base.code_version, "chip": space.base.chip,
                 "time_target": space.base.time_target},
        "dims": [{"knob": d.knob, "base": None if d.base_value is None else fmt_value(d.base_value),
                  "values": [fmt_value(v) for v in d.values]} for d in space.dims],
        "max_changes": space.max_changes,
        "models": {k: v for k, v in (getattr(models, "meta", {}) or {}).items() if k != "support"},
        "support_knobs": sorted(quality_support(models)),
        "n_in_support": sum(1 for r in good if r.get("in_support")),
        "support_levels": levels,
        "ranking": "analytic: official_mean = quality mean at the model's step count + offset mean (rng-invariant); "
                   "official_mc_mean is the Monte Carlo mean; sd, percentiles and P(...) are Monte Carlo",
        "crn": "standardised (mean 0, sd 1 per z vector and per coefficient-draw column; offset centred on the "
               "model mean and sd)",
        "min_value_runs": min_value_runs(models.quality),
        "train_py_rules": TRAIN_PY_SOURCE if validate else None,
        "base_violations": base_violations,
        "queue": queue_meta(resolve_queue(space)),
    }
    return {"meta": meta, "base": base_row, "rows": good, "errors": errors, "invalid": invalid}


def top_by(results: Dict[str, Any], key: str = "official_mean", k: int = 20, in_support: Optional[bool] = None,
           exclude_base: bool = True) -> List[Dict[str, Any]]:
    rows = [r for r in results["rows"] if not (exclude_base and r["n_changes"] == 0)]
    if in_support is not None:
        rows = [r for r in rows if bool(r.get("in_support")) == in_support]
    reverse = key in ("p_beat_best", "p_beat_base")
    rows = sorted(rows, key=lambda r: (-r[key] if reverse else r[key], r["official_mean"], r["name"]))
    return rows[:k]
