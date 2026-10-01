"""Virtual charged clock for the GPU proxy (pure python, no torch; embedded verbatim into train_gpu.py).

train.py's time schedule keys every switch on the CHARGED clock: accumulation phases
(FF_ACCUM_SCHED, k1 until 6% and k2 until 20% of FF_TIME_TARGET), FF_MTP stages, the stepped
cooldown level (and with it the EMA insurance level), the insurance save and the stop. On a GPU the
wall clock means nothing, so the patched trainer replaces the charged reading with

    virtual_charged(after step s) = sum over steps 5..s of T[k(step)]

where T[k] is the Trainium seconds per step of accumulation phase k (k1/k2/k4) and steps 0-4 are
excluded exactly as the official runner excludes them (train.py resets budget_started after step 4).
With the chip's own per-phase seconds every switch and the stop land on the same step numbers as on
chip C, so a GPU delta means the same thing as a Trainium delta (equal steps, same switch points).

Two per-phase sources:
  * phase_seconds_from_log(train.log): the per-k MEAN seconds of a real chip run (charged span of
    each phase / its step count). Reproduces that run's boundaries exactly (C40: all switches and
    the stop step within +-1).
  * a fixed triple (--step-seconds 0.30,0.56,0.93) or the ffsim step-time model's medians. Medians
    run ~0.3-1% under the means (MEDIAN_TO_MEAN_OVERHEAD in ffsim/steptime.py), which drifts the
    late switches by a few steps and the stop by ~0.5%; use --stop-step to pin the total.

walk_schedule replays train.py's loop arithmetic (reached_min_steps, hs, the control all_reduce
order of MTP -> accum -> level, the stop rule with step_time_estimate) on the virtual clock and is
the same code the patched trainer runs for --dry-run; the vc_* replicas of cooldown_level /
mtp_stage_of / accum_phase_of / accum_k_of / step_time_estimate are copied from the K60 file and
the trainer passes its OWN functions instead (fns=...), so the dry-run schedule is the real one.

No `from __future__ import annotations` here on purpose: this file is embedded mid-module.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

VCLOCK_VERSION = "2026-09-29.1"

# chip C, C40_cold_K59 (TT 1793, 2357 steps): per-phase MEAN seconds derived from the switch lines
# (13.9 s / 15 warm-up k4 steps; (107.7-13.9)/316 k1; (358.7-107.7)/492 k2; (1787.1-358.7)/1529 k4).
DEFAULT_STEP_SECONDS: Dict[int, float] = {1: 0.29684, 2: 0.51016, 4: 0.93414}
DEFAULT_STEP_SECONDS_SOURCE = "chip C C40_cold_K59 phase means (research/sim-data/chipC/C40_cold_K59/train.log)"


# ----------------------------------------------------------------------------------------------
# train.py schedule arithmetic, replicated (names prefixed vc_ so the embedded copy never shadows
# the trainer's own functions)
# ----------------------------------------------------------------------------------------------
def vc_parse_accum_sched(raw: str) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float], Optional[int]]:
    """train.py's FF_ACCUM_SCHED parser: 'K' | 'k0:u0,K' | 'k0:u0,k1:u1,K' -> (k0, u0, k1, u1, K)."""
    raw = (raw or "").strip()
    if not raw:
        return None, None, None, None, None
    parts = [v.strip() for v in raw.split(",")]
    if len(parts) == 1:
        return None, None, None, None, int(parts[0])
    if len(parts) == 2 and parts[0].count(":") == 1:
        k0, u0 = parts[0].split(":")
        return int(k0), float(u0), None, None, int(parts[1])
    if len(parts) == 3 and all(v.count(":") == 1 for v in parts[:2]):
        (k0, u0), (k1, u1) = (v.split(":") for v in parts[:2])
        return int(k0), float(u0), int(k1), float(u1), int(parts[2])
    raise ValueError("FF_ACCUM_SCHED=%r must be 'K', 'k0:until,K' or 'k0:u0,k1:u1,K'" % raw)


@dataclass
class Recipe:
    """The schedule-relevant knobs of a train.py, defaults = the K60 file's baked defaults."""
    time_target: float = 1793.0
    warmup: int = 20
    accum_sched: str = "1:0.06,2:0.2,4"
    accum_from: Optional[int] = None          # None -> warmup (train.py: FF_ACCUM_FROM defaults to WARMUP_STEPS)
    accum_force_steps: Tuple[int, ...] = ()
    mtp: int = 1
    mtp_phases: Tuple[float, float] = (0.2, 0.45)
    mtp_levels: int = 4
    cooldown_fraction: float = 0.6
    cooldown_floor: int = 1
    cooldown_shape: str = "linear"
    insurance_at: Tuple[float, ...] = (9.0,)
    startup_excluded: int = 5                 # prepare.TRAIN_STARTUP_STEPS_EXCLUDED
    reserve: float = 5.0                      # train.py CHECKPOINT_RESERVE_SECONDS
    async_loop: int = 0
    num_steps: int = 1_000_000
    schedule: str = "time"

    @property
    def accum(self):
        return vc_parse_accum_sched(self.accum_sched)

    @property
    def accum_on(self) -> bool:
        return self.accum[0] is not None

    @property
    def ks(self) -> List[int]:
        k0, _, k1, _, kf = self.accum
        return sorted({k for k in (k0, k1, kf) if k is not None})

    @classmethod
    def from_env(cls, env: Dict[str, str], **overrides: Any) -> "Recipe":
        """Read the knobs the way train.py reads them (empty string = unset = default)."""
        def get(name, conv, default):
            v = env.get(name, "")
            return conv(v) if v else default
        d = cls()
        r = cls(
            time_target=get("FF_TIME_TARGET", float, d.time_target),
            warmup=get("FF_WARMUP", int, d.warmup),
            accum_sched=env.get("FF_ACCUM_SCHED", d.accum_sched).strip(),
            accum_from=get("FF_ACCUM_FROM", int, None),
            accum_force_steps=tuple(int(v) for v in env.get("FF_ACCUM_FORCE_STEPS", "").split(",") if v.strip()),
            mtp=get("FF_MTP", int, d.mtp),
            mtp_phases=tuple(float(v) for v in env.get("FF_MTP_PHASES", "0.2,0.45").split(",")),
            mtp_levels=get("FF_MTP_LEVELS", int, d.mtp_levels),
            cooldown_fraction=get("FF_COOLDOWN_FRAC", float, d.cooldown_fraction),
            cooldown_floor=get("FF_COOLDOWN_FLOOR", int, d.cooldown_floor),
            cooldown_shape=env.get("FF_COOLDOWN_SHAPE", d.cooldown_shape),
            insurance_at=tuple(sorted(float(v) for v in env.get("FF_INSURANCE_AT", "9").split(",") if v.strip())),
            async_loop=get("FF_ASYNC_LOOP", int, d.async_loop),
            schedule=env.get("FF_SCHEDULE", d.schedule),
        )
        for k, v in overrides.items():
            setattr(r, k, v)
        return r


def vc_cooldown_level(progress: float, cooldown_fraction: float, floor: int = 1, shape: str = "linear") -> int:
    """train.py cooldown_level: stepped level (multiplier = level / 20) for charged-time progress."""
    start = 1.0 - cooldown_fraction
    if progress < start:
        return 20
    fraction = (progress - start) / max(cooldown_fraction, 1e-9)
    if shape == "sqrt":
        fraction = max(0.0, fraction) ** 0.5
    return max(floor, 19 - int(fraction * 19))


def vc_mtp_stage_of(progress: float, phases: Sequence[float] = (0.2, 0.45), levels: int = 4) -> int:
    """train.py mtp_stage_of."""
    p1, p2 = phases
    n = levels
    if progress >= p2:
        return 2 * n
    if progress >= p1:
        return n + min(n - 1, int((progress - p1) / (p2 - p1) * n))
    return min(n - 1, int(max(0.0, progress) / p1 * n))


def vc_accum_phase_of(next_step: int, progress: float, accum_from: int, until0: Optional[float],
                      until1: Optional[float], force_steps: Sequence[int] = ()) -> int:
    """train.py accum_phase_of: 0 warm-up (k=K), 1 k0 phase, 2 k1 phase (RAMP3) or the rest, 3 the rest."""
    if force_steps:
        a, b, *c = force_steps
        return 0 if next_step < a else (1 if next_step < b else (2 if not c or next_step < c[0] else 3))
    if next_step < accum_from:
        return 0
    if progress < until0:
        return 1
    return 2 if until1 is None or progress < until1 else 3


def vc_accum_k_of(phase: int, k0: Optional[int], k1: Optional[int], k_final: int) -> int:
    """train.py accum_k_of."""
    return k0 if phase == 1 else (k1 if phase == 2 and k1 is not None else k_final)


def vc_step_time_estimate(recent: Sequence[float], steady: Sequence[float]) -> float:
    """train.py step_time_estimate: max of the last 20 step times, bounded by 3x the median of the last 50 steady."""
    window = recent[-20:]
    worst = max(window)
    tail = steady[-50:]
    if len(tail) >= 5:
        med = sorted(tail)[len(tail) // 2]
        return min(worst, 3.0 * med)
    return worst


# ----------------------------------------------------------------------------------------------
# The clock
# ----------------------------------------------------------------------------------------------
class VirtualClock:
    """Cumulative Trainium seconds of the completed steps, by accumulation phase.

    record(step, k) after step `step` ran with k micro-batches per rank: adds T[k] to `charged`
    when step >= startup_excluded (steps 0-4 are outside the budget, like the real clock after
    budget_started is reset) and mirrors train.py's recent_step_times / steady_step_times lists so
    step_time_estimate sees virtual step times in the stop rule."""

    def __init__(self, phase_seconds: Dict[int, float], startup_excluded: int = 5, steady_from: int = 3,
                 source: str = ""):
        self.T = {int(k): float(v) for k, v in phase_seconds.items()}
        for k, v in self.T.items():
            if not (math.isfinite(v) and v > 0):
                raise ValueError("step seconds for k=%d must be a positive finite number, got %r" % (k, v))
        self.startup_excluded = int(startup_excluded)
        self.steady_from = int(steady_from)
        self.source = source
        self.charged = 0.0
        self.recent: List[float] = []
        self.steady: List[float] = []
        self.last_dt: Optional[float] = None
        self.steps = 0

    def dt_for(self, k: int) -> float:
        try:
            return self.T[int(k)]
        except KeyError:
            raise KeyError("virtual clock has no step seconds for k=%d: it knows %s; pass --step-seconds k%d=... "
                           "(or a triple covering every FF_ACCUM_SCHED phase)" % (k, sorted(self.T), k)) from None

    def record(self, step: int, k: int) -> float:
        dt = self.dt_for(k)
        self.recent.append(dt)
        if step >= self.steady_from:
            self.steady.append(dt)
        if step >= self.startup_excluded:
            self.charged += dt
        self.last_dt = dt
        self.steps = step + 1
        return dt

    def windows(self, recent: Sequence[float], steady: Sequence[float]) -> Tuple[List[float], List[float]]:
        """The (recent, steady) lists the stop rule should use: the virtual ones."""
        return self.recent, self.steady

    def describe(self) -> Dict[str, Any]:
        return {"step_seconds": {str(k): v for k, v in sorted(self.T.items())}, "source": self.source,
                "startup_excluded": self.startup_excluded}


def parse_step_seconds(spec: str, ks: Sequence[int]) -> Dict[int, float]:
    """'0.30,0.56,0.93' (positional over the schedule's sorted ks) or 'k1=0.30,k2=0.56,k4=0.93'."""
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("--step-seconds is empty")
    out: Dict[int, float] = {}
    if all("=" in p for p in parts):
        for p in parts:
            name, val = p.split("=", 1)
            name = name.strip().lower()
            k = int(name[1:] if name.startswith("k") else name)
            out[k] = float(val)
        return out
    if any("=" in p for p in parts):
        raise ValueError("--step-seconds %r: mix of named and positional values" % spec)
    ks = list(ks)
    if len(parts) != len(ks):
        raise ValueError("--step-seconds %r: %d values for the schedule's phases k=%s (give one per phase in "
                         "ascending k, or name them k1=...,k2=...)" % (spec, len(parts), ks))
    return {k: float(v) for k, v in zip(ks, parts)}


# ----------------------------------------------------------------------------------------------
# Chip log parsing
# ----------------------------------------------------------------------------------------------
STEP_LINE = re.compile(r"^step (\d+) \| loss ([-\d.na]+) \| lrm ([\d.]+) \| dt ([\d.]+)s \| tok/s ([\d,]+) \| "
                       r"charged ([\d.]+)s \| next_level (\d+)")
ACCUM_LINE = re.compile(r"^FF_ACCUM_SCHED phase (\d+): (\d+) micro-batches per update .* from step (\d+) \(charged ([\d.]+)s\)")
MTP_LINE = re.compile(r"^FF_MTP stage (\d+)/(\d+): .* from step (\d+) \(charged ([\d.]+)s\)")
INSURANCE_LINE = re.compile(r"^insurance checkpoint \((\w+), level (\d+)\) written in [\d.]+s at charged ([\d.]+)s")
HEADER_LINE = re.compile(r"^schedule=(\w+) time_target=([\d.]+)s num_steps=(\d+) cooldown_fraction=([\d.]+)")
ACCUM_K_LINE = re.compile(r"grad_accum_steps=(\d+)")


def parse_chip_log(path: str) -> Dict[str, Any]:
    """Switch events and end state of a train.log (chip or GPU). Steps are 'from step N' (the first step
    that runs under the new phase / stage / level). Levels come from the step lines (train.py prints a
    step line whenever next_level changes), as [level, from_step, charged]."""
    accum: List[List[Any]] = []
    mtp: List[List[Any]] = []
    levels: List[List[Any]] = []
    insurance: List[List[Any]] = []
    summary: Optional[Dict[str, Any]] = None
    last_step = -1
    last_charged = None
    n_step_lines = 0
    time_target = None
    schedule = None
    grad_accum = None
    prev_level = 20
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = STEP_LINE.match(line)
            if m:
                n_step_lines += 1
                s = int(m.group(1))
                last_step = max(last_step, s)
                last_charged = float(m.group(6))
                lvl = int(m.group(7))
                if lvl != prev_level:
                    levels.append([lvl, s + 1, float(m.group(6))])
                    prev_level = lvl
                continue
            m = ACCUM_LINE.match(line)
            if m:
                accum.append([int(m.group(1)), int(m.group(3)), float(m.group(4)), int(m.group(2))])
                continue
            m = MTP_LINE.match(line)
            if m:
                mtp.append([int(m.group(1)), int(m.group(3)), float(m.group(4))])
                continue
            m = INSURANCE_LINE.match(line)
            if m:
                insurance.append([m.group(1), int(m.group(2)), float(m.group(3))])
                continue
            m = HEADER_LINE.match(line)
            if m:
                schedule, time_target = m.group(1), float(m.group(2))
                continue
            m = ACCUM_K_LINE.search(line)
            if m and grad_accum is None and line.startswith("micro_tokens="):
                grad_accum = int(m.group(1))
                continue
            if line.startswith("ff_summary: "):
                try:
                    summary = json.loads(line[len("ff_summary: "):])
                except json.JSONDecodeError:
                    summary = None
    if summary:
        # the summary carries the switches with one decimal (the log lines round charged to whole seconds)
        acc = summary.get("accum_sched", {}).get("switches")
        if acc:
            accum = [[int(a), int(b), float(c), int(d)] for a, b, c, d in acc]
        ms = summary.get("mtp", {}).get("switches")
        if ms:
            mtp = [[int(a), int(b), float(c)] for a, b, c in ms]
        if grad_accum is None:
            grad_accum = summary.get("grad_accum_steps")
        if time_target is None:
            time_target = summary.get("time_target_seconds")
    charged_final = None
    if summary and summary.get("training_seconds_before_save") is not None:
        charged_final = float(summary["training_seconds_before_save"])
    elif last_charged is not None:
        charged_final = last_charged
    steps = int(summary["steps"]) if summary and summary.get("steps") else (last_step + 1 if last_step >= 0 else 0)
    return {"accum": accum, "mtp": mtp, "levels": levels, "insurance": insurance, "steps": steps,
            "charged_final": charged_final, "summary": summary, "time_target": time_target, "schedule": schedule,
            "grad_accum_steps": grad_accum, "n_step_lines": n_step_lines,
            "median_by_k": {int(k): float(v) for k, v in
                            (summary or {}).get("accum_sched", {}).get("median_step_seconds_by_k", {}).items()}}


def phase_seconds_from_log(path: str, startup_excluded: int = 5) -> Dict[int, float]:
    """Per-k MEAN seconds of a chip run: the charged span between consecutive switch lines divided by
    the steps in between, pooled by k (the warm-up runs at k = K, so it pools with the final phase).
    The charged clock starts after step startup_excluded - 1; a switch 'from step N (charged C)' was
    decided after step N - 1 at charged C; the final segment ends at training_seconds_before_save."""
    log = parse_chip_log(path)
    if not log["accum"]:
        raise ValueError("%s: no FF_ACCUM_SCHED switch lines (not a time-schedule run with FF_ACCUM_SCHED?)" % path)
    if log["charged_final"] is None or not log["steps"]:
        raise ValueError("%s: no step lines / final charged seconds" % path)
    k_final = log["grad_accum_steps"] or max(k for _, _, _, k in log["accum"])
    bounds = [(startup_excluded, 0.0, k_final)]
    for phase, step, charged, k in log["accum"]:
        bounds.append((step, charged, k))
    seconds: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for i, (start, c0, k) in enumerate(bounds):
        if i + 1 < len(bounds):
            end, c1, _ = bounds[i + 1]
        else:
            end, c1 = log["steps"], log["charged_final"]
        n = end - start
        if n <= 0:
            continue
        seconds[k] = seconds.get(k, 0.0) + (c1 - c0)
        counts[k] = counts.get(k, 0) + n
    return {k: seconds[k] / counts[k] for k in sorted(seconds)}


# ----------------------------------------------------------------------------------------------
# The walk
# ----------------------------------------------------------------------------------------------
def default_fns(recipe: Recipe) -> Dict[str, Callable]:
    k0, u0, k1, u1, kf = recipe.accum
    accum_from = recipe.warmup if recipe.accum_from is None else recipe.accum_from
    return {
        "cooldown_level": lambda p: vc_cooldown_level(p, recipe.cooldown_fraction, recipe.cooldown_floor, recipe.cooldown_shape),
        "mtp_stage_of": lambda p: vc_mtp_stage_of(p, recipe.mtp_phases, recipe.mtp_levels),
        "accum_phase_of": lambda s, p: vc_accum_phase_of(s, p, accum_from, u0, u1, recipe.accum_force_steps),
        "accum_k_of": lambda ph, k_final: vc_accum_k_of(ph, k0, k1, k_final),
        "step_time_estimate": vc_step_time_estimate,
    }


def walk_schedule(recipe: Recipe, phase_seconds: Dict[int, float], stop_step: Optional[int] = None,
                  fns: Optional[Dict[str, Callable]] = None, k_final: Optional[int] = None) -> Dict[str, Any]:
    """Replay train.py's loop on the virtual clock. Returns the switch events in train.log's convention
    ('from step N') plus the step count and the final virtual charged seconds.

    stop_step: when set the clock's stop rule is disabled and the run ends after exactly stop_step
    steps (train_gpu.py --stop-step), every other switch still following the clock."""
    if recipe.schedule != "time":
        raise ValueError("walk_schedule: FF_SCHEDULE must be 'time' (a steps schedule has no clock-keyed switches)")
    fns = dict(fns or default_fns(recipe))
    k0, u0, k1, u1, kf = recipe.accum
    if k_final is None:
        k_final = kf if kf is not None else max(phase_seconds)
    accum_on = k0 is not None
    mtp_on = bool(recipe.mtp)
    TT = float(recipe.time_target)
    vc = VirtualClock(phase_seconds, recipe.startup_excluded)
    num_steps = recipe.num_steps if stop_step is None else min(recipe.num_steps, int(stop_step))
    step = 0
    next_level = 20
    mtp_stage = 0
    accum_phase = fns["accum_phase_of"](0, 0.0) if accum_on else 0
    accum_k = fns["accum_k_of"](accum_phase, k_final) if accum_on else k_final
    ins_next = 0
    accum_ev: List[List[Any]] = []
    mtp_ev: List[List[Any]] = []
    level_ev: List[List[Any]] = []
    ins_ev: List[List[Any]] = []
    steps_by_k: Dict[int, int] = {}
    stop_reason = "num_steps"
    while step < num_steps:
        step_k = accum_k
        hs = (not recipe.async_loop or step + 1 <= recipe.startup_excluded
              or (step + 1) % recipe.async_loop == 0 or step + 1 >= num_steps)
        vc.record(step, step_k)
        charged = vc.charged
        steps_by_k[step_k] = steps_by_k.get(step_k, 0) + 1
        reached = step + 1 >= recipe.startup_excluded
        level_next = fns["cooldown_level"](charged / TT) if reached else 20
        if step + 1 >= num_steps:
            stop = True
            stop_reason = "stop_step" if stop_step is not None and step + 1 >= stop_step else "num_steps"
        elif stop_step is not None:
            stop = False
        else:
            mult = max(1, recipe.async_loop) if len(vc.steady) >= 5 else 1
            stop = reached and (charged >= TT
                                or charged + mult * fns["step_time_estimate"](vc.recent, vc.steady) + recipe.reserve >= TT)
            if stop:
                stop_reason = "clock"
        insurance_due = reached and ins_next < len(recipe.insurance_at) and charged / TT >= recipe.insurance_at[ins_next]
        mtp_next = fns["mtp_stage_of"](charged / TT) if (mtp_on and reached) else 0
        accum_next = fns["accum_phase_of"](step + 1, charged / TT if reached else 0.0) if accum_on else 0
        if hs:
            if mtp_on and mtp_next != mtp_stage:
                mtp_ev.append([mtp_next, step + 1, round(charged, 1)])
                mtp_stage = mtp_next
            if accum_on and accum_next > accum_phase:
                k_new = fns["accum_k_of"](accum_next, k_final)
                accum_ev.append([accum_next, step + 1, round(charged, 1), k_new])
                accum_phase, accum_k = accum_next, k_new
            if level_next != next_level:
                level_ev.append([level_next, step + 1, round(charged, 1)])
                next_level = level_next
            stop_flag = stop
            if insurance_due and not stop_flag:
                ins_ev.append([ins_next, step + 1, round(charged, 1)])
                ins_next += 1
        else:
            stop_flag = False
        step += 1
        if stop_flag:
            break
        if step > 50_000_000:
            raise RuntimeError("walk_schedule did not terminate")
    return {"steps": step, "charged_final": round(vc.charged, 3), "accum": accum_ev, "mtp": mtp_ev,
            "levels": level_ev, "insurance": ins_ev, "steps_by_k": steps_by_k, "final_level": next_level,
            "final_k": accum_k, "stop_reason": stop_reason, "phase_seconds": {k: v for k, v in sorted(vc.T.items())}}


def rescale_to_steps(recipe: Recipe, phase_seconds: Dict[int, float], target_steps: int,
                     fns: Optional[Dict[str, Callable]] = None, k_final: Optional[int] = None,
                     lo: float = 0.25, hi: float = 4.0, iters: int = 60) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
    """Scale every phase's seconds by one factor so the clock-driven run ends at target_steps
    (train_gpu.py --steps N: both arms of a pair at the same step count with every switch at the same
    charged FRACTION of FF_TIME_TARGET, i.e. the same relative position in the run). Steps are monotone
    non-increasing in the scale, so a bisection finds the exact count when one exists; the walk that comes
    back is at the chosen scale and the caller pins the stop at target_steps as well (walk_schedule's
    stop_step), so a boundary case can never overshoot by a step. Returns (scaled seconds, scale, walk)."""
    if target_steps < recipe.startup_excluded + 1:
        raise ValueError("target steps %d is below the excluded startup steps" % target_steps)

    def steps_at(scale: float) -> int:
        return walk_schedule(recipe, {k: v * scale for k, v in phase_seconds.items()}, fns=fns, k_final=k_final)["steps"]

    s_lo, s_hi = lo, hi
    n_lo, n_hi = steps_at(s_lo), steps_at(s_hi)
    if not (n_hi <= target_steps <= n_lo):
        raise ValueError("target steps %d is outside the reachable range [%d (scale %.2f), %d (scale %.2f)]"
                         % (target_steps, n_hi, s_hi, n_lo, s_lo))
    scale = 1.0
    for _ in range(iters):
        mid = 0.5 * (s_lo + s_hi)
        n_mid = steps_at(mid)
        if n_mid >= target_steps:
            s_lo, n_lo = mid, n_mid
        else:
            s_hi, n_hi = mid, n_mid
        if n_lo == target_steps and s_hi - s_lo < 1e-9:
            break
    scale = s_lo if n_lo == target_steps else (s_lo if abs(n_lo - target_steps) <= abs(n_hi - target_steps) else s_hi)
    scaled = {k: v * scale for k, v in phase_seconds.items()}
    walk = walk_schedule(recipe, scaled, stop_step=target_steps, fns=fns, k_final=k_final)
    return scaled, scale, walk


def compare_schedules(walk: Dict[str, Any], chip: Dict[str, Any]) -> Dict[str, Any]:
    """Line up the walk's events with a chip log's by id (accum phase, MTP stage, cooldown level) and
    report the step differences (walk - chip). Missing partners are reported, not hidden."""
    rows: List[Dict[str, Any]] = []

    def pair(kind, a, b):
        da = {e[0]: e[1] for e in a}
        db = {e[0]: e[1] for e in b}
        for ident in sorted(set(da) | set(db)):
            ga, gb = da.get(ident), db.get(ident)
            rows.append({"kind": kind, "id": ident, "gpu_step": ga, "chip_step": gb,
                         "diff": None if ga is None or gb is None else ga - gb})

    pair("accum", walk["accum"], chip["accum"])
    pair("mtp", walk["mtp"], chip["mtp"])
    pair("level", walk["levels"], chip["levels"])
    rows.append({"kind": "steps", "id": "total", "gpu_step": walk["steps"], "chip_step": chip["steps"],
                 "diff": walk["steps"] - chip["steps"] if chip.get("steps") else None})
    diffs = [abs(r["diff"]) for r in rows if r["diff"] is not None and r["kind"] != "steps"]
    return {"rows": rows, "max_abs_switch_diff": max(diffs) if diffs else None,
            "missing": [r for r in rows if r["diff"] is None], "steps_diff": rows[-1]["diff"]}


def format_comparison(cmp: Dict[str, Any]) -> str:
    lines = ["%-6s %6s %6s %6s %6s" % ("kind", "id", "gpu", "chip", "diff")]
    for r in cmp["rows"]:
        lines.append("%-6s %6s %6s %6s %6s" % (r["kind"], r["id"], r["gpu_step"], r["chip_step"],
                                               "" if r["diff"] is None else r["diff"]))
    lines.append("max |switch diff| = %s, steps diff = %s, unmatched = %d"
                 % (cmp["max_abs_switch_diff"], cmp["steps_diff"], len(cmp["missing"])))
    return "\n".join(lines)
