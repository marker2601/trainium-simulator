"""Parse a harvested chip run directory (research/sim-data/chip{C,D}/<run>/) into a RunRecord.

Files read (all optional except train.log): train.log, eval.log, eval20.log, overrides, code.sha256,
full_run.status, t_* (epoch stamps), and _meta/base.env found in the run dir, the chip dir or the
sim-data dir.  Pure stdlib.

Step-time rules (see CONTRACT.md):
  * each logged `step N | ... | dt Ds | tok/s T | charged Cs` line is assigned an accumulation k;
    k comes from the `FF_ACCUM_SCHED phase P: K micro-batches per update ... from step S` lines
    (initial k = `grad_accum_steps` from the `micro_tokens=... grad_accum_steps=K` echo, or the
    first stage of the FF_ACCUM_SCHED echo from the warm-up end), cross-checked per step against
    tokens/step = dt * tok/s (the tok/s column is exact to 6 digits, the printed dt only to 0.01 s,
    so the per-step dt used for the medians is k * micro_tokens / tok_s when it agrees with the
    printed dt within rounding);
  * line-based medians per k phase EXCLUDE steps 0-4 (startup), the first 2 logged step lines
    at/after any phase switch, and any step with dt > 3x the phase median (recompiles);
  * when the log ends with `ff_summary: {...}`, its `accum_sched.median_step_seconds_by_k` (median
    over ALL steps, 4 decimals) is used for StepTime instead, with n_* = `steps_by_k`; S60 screens
    log too few lines for the line-based medians (10-step forced phases), so this is what makes
    them usable. The line-based values stay in TrainLog.step_time_lines and in notes
    (`lines_median_by_k`); notes carry `step_time_source=summary|lines`;
  * steps = summary `steps` when present, else last step number + 1;
  * charged_seconds = summary `training_seconds_before_save` when present (the clock at the end of
    training), else the last logged `charged` value; startup_seconds = dt of step 0 when >= 100 s
    (a warm compile cache gives ~40 s and is not a cold start).
Other files: eval.log -> bpb_2m, eval_ema.log -> bpb_2m_ema, eval20.log -> bpb_20m, overrides
(the full effective env, one KEY=VALUE per line, later lines win) + _meta/base.env -> knobs,
code.sha256 -> code_sha, FF_CODE_DIR -> code_version, t_launch -> date_utc, full_run.status and
timing.txt -> notes.

Env provenance (added by the dataset build, 29 Sep): the queue runner (`_meta/queue/queue_runner.sh`) writes
base.env + the job env verbatim into `overrides`, so a file that covers every base.env key IS the effective
env (`env_source=queue`, knobs_complete=True). When the harvest carries the queue's runner.log, a run absent
from its START lines was launched by a driver/suite script that never read base.env (`env_source=driver`:
base.env is NOT layered, knobs_complete=False); a run in the START lines whose `overrides` is missing gets
the START override list (recovered). A dir with `launch-command.txt` and no `overrides` is a hand-launched
rehearsal whose env is baked into train.py (`env_source=hand`). Without a runner.log the legacy behaviour
stands (base.env layered under overrides). Whenever provenance is decidable (or no overrides exist), the
knobs are reconciled against what train_ff.py echoed (schedule=/ema=/async_sync= lines, model_config JSON,
ff_summary): missing knobs are filled from the echo (`knobs_from_echo=`), and a knob that contradicts the
echo is replaced by the echo value with `env_mismatch=KEY:env->echo` and knobs_complete=False (the base.env
in force at run time differed from the harvested one). Logs with step lines but neither ff_summary nor an
eval are tagged `log_incomplete` (still running or died).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .schema import RunRecord, StepTime

__all__ = [
    "TrainLog",
    "parse_train_log",
    "parse_eval_log",
    "parse_overrides",
    "parse_env_file",
    "read_code_sha",
    "compute_step_time",
    "is_screen_name",
    "is_run_dir",
    "parse_run_dir",
    "find_base_env",
    "find_runner_log",
    "parse_runner_log",
    "load_runner_log",
    "echo_env",
    "knob_vocab",
    "canonical_value",
]

DEFAULT_MICRO_TOKENS = 65_536
STARTUP_STEPS = 5            # steps 0-4 are excluded from the medians
SWITCH_SKIP_LINES = 2        # logged step lines skipped at/after each phase switch
RECOMPILE_FACTOR = 3.0       # dt > 3x the phase median is a recompile
STARTUP_MIN_SECONDS = 100.0  # step-0 dt at or above this is the compile/startup time
SCREEN_MAX_STEPS = 200
DT_ROUNDING_TOL = 0.0075     # |k*micro_tokens/tok_s - printed dt| within this = same step

STEP_RE = re.compile(
    r"^step\s+(\d+)\s*\|\s*loss\s+([-+0-9.eE]+|nan|inf)\s*\|\s*lrm\s+([-+0-9.eE]+)"
    r"\s*\|\s*dt\s+([0-9.]+)s\s*\|\s*tok/s\s+([0-9,]+)\s*\|\s*charged\s+([0-9.]+)s"
    r"(?:\s*\|\s*next_level\s+(\d+))?"
)
PHASE_RE = re.compile(
    r"^FF_ACCUM_SCHED phase (\d+):\s*(\d+)\s+micro-batch(?:es)?\s+per update.*?from step (\d+)"
    r"(?:\s*\(charged\s+([0-9.]+)s\))?"
)
SCHED_ECHO_RE = re.compile(
    r"^FF_ACCUM_SCHED=([0-9.:,]+):\s*(\d+)\s+micro-batch(?:es)?.*?from step (\d+)"
)
ACCUM_RE = re.compile(r"micro_tokens=([0-9,]+)\s+grad_accum_steps=(\d+)")
NUM_PARAMS_RE = re.compile(r"num_params=([0-9,]+)")
KV_TOKEN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")
ECHO_KNOB_RE = re.compile(r"^(FF_[A-Z0-9_]+)=([^\s:]+)(?::|\s|$)")
ECHO_FLAG_RE = re.compile(r"^(FF_[A-Z0-9_]+):\s")
SUMMARY_RE = re.compile(r"^ff_summary:\s*(\{.*\})\s*$")
SUMMARY_STEPS_RE = re.compile(r'"steps":\s*(\d+)')
DATE_W_RE = re.compile(r"^\[?W(\d{3,4}) \d{2}:\d{2}:\d{2}")   # torch stamps: W0926 18:37:54 (MMDD) / [W926 18:37:54 (MDD)
VAL_BPB_RE = re.compile(r"val_bpb[\"'\s:=]*([0-9]+\.[0-9]+)")
# The S60 screen marker is matched case-sensitively: a lowercase `_s60` is the seed suffix of a full run
# (run names end in `_s<seed>`, e.g. `2147_D_s60` with FF_SEED=60), never a screen marker.
SCREEN_NAME_RE = re.compile(
    r"(?i)(?:^|[_\-])(?:(?-i:S60)|driver|screen|probe|prof|trace)(?:$|[_\-.0-9])|^S\d{1,3}(?:[_\-]|$)"
)
NOISE_RE = re.compile(
    r"OperatorEntry|registered at|dispatch key|new kernel|previous kernel|operator:"
    r"|torch/distributed/run\.py|ShardIndexInjection|Overriding a previously registered kernel"
)


@dataclass
class TrainLog:
    """Everything parse_train_log() extracts; parse_run_dir() folds it into a RunRecord."""
    rows: List[dict] = field(default_factory=list)   # step, loss, lrm, dt, tok_s, charged, k, dt_precise
    switches: List[Tuple[int, int]] = field(default_factory=list)  # (step, k) in log order
    initial_k: Optional[int] = None
    micro_tokens: int = DEFAULT_MICRO_TOKENS
    schedule: Optional[str] = None      # FF_ACCUM_SCHED string as echoed, e.g. "2:0.2,4"
    config: Dict[str, str] = field(default_factory=dict)   # key=value tokens from config echo lines
    model_config: Dict[str, object] = field(default_factory=dict)  # the model_config={...} JSON echo
    knob_hints: Dict[str, str] = field(default_factory=dict)  # FF_X=V echoed by train_ff.py
    echo_flags: List[str] = field(default_factory=list)       # FF_X: ... lines (enabled, value unknown)
    summary: Optional[dict] = None
    summary_steps: Optional[int] = None
    step_time: StepTime = field(default_factory=StepTime)         # final: summary medians when present, else lines
    step_time_lines: StepTime = field(default_factory=StepTime)   # always the logged-line computation
    step_time_source: str = "lines"
    steps: Optional[int] = None
    charged_seconds: Optional[float] = None
    startup_seconds: Optional[float] = None
    time_target: Optional[float] = None
    num_steps: Optional[int] = None
    date_mmdd: Optional[Tuple[int, int]] = None
    k_mismatches: int = 0
    other_k: Dict[int, int] = field(default_factory=dict)  # k values outside {1,2,4}: count
    notes: List[str] = field(default_factory=list)

    @property
    def loss_curve(self) -> List[List[float]]:
        return [[float(r["step"]), float(r["loss"])] for r in self.rows if r["loss"] == r["loss"]]


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").rstrip("s"))
    except (ValueError, AttributeError):
        return None


def _to_int(s) -> Optional[int]:
    try:
        return int(float(str(s).replace(",", "")))
    except (ValueError, TypeError):
        return None


def _median(xs: List[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def _parse_schedule(sched: str) -> List[int]:
    """'1:0.06,2:0.2,4' -> [1, 2, 4]; '4' -> [4]."""
    ks = []
    for part in sched.split(","):
        part = part.strip()
        if not part:
            continue
        k = part.split(":")[0]
        if k.isdigit():
            ks.append(int(k))
    return ks


def parse_train_log(text: str) -> TrainLog:
    """Parse train.log text. Tolerates truncated logs (SSM 24,000-char cap) and noise lines."""
    log = TrainLog()
    sched_first_k: Optional[int] = None
    sched_from_step: Optional[int] = None
    in_json = False
    json_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line:
            continue
        if log.date_mmdd is None and line.startswith(("W", "[W")):
            m = DATE_W_RE.match(line)          # the date stamp sits on noise lines, so read it first
            if m:
                md = m.group(1)
                log.date_mmdd = (int(md[:-2]), int(md[-2:]))
        if NOISE_RE.search(line):
            continue
        if in_json:
            json_lines.append(line)
            if line.startswith("}"):
                in_json = False
                _set_model_config(log, "\n".join(json_lines))
                json_lines = []
            continue
        m = STEP_RE.match(line)
        if m:
            step = int(m.group(1))
            loss = _to_float(m.group(2))
            row = {
                "step": step,
                "loss": float("nan") if loss is None else loss,
                "lrm": _to_float(m.group(3)),
                "dt": float(m.group(4)),
                "tok_s": _to_float(m.group(5)),
                "charged": float(m.group(6)),
                "next_level": _to_int(m.group(7)) if m.group(7) else None,
            }
            log.rows.append(row)
            continue
        m = PHASE_RE.match(line)
        if m:
            log.switches.append((int(m.group(3)), int(m.group(2))))
            continue
        m = SCHED_ECHO_RE.match(line)
        if m:
            log.schedule = m.group(1)
            sched_first_k = int(m.group(2))
            sched_from_step = int(m.group(3))
            log.knob_hints.setdefault("FF_ACCUM_SCHED", log.schedule)
            continue
        m = ACCUM_RE.search(line)
        if m:
            log.micro_tokens = int(m.group(1).replace(",", ""))
            log.initial_k = int(m.group(2))
            log.config["micro_tokens"] = str(log.micro_tokens)
            log.config["grad_accum_steps"] = m.group(2)
            continue
        m = SUMMARY_RE.match(line)
        if m:
            try:
                log.summary = json.loads(m.group(1))
            except json.JSONDecodeError:
                ms = SUMMARY_STEPS_RE.search(m.group(1))
                log.summary = {"steps": int(ms.group(1))} if ms else {}
                log.notes.append("ff_summary JSON malformed")
            continue
        if line.startswith("model_config={"):
            body = line[len("model_config="):]
            if body.rstrip().endswith("}") and body.count("{") == body.count("}"):
                _set_model_config(log, body)          # single-line form
            else:
                in_json = True
                json_lines = ["{"]
            continue
        if line.startswith("FF_"):
            m = ECHO_KNOB_RE.match(line)
            if m:
                log.knob_hints.setdefault(m.group(1), m.group(2))
                continue
            m = ECHO_FLAG_RE.match(line)
            if m:
                log.echo_flags.append(m.group(1))
                continue
        tokens = line.split()
        if len(tokens) >= 2 and all(KV_TOKEN_RE.fullmatch(t) for t in tokens):
            for t in tokens:                       # config echo lines: schedule=time time_target=1788.0s ...
                km = KV_TOKEN_RE.fullmatch(t)
                log.config.setdefault(km.group(1), km.group(2))
    # ---- derived scalars
    if log.initial_k is None:
        if log.schedule:
            ks = _parse_schedule(log.schedule)
            log.initial_k = ks[-1] if ks else 4   # the schedule's last stage is the base k
        elif log.summary and log.summary.get("grad_accum_steps"):
            log.initial_k = _to_int(log.summary["grad_accum_steps"])
        else:
            log.initial_k = 4
    # the warm-up end switch implied by the FF_ACCUM_SCHED echo when no phase line covers it
    if sched_first_k is not None and sched_from_step is not None:
        if not any(s == sched_from_step for s, _ in log.switches):
            log.switches.insert(0, (sched_from_step, sched_first_k))
    log.switches.sort(key=lambda t: t[0])
    tt = log.config.get("time_target")
    log.time_target = _to_float(tt) if tt else None
    ns = log.config.get("num_steps")
    log.num_steps = _to_int(ns) if ns else None
    if log.summary:
        log.summary_steps = _to_int(log.summary.get("steps"))
        if log.time_target is None:
            log.time_target = _to_float(str(log.summary.get("time_target_seconds", "")))
    if log.rows:
        last = log.rows[-1]
        log.steps = log.summary_steps if log.summary_steps else last["step"] + 1
        log.charged_seconds = last["charged"]
        first = log.rows[0]
        if first["step"] == 0 and first["dt"] >= STARTUP_MIN_SECONDS:
            log.startup_seconds = first["dt"]
    if log.summary:
        # the summary's charged clock is the value at the end of training; the last logged line can be
        # up to ~10 steps earlier
        for key in ("training_seconds_before_save", "charged_seconds_est_after_save"):
            v = _to_float(str(log.summary.get(key, "")))
            if v is not None and v > 0:
                log.charged_seconds = v
                break
    _assign_k(log)
    log.step_time_lines = compute_step_time(log.rows, log.switches)
    log.step_time = _summary_step_time(log)
    return log


def _summary_step_time(log: TrainLog) -> StepTime:
    """Prefer the log's own `median_step_seconds_by_k` (all steps, 4 decimals; screens log too few
    lines for the line-based medians) and fall back per phase to the logged-line computation."""
    st = StepTime(**vars(log.step_time_lines))
    acc = (log.summary or {}).get("accum_sched") or {}
    med = acc.get("median_step_seconds_by_k") or {}
    nby = acc.get("steps_by_k") or {}
    used = []
    for k in (1, 2, 4):
        v = _to_float(str(med.get(str(k), "")))
        if v is None or v <= 0:
            continue
        n = _to_int(nby.get(str(k))) or getattr(st, f"n_k{k}")
        setattr(st, f"k{k}", v)
        setattr(st, f"n_k{k}", n)
        used.append(k)
    log.step_time_source = "summary" if used else "lines"
    return st


def _assign_k(log: TrainLog) -> None:
    """Assign k per logged step from the phase lines, cross-checked against dt * tok/s."""
    switches = list(log.switches)
    for row in log.rows:
        k_lines = log.initial_k
        for s, k in switches:
            if row["step"] >= s:
                k_lines = k
        k_tok = None
        if row["tok_s"] and row["dt"] > 0 and log.micro_tokens:
            est = row["dt"] * row["tok_s"] / log.micro_tokens
            cand = int(round(est))
            if cand >= 1 and abs(est - cand) <= 0.15 * cand:
                k_tok = cand
        k = k_lines
        if k_tok is not None and k_tok != k_lines:
            log.k_mismatches += 1
            k = k_tok
        row["k_lines"] = k_lines
        row["k_tok"] = k_tok
        row["k"] = k
        dt_precise = None
        if row["tok_s"] and k:
            cand = k * log.micro_tokens / row["tok_s"]
            if abs(cand - row["dt"]) <= DT_ROUNDING_TOL:   # printed dt is rounded to 0.01 s
                dt_precise = cand
        row["dt_precise"] = dt_precise if dt_precise is not None else row["dt"]
        if k not in (1, 2, 4):
            log.other_k[k] = log.other_k.get(k, 0) + 1
    if log.k_mismatches:
        log.notes.append(f"k from tok/s disagreed with phase lines on {log.k_mismatches} step lines")


def compute_step_time(rows: List[dict], switches: List[Tuple[int, int]]) -> StepTime:
    """Median dt per k phase with the startup / switch / recompile exclusions (see module doc)."""
    switch_steps = sorted({s for s, _ in switches})
    skip_idx = set()
    for s in switch_steps:
        n = 0
        for i, row in enumerate(rows):
            if row["step"] >= s:
                skip_idx.add(i)
                n += 1
                if n >= SWITCH_SKIP_LINES:
                    break
    by_k: Dict[int, List[float]] = {}
    for i, row in enumerate(rows):
        if row["step"] < STARTUP_STEPS or i in skip_idx or row.get("k") is None:
            continue
        by_k.setdefault(row["k"], []).append(row.get("dt_precise", row["dt"]))
    st = StepTime()
    for k, dts in by_k.items():
        med = _median(dts)
        kept = [d for d in dts if d <= RECOMPILE_FACTOR * med] if med else dts
        med = _median(kept)
        if k == 1:
            st.k1, st.n_k1 = med, len(kept)
        elif k == 2:
            st.k2, st.n_k2 = med, len(kept)
        elif k == 4:
            st.k4, st.n_k4 = med, len(kept)
    return st


def parse_eval_log(text: str) -> Tuple[Optional[float], Optional[float]]:
    """(final val_bpb, ema val_bpb-or-None). A line naming EMA feeds the second value."""
    plain: List[float] = []
    ema: List[float] = []
    for line in text.splitlines():
        if NOISE_RE.search(line):
            continue
        for m in VAL_BPB_RE.finditer(line):
            v = float(m.group(1))
            if re.search(r"(?i)\bema\b|_ema|ema_", line):
                ema.append(v)
            else:
                plain.append(v)
    return (plain[-1] if plain else None), (ema[-1] if ema else None)


def _kv_tokens(tokens: List[str], out: Dict[str, str]) -> None:
    last_key = None
    for tok in tokens:
        if tok.startswith("export") and tok == "export":
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", tok, re.S)
        if m:
            last_key = m.group(1)
            out[last_key] = m.group(2)
        elif last_key is not None:
            out[last_key] = (out[last_key] + " " + tok).strip()   # value with spaces (FF_CC_FLAGS)


def parse_overrides(text: str) -> Dict[str, str]:
    """`KEY=VALUE` tokens, one or many per line, applied in order (last FF_SEED wins).

    Accepts `export KEY=VALUE`, quoted values, `#` comments and values containing spaces
    (tokens that are not KEY=VALUE extend the previous value, as in FF_CC_FLAGS=--x -O3).
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        try:
            tokens = shlex.split(line, comments=False, posix=True)
        except ValueError:
            tokens = line.split()
        _kv_tokens(tokens, out)
    return out


parse_env_file = parse_overrides   # base.env is the same KEY=VALUE format


def read_code_sha(path: str) -> Optional[str]:
    """First hex token (>= 32 chars) of code.sha256 (`<sha>  train_ff.py` or bare)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    m = re.search(r"\b([0-9a-fA-F]{32,64})\b", text)
    return m.group(1).lower() if m else None


def is_screen_name(run: str) -> bool:
    return bool(SCREEN_NAME_RE.search(run))


def _find_meta_file(run_dir: str, name: str) -> Optional[str]:
    """`_meta/<name>` or `_meta/queue/<name>` (the chip's queue dir copied verbatim) in the run dir, its chip
    dir, or the sim-data dir (first found)."""
    d = os.path.abspath(run_dir)
    for up in range(3):
        for sub in (("_meta",), ("_meta", "queue")):
            cand = os.path.join(d, *sub, name)
            if os.path.isfile(cand):
                return cand
        d = os.path.dirname(d)
    return None


def find_base_env(run_dir: str) -> Optional[str]:
    """_meta/base.env (or _meta/queue/base.env) in the run dir, its chip dir, or the sim-data dir."""
    return _find_meta_file(run_dir, "base.env")


def find_runner_log(run_dir: str) -> Optional[str]:
    """The queue's runner.log (`START <run> HH:MM:SS :: KEY=V ...` / `END <run> HH:MM:SS bpb=...`)."""
    return _find_meta_file(run_dir, "runner.log")


RUNNER_START_RE = re.compile(r"^START\s+(\S+)\s+(\d{2}:\d{2}:\d{2})\s+::\s*(.*)$")
RUNNER_END_RE = re.compile(r"^END\s+(\S+)\s+(\d{2}:\d{2}:\d{2})\s+bpb=(\S+)")
RUNNER_DATE_RE = re.compile(r"^runner start (\d{4}-\d{2}-\d{2})")
_RUNNER_CACHE: Dict[Tuple[str, float, int], Dict[str, dict]] = {}


def parse_runner_log(text: str) -> Dict[str, dict]:
    """runner.log -> {run: {"overrides": "<KEY=VALUE ...>", "start": "HH:MM:SS", "end_bpb": float|None,
    "runner_date": "YYYY-MM-DD"|None}} (the date of the most recent `runner start` line before the job)."""
    out: Dict[str, dict] = {}
    date = None
    for raw in text.splitlines():
        line = raw.strip()
        m = RUNNER_DATE_RE.match(line)
        if m:
            date = m.group(1)
            continue
        m = RUNNER_START_RE.match(line)
        if m:
            out[m.group(1)] = {"overrides": m.group(3).strip(), "start": m.group(2), "end_bpb": None,
                               "runner_date": date}
            continue
        m = RUNNER_END_RE.match(line)
        if m and m.group(1) in out:
            out[m.group(1)]["end_bpb"] = _to_float(m.group(3))
    return out


def load_runner_log(path: str) -> Dict[str, dict]:
    """parse_runner_log() of a file, cached by (path, mtime, size)."""
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime, st.st_size)
    except OSError:
        return {}
    if key not in _RUNNER_CACHE:
        txt = _read(path)
        _RUNNER_CACHE[key] = parse_runner_log(txt) if txt else {}
    return _RUNNER_CACHE[key]


def _set_model_config(log: TrainLog, body: str) -> None:
    try:
        mc = json.loads(body)
        if isinstance(mc, dict):
            log.model_config = mc
    except json.JSONDecodeError:
        log.notes.append("model_config JSON malformed")


def _fmt_num(v: object) -> str:
    """A knob-style string: 9 -> '9', 1760.0 -> '1760', 0.6 -> '0.6', True -> '1', 'fnm' -> 'fnm'."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) and abs(v) < 1e15 else ("%.10g" % v)
    if isinstance(v, (list, tuple)):
        return ",".join(_fmt_num(x) for x in v)
    return str(v).strip()


def _same_value(a: str, b: str) -> bool:
    """'0.60' == '0.6', '1760' == '1760.0', 'on' == '1'; comma lists element-wise; else string equality."""
    on_off = {"on": "1", "off": "0", "true": "1", "false": "0"}
    a, b = on_off.get(str(a).strip().lower(), str(a).strip()), on_off.get(str(b).strip().lower(), str(b).strip())
    if a == b:
        return True
    pa, pb = a.split(","), b.split(",")
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        try:
            if abs(float(x) - float(y)) > 1e-9 * max(1.0, abs(float(x))):
                return False
        except ValueError:
            if x.strip() != y.strip():
                return False
    return True


_VOCAB_CACHE: Dict[int, Dict[str, List[str]]] = {}


def knob_vocab(base_env: Optional[Dict[str, str]], queue: Optional[Dict[str, dict]]) -> Dict[str, List[str]]:
    """knob -> value strings seen in base.env and in every runner.log START job (first-seen order)."""
    key = (id(base_env), id(queue))
    if key in _VOCAB_CACHE:
        return _VOCAB_CACHE[key]
    vocab: Dict[str, List[str]] = {}
    for env in [base_env or {}] + [parse_overrides(j.get("overrides", "")) for j in (queue or {}).values()]:
        for k, v in env.items():
            lst = vocab.setdefault(k, [])
            if v not in lst:
                lst.append(v)
    _VOCAB_CACHE[key] = vocab
    return vocab


def canonical_value(knob: str, value: str, vocab: Dict[str, List[str]]) -> str:
    """The campaign's own spelling of a numerically equal value when one exists ('0.6' -> '0.60'), else value."""
    for s in vocab.get(knob, ()):
        if _same_value(s, value):
            return s
    return value


def echo_env(log: TrainLog) -> Dict[str, str]:
    """FF_* values implied by what train_ff.py echoed: the `schedule=... cooldown_fraction=...` line, the
    `ema=on ... every=32` / `async_sync=on ... fused_ce=8` lines, the model_config JSON and ff_summary.
    These are the values the run actually used, whatever the env files say."""
    s = log.summary or {}
    mc = log.model_config or {}
    cfg = log.config
    out: Dict[str, str] = {}

    def put(knob: str, val: object) -> None:
        if val is None or val == "":
            return
        out[knob] = _fmt_num(val)

    put("FF_DEPTH", s.get("depth", mc.get("n_layer")))
    put("FF_COOLDOWN_FRAC", cfg.get("cooldown_fraction", s.get("cooldown_fraction")))
    tt = cfg.get("time_target")
    put("FF_TIME_TARGET", _to_float(tt) if tt else s.get("time_target_seconds"))
    put("FF_SCHEDULE", cfg.get("schedule", s.get("schedule")))
    put("FF_MB", s.get("microbatch"))
    put("FF_TOTAL_BATCH", s.get("total_batch"))
    put("FF_ADAMW_LR_SCALE", s.get("adamw_lr_scale"))
    put("FF_MATRIX_LR_SCALE", s.get("matrix_lr_scale"))
    put("FF_SEED", s.get("seed"))
    put("FF_WARMUP", s.get("warmup_steps"))
    put("FF_MASK_BOS_TARGET", s.get("mask_bos_target"))
    acc = s.get("accum_sched") or {}
    put("FF_ACCUM_SCHED", acc.get("schedule") or log.schedule)
    put("FF_ACCUM_LR", acc.get("lr_scale"))
    mtp = s.get("mtp") or {}
    put("FF_MTP", mtp.get("mode"))
    put("FF_MTP_PHASES", mtp.get("phases"))
    put("FF_QK_GAIN", mc.get("qk_gain"))
    put("FF_KV_HEADS", mc.get("n_kv_heads"))
    put("FF_VALUE_EMBED", mc.get("value_embeds"))
    put("FF_MLP_MULT", mc.get("mlp_mult"))
    if mc.get("key_offset"):
        put("FF_KEY_OFFSET_RAD", mc.get("key_offset_rad"))
    if mc.get("mlp_leaky"):
        put("FF_LEAKY_RELU2", mc.get("mlp_leaky"))
        put("FF_LEAKY_FORM", mc.get("mlp_leaky_form"))
    ne, nh = mc.get("n_embd"), mc.get("n_head")
    if isinstance(ne, int) and isinstance(nh, int) and nh > 0 and ne % nh == 0:
        put("FF_HEAD_DIM", ne // nh)
    if cfg.get("ema") in ("on", "off"):
        put("FF_EMA", cfg["ema"] == "on")
    put("FF_EMA_EVERY", cfg.get("every"))
    put("FF_FUSED_CE", cfg.get("fused_ce"))
    if cfg.get("async_sync") in ("on", "off"):
        put("FF_ASYNC_SYNC", cfg["async_sync"] == "on")
    return out


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _date_from_epoch_files(run_dir: str) -> Optional[str]:
    for name in ("t_launch", "t_step5", "t_end", "t_eval_start"):
        txt = _read(os.path.join(run_dir, name))
        if txt:
            m = re.search(r"([0-9]{9,10})(?:\.[0-9]+)?", txt)
            if m:
                return datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc).strftime("%Y-%m-%d")
    return None


def _chip_from_path(run_dir: str) -> Optional[str]:
    parent = os.path.basename(os.path.dirname(os.path.abspath(run_dir)))
    m = re.fullmatch(r"(?i)chip[_-]?([A-D])", parent)
    return m.group(1).upper() if m else None


RUN_DIR_FILES = ("train.log", "eval.log", "eval20.log", "overrides")


def is_run_dir(path: str) -> bool:
    """A harvested run dir: not `_meta`/`_transfer`-style bookkeeping, and holds at least one run file."""
    if os.path.basename(os.path.normpath(path)).startswith("_"):
        return False
    return any(os.path.isfile(os.path.join(path, f)) for f in RUN_DIR_FILES)


def parse_run_dir(run_dir: str, base_env: Optional[Dict[str, str]] = None,
                  chip: Optional[str] = None) -> Optional[RunRecord]:
    """Build a RunRecord from a harvested run directory. Chip comes from the parent dir chipC/chipD
    unless given. base_env overrides the _meta/base.env lookup when supplied.
    Returns None for directories that are not run dirs (`_meta`, `_transfer`, empty dirs)."""
    run_dir = os.path.normpath(run_dir)
    if not is_run_dir(run_dir):
        return None
    run = os.path.basename(run_dir)
    chip = (chip or _chip_from_path(run_dir) or "?").upper()
    rec = RunRecord(run_id=f"{chip}:{run}", chip=chip, run=run, sources=["chip_log"])
    notes: List[str] = []

    train_text = _read(os.path.join(run_dir, "train.log"))
    log = parse_train_log(train_text) if train_text is not None else None
    if train_text is None:
        notes.append("no train.log")

    overrides_text = _read(os.path.join(run_dir, "overrides"))
    overrides = parse_overrides(overrides_text) if overrides_text is not None else None
    if base_env is None:
        be_path = find_base_env(run_dir)
        be_text = _read(be_path) if be_path else None
        base_env = parse_env_file(be_text) if be_text is not None else None
    rl_path = find_runner_log(run_dir)
    queue = load_runner_log(rl_path) if rl_path else None     # None: provenance undecidable (no runner.log)
    hand = os.path.isfile(os.path.join(run_dir, "launch-command.txt"))
    provenance = "unknown"
    if overrides is None and queue and run in queue:
        overrides = parse_overrides(queue[run]["overrides"])
        provenance = "queue"
        notes.append("overrides recovered from runner.log START line")
    elif overrides is not None and base_env and set(base_env) <= set(overrides):
        provenance = "queue"                # the queue runner writes base.env + job env verbatim
    elif overrides is not None and queue is not None:
        provenance = "queue" if run in queue else "driver"
    elif overrides is None and hand:
        provenance = "hand"
    knobs: Dict[str, str] = {}
    if base_env and provenance in ("queue", "unknown"):
        knobs.update(base_env)
    if overrides:
        knobs.update(overrides)
    rec.knobs_complete = bool(base_env) and overrides is not None and provenance in ("queue", "unknown")
    if overrides is None:
        notes.append("no overrides file" + ("; hand-launched (launch-command.txt): env baked into train.py" if hand else ""))
    elif provenance == "driver":
        notes.append("launcher=driver/suite (not in runner.log): base.env not applied")
    notes.append(f"env_source={provenance}")

    if log is not None:
        for k, v in log.knob_hints.items():
            knobs.setdefault(k, v)
        if queue is not None or overrides is None:   # provenance decidable: reconcile with the echo
            mism: List[str] = []
            filled: List[str] = []
            vocab = knob_vocab(base_env, queue)
            for k, v in echo_env(log).items():
                v = canonical_value(k, v, vocab)
                cur = knobs.get(k)
                if cur is None:
                    knobs[k] = v
                    filled.append(k)
                elif not _same_value(cur, v):
                    mism.append(f"{k}:{cur}->{v}")
                    knobs[k] = v
            for k in log.echo_flags:
                if k not in knobs:
                    knobs[k] = "1"
                    filled.append(k)
            if mism:
                rec.knobs_complete = False
                notes.append("env_mismatch=" + ",".join(mism))
            if filled:
                notes.append("knobs_from_echo=" + ",".join(sorted(set(filled))))
        rec.step_time = log.step_time
        rec.loss_curve = log.loss_curve
        rec.steps = log.steps
        rec.charged_seconds = log.charged_seconds
        rec.startup_seconds = log.startup_seconds
        rec.time_target = log.time_target
        notes.append(f"step_time_source={log.step_time_source}")
        if log.step_time_source == "summary":
            lm = {f"k{k}": round(getattr(log.step_time_lines, f"k{k}"), 4)
                  for k in (1, 2, 4) if getattr(log.step_time_lines, f"k{k}") is not None}
            notes.append("lines_median_by_k=" + json.dumps(lm, separators=(",", ":"), sort_keys=True))
        if log.summary:
            sd = log.summary.get("seed")
            if sd is not None and "FF_SEED" not in (overrides or {}):
                rec.seed = _to_int(sd)
            acc = log.summary.get("accum_sched") or {}
            if acc.get("steps_by_k"):
                notes.append("summary_steps_by_k=" + json.dumps(acc["steps_by_k"], separators=(",", ":"), sort_keys=True))
        if log.echo_flags:
            notes.append("echo_flags=" + ",".join(sorted(set(log.echo_flags))))
        for key in ("schedule", "cooldown_fraction", "num_params"):
            if key in log.config:
                notes.append(f"{key}={log.config[key]}")
        if log.other_k:
            notes.append("steps at k outside 1/2/4: " + json.dumps(log.other_k, sort_keys=True))
        notes.extend(log.notes)
        if not log.rows:
            notes.append("train.log has no step lines")
    if overrides and "FF_SEED" in overrides:
        rec.seed = _to_int(overrides["FF_SEED"])
    elif rec.seed is None and "FF_SEED" in knobs:
        rec.seed = _to_int(knobs["FF_SEED"])
    if rec.time_target is None and knobs.get("FF_TIME_TARGET"):
        rec.time_target = _to_float(knobs["FF_TIME_TARGET"])
    rec.knobs = knobs

    code_dir = (overrides or {}).get("FF_CODE_DIR") or knobs.get("FF_CODE_DIR")
    rec.code_version = os.path.basename(code_dir.rstrip("/")) if code_dir else "code"
    rec.code_sha = read_code_sha(os.path.join(run_dir, "code.sha256"))

    ev = _read(os.path.join(run_dir, "eval.log"))
    if ev is not None:
        rec.bpb_2m, rec.bpb_2m_ema = parse_eval_log(ev)
    ev_ema = _read(os.path.join(run_dir, "eval_ema.log"))     # the EMA weights are scored into their own file
    if ev_ema is not None:
        v, v_ema = parse_eval_log(ev_ema)
        if v is not None or v_ema is not None:
            rec.bpb_2m_ema = v if v is not None else v_ema
    ev20 = _read(os.path.join(run_dir, "eval20.log"))
    if ev20 is not None:
        rec.bpb_20m, _ = parse_eval_log(ev20)
    if log is not None and log.rows and log.summary is None and rec.bpb_2m is None:
        rec.tags.append("log_incomplete")
        ended = os.path.isfile(os.path.join(run_dir, "t_end"))
        notes.append(f"train.log has no ff_summary and no eval: {'ended' if ended else 'still running at harvest'} "
                     f"at step {log.rows[-1]['step']}")
    status = _read(os.path.join(run_dir, "full_run.status"))
    if status:
        notes.append("status=" + " ".join(status.split())[:80])
    timing = _read(os.path.join(run_dir, "timing.txt"))       # harness clock: startup_s=... charged_s=...
    if timing:
        kv = parse_overrides(timing)
        if kv:
            notes.append("timing=" + " ".join(f"{k}={v}" for k, v in kv.items()))

    # screens: FF_STEP_TOTAL / num_steps / NEURON_COMPETITION_R1_NUM_STEPS <= 200, name pattern, steps <= 200
    small_total = False
    for key in ("FF_STEP_TOTAL", "NEURON_COMPETITION_R1_NUM_STEPS"):
        v = _to_int(knobs.get(key)) if knobs.get(key) else None
        if v is not None and v <= SCREEN_MAX_STEPS:
            small_total = True
    if log is not None and log.num_steps is not None and log.num_steps <= SCREEN_MAX_STEPS:
        small_total = True
    rec.is_screen = bool(small_total or is_screen_name(run) or (rec.steps is not None and rec.steps <= SCREEN_MAX_STEPS))

    rec.date_utc = _date_from_epoch_files(run_dir)
    if rec.date_utc is None and log is not None and log.date_mmdd:
        try:
            year = datetime.fromtimestamp(os.path.getmtime(os.path.join(run_dir, "train.log")), tz=timezone.utc).year
        except OSError:
            year = datetime.now(tz=timezone.utc).year
        rec.date_utc = f"{year:04d}-{log.date_mmdd[0]:02d}-{log.date_mmdd[1]:02d}"

    m = re.search(r"(?<![A-Za-z0-9])(K\d{2,3}[A-Za-z]*)(?![0-9])", run)
    if m:
        rec.submission = m.group(1)
    if re.search(r"(?i)cold", run):
        rec.tags.append("rehearsal")
    if rec.is_screen:
        rec.tags.append("screen")
    rec.notes = " | ".join(n for n in notes if n)
    return rec
