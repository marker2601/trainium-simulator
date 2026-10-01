"""GPU proxy fleet: plan / run / report / from-search / verdict for equal-steps CUDA runs.

Route 2 of docs/simulator-spec.md.  Every run is one subprocess of
``ffsim/gpu/train_gpu.py`` (written by another agent) pinned to a GPU through
CUDA_VISIBLE_DEVICES, N runs per GPU sequential, failures recorded and never fatal.

The command each run gets (the train_gpu.py contract this file assumes)::

    python ffsim/gpu/train_gpu.py --clock virtual --step-seconds k1=K1,k2=K2,k4=K4 --steps N
        --data-world 8 --results RESULTS.jsonl --out-dir DIR --label L --arm A [config args]

* ``--clock virtual --step-seconds k1=..,k2=..,k4=..`` (named, so an arm whose FF_ACCUM_SCHED
  has fewer phases, e.g. noramp 2:0.2,4, still parses; vclock.parse_step_seconds accepts both
  forms): train.py keeps its charged-time schedule
  (FF_SCHEDULE=time), but the charged clock is virtual: charged(step) = sum over
  completed steps of the Trainium seconds of that step's accumulation phase, so every
  clock-keyed switch (accum phase 1->2->4 at 6%/20% of FF_TIME_TARGET, MTP stages,
  cooldown start, EMA levels) fires at the same step number as on Trainium.
* ``--steps N``: the equal-steps target (the Trainium pair's own step count).  train_gpu.py
  is expected to rescale the virtual clock so that charged(N) == FF_TIME_TARGET (the raw
  medians under-count per-step overhead by ~2%: K60 C41 ended at 2361 steps, the raw
  triple alone would walk to ~2410), which keeps every switch at its Trainium step fraction
  and ends the run at exactly N.
* The seed travels as env FF_SEED (train.py reads it); fleet also records it in its own
  manifest so calibrate.py never has to guess it from the results row.
* Neuron-only env is stripped loudly through ``ffsim.gpu.gpu_env.strip(env) -> (kept,
  removed)`` (imported lazily; a local fallback list is used, and announced, if that
  module is absent).

Config format (JSON): ``{"defaults": {"steps", "env", "args"}, "runs": [{label, arm,
pair, seed, env, gpu, steps}], "pairs": [{pair_id, family, treatment_arm, control_arm,
trainium: [...], gating}]}``.  ``pairs`` is read by calibrate.py; ``run`` ignores it.

Nothing here launches or touches an AWS resource.  ``prices`` only *reads* the public
pricing API (aws pricing get-products, us-east-1 endpoint) and ``plan`` spends nothing.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN_GPU = HERE / "train_gpu.py"
CONFIG_DIR = HERE / "configs"
DEFAULT_MODELS = REPO / "research" / "sim-data" / "models.pkl"
DEFAULT_SEARCH_GLOB = str(REPO / "research" / "sim-data" / "search-*.csv")

K60_TIME_TARGET = 1793.0
K60_ACCUM_FRACTIONS = (0.06, 0.20)       # FF_ACCUM_SCHED=1:0.06,2:0.2,4
K60_STEPS = 2361                          # C41_cold_K60 rehearsal, chip C
K60_FLOPS_PER_TOKEN = 7.581204e8          # train.log flops_per_token of the K60 code (PROTOCOL.md 2.1)
MICRO_TOKENS = 65_536                     # tokens per micro-batch: a k-step consumes k x 65,536
SCHEDULE_K60 = CONFIG_DIR / "schedule-K60-TT1793.json"
CHIP_LOG_DIR = REPO / "research" / "sim-data" / "chipC"
# The K60 triple (chip-C seconds per optimizer step by accumulation phase) is DERIVED from a harvested
# log's own switch lines (vclock.phase_seconds_from_log), never typed in, and lives in SCHEDULE_K60
# together with the walk it produces (`python -m ffsim.gpu.fleet schedule --write`).  The reference is
# C41_cold_K60 (the K60 rehearsal, 2,361 steps); while its train.log is not harvested the fallback is
# C40_cold_K59: K59 on the same master M14, and AS-a leaves the step time unchanged (PROTOCOL.md 2.2).
K60_REFERENCE_LOGS = ("C41_cold_K60", "C40_cold_K59")
K60_REFERENCE = {"run": "C41_cold_K60", "steps": K60_STEPS, "bpb": 0.95910, "charged_seconds": 1790.5,
                 "startup_seconds": 497.3, "time_target": K60_TIME_TARGET}
# Hand-assembled medians, used ONLY if neither SCHEDULE_K60 nor a reference log exists: k1/k2
# 1639_R_m14asa_s73 (0.2963 / 0.5057, n=29/26); k4 lk35 s73 full run 0.9279, S60 ASa screen 0.9274.
FALLBACK_STEP_SECONDS = (0.295, 0.506, 0.928)

# Neuron-only env, used ONLY if ffsim.gpu.gpu_env is missing.  Deliberately does not touch
# FF_SCHEDULE / FF_TIME_TARGET: the virtual clock needs them.
FALLBACK_NEURON_ONLY = (
    "FF_CC_FLAGS", "FF_NKI_ATTN", "FF_BF16_PARAMS", "FF_PROFILE_TRACE", "FF_COMPILE_OPT",
    "FF_CODE_DIR", "FF_ASYNC_SYNC", "FF_XU_QUEUE",
)
FALLBACK_NEURON_PREFIXES = ("NEURON_", "XLA_", "NEURONX_")
# Mirrors gpu_env.ALLOWED_NEURON: the NEURON_COMPETITION_R1_* names prepare.py reads for data /
# tokenizer / output locations.  launch.sh push puts the shards under $NEURON_COMPETITION_R1_CACHE_DIR
# (~/ff/env.sh); stripping it from the inherited env makes every run fail at data load.
FALLBACK_ALLOWED_NEURON = (
    "NEURON_COMPETITION_R1_CACHE_DIR", "NEURON_COMPETITION_R1_TOKENIZER_PATH",
    "NEURON_COMPETITION_R1_TOKENIZER_URL", "NEURON_COMPETITION_R1_TOKENIZER_SHA256",
    "NEURON_COMPETITION_R1_TOKENIZER_PKL_SHA256", "NEURON_COMPETITION_R1_TOKEN_BYTES_SHA256",
    "NEURON_COMPETITION_R1_OUT_DIR", "NEURON_COMPETITION_R1_EVAL_PUBLIC", "NEURON_COMPETITION_R1_EVAL_TOKENS",
)

# On-demand Linux prices, USD per instance-hour.  ``live``: aws pricing get-products,
# us-east-1 endpoint, location "US East (N. Virginia)", read 29 Sep 2026 from our own account.
# ``quote_20sep``: the 20 Sep us-west-2 list quotes (see PROTOCOL.md, which supersedes the unpublished
# 20 Sep proxy protocol).
INSTANCES = (
    # name, GPUs, live price, 20-Sep quote
    ("g6e.xlarge", 1, 1.861, 1.861),
    ("g6e.12xlarge", 4, 10.49264, 10.4912),
    ("g6e.48xlarge", 8, 30.13118, 30.1312),
)
PRICE_SOURCE = "aws pricing get-products (us-east-1 endpoint, US East (N. Virginia), 29 Sep 2026)"
PRICE_SOURCE_FALLBACK = "20-Sep us-west-2 list quotes (PROTOCOL.md)"

EVAL_SECONDS = 210.0        # 2,097,152 val tokens at eval batch 1 (protocol: 3.5 min)
STARTUP_SECONDS = 60.0      # import, model build, first parquet read
SETUP_SECONDS = 1800.0      # once per instance: pip, shards, tokenizer


# ----------------------------------------------------------------------------- configs

def load_config(path: Path) -> Dict[str, Any]:
    """Expand defaults into every run; refuse duplicate labels (they name run dirs and rows)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"runs": raw}
    defaults = raw.get("defaults", {})
    runs: List[Dict[str, Any]] = []
    seen = set()
    for i, r in enumerate(raw.get("runs", [])):
        env = {**defaults.get("env", {}), **r.get("env", {})}
        seed = r.get("seed")
        if seed is None and "FF_SEED" in env:
            seed = int(env["FF_SEED"])
        if seed is None:
            seed = int(defaults.get("seed", 73))
        env["FF_SEED"] = str(seed)
        cfg = {
            "label": r.get("label") or f"run{i:03d}",
            "arm": r.get("arm", ""),
            "pair": r.get("pair", ""),
            "seed": int(seed),
            "steps": int(r.get("steps", defaults.get("steps", K60_STEPS))),
            "env": {k: str(v) for k, v in env.items()},
            "args": list(defaults.get("args", [])) + list(r.get("args", [])),
        }
        if r.get("gpu") is not None:
            cfg["gpu"] = int(r["gpu"])
        if cfg["label"] in seen:
            raise SystemExit(f"duplicate label {cfg['label']!r} in {path}")
        seen.add(cfg["label"])
        runs.append(cfg)
    return {"defaults": defaults, "runs": runs, "pairs": list(raw.get("pairs", [])),
            "meta": {k: v for k, v in raw.items() if k.startswith("_")}}


def load_runs(path: Path) -> List[Dict[str, Any]]:
    return load_config(path)["runs"]


# ----------------------------------------------------------------------------- env strip

def strip_env(env: Dict[str, str]) -> Tuple[Dict[str, str], List[str], str]:
    """(kept, removed, how).  Lazy import so gpu_env can be written concurrently (import_module,
    not ``from ffsim.gpu import``, so a sys.modules override is honoured even after a real import)."""
    try:
        import importlib
        gpu_env = importlib.import_module("ffsim.gpu.gpu_env")
        kept, removed = gpu_env.strip(dict(env))
        return {k: str(v) for k, v in kept.items()}, [str(x) for x in removed], "ffsim.gpu.gpu_env.strip"
    except ImportError:
        kept, removed = {}, []
        for k, v in env.items():
            if k in FALLBACK_ALLOWED_NEURON:
                kept[k] = str(v)
            elif k in FALLBACK_NEURON_ONLY or k.startswith(FALLBACK_NEURON_PREFIXES):
                removed.append(f"{k}={v}")
            else:
                kept[k] = str(v)
        return kept, removed, "FALLBACK list (ffsim.gpu.gpu_env not importable)"


# ----------------------------------------------------------------------------- step seconds

def reference_log() -> Optional[Path]:
    """First harvested K60 reference log (C41_cold_K60, else C40_cold_K59), or None."""
    for run in K60_REFERENCE_LOGS:
        cand = CHIP_LOG_DIR / run / "train.log"
        if cand.exists():
            return cand
    return None


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def derive_step_seconds(log: Path) -> Tuple[Tuple[float, float, float], str]:
    """(k1, k2, k4) per-phase MEAN seconds from a chip log's switch lines (vclock.phase_seconds_from_log),
    rounded to 6 decimals so the JSON round-trips exactly; plus the provenance string."""
    from ffsim.gpu import vclock
    means = vclock.phase_seconds_from_log(str(log))
    if sorted(means) != [1, 2, 4]:
        raise ValueError(f"{log}: phases {sorted(means)} are not the K60 1/2/4 accumulation phases")
    trip = tuple(round(float(means[k]), 6) for k in (1, 2, 4))
    return trip, f"phase_seconds_from_log({_rel(log)}): per-phase mean seconds between the FF_ACCUM_SCHED switch lines"


def _default_step_seconds() -> Tuple[Tuple[float, float, float], str]:
    """The schedule table's triple (single source), else derived from the reference log, else FALLBACK."""
    try:
        d = json.loads(SCHEDULE_K60.read_text(encoding="utf-8"))
        s = d["step_seconds_input"]
        return (float(s["1"]), float(s["2"]), float(s["4"])), f"{SCHEDULE_K60.name}: {d['step_seconds_source']}"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    ref = reference_log()
    if ref is not None:
        try:
            return derive_step_seconds(ref)
        except (OSError, ValueError, ImportError):
            pass
    return FALLBACK_STEP_SECONDS, "FALLBACK hand-assembled K60 lineage chip C medians (no schedule table, no reference log)"


DEFAULT_STEP_SECONDS, DEFAULT_STEP_SECONDS_SOURCE = _default_step_seconds()


def schedule_table(step_seconds: Sequence[float], steps: int = K60_STEPS, source: str = "",
                   reference: Optional[Path] = None) -> Dict[str, Any]:
    """The schedule train_gpu.py actually walks for ``--clock virtual --steps N --step-seconds k1,k2,k4``:
    vclock.rescale_to_steps (one factor on the triple so the clock-driven stop lands on N, every switch at
    its charged fraction of FF_TIME_TARGET) with the stop pinned at N.  Steps by k, tokens and FLOPs follow
    from the walk; nothing here is hand-computed.  JSON-serialisable (int keys become strings)."""
    from ffsim.gpu import vclock
    k1, k2, k4 = (float(v) for v in step_seconds)
    recipe = vclock.Recipe()
    scaled, scale, walk = vclock.rescale_to_steps(recipe, {1: k1, 2: k2, 4: k4}, int(steps))
    by_k = {str(k): int(n) for k, n in sorted(walk["steps_by_k"].items())}
    tokens_by_k = {k: n * int(k) * MICRO_TOKENS for k, n in by_k.items()}
    tokens = sum(tokens_by_k.values())
    all_k4 = int(steps) * 4 * MICRO_TOKENS
    levels = {int(e[0]): e for e in walk["levels"]}
    ema_level = max((lv for lv in levels if lv <= 3), default=None)
    ref_run = reference.parent.name if reference is not None else None
    provisional = ref_run != K60_REFERENCE["run"]
    return {
        "_comment": "K60 pinned schedule on the GPU proxy's virtual charged clock. GENERATED by "
                    "`python -m ffsim.gpu.fleet schedule --write`; tests/test_ffsim_gpu_fleet.py regenerates it and pins "
                    "PROTOCOL.md 2.2 / README.md to it. Do not hand-edit: re-run the command after a harvest.",
        "recipe": "K60 (vclock.Recipe defaults: FF_TIME_TARGET %.0f, warmup %d, FF_ACCUM_SCHED %s, MTP phases %s, "
                  "cooldown %.2f %s, insurance at %s, startup excluded %d)" % (
                      recipe.time_target, recipe.warmup, recipe.accum_sched, list(recipe.mtp_phases),
                      recipe.cooldown_fraction, recipe.cooldown_shape, list(recipe.insurance_at), recipe.startup_excluded),
        "vclock_version": vclock.VCLOCK_VERSION,
        "reference": dict(K60_REFERENCE, log=(_rel(reference) if reference is not None else None),
                          derived_from=ref_run, provisional=provisional,
                          note=("the K60 rehearsal's own train.log is harvested: the triple is K60's" if not provisional else
                                "PROVISIONAL: research/sim-data/chipC has no C41_cold_K60/train.log (only its step count and "
                                "charged seconds are known, from the monitor), so the triple comes from %s. After "
                                "`python -m ffsim harvest --chip C` re-run `python -m ffsim.gpu.fleet schedule --write`." % ref_run)),
        "step_seconds_input": {"1": k1, "2": k2, "4": k4},
        "step_seconds_source": source,
        "steps": int(steps),
        "scale": round(float(scale), 9),
        "step_seconds_scaled": {str(k): round(float(v), 6) for k, v in sorted(scaled.items())},
        "charged_final": walk["charged_final"],
        "stop_reason": walk["stop_reason"],
        "accum": walk["accum"],
        "mtp": walk["mtp"],
        "levels": walk["levels"],
        "insurance": walk["insurance"],
        "phase2_step": walk["accum"][1][1] if len(walk["accum"]) > 1 else None,
        "phase3_step": walk["accum"][2][1] if len(walk["accum"]) > 2 else None,
        "cooldown_start_step": walk["levels"][0][1] if walk["levels"] else None,
        "ema_save_level_step": levels[ema_level][1] if ema_level is not None else None,
        "level1_step": levels[1][1] if 1 in levels else None,
        "final_level": walk["final_level"],
        "steps_by_k": by_k,
        "tokens_by_k": tokens_by_k,
        "tokens": tokens,
        "all_k4_tokens": all_k4,
        "tokens_fraction_of_all_k4": round(tokens / all_k4, 4),
        "flops_per_token": K60_FLOPS_PER_TOKEN,
        "flops": K60_FLOPS_PER_TOKEN * tokens,
    }


def format_schedule(t: Dict[str, Any]) -> str:
    s = t["step_seconds_input"]
    lines = [f"K60 pinned schedule, virtual clock (train_gpu.py --clock virtual --steps {t['steps']} "
             f"--step-seconds k1={s['1']:.6f},k2={s['2']:.6f},k4={s['4']:.6f}):",
             f"  triple: {t['step_seconds_source']}",
             f"  reference: {t['reference']['derived_from']}"
             + (" [PROVISIONAL, C41_cold_K60 not harvested]" if t["reference"]["provisional"] else ""),
             f"  clock x{t['scale']:.6f} so the clock-driven stop lands on {t['steps']}; then pinned there "
             f"({t['stop_reason']}); virtual charged at the stop {t['charged_final']:.1f} s",
             f"  {'event':<36}{'step':>6}   virtual charged"]
    labels = {1: "phase 1 (k=1, LR x 0.5)", 2: "phase 2 (k=2, LR x 0.7071)", 3: "phase 3 (k=4) = MTP stage 4"}
    for ph, step, charged, k in t["accum"]:
        lines.append(f"  {labels.get(ph, 'phase %d (k=%d)' % (ph, k)):<36}{step:>6}   {charged:.1f} s")
    mtp = {m[0]: m[1] for m in t["mtp"]}
    lines.append("  %-36s%s" % ("MTP stages 1,2,3 / 5,6,7,8", ", ".join(str(mtp.get(i, "-")) for i in (1, 2, 3))
                                 + " / " + ", ".join(str(mtp.get(i, "-")) for i in (5, 6, 7, 8))))
    lv = {e[0]: e for e in t["levels"]}
    for name, key in (("cooldown level 19 (start)", 19), ("EMA save level <= 3", None), ("level 1 (floor)", 1)):
        e = lv.get(key) if key is not None else next((e for e in t["levels"] if e[0] <= 3), None)
        if e:
            lines.append(f"  {name:<36}{e[1]:>6}   {e[2]:.1f} s")
    lines.append(f"  {'stop (pinned)':<36}{t['steps']:>6}   {t['charged_final']:.1f} s")
    by = t["steps_by_k"]
    lines.append("  steps by k: " + ", ".join(f"k{k} {by[k]:,}" for k in sorted(by, key=int))
                 + f" (k4 incl. the warm-up steps) -> {t['tokens']:,} tokens "
                 f"({100 * t['tokens_fraction_of_all_k4']:.1f}% of all-k4 {t['all_k4_tokens']:,}); "
                 f"{t['flops']:.4e} FLOPs at {t['flops_per_token']:.6e}/token")
    return "\n".join(lines)


def parse_step_seconds(spec: str, chip: str = "C", code_version: str = "M14",
                       knobs: Optional[Dict[str, str]] = None) -> Tuple[Tuple[float, float, float], str]:
    """'k1,k2,k4' triple, or 'models.pkl' / 'models:<path>[:<code_version>]' -> steptime.predict."""
    spec = (spec or "").strip()
    if not spec or spec == "default":
        return DEFAULT_STEP_SECONDS, DEFAULT_STEP_SECONDS_SOURCE
    parts = spec.split(",")
    if len(parts) == 3:
        try:
            k1, k2, k4 = (float(p) for p in parts)
            return (k1, k2, k4), "explicit triple"
        except ValueError:
            pass
    path = spec
    if spec.startswith("models:"):
        rest = spec[len("models:"):]
        if ":" in rest and not rest[1:3] == ":\\":
            path, code_version = rest.rsplit(":", 1)
        else:
            path = rest
    if path == "models.pkl":
        path = str(DEFAULT_MODELS)
    from ffsim.simulate import load_models  # numpy only
    models = load_models(path)
    pred = models.steptime.predict(code_version, knobs or {}, chip)
    trip = (float(pred["k1"]), float(pred["k2"]), float(pred["k4"]))
    return trip, f"models.pkl steptime.predict({code_version}, chip {chip}); {pred.get('notes') or 'no notes'}"


def virtual_steps(step_seconds: Sequence[float], time_target: float = K60_TIME_TARGET,
                  fractions: Sequence[float] = K60_ACCUM_FRACTIONS, excluded_steps: int = 5) -> int:
    """Steps the raw triple walks to on the virtual clock (no overhead).  Diagnostic only:
    train_gpu.py rescales the clock to the --steps target."""
    k1, k2, k4 = step_seconds
    t1 = time_target * fractions[0]
    t2 = time_target * fractions[1]
    n = t1 / k1 + (t2 - t1) / k2 + (time_target - t2) / k4
    return int(round(n)) + excluded_steps


# ----------------------------------------------------------------------------- commands

def build_command(cfg: Dict[str, Any], python: str, results: Path, out_root: Path,
                  step_seconds: Sequence[float], extra_args: Sequence[str] = ()) -> List[str]:
    run_dir = out_root / cfg["label"]
    args = list(cfg["args"])
    cmd = [python, str(TRAIN_GPU),
           "--clock", "virtual",
           "--step-seconds", ",".join(f"k{k}={s:.4f}" for k, s in zip((1, 2, 4), step_seconds)),
           "--steps", str(cfg["steps"]),
           "--out-dir", str(run_dir / "out"),
           "--results", str(results),
           "--label", cfg["label"],
           "--arm", cfg["arm"]]
    if "--data-world" not in args:
        cmd += ["--data-world", "8"]
    return cmd + args + list(extra_args)


def gpu_count() -> int:
    n = os.environ.get("FLEET_GPUS")
    if n:
        return int(n)
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=30)
        return max(1, len([ln for ln in out.stdout.splitlines() if ln.strip()]))
    except Exception:
        return 1


def skip_reason(cfg: Dict[str, Any], ngpu: int) -> str:
    """Non-empty when the config is pinned to a GPU the fleet does not have."""
    want = cfg.get("gpu")
    if want is not None and int(want) >= max(1, ngpu):
        return f"needs gpu {int(want)} (fleet has {max(1, ngpu)} GPU(s); PROTOCOL 3.2: a pinned null runs only on a box with that GPU)"
    return ""


def assign_lanes(runs: List[Dict[str, Any]], ngpu: int,
                 skipped: Optional[List[Dict[str, Any]]] = None) -> List[List[Dict[str, Any]]]:
    """Round robin, honouring explicit gpu pins (nulls need same/different physical GPUs).
    A config pinned to a GPU the fleet lacks is never folded onto another GPU (repeat2, the
    cross-device null, must not run on gpu 0 and be read as the across-GPU null): it is dropped
    from every lane and appended to ``skipped`` when a list is given."""
    lanes: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, ngpu))]
    free = 0
    for cfg in runs:
        want = cfg.get("gpu")
        if want is None:
            lanes[free % len(lanes)].append(cfg)
            free += 1
        elif skip_reason(cfg, ngpu):
            if skipped is not None:
                skipped.append(cfg)
        else:
            lanes[int(want)].append(cfg)
    return lanes


def record_skipped(cfg: Dict[str, Any], reason: str, manifest: Path) -> Dict[str, Any]:
    """Manifest row for a run the fleet refused to place; no results row is written, so
    launch.sh's resume filter re-queues it (and skips it again) until a box has the GPU."""
    record = {"label": cfg["label"], "arm": cfg["arm"], "pair": cfg.get("pair", ""), "seed": cfg["seed"],
              "steps": cfg["steps"], "gpu": cfg.get("gpu"), "status": f"skipped: {reason}", "returncode": None,
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def run_one(cfg: Dict[str, Any], gpu: int, out_root: Path, results: Path, manifest: Path,
            python: str, step_seconds: Sequence[float], extra_args: Sequence[str],
            dry: bool = False, runner=None) -> Dict[str, Any]:
    run_dir = out_root / cfg["label"]
    env, removed, how = strip_env(cfg["env"])
    cmd = build_command(cfg, python, results, out_root, step_seconds, extra_args)
    record: Dict[str, Any] = {
        "label": cfg["label"], "arm": cfg["arm"], "pair": cfg.get("pair", ""),
        "seed": cfg["seed"], "steps": cfg["steps"], "gpu": gpu,
        "cmd": " ".join(shlex.quote(c) for c in cmd), "env": env,
        "stripped": removed, "strip_how": how, "step_seconds": list(step_seconds),
    }
    if dry:
        return {**record, "status": "planned"}
    # The inherited environment goes through the same classifier as the config env: gpu_env keeps
    # the prepare.py allowlist (NEURON_COMPETITION_R1_CACHE_DIR etc., where launch.sh push put the
    # shards) and removes the Neuron-only rest; the config env always wins over the inherited one.
    inherited, inherited_removed, _ = strip_env({k: v for k, v in os.environ.items() if k not in env})
    full_env = {**inherited, **env, "CUDA_VISIBLE_DEVICES": str(gpu)}
    full_env.setdefault("OMP_NUM_THREADS", "4")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cmd.txt").write_text(record["cmd"] + "\n", encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")
    log = run_dir / "train.log"
    t0 = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as fh:
            fh.write(f"# fleet: env strip via {how}\n")
            if removed:
                fh.write(f"# fleet stripped Neuron-only env: {', '.join(removed)}\n")
            if inherited_removed:
                fh.write(f"# fleet stripped inherited Neuron-only env: {', '.join(inherited_removed)}\n")
            fh.flush()
            if runner is not None:
                rc = int(runner(cmd, full_env, fh))
            else:
                rc = subprocess.run(cmd, env=full_env, stdout=fh, stderr=subprocess.STDOUT).returncode
        status = "ok" if rc == 0 else f"exit {rc}"
    except Exception as exc:                           # a missing interpreter, a killed box
        rc, status = -1, f"error {type(exc).__name__}: {exc}"
    record.update({"status": status, "returncode": rc, "seconds": round(time.monotonic() - t0, 1),
                   "log": str(log), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def cmd_run(args) -> int:
    cfg = load_config(Path(args.configs))
    runs = cfg["runs"]
    out_root = Path(args.out_root)
    results = Path(args.results)
    manifest = Path(args.manifest) if args.manifest else results.with_name(results.stem + "-manifest.jsonl")
    step_seconds, how = parse_step_seconds(args.step_seconds)
    ngpu = args.gpus or gpu_count()
    skipped: List[Dict[str, Any]] = []
    lanes = assign_lanes(runs, ngpu, skipped)
    print(f"fleet: {len(runs)} runs over {ngpu} GPU(s), {args.per_gpu} per GPU sequential-per-slot; "
          f"results -> {results}; manifest -> {manifest}")
    print(f"step-seconds k1,k2,k4 = {step_seconds} ({how}); raw virtual walk = "
          f"{virtual_steps(step_seconds)} steps at TT {K60_TIME_TARGET:.0f}s (train_gpu.py rescales to --steps)")
    for g, lane in enumerate(lanes):
        print(f"  gpu {g}: {len(lane)} runs: {', '.join(c['label'] for c in lane)}")
    for c in skipped:
        print(f"  SKIPPED: {c['label']} (arm={c['arm']}) {skip_reason(c, ngpu)}", flush=True)
    if args.dry_run:
        for g, lane in enumerate(lanes):
            for c in lane:
                rec = run_one(c, g, out_root, results, manifest, args.python, step_seconds, args.extra, dry=True)
                print(f"\n[gpu {g}] {rec['label']}  arm={rec['arm']} seed={rec['seed']} steps={rec['steps']}")
                if rec["stripped"]:
                    print(f"  stripped Neuron-only env ({rec['strip_how']}): {', '.join(rec['stripped'])}")
                print(f"  env: " + " ".join(f"{k}={v}" for k, v in sorted(rec["env"].items())))
                print(f"  {rec['cmd']}")
        return 0
    if not TRAIN_GPU.exists():
        print(f"fleet: {TRAIN_GPU} does not exist yet; nothing run", file=sys.stderr)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)
    for c in skipped:
        record_skipped(c, skip_reason(c, ngpu), manifest)
    done: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(gpu: int, lane: List[Dict[str, Any]], slot: int) -> None:
        for c in lane[slot::args.per_gpu]:
            rec = run_one(c, gpu, out_root, results, manifest, args.python, step_seconds, args.extra)
            with lock:
                done.append(rec)
                mark = "ok  " if rec["status"] == "ok" else "FAIL"
                print(f"[{mark}] gpu {gpu} {rec['label']} in {rec.get('seconds', 0):.0f}s"
                      f"{'' if rec['status'] == 'ok' else '  -> ' + rec['status'] + ' (' + rec.get('log', '') + ')'}",
                      flush=True)

    threads = []
    for g, lane in enumerate(lanes):
        for slot in range(args.per_gpu):
            th = threading.Thread(target=worker, args=(g, lane, slot), daemon=False)
            th.start()
            threads.append(th)
    for th in threads:
        th.join()
    failed = [r for r in done if r["status"] != "ok"]
    print(f"\nfleet done: {len(done) - len(failed)} ok, {len(failed)} failed, {len(skipped)} skipped (pinned to an absent GPU)")
    for r in failed:
        print(f"  FAILED {r['label']}: {r['status']} (log: {r.get('log')})")
    for c in skipped:
        print(f"  SKIPPED {c['label']}: {skip_reason(c, ngpu)}")
    return 1 if failed else 0


# ----------------------------------------------------------------------------- plan / cost

def cost_table(runs: List[Dict[str, Any]], seconds_per_step: float, price_source: str = "live",
               prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """GPU-hours and dollars, no MFU guesswork: the caller supplies seconds per optimizer step."""
    per_run_s = [c["steps"] * seconds_per_step + EVAL_SECONDS + STARTUP_SECONDS for c in runs]
    gpu_hours = sum(per_run_s) / 3600.0
    by_steps: Dict[int, int] = {}
    for c in runs:
        by_steps[c["steps"]] = by_steps.get(c["steps"], 0) + 1
    rows = []
    for name, ngpu, live, quote in INSTANCES:
        price = (prices or {}).get(name)
        if price is None:
            price = live if price_source == "live" else quote
        # Longest lane after round robin: ceil(runs / GPUs) of the mean per-run time.
        waves = math.ceil(len(runs) / ngpu) if runs else 0
        wall_h = (SETUP_SECONDS + waves * (sum(per_run_s) / max(1, len(per_run_s)))) / 3600.0
        rows.append({"instance": name, "gpus": ngpu, "usd_per_hour": price, "waves": waves,
                     "wall_hours": round(wall_h, 2), "usd": round(wall_h * price, 2),
                     "usd_per_run_marginal": round((per_run_s and max(per_run_s) or 0) / 3600.0 * price / ngpu, 2)})
    return {"runs": len(runs), "steps_total": sum(c["steps"] for c in runs), "by_steps": by_steps,
            "seconds_per_step": seconds_per_step, "gpu_hours": round(gpu_hours, 2),
            "per_run_hours_max": round(max(per_run_s) / 3600.0, 3) if per_run_s else 0.0,
            "instances": rows, "price_source": PRICE_SOURCE if price_source == "live" else PRICE_SOURCE_FALLBACK}


def cmd_plan(args) -> int:
    cfg = load_config(Path(args.configs))
    runs = cfg["runs"]
    step_seconds, how = parse_step_seconds(args.step_seconds)
    print(f"{len(runs)} runs from {args.configs}")
    print(f"virtual clock: k1,k2,k4 = {step_seconds} ({how}); raw walk {virtual_steps(step_seconds)} steps "
          f"at TT {K60_TIME_TARGET:.0f}s; train_gpu.py rescales to each run's --steps")
    print()
    print(format_schedule(schedule_table(step_seconds, K60_STEPS, how, reference_log())))
    for n in sorted({int(c["steps"]) for c in runs} - {K60_STEPS}):
        t = schedule_table(step_seconds, n, how, reference_log())
        print(f"  --steps {n}: phase 2 at {t['phase2_step']}, phase 3 at {t['phase3_step']}, cooldown at "
              f"{t['cooldown_start_step']}, {t['tokens']:,} tokens (clock x{t['scale']:.6f})")
    print(f"\n{'label':<22}{'arm':<10}{'pair':<22}{'seed':>5}{'steps':>7}{'gpu':>4}  env (on top of K60 defaults)")
    for c in runs:
        env = {k: v for k, v in c["env"].items() if k != "FF_SEED"}
        print(f"{c['label']:<22}{c['arm']:<10}{c.get('pair', ''):<22}{c['seed']:>5}{c['steps']:>7}"
              f"{str(c.get('gpu', '-')):>4}  " + (" ".join(f"{k}={v!r}" for k, v in sorted(env.items())) or "(base)"))
    if cfg["pairs"]:
        print(f"\n{len(cfg['pairs'])} pairs: " + ", ".join(
            f"{p['pair_id']}[{p.get('family', '')}: {p['treatment_arm']} vs {p['control_arm']}"
            f"{'' if p.get('gating', True) else ', non-gating'}]" for p in cfg["pairs"]))
    table = cost_table(runs, args.seconds_per_step, "quote" if args.quote_prices else "live")
    print(f"\ncost at {args.seconds_per_step:.3f} s/step on the GPU (+{EVAL_SECONDS / 60:.1f} min eval "
          f"+{STARTUP_SECONDS / 60:.0f} min startup per run; {SETUP_SECONDS / 60:.0f} min setup once):")
    print(f"  {table['steps_total']:,} optimizer steps, {table['gpu_hours']:.2f} GPU-hours "
          f"(max per run {table['per_run_hours_max']:.2f} h)")
    print(f"  prices: {table['price_source']}")
    print(f"  {'instance':<14}{'GPUs':>5}{'$/h':>10}{'waves':>7}{'wall h':>9}{'USD':>10}{'$/run marg.':>13}")
    for r in table["instances"]:
        print(f"  {r['instance']:<14}{r['gpus']:>5}{r['usd_per_hour']:>10.4f}{r['waves']:>7}"
              f"{r['wall_hours']:>9.2f}{r['usd']:>10.2f}{r['usd_per_run_marginal']:>13.2f}")
    print("\n  s/step is an ESTIMATE until one run is measured (protocol section 5: the L40S dense bf16 figure "
          "is 181 TFLOP/s; a 768-wide eager model at 20% MFU is ~4.5 s/step for 262k tokens); "
          "re-plan from the first run's median_step_seconds.  Nothing above was launched.")
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": runs, "cost": table}, indent=1), encoding="utf-8")
    return 0


def cmd_schedule(args) -> int:
    if args.step_seconds:
        step_seconds, how = parse_step_seconds(args.step_seconds)
        ref = reference_log()
    else:
        ref = reference_log() if args.from_log == "auto" else Path(args.from_log)
        if ref is None or not ref.exists():
            print(f"no reference log: {args.from_log} (harvest research/sim-data/chipC/{K60_REFERENCE_LOGS[0]}/train.log "
                  f"with `python -m ffsim harvest --chip C`, or pass --step-seconds)", file=sys.stderr)
            return 2
        step_seconds, how = derive_step_seconds(ref)
    table = schedule_table(step_seconds, args.steps, how, ref)
    if args.json:
        print(json.dumps(table, indent=1))
    else:
        print(format_schedule(table))
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


def fetch_prices(profile: str = os.environ.get("AWS_PROFILE", "default"), region_endpoint: str = "us-east-1",
                 location: str = "US East (N. Virginia)") -> Dict[str, Any]:
    """Read-only: aws pricing get-products for the three g6e sizes.  Never launches anything."""
    out: Dict[str, Any] = {"source": f"aws pricing get-products endpoint {region_endpoint}, {location}", "prices": {}}
    for name, _n, _live, _q in INSTANCES:
        cmd = ["aws", "pricing", "get-products", "--service-code", "AmazonEC2", "--region", region_endpoint,
               "--profile", profile, "--output", "json", "--max-results", "5", "--filters",
               f"Type=TERM_MATCH,Field=instanceType,Value={name}",
               f"Type=TERM_MATCH,Field=location,Value={location}",
               "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
               "Type=TERM_MATCH,Field=tenancy,Value=Shared",
               "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
               "Type=TERM_MATCH,Field=capacitystatus,Value=Used"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            data = json.loads(res.stdout)
            for pl in data.get("PriceList", []):
                p = json.loads(pl)
                for term in p["terms"].get("OnDemand", {}).values():
                    for dim in term["priceDimensions"].values():
                        out["prices"][name] = float(dim["pricePerUnit"]["USD"])
        except Exception as exc:
            out.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"
    return out


def cmd_prices(args) -> int:
    res = fetch_prices(args.profile)
    print(json.dumps(res, indent=1))
    if len(res["prices"]) < len(INSTANCES):
        print(f"\nlookup incomplete; fallback = {PRICE_SOURCE_FALLBACK}: "
              + ", ".join(f"{n} ${q}" for n, _g, _l, q in INSTANCES))
        return 1
    return 0


# ----------------------------------------------------------------------------- report

def read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """(rows, n_bad_lines).  Partial / truncated files are the normal case mid-fleet."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    p = Path(path)
    if not p.exists():
        return rows, 0
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    return rows, bad


def merge_results(results: Path, manifest: Optional[Path] = None,
                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """label -> row with arm / seed / steps / val_bpb / status resolved from results.jsonl,
    the fleet manifest and the config (in that priority for values, last row wins per file)."""
    rows, _ = read_jsonl(results)
    man_rows, _ = read_jsonl(manifest) if manifest else ([], 0)
    by_cfg = {c["label"]: c for c in (config or {}).get("runs", [])}
    out: Dict[str, Dict[str, Any]] = {}
    for m in man_rows:
        out.setdefault(m.get("label", ""), {}).update(
            {"label": m.get("label"), "arm": m.get("arm"), "pair": m.get("pair"), "seed": m.get("seed"),
             "steps": m.get("steps"), "status": m.get("status"), "gpu": m.get("gpu"), "fleet_seconds": m.get("seconds")})
    for r in rows:
        label = r.get("label", "")
        cfg = r.get("config", {}) if isinstance(r.get("config"), dict) else {}
        seed = r.get("seed", cfg.get("seed"))
        if seed is None and isinstance(r.get("env"), dict):
            seed = r["env"].get("FF_SEED")
        row = out.setdefault(label, {"label": label})
        row.update({k: v for k, v in {"arm": r.get("arm"), "steps": r.get("steps"), "val_bpb": r.get("val_bpb"),
                                      "median_step_seconds": r.get("median_step_seconds"),
                                      "eval_denominator_ok": r.get("eval_denominator_ok"),
                                      "seconds": r.get("seconds")}.items() if v is not None})
        if seed is not None:
            row["seed"] = int(seed)
        if r.get("val_bpb") is not None:
            row["status"] = "ok"
        elif row.get("status") in (None, "ok"):
            row["status"] = r.get("status") or "no val_bpb"
    for label, row in out.items():
        c = by_cfg.get(label)
        if c:
            for k in ("arm", "pair", "seed", "steps"):
                row.setdefault(k, c.get(k))
        if row.get("seed") is None:
            import re
            m = re.search(r"_s(\d+)(?:_|$)", label)
            if m:
                row["seed"] = int(m.group(1))
    for label, c in by_cfg.items():
        if label not in out:
            out[label] = {"label": label, "arm": c["arm"], "pair": c.get("pair"), "seed": c["seed"],
                          "steps": c["steps"], "status": "pending"}
    return out


def cmd_report(args) -> int:
    config = load_config(Path(args.configs)) if args.configs else None
    results = Path(args.results)
    manifest = Path(args.manifest) if args.manifest else results.with_name(results.stem + "-manifest.jsonl")
    _, bad = read_jsonl(results)
    index = merge_results(results, manifest, config)
    if not index:
        print("no results yet")
        return 0
    if bad:
        print(f"({bad} unparseable line(s) in {results} skipped)")
    print(f"{'label':<22}{'arm':<10}{'seed':>5}{'steps':>7}{'val_bpb':>10}{'s/step':>8}{'status':<14}")
    n_ok = n_fail = n_pend = 0
    for label in sorted(index):
        r = index[label]
        st = r.get("status", "?")
        n_ok += st == "ok"
        n_pend += st == "pending"
        n_fail += st not in ("ok", "pending")
        bpb = r.get("val_bpb")
        mss = r.get("median_step_seconds")
        print(f"{label:<22}{str(r.get('arm', '')):<10}{str(r.get('seed', '')):>5}{str(r.get('steps', '')):>7}"
              f"{('%.5f' % bpb) if isinstance(bpb, (int, float)) else '-':>10}"
              f"{('%.3f' % mss) if isinstance(mss, (int, float)) else '-':>8} {st:<14}")
    print(f"\n{n_ok} ok, {n_fail} failed, {n_pend} pending")
    bad_eval = [l for l, r in index.items() if r.get("eval_denominator_ok") is False]
    if bad_eval:
        print("WRONG EVAL TOKEN SET (protocol gate 0): " + ", ".join(bad_eval))
    return 0


# ----------------------------------------------------------------------------- from-search

def parse_changes(changes: str) -> Dict[str, str]:
    """'FF_A=1 FF_B=0.3' -> {'FF_A': '1', 'FF_B': '0.3'} (also accepts '+' separators)."""
    out: Dict[str, str] = {}
    for tok in changes.replace("+", " ").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_search_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    r["_official_mean"] = float(r.get("official_mean") or "nan")
                except ValueError:
                    r["_official_mean"] = float("nan")
                r["_source"] = str(path)
                rows.append(r)
    rows = [r for r in rows if not math.isnan(r["_official_mean"])]
    rows.sort(key=lambda r: r["_official_mean"])
    return rows


def candidates_config(rows: List[Dict[str, Any]], top_k: int, seeds: Sequence[int], steps: int,
                      base_arm: str = "k60", base_label: str = "K60 (recipes/K60/train.py)") -> Dict[str, Any]:
    """Paired configs: each candidate arm vs the K60 base arm, matched seeds, equal steps."""
    seen_changes = set()
    picked = []
    for r in rows:
        key = r.get("changes", "")
        if key in seen_changes:
            continue
        seen_changes.add(key)
        picked.append(r)
        if len(picked) >= top_k:
            break
    runs: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    for s in seeds:
        runs.append({"label": f"{base_arm}_s{s}", "arm": base_arm, "pair": "", "seed": s, "env": {}})
    for i, r in enumerate(picked, start=1):
        arm = f"cand{i:02d}"
        env = parse_changes(r.get("changes", ""))
        pair_id = f"{arm}_vs_{base_arm}"
        for s in seeds:
            runs.append({"label": f"{arm}_s{s}", "arm": arm, "pair": pair_id, "seed": s, "env": env})
        pairs.append({"pair_id": pair_id, "family": "ffsim-candidate", "treatment_arm": arm,
                      "control_arm": base_arm, "gating": False,
                      "ffsim": {"name": r.get("name"), "changes": r.get("changes"), "rank": r.get("rank"),
                                "official_mean": r["_official_mean"], "official_sd": r.get("official_sd"),
                                "delta_vs_base": r.get("delta_vs_base"), "p_beat_base": r.get("p_beat_base"),
                                "steps_mean": r.get("steps_mean"), "source": r["_source"]},
                      "trainium": []})
    return {
        "_comment": "GENERATED by `python -m ffsim.gpu.fleet from-search`: top-K ffsim search rows (by "
                    "official_mean) as candidate arms vs the K60 base, matched seeds, equal steps. "
                    "Candidates are screened, never gated: calibrate.py turns a GPU delta into "
                    "P(Trainium delta < 0) only after gate-k60 has accepted the proxy.",
        "_base": base_label + "; base arm env is empty because the K60 knobs are baked into train.py",
        "_steps_policy": f"both arms at {steps} steps (K60 C41 rehearsal count); virtual clock rescaled by train_gpu.py",
        "defaults": {"steps": steps, "env": {}, "args": ["--data-world", "8", "--eval-tokens", "2097152",
                                                          "--eval-emulate-ranks", "8", "--compile", "none"]},
        "runs": runs, "pairs": pairs,
    }


def cmd_from_search(args) -> int:
    paths = sorted(set(sum((glob.glob(p) for p in args.search), [])))
    if not paths:
        print(f"no search CSV matches {args.search}", file=sys.stderr)
        return 2
    rows = read_search_rows(paths)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = candidates_config(rows, args.top_k, seeds, args.steps)
    cfg["_generator"] = {"search": paths, "top_k": args.top_k, "seeds": seeds, "steps": args.steps,
                         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"{len(cfg['runs'])} runs / {len(cfg['pairs'])} pairs from {len(rows)} search rows in {len(paths)} file(s) -> {out}")
    for p in cfg["pairs"]:
        f = p["ffsim"]
        print(f"  {p['pair_id']}: {f['changes']}  official_mean {f['official_mean']:.5f}  p_beat_base {f['p_beat_base']}")
    return 0


# ----------------------------------------------------------------------------- verdict

def cmd_verdict(args) -> int:
    from ffsim.gpu import calibrate
    argv = ["--results", args.results, "--configs", args.configs, "--pairs", args.pairs]
    if args.manifest:
        argv += ["--manifest", args.manifest]
    if args.json:
        argv += ["--json", args.json]
    return calibrate.main(argv)


# ----------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ffsim.gpu.fleet", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    def common(p, configs_default=None):
        p.add_argument("--configs", default=configs_default or str(CONFIG_DIR / "gate-k60.json"))
        p.add_argument("--step-seconds", default="default",
                       help="k1,k2,k4 Trainium seconds per step, 'default', 'models.pkl' or 'models:<path>[:<code>]'")

    p = sub.add_parser("plan", help="print the run list and cost; runs nothing")
    common(p)
    p.add_argument("--seconds-per-step", type=float, default=4.5, help="GPU wall seconds per optimizer step (estimate)")
    p.add_argument("--quote-prices", action="store_true", help="use the 20-Sep list quotes (PROTOCOL.md) instead of the live lookup")
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("schedule", help="print the K60 pinned schedule the code walks; --write regenerates the JSON")
    p.add_argument("--steps", type=int, default=K60_STEPS)
    p.add_argument("--step-seconds", default=None,
                   help="k1,k2,k4 / 'models.pkl' / 'models:<path>'; default: derive from --from-log")
    p.add_argument("--from-log", default="auto",
                   help="chip-C train.log whose switch lines give the triple ('auto': C41_cold_K60, else C40_cold_K59)")
    p.add_argument("--write", nargs="?", const=str(SCHEDULE_K60), default=None,
                   help="write the table as JSON (default path: configs/schedule-K60-TT1793.json)")
    p.add_argument("--json", action="store_true", help="print the table as JSON instead of text")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("run", help="run every config, one subprocess per run pinned to a GPU")
    common(p)
    p.add_argument("--results", required=True)
    p.add_argument("--manifest", default=None, help="fleet manifest jsonl (default: <results>-manifest.jsonl)")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--per-gpu", type=int, default=1)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra train_gpu.py args")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="table of results so far (partial files are fine)")
    p.add_argument("--results", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--configs", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("from-search", help="ffsim search CSV top-K -> paired candidate config")
    p.add_argument("--search", nargs="+", default=[DEFAULT_SEARCH_GLOB])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seeds", default="73,58")
    p.add_argument("--steps", type=int, default=K60_STEPS)
    p.add_argument("--out", default=str(CONFIG_DIR / "candidates-from-ffsim.json"))
    p.set_defaults(func=cmd_from_search)

    p = sub.add_parser("prices", help="read-only aws pricing lookup for the g6e sizes")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "default"))
    p.set_defaults(func=cmd_prices)

    p = sub.add_parser("verdict", help="calibration verdict (delegates to ffsim.gpu.calibrate)")
    p.add_argument("--results", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--configs", default=str(CONFIG_DIR / "gate-k60.json"))
    p.add_argument("--pairs", default=str(REPO / "research" / "sim-data" / "validation-pairs.json"))
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_verdict)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
