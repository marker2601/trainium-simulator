"""Deterministic patcher: submission train.py -> ffsim/gpu/train_gpu.py (the GPU proxy trainer).

    python -m ffsim.gpu.make_train_gpu [--source recipes/K60/train.py] [--out ffsim/gpu/train_gpu.py] [--check]

Not a hand-edited fork. An ORDERED list of textual patches, each with an anchor that must match the
source exactly `count` times (1 unless stated), is applied in sequence; a miss or a surplus match
raises PatchError naming the patch, so the same patcher can be re-run on the K61/K62 files and any
drift in the trainer shows up as a loud failure instead of a silently wrong fork. What each patch does
and why is in the Patch.why field and in ffsim/gpu/PATCH_NOTES.md.

The generated file is single-file on purpose (only prepare.py beside it): ffsim/gpu/vclock.py and
ffsim/gpu/datasplit.py are embedded verbatim (they are pure python, no `from __future__`), and the
env refusal table from ffsim/gpu/gpu_env.py is rendered into it as literals.

What stays untouched (checked, not assumed): the init RNG order, the optimizer, the loss / softcap /
MTP head, the custom autograd Functions (_Relu2Fn, _LeakySq, _FusedChunkedCE, _AccSeed: plain torch
ops, no Neuron-only op; the NKI ones are gated on device.type == "neuron" and their env is refused),
FF_ATTN_SRC, the EMA, the cooldown arithmetic. The patches only (1) replace the charged-clock READ
with the virtual clock, (2) force cuda|cpu, (3) refuse Neuron-only env, (4) emulate the 8-way data
split on fewer processes, (5) score and append a results line, (6) add --dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PATCHER_VERSION = "2026-09-29.1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_SOURCE = REPO / "recipes" / "K60" / "train.py"
DEFAULT_OUT = HERE / "train_gpu.py"

sys.path.insert(0, str(REPO))
from ffsim.gpu import gpu_env  # noqa: E402


class PatchError(RuntimeError):
    pass


@dataclass
class Patch:
    name: str
    anchor: str
    replacement: str
    why: str
    count: int = 1
    mode: str = "replace"          # replace | insert_before | insert_after | prepend

    def apply(self, text: str) -> str:
        if self.mode == "prepend":
            return self.replacement + text
        n = text.count(self.anchor)
        if n != self.count:
            snippet = self.anchor.strip().splitlines()[0][:100]
            raise PatchError(f"patch {self.name!r}: anchor matched {n} time(s), expected {self.count} "
                             f"(first anchor line: {snippet!r})")
        if self.mode == "replace":
            new = self.replacement
        elif self.mode == "insert_before":
            new = self.replacement + self.anchor
        elif self.mode == "insert_after":
            new = self.anchor + self.replacement
        else:
            raise PatchError(f"patch {self.name!r}: unknown mode {self.mode}")
        return text.replace(self.anchor, new)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_module(path: Path) -> str:
    """A pure-python module's text, ready to sit mid-file (its docstring becomes a bare string)."""
    text = path.read_text(encoding="utf-8")
    if re.search(r"^from __future__ import", text, re.M):
        raise PatchError(f"{path.name} must not use a `from __future__` import (it is embedded mid-module)")
    return f"# ---- embedded from ffsim/gpu/{path.name} (do not edit here; edit the module and re-run the patcher) ----\n" \
           f"{text.rstrip()}\n# ---- end of ffsim/gpu/{path.name} ----\n"


def _top_level_names(text: str) -> List[str]:
    return re.findall(r"^(?:def|class)\s+([A-Za-z_]\w*)|^([A-Z_][A-Z0-9_]*)\s*[:=]", text, re.M)


def _source_defaults(source: str) -> Dict[str, str]:
    """Defaults the patcher needs from the source itself (so K61/K62 files keep their own)."""
    out: Dict[str, str] = {}
    m = re.search(r'ATTN_SRC_RAW = os\.environ\.get\("FF_ATTN_SRC", "([^"]*)"\)', source)
    out["FF_ATTN_SRC"] = m.group(1) if m else ""
    for name in ("FF_MUON_ZERO", "FF_MUON_ZERO2"):
        m = re.search(r'_env_flag\("%s", (True|False)\)' % name, source)
        out[name] = m.group(1) if m else "False"
    m = re.search(r'ADAMW_ZERO = _env_int\("FF_ADAMW_ZERO", (\d+)\)', source)
    out["FF_ADAMW_ZERO"] = ("True" if m and int(m.group(1)) else "False")
    return out


# ==============================================================================================
# The inserted code
# ==============================================================================================
IMPORT_BLOCK = '''import os
import sys
# ==== ffsim/gpu GPU proxy: import-time block (P01). Generated; edit make_train_gpu.py, not this file. ====
_GPU_PATCHER_VERSION = "@@PATCHER_VERSION@@"
_GPU_SOURCE_PATH = "@@SOURCE_PATH@@"
_GPU_SOURCE_SHA256 = "@@SOURCE_SHA256@@"
_GPU_DRY_RUN = "--dry-run" in sys.argv
_GPU_DRY_RUN_ENV = @@DRY_RUN_ENV@@
if _GPU_DRY_RUN:
    for _gpu_k, _gpu_v in _GPU_DRY_RUN_ENV.items():
        os.environ.setdefault(_gpu_k, _gpu_v)
if "--deterministic" in sys.argv:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_GPU_ENV_SNAPSHOT = {k: v for k, v in sorted(os.environ.items())
                     if k.startswith(("FF_", "NEURON", "TORCH_NEURONX_", "XLA_", "CUBLAS_", "WORLD_SIZE", "RANK", "LOCAL_RANK"))}
_GPU_REFUSED = @@REFUSED@@
_GPU_VALUE_ALWAYS = @@VALUE_ALWAYS@@
_GPU_STRIPPED = @@STRIPPED@@
_GPU_REFUSED_PREFIXES = @@REFUSED_PREFIXES@@
_GPU_ALLOWED_NEURON = @@ALLOWED_NEURON@@
_GPU_OFF_VALUES = ("", "0", "false", "False")


def _gpu_classify(name, value):
    """'' (allowed / tolerated) or the reason train_gpu.py refuses this variable (ffsim/gpu/gpu_env.py)."""
    if name in _GPU_REFUSED:
        if name in _GPU_VALUE_ALWAYS or value not in _GPU_OFF_VALUES:
            return _GPU_REFUSED[name]
        return ""
    if name in _GPU_STRIPPED:
        return ""
    if name.startswith(("NEURON_", "NEURONX_", "TORCH_NEURONX_", "XLA_")):
        if name in _GPU_ALLOWED_NEURON:
            return ""
        for prefix, reason in _GPU_REFUSED_PREFIXES.items():
            if name.startswith(prefix):
                return reason
        return "NEURON_* variable outside the prepare.py allowlist"
    return ""


_gpu_bad = {k: _gpu_classify(k, os.environ[k]) for k in sorted(os.environ) if _gpu_classify(k, os.environ[k])}
_gpu_tolerated = sorted(k for k in os.environ if k in _GPU_STRIPPED or (k in _GPU_REFUSED and not _gpu_classify(k, os.environ[k])))
if _gpu_bad and __name__ == "__main__":
    raise SystemExit("train_gpu.py refuses the Neuron-only / diagnostic environment it was given (unset them, or let "
                     "fleet.py strip them):\\n" + "\\n".join("  %s=%r: %s" % (k, os.environ.get(k), why)
                                                             for k, why in _gpu_bad.items()))
# NEURON_CC_FLAGS is never set here: there is no neuronx-cc on this box.
# ==== end of P01 ====
'''

BF16_BLOCK = '''if _BF16_ON and __name__ == "__main__":
    raise SystemExit("train_gpu.py: FF_BF16_PARAMS / BF16_PARAMS_DEFAULT need NEURON_RT_STOCHASTIC_ROUNDING_EN "
                     "(stochastic rounding of the bf16 weight update), which has no CUDA equivalent: refused")
'''

XU_BLOCK = '''# train_gpu.py (P03): NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS is not set on a GPU (FF_XU_QUEUE is refused above).
'''

DEVICE_BLOCK = '''def detect_device_type(requested: str = "") -> str:
    """train_gpu.py (P04): cuda or cpu only. A neuron request, or a box where only neuron is available,
    is refused: the GPU proxy exists to run this file somewhere that is not Trainium."""
    if requested == "neuron":
        raise SystemExit("train_gpu.py runs on cuda or cpu only (--device cuda|cpu); use the submission train.py on Trainium")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if neuron_available():
        raise SystemExit("train_gpu.py: no CUDA device and torch reports a neuron device: this is the Trainium box; "
                         "pass --device cpu if a CPU smoke test is really what you want")
    return "cpu"
'''

ZERO_BLOCK = '''# ==== train_gpu.py (P05): sharded-optimizer flags need torch.distributed world > 1; on one process rank 0
# owns every row and the unsharded eager path is the same update (see ffsim/gpu/gpu_env.py). ====
_GPU_WORLD_SIZE = get_dist_info()[3]
_GPU_SHARDING_DISABLED: list = []
if _GPU_WORLD_SIZE == 1:
    for _gpu_k, _gpu_default_on in (("FF_MUON_ZERO", @@MUON_ZERO_DEFAULT@@), ("FF_MUON_ZERO2", @@MUON_ZERO2_DEFAULT@@),
                                    ("FF_ADAMW_ZERO", @@ADAMW_ZERO_DEFAULT@@)):
        _gpu_v = os.environ.get(_gpu_k, "")
        if (_gpu_v not in ("0", "false", "False")) if _gpu_v else _gpu_default_on:
            _GPU_SHARDING_DISABLED.append(_gpu_k)
        os.environ[_gpu_k] = "0"
# ==== end of P05 ====
'''

RUNTIME_BLOCK = '''

# ==============================================================================================
# ==== train_gpu.py (P06): GPU proxy runtime: virtual clock, data-split emulation, scoring, results ====
# ==============================================================================================
import contextlib
import hashlib
import platform
import socket

@@VCLOCK_MODULE@@
@@DATASPLIT_MODULE@@

_GPU_SHARES = 1
_GPU_DATA_WORLD = None
_GPU_VCLOCK = None
_GPU_STOP_PINNED = False
_GPU_ARGS = None
_GPU_LEVEL_SWITCHES: list = []
_GPU_LEVEL_LAST: list = [20]
_GPU_CLOCK_SCALE: list = [1.0]
_GPU_ACC_IN_GRAPH_DISABLED: list = [False]
_GPU_EXPECTED_EVAL_BYTES = {2097152: 8387440}   # public_val, 8 emulated ranks, 2,097,152 tokens (monitor 19 Sep, protocol gate 0)


def _gpu_add_args(parser) -> None:
    g = parser.add_argument_group("GPU proxy (ffsim/gpu)")
    g.add_argument("--clock", type=str, default="virtual", choices=["virtual", "wall"],
                   help="virtual: charged = cumulative Trainium seconds per phase (equal steps, same switch points); "
                        "wall: the original wall-clock reading (smoke tests only)")
    g.add_argument("--step-seconds", type=str, default="",
                   help="Trainium seconds per step by accumulation phase: '0.30,0.56,0.93' (ascending k of "
                        "FF_ACCUM_SCHED) or 'k1=0.30,k2=0.56,k4=0.93'")
    g.add_argument("--step-seconds-from-log", type=str, default="",
                   help="derive per-phase MEAN seconds from a chip train.log (reproduces that run's switch steps)")
    g.add_argument("--step-seconds-from-model", type=str, default="",
                   help="MODELS.pkl:CODE_VERSION:CHIP -> ffsim step-time model medians (needs the ffsim package importable)")
    g.add_argument("--stop-step", type=int, default=0,
                   help="end after exactly N steps (clock stop disabled; switches still follow the UNscaled virtual clock)")
    g.add_argument("--steps", type=int, default=0,
                   help="equal-steps target: rescale the virtual clock by one factor so the clock-driven run ends at "
                        "exactly N steps (every switch at the same charged fraction of FF_TIME_TARGET), and pin the stop at N")
    g.add_argument("--data-world", type=int, default=8,
                   help="emulate an N-rank training data split with N/world_size generators per process (0 = plain)")
    g.add_argument("--results", type=str, default="", help="append one JSON line per run to this file (file-locked)")
    g.add_argument("--label", type=str, default="", help="run name in the results line")
    g.add_argument("--arm", type=str, default="", help="treatment|control (or any arm name) in the results line")
    g.add_argument("--pair", type=str, default="", help="pair id (validation-pairs.json pair_id) in the results line")
    g.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True,
                   help="score with prepare.evaluate_bpb on public_val after training (default on)")
    g.add_argument("--eval-batch-size", type=int, default=0, help="eval micro-batch rows (0 = FF_MB)")
    g.add_argument("--eval-emulate-ranks", type=int, default=8,
                   help="reproduce the N-way eval sharding of the Trainium eval in one process (0 = plain)")
    g.add_argument("--causality-check", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--f32-matmul", type=str, default="medium", choices=["highest", "high", "medium"],
                   help="torch fp32 matmul precision on CUDA; medium (bf16 internally) is the closest analogue of "
                        "neuronx-cc --auto-cast=matmult --auto-cast-type=bf16")
    g.add_argument("--deterministic", action="store_true",
                   help="torch.use_deterministic_algorithms(True, warn_only=True) + CUBLAS_WORKSPACE_CONFIG")
    g.add_argument("--dry-run", action="store_true",
                   help="tiny model on CPU, 3 synthetic steps, print the virtual-clock switch schedule, exit")
    g.add_argument("--compare-log", type=str, default="",
                   help="with --dry-run: a chip train.log to diff the predicted switch steps against")
    g.add_argument("--wall-cap", type=float, default=0.0,
                   help="abort (results line with status=aborted) after this many wall seconds of training (0 = off)")


def _gpu_resolve_step_seconds(args, ks):
    if args.step_seconds_from_log:
        return phase_seconds_from_log(args.step_seconds_from_log, TRAIN_STARTUP_STEPS_EXCLUDED), \\
            "phase means of " + args.step_seconds_from_log
    if args.step_seconds_from_model:
        try:
            path, code_version, chip = args.step_seconds_from_model.split(":")
            import pickle
            with open(path, "rb") as fh:
                models = pickle.load(fh)
            knobs = {k: v for k, v in _GPU_ENV_SNAPSHOT.items() if k.startswith("FF_")}
            pred = models["steptime"].predict(code_version, knobs, chip)
            out = {int(k[1:]): float(v) for k, v in pred.items() if k in ("k1", "k2", "k4")}
            return out, "ffsim step-time model %s %s chip %s: %s" % (path, code_version, chip, pred.get("notes", ""))
        except Exception as exc:
            raise SystemExit("--step-seconds-from-model %r failed (%s: %s); the ffsim package must be importable "
                             "for models.pkl" % (args.step_seconds_from_model, type(exc).__name__, exc))
    if args.step_seconds:
        return parse_step_seconds(args.step_seconds, ks), "--step-seconds " + args.step_seconds
    return dict(DEFAULT_STEP_SECONDS), DEFAULT_STEP_SECONDS_SOURCE


def _gpu_normalise_args(args):
    """The GPU flags -> the trainer's own attributes; refusals that cannot wait for the model build."""
    global _GPU_ARGS, _GPU_VCLOCK, _GPU_STOP_PINNED
    _GPU_ARGS = args
    if args.nki_relu2:
        raise SystemExit("train_gpu.py: --nki-relu2 / NEURON_COMPETITION_R1_NKI_RELU2 is a Neuron kernel: refused")
    args.compile = args.compile != "none"
    args.eval_public = bool(args.eval)
    if args.dry_run:
        args.device_type = args.device_type or "cpu"
        args.compile = False
        args.compile_only = False
    if not args.compile and ACC_IN_GRAPH:
        # accumulate-in-graph lives in the COMPILED backward; eager accumulation (AccumulateGrad: p.grad += g,
        # the same fp32 sum in the same order) is the plain path. Turned off and recorded, never silent.
        globals()["ACC_IN_GRAPH"] = 0
        _GPU_ACC_IN_GRAPH_DISABLED[0] = True
        print0("train_gpu.py: --compile none: FF_ACC_IN_GRAPH=%d needs the compiled backward; accumulating eagerly "
               "(same sum, same order); recorded as acc_in_graph_disabled" % int(_GPU_ENV_SNAPSHOT.get("FF_ACC_IN_GRAPH", "1") or 1))
    if args.stop_step and args.steps:
        raise SystemExit("--stop-step and --steps are alternatives: --steps rescales the clock to end at N, "
                         "--stop-step only pins the count")
    if args.stop_step:
        if args.stop_step < 1:
            raise SystemExit("--stop-step must be >= 1")
        args.num_steps = min(args.num_steps, args.stop_step)
        _GPU_STOP_PINNED = True
    if SCHEDULE == "time" and args.clock == "virtual":
        ks = sorted({k for k in (ACCUM_K0, ACCUM_K1, ACCUM_K_FINAL) if k is not None})
        seconds, source = _gpu_resolve_step_seconds(args, ks)
        if args.steps:
            if args.steps < TRAIN_STARTUP_STEPS_EXCLUDED + 1:
                raise SystemExit("--steps must be > %d" % TRAIN_STARTUP_STEPS_EXCLUDED)
            k_final = ACCUM_K_FINAL if ACCUM_K_FINAL is not None else (
                TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * TRAIN_SEQ * (args.data_world or 1)))
            try:
                seconds, scale, _walk = rescale_to_steps(_gpu_recipe(args, k_final), seconds, args.steps,
                                                         fns=_gpu_fns(), k_final=k_final)
            except ValueError as exc:
                raise SystemExit("--steps %d: %s" % (args.steps, exc))
            source = "%s x %.6f (rescaled to %d steps)" % (source, scale, args.steps)
            _GPU_CLOCK_SCALE[0] = scale
            args.num_steps = min(args.num_steps, args.steps)
            _GPU_STOP_PINNED = True
        _GPU_VCLOCK = VirtualClock(seconds, TRAIN_STARTUP_STEPS_EXCLUDED, source=source)
    elif args.clock == "virtual":
        print0("train_gpu.py: FF_SCHEDULE=%s has no clock-keyed switches; --clock virtual is inert" % SCHEDULE)
        if args.steps:
            args.num_steps = min(args.num_steps, args.steps)
            _GPU_STOP_PINNED = True
    elif args.steps:
        raise SystemExit("--steps needs --clock virtual (a wall-clock run cannot be rescaled)")
    return args


def _gpu_after_init(args, world_size, rank, device) -> None:
    global _GPU_SHARES, _GPU_DATA_WORLD
    dw = args.data_world if args.data_world > 0 else world_size
    if dw % world_size:
        raise SystemExit("--data-world %d must be a multiple of the process world size %d" % (dw, world_size))
    _GPU_DATA_WORLD = dw
    _GPU_SHARES = dw // world_size
    if device.type == "cuda":
        torch.set_float32_matmul_precision(args.f32_matmul)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = args.f32_matmul != "highest"
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    vc = "wall clock (smoke test)" if _GPU_VCLOCK is None else "virtual: %s <- %s" % (
        {("k%d" % k): round(v, 5) for k, v in sorted(_GPU_VCLOCK.T.items())}, _GPU_VCLOCK.source)
    print0("train_gpu.py %s: source %s sha256 %s | clock %s | data_world %d = %d process(es) x %d share(s) | "
           "compile %s | f32_matmul %s | sharding auto-off %s | stop %s | tolerated env %s"
           % (_GPU_PATCHER_VERSION, os.path.basename(_GPU_SOURCE_PATH), _GPU_SOURCE_SHA256[:16], vc, dw, world_size,
              _GPU_SHARES, "inductor" if args.compile else "none", args.f32_matmul,
              _GPU_SHARDING_DISABLED or "none",
              ("rescaled x%.5f and pinned at %d steps" % (_GPU_CLOCK_SCALE[0], args.steps)) if args.steps else
              (("pinned at %d steps" % args.stop_step) if args.stop_step else "clock"),
              _gpu_tolerated or "none"))


def _gpu_recipe(args, k_final):
    return Recipe(time_target=TIME_TARGET_SECONDS, warmup=WARMUP_STEPS, accum_sched=ACCUM_SCHED_RAW,
                  accum_from=ACCUM_FROM, accum_force_steps=ACCUM_FORCE_STEPS, mtp=MTP, mtp_phases=MTP_PHASES,
                  mtp_levels=MTP_LEVELS, cooldown_fraction=COOLDOWN_FRACTION, cooldown_floor=COOLDOWN_FLOOR,
                  cooldown_shape=COOLDOWN_SHAPE, insurance_at=INSURANCE_AT,
                  startup_excluded=TRAIN_STARTUP_STEPS_EXCLUDED, reserve=CHECKPOINT_RESERVE_SECONDS,
                  async_loop=ASYNC_LOOP, num_steps=args.num_steps, schedule=SCHEDULE)


def _gpu_fns():
    """The trainer's OWN schedule functions, so the predicted schedule is the one the loop will run."""
    return {"cooldown_level": cooldown_level, "mtp_stage_of": mtp_stage_of, "accum_phase_of": accum_phase_of,
            "accum_k_of": accum_k_of, "step_time_estimate": step_time_estimate}


def _gpu_predict_schedule(args, k_final):
    if _GPU_VCLOCK is None or SCHEDULE != "time":
        return None
    return walk_schedule(_gpu_recipe(args, k_final), _GPU_VCLOCK.T, stop_step=(args.stop_step or args.steps or None),
                         fns=_gpu_fns(), k_final=k_final)


def _gpu_check_grad_accum(grad_accum_steps) -> None:
    if _GPU_VCLOCK is not None:
        for k in sorted({k for k in (ACCUM_K0, ACCUM_K1, ACCUM_K_FINAL, grad_accum_steps) if k is not None}):
            _GPU_VCLOCK.dt_for(k)
    print0("train_gpu.py: %d micro-batches per optimizer step on this process at k=%d (%d share(s) x k)"
           % (grad_accum_steps * _GPU_SHARES, grad_accum_steps, _GPU_SHARES))
    walk = _gpu_predict_schedule(_GPU_ARGS, grad_accum_steps)
    if walk is not None:
        print0("gpu_schedule_predicted: " + json.dumps(walk, sort_keys=True))


@contextlib.contextmanager
def _gpu_pseudo_rank(rank: int, world: int):
    """prepare.get_dist_info() reports (rank, world) inside the block: the loaders pick their row groups
    from it on their first next(), so build AND prime a generator inside."""
    saved = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _GpuDataWorldLoader:
    """shares generators (one per pseudo-rank) behind one next()-able object; every batch is copied into ONE
    fixed device pair so the compiled graph always receives the same tensor objects (the chip's rule)."""

    def __init__(self, factory, fargs, shares, rank, data_world, k_now):
        self.cursor = ShareCursor(shares, k_now)
        self.gens = []
        self.pending = []
        for s in range(shares):
            pr = rank * shares + s
            with _gpu_pseudo_rank(pr, data_world):
                gen = factory(*fargs)
                first = next(gen)
            self.gens.append(gen)
            self.pending.append(first)
        x0, y0, _ = self.pending[0]
        self.x = torch.empty_like(x0)
        self.y = torch.empty_like(y0)

    def set_k(self, k: int) -> None:
        self.cursor.set_k(k)

    def __iter__(self):
        return self

    def __next__(self):
        share = self.cursor.next_share()
        item = self.pending[share]
        if item is not None:
            self.pending[share] = None
        else:
            item = next(self.gens[share])
        x, y, st = item
        self.x.copy_(x)
        self.y.copy_(y)
        return self.x, self.y, st


def _gpu_make_loader(factory, *fargs):
    if _GPU_SHARES == 1:
        return factory(*fargs)
    _ddp, rank, _lr, world = get_dist_info()
    k_now = fargs[-1] if isinstance(fargs[-1], int) else (TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * TRAIN_SEQ * _GPU_DATA_WORLD))
    loader = _GpuDataWorldLoader(factory, fargs, _GPU_SHARES, rank, _GPU_DATA_WORLD, k_now)
    print0("train_gpu.py: --data-world %d: %d generator(s) on this process, pseudo-ranks %s of %d; micro j of a step "
           "comes from share j // k, so an optimizer step reads the same rows in the same per-stream order as the "
           "%d-rank Trainium step" % (_GPU_DATA_WORLD, _GPU_SHARES, [rank * _GPU_SHARES + s for s in range(_GPU_SHARES)],
                                      _GPU_DATA_WORLD, _GPU_DATA_WORLD))
    return loader


def _gpu_loader_set_k(loader, k: int) -> None:
    if isinstance(loader, _GpuDataWorldLoader):
        loader.set_k(k)


def _gpu_est_windows(recent, steady):
    return (recent, steady) if _GPU_VCLOCK is None else _GPU_VCLOCK.windows(recent, steady)


def _gpu_wall_tag(charged_wall: float) -> str:
    return "" if _GPU_VCLOCK is None else " | wall %.0fs" % charged_wall


def _gpu_dry_run(model, acc_ctl, optimizer, synthetic_x, synthetic_y, mtp_buf, grad_accum_steps, device, args) -> None:
    """Two more synthetic steps (three in all), then the virtual-clock switch schedule of the REAL recipe."""
    losses = []
    for s in range(2):
        if acc_ctl is not None:
            acc_ctl.begin_step()
        model.zero_grad(set_to_none=True)
        loss = None
        for micro in range(grad_accum_steps * _GPU_SHARES):
            if acc_ctl is not None:
                loss = acc_ctl.micro(model, synthetic_x, synthetic_y, micro, mtp_buf)
            else:
                loss = model(synthetic_x, synthetic_y) if mtp_buf is None else model(synthetic_x, synthetic_y, mtp_w=mtp_buf)
            backward_scaled(loss, grad_accum_steps * _GPU_SHARES)
        sync_gradients(model)
        optimizer.step()
        synchronize(device)
        losses.append(float(loss.detach().float().item()))
    print0("dry-run: 3 synthetic optimizer steps on %s, losses after steps 2-3: %s" % (device, losses))
    print0("dry-run env overrides (setdefault, only when unset): %s" % json.dumps(_GPU_DRY_RUN_ENV, sort_keys=True))
    print0("dry-run env snapshot: %s" % json.dumps({k: v for k, v in _GPU_ENV_SNAPSHOT.items() if k.startswith("FF_")}, sort_keys=True))
    walk = _gpu_predict_schedule(args, grad_accum_steps)
    if walk is None:
        print0("dry-run: no virtual clock (--clock wall or FF_SCHEDULE=%s): no switch schedule to print" % SCHEDULE)
        return
    print0("dry-run virtual clock: %s" % json.dumps(_GPU_VCLOCK.describe(), sort_keys=True))
    print0("dry-run schedule: %d steps, virtual charged %.1fs, stop by %s, final level %d"
           % (walk["steps"], walk["charged_final"], walk["stop_reason"], walk["final_level"]))
    for phase, step, charged, k in walk["accum"]:
        print0("  FF_ACCUM_SCHED phase %d: %d micro-batches per update from step %d (charged %.0fs)" % (phase, k, step, charged))
    for stage, step, charged in walk["mtp"]:
        print0("  FF_MTP stage %d/%d from step %d (charged %.0fs)" % (stage, 2 * MTP_LEVELS, step, charged))
    for level, step, charged in walk["levels"]:
        print0("  cooldown level %d from step %d (charged %.0fs)%s" % (level, step, charged,
               "  <- EMA no longer saved above this level" if level == EMA_SAVE_MAX_LEVEL else ""))
    for idx, step, charged in walk["insurance"]:
        print0("  insurance save %d after step %d (charged %.0fs)" % (idx, step - 1, charged))
    print0("gpu_schedule_predicted: " + json.dumps(walk, sort_keys=True))
    if args.compare_log:
        chip = parse_chip_log(args.compare_log)
        cmp = compare_schedules(walk, chip)
        print0("dry-run schedule vs %s (diff = gpu - chip):" % args.compare_log)
        print0(format_comparison(cmp))
        print0("gpu_schedule_diff: " + json.dumps({"max_abs_switch_diff": cmp["max_abs_switch_diff"],
                                                   "steps_diff": cmp["steps_diff"], "unmatched": len(cmp["missing"])}))


def _gpu_score(model, tokenizer, device, args) -> dict:
    """prepare.evaluate_bpb on public_val, the same tokens the Trainium eval scored: 8 ranks each taking
    every 8th row group and 1/8 of the token budget (--eval-emulate-ranks 8 runs them in sequence here
    and sums nats / bytes as the all-reduce does). A distributed eval shards by the LIVE world size, so a
    torchrun launch whose world differs from the emulated count is refused."""
    n = args.eval_emulate_ranks
    bs = args.eval_batch_size or DEVICE_BATCH_SIZE
    live = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 0
    if live:
        if n > 0 and live != n:
            raise SystemExit("--eval-emulate-ranks %d but torch.distributed world_size is %d: the eval would score a "
                             "different token set. Launch with --nproc_per_node=%d, or --eval-emulate-ranks 0 and "
                             "accept an incomparable row." % (n, live, n))
        out = evaluate_bpb(model, tokenizer, bs, SEQ_LEN, device, split="public_val", max_tokens=args.eval_tokens,
                           timeout_seconds=EVAL_TIMEOUT_SECONDS)
        out.update({"eval_ranks_actual": live, "eval_ranks_how": "torch.distributed"})
    elif n <= 0:
        out = evaluate_bpb(model, tokenizer, bs, SEQ_LEN, device, split="public_val", max_tokens=args.eval_tokens,
                           timeout_seconds=EVAL_TIMEOUT_SECONDS)
        out.update({"eval_ranks_actual": 1, "eval_ranks_how": "single"})
    else:
        per_rank = max(1, math.ceil(args.eval_tokens / n))
        nats = 0.0
        nbytes = 0
        ntok = 0
        secs = 0.0
        for r in range(n):
            with _gpu_pseudo_rank(r, n):
                part = evaluate_bpb(model, tokenizer, bs, SEQ_LEN, device, split="public_val", max_tokens=per_rank,
                                    timeout_seconds=EVAL_TIMEOUT_SECONDS)
            nats += part["total_nats"]
            nbytes += part["num_bytes"]
            ntok += part["num_tokens"]
            secs += part["eval_seconds"]
        out = {"val_bpb": nats / (math.log(2) * nbytes) if nbytes > 0 else float("inf"), "num_tokens": ntok,
               "num_bytes": nbytes, "total_nats": nats, "eval_seconds": secs, "eval_ranks_actual": n,
               "eval_ranks_how": "pseudo-rank emulation"}
    expected = _GPU_EXPECTED_EVAL_BYTES.get(args.eval_tokens)
    out["eval_denominator_ok"] = None if expected is None else (out["num_bytes"] == expected)
    out["eval_batch_size"] = bs
    return out


def _gpu_sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_append_result(path, record: dict) -> None:
    """One JSON object per line under an exclusive lock (fcntl on POSIX, msvcrt on Windows)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\\n"
    with p.open("a", encoding="utf-8") as fh:
        locked = None
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            locked = "fcntl"
        except ImportError:
            try:
                import msvcrt
                fh.seek(0, os.SEEK_END)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                locked = "msvcrt"
            except (ImportError, OSError):
                locked = None
        try:
            fh.seek(0, os.SEEK_END)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if locked == "fcntl":
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            elif locked == "msvcrt":
                import msvcrt
                try:
                    fh.seek(0, os.SEEK_END)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass


def _gpu_write_results(args, summary: dict, public_eval, causal, eval_seconds: float, device, config, status: str = "ok") -> None:
    try:
        own_sha = _gpu_sha256_file(os.path.abspath(__file__))
    except OSError:
        own_sha = None
    record = {
        "status": status,
        "label": args.label or Path(args.out_dir).name,
        "arm": args.arm,
        "pair": args.pair,
        "seed": SEED,
        "steps": summary.get("steps"),
        "steps_target": args.steps or args.stop_step or None,
        "stop_pinned": _GPU_STOP_PINNED,
        "clock_scale": _GPU_CLOCK_SCALE[0],
        "acc_in_graph_disabled": _GPU_ACC_IN_GRAPH_DISABLED[0],
        "final_train_loss": summary.get("final_smoothed_loss"),
        "val_bpb": None if public_eval is None else float(public_eval["val_bpb"]),
        "eval_num_tokens": None if public_eval is None else int(public_eval["num_tokens"]),
        "eval_num_bytes": None if public_eval is None else int(public_eval["num_bytes"]),
        "eval_denominator_ok": None if public_eval is None else public_eval.get("eval_denominator_ok"),
        "eval_ranks_actual": None if public_eval is None else public_eval.get("eval_ranks_actual"),
        "eval_ranks_how": None if public_eval is None else public_eval.get("eval_ranks_how"),
        "eval_tokens_requested": args.eval_tokens,
        "eval_batch_size": None if public_eval is None else public_eval.get("eval_batch_size"),
        "eval_seconds": round(eval_seconds, 1),
        "causality_passed": None if causal is None else bool(causal["passed"]),
        "train_wall_seconds": summary.get("training_seconds_before_save"),
        "wall_total_seconds": round(time.monotonic() - _PROCESS_STARTED, 1),
        "median_step_seconds_wall": summary.get("median_step_seconds"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or "cpu"),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "host": os.environ.get("FF_HOST_LABEL", "<HOST>"),  # never the real hostname: outputs get published
        "clock": "wall" if _GPU_VCLOCK is None else "virtual",
        "step_seconds": None if _GPU_VCLOCK is None else {str(k): v for k, v in sorted(_GPU_VCLOCK.T.items())},
        "step_seconds_source": None if _GPU_VCLOCK is None else _GPU_VCLOCK.source,
        "virtual_charged_final": None if _GPU_VCLOCK is None else round(_GPU_VCLOCK.charged, 2),
        "time_target": TIME_TARGET_SECONDS,
        "schedule": SCHEDULE,
        "data_world": _GPU_DATA_WORLD,
        "world_size": summary.get("world_size", get_dist_info()[3]),
        "shares": _GPU_SHARES,
        "compile": "inductor" if args.compile else "none",
        "f32_matmul": args.f32_matmul,
        "deterministic": bool(args.deterministic),
        "sharding_disabled": list(_GPU_SHARDING_DISABLED),
        "switches": {"accum": summary.get("accum_sched", {}).get("switches"), "mtp": summary.get("mtp", {}).get("switches"),
                     "levels": list(_GPU_LEVEL_SWITCHES)},
        "final_level": summary.get("final_level"),
        "saved_weights": summary.get("saved_weights"),
        "checkpoint": summary.get("checkpoint"),
        "num_params": summary.get("num_params"),
        "total_tokens": summary.get("total_tokens"),
        "grad_accum_steps": summary.get("grad_accum_steps"),
        "micro_per_step": None if summary.get("grad_accum_steps") is None else summary["grad_accum_steps"] * _GPU_SHARES,
        "model_config": config_dict(config),
        "env": {k: v for k, v in _GPU_ENV_SNAPSHOT.items() if k.startswith(("FF_", "NEURON_COMPETITION"))},
        "train_gpu_sha256": own_sha,
        "source_train_sha256": _GPU_SOURCE_SHA256,
        "source_train_path": _GPU_SOURCE_PATH,
        "patcher_version": _GPU_PATCHER_VERSION,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if summary.get("diagnostic_probe"):
        record["diagnostic_probe"] = summary["diagnostic_probe"]
    print("gpu_summary: " + json.dumps(record, sort_keys=True), flush=True)
    if args.results:
        _gpu_append_result(args.results, record)
        print("appended to %s" % args.results, flush=True)
    if public_eval is not None and public_eval.get("eval_denominator_ok") is False:
        print("WARNING: eval scored %d bytes, expected %d for %d tokens on public_val with 8 ranks: the token set differs "
              "from the Trainium eval's; the row is marked eval_denominator_ok=false"
              % (public_eval["num_bytes"], _GPU_EXPECTED_EVAL_BYTES.get(args.eval_tokens, -1), args.eval_tokens), flush=True)
# ==== end of P06 ====
'''

STOP_ANCHOR = '''            stop_after_step = step + 1 >= args.num_steps or (
                reached_min_steps
                and (
                    charged >= TIME_TARGET_SECONDS
                    or charged + (max(1, ASYNC_LOOP) if len(steady_step_times) >= 5 else 1)
                    * step_time_estimate(recent_step_times, steady_step_times)
                    + CHECKPOINT_RESERVE_SECONDS >= TIME_TARGET_SECONDS
                )
            )
'''
STOP_REPLACEMENT = '''            stop_after_step = step + 1 >= args.num_steps or (
                reached_min_steps and not _GPU_STOP_PINNED
                and (
                    charged >= TIME_TARGET_SECONDS
                    or charged + (max(1, ASYNC_LOOP) if len(steady_step_times) >= 5 else 1)
                    * step_time_estimate(*_gpu_est_windows(recent_step_times, steady_step_times))
                    + CHECKPOINT_RESERVE_SECONDS >= TIME_TARGET_SECONDS
                )
            )
            if _GPU_ARGS.wall_cap and time.monotonic() - loop_started > _GPU_ARGS.wall_cap:
                raise RuntimeError("train_gpu.py: --wall-cap %.0fs exceeded at step %d" % (_GPU_ARGS.wall_cap, step))
'''

EVAL_ANCHOR = '''    public_eval = None
    causal = None
    if args.eval_public:
        causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = evaluate_bpb(
            orig_model,
            tokenizer,
            DEVICE_BATCH_SIZE,
            SEQ_LEN,
            device,
            split="public_val",
            max_tokens=args.eval_tokens,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
        )
'''
EVAL_REPLACEMENT = '''    public_eval = None
    causal = None
    _gpu_t_eval = time.monotonic()
    if args.eval_public and not probes:
        if args.causality_check:
            causal = causality_check(orig_model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, device)
        public_eval = _gpu_score(orig_model, tokenizer, device, args)
        print0("val_bpb=%.6f over %s tokens / %s bytes (%s) in %.1fs" % (
            public_eval["val_bpb"], format(public_eval["num_tokens"], ","), format(public_eval["num_bytes"], ","),
            public_eval["eval_ranks_how"], time.monotonic() - _gpu_t_eval))
    _gpu_eval_seconds = time.monotonic() - _gpu_t_eval
'''

COMPILE_ONLY_TAIL_ANCHOR = '''        print0("compile-only synthetic training step completed")
        cleanup_runtime()
        return
'''
COMPILE_ONLY_TAIL_REPLACEMENT = '''        print0("compile-only synthetic training step completed")
        if args.dry_run:
            _gpu_dry_run(model, acc_ctl, optimizer, synthetic_x, synthetic_y, mtp_buf, grad_accum_steps, device, args)
        cleanup_runtime()
        return
'''


def build_patches(source: str, source_path: str) -> List[Patch]:
    defaults = _source_defaults(source)
    dry_env = {"FF_DEPTH": "3", "FF_ASPECT_RATIO": "32", "FF_HEAD_DIM": "64", "FF_SKIP_SAVE": "1", "FF_ACC_IN_GRAPH": "0"}
    if defaults["FF_ATTN_SRC"]:
        dry_env["FF_ATTN_SRC"] = "1:2"
    import_block = (IMPORT_BLOCK
                    .replace("@@PATCHER_VERSION@@", PATCHER_VERSION)
                    .replace("@@SOURCE_PATH@@", source_path.replace("\\", "/"))
                    .replace("@@SOURCE_SHA256@@", sha256_text(source))
                    .replace("@@DRY_RUN_ENV@@", repr(dry_env))
                    .replace("@@REFUSED@@", repr(gpu_env.REFUSED))
                    .replace("@@VALUE_ALWAYS@@", repr(tuple(gpu_env.VALUE_ALWAYS)))
                    .replace("@@STRIPPED@@", repr(gpu_env.STRIPPED))
                    .replace("@@REFUSED_PREFIXES@@", repr(gpu_env.REFUSED_PREFIXES))
                    .replace("@@ALLOWED_NEURON@@", repr(tuple(gpu_env.ALLOWED_NEURON))))
    zero_block = (ZERO_BLOCK
                  .replace("@@MUON_ZERO_DEFAULT@@", defaults["FF_MUON_ZERO"])
                  .replace("@@MUON_ZERO2_DEFAULT@@", defaults["FF_MUON_ZERO2"])
                  .replace("@@ADAMW_ZERO_DEFAULT@@", defaults["FF_ADAMW_ZERO"]))
    runtime_block = (RUNTIME_BLOCK
                     .replace("@@VCLOCK_MODULE@@", _embed_module(HERE / "vclock.py"))
                     .replace("@@DATASPLIT_MODULE@@", _embed_module(HERE / "datasplit.py")))
    header = (f"# GENERATED by ffsim/gpu/make_train_gpu.py {PATCHER_VERSION} from {source_path.replace(chr(92), '/')}\n"
              f"# source sha256 {sha256_text(source)}. Do not edit: edit the patcher and re-run it.\n"
              f"# GPU proxy trainer (route 2): equal steps under a virtual charged clock. See ffsim/gpu/PATCH_NOTES.md.\n")
    return [
        Patch("P00 header", "", header, "provenance: source path + sha256 at the top of the generated file", mode="prepend"),
        Patch("P01 import block + env refusal",
              'import os\nif os.environ.get("FF_CC_FLAGS") is not None and not os.environ.get("NEURON_CC_FLAGS"):\n'
              '    os.environ["NEURON_CC_FLAGS"] = os.environ["FF_CC_FLAGS"]\n'
              'elif not os.environ.get("NEURON_CC_FLAGS") and CC_FLAGS_DEFAULT != "":\n'
              '    os.environ["NEURON_CC_FLAGS"] = CC_FLAGS_DEFAULT\n',
              import_block,
              "no neuronx-cc flags; refuse Neuron-only env on sight; --dry-run env defaults; env snapshot for the results line"),
        Patch("P02 bf16 stochastic rounding refusal",
              'if _BF16_ON and _IS_TRAINER:\n    os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")\n'
              'elif _IS_TRAINER:\n    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"\n',
              BF16_BLOCK, "FF_BF16_PARAMS needs stochastic rounding (Neuron runtime): refused, never downgraded"),
        Patch("P03 no XU queue env",
              'if _XU_Q > 0 and _IS_TRAINER:\n    os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", str(_XU_Q))\n',
              XU_BLOCK, "NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS is a Neuron runtime queue setting"),
        Patch("P04 device cuda|cpu",
              'def detect_device_type(requested: str = "") -> str:\n    if requested:\n        return requested\n'
              '    if torch.cuda.is_available():\n        return "cuda"\n    if neuron_available():\n        return "neuron"\n'
              '    return "cpu"\n',
              DEVICE_BLOCK, "refuse neuron; cuda when available else cpu"),
        Patch("P05 sharded-optimizer auto-off at world 1",
              'MUON_ZERO = _env_flag("FF_MUON_ZERO", True)\n', zero_block,
              "FF_MUON_ZERO/ZERO2/ADAMW_ZERO assert world > 1; on one process the unsharded path is the same update",
              mode="insert_before"),
        Patch("P06 runtime block", "\n\ndef main() -> None:\n", runtime_block,
              "virtual clock + data-split loader + scoring + results, embedded before main()", mode="insert_before"),
        Patch("P07 --device", '    parser.add_argument("--device-type", type=str, default="")\n',
              '    parser.add_argument("--device-type", "--device", dest="device_type", type=str, default="", choices=["", "cuda", "cpu"])\n',
              "--device alias, neuron not a choice"),
        Patch("P08 --compile inductor|none",
              '    parser.add_argument("--compile", action="store_true", default=os.environ.get("NEURON_COMPETITION_R1_COMPILE", "1") != "0",\n'
              '                        help="enable torch.compile on the full model")\n',
              '    parser.add_argument("--compile", type=str, default="inductor", choices=["inductor", "none"],\n'
              '                        help="torch.compile the full model with inductor, or run eager (none)")\n',
              "compile is a CLI decision on the GPU (normalised to the bool the loop reads)"),
        Patch("P09 GPU args + normalise", "    args = parser.parse_args()\n\n    global USE_NKI_RELU2\n",
              "    _gpu_add_args(parser)\n    args = _gpu_normalise_args(parser.parse_args())\n\n    global USE_NKI_RELU2\n",
              "add the GPU flags; build the virtual clock; refuse --nki-relu2; --stop-step; --dry-run forces cpu/eager"),
        Patch("P10 after init_runtime", "    ddp, rank, _local_rank, world_size, device = init_runtime(device_type)\n",
              "    _gpu_after_init(args, world_size, rank, device)\n",
              "data_world / shares, fp32 matmul precision, determinism, the banner", mode="insert_after"),
        Patch("P11 world tokens per micro from data_world",
              "    world_tokens_per_micro = DEVICE_BATCH_SIZE * TRAIN_SEQ * world_size\n",
              "    world_tokens_per_micro = DEVICE_BATCH_SIZE * TRAIN_SEQ * _GPU_DATA_WORLD\n",
              "grad_accum_steps = the Trainium k (FF_TOTAL_BATCH over the emulated world), not 32 on one GPU"),
        Patch("P12 grad-accum check + predicted schedule",
              '    print0(f"micro_tokens={world_tokens_per_micro:,} grad_accum_steps={grad_accum_steps}")\n',
              "    _gpu_check_grad_accum(grad_accum_steps)\n",
              "virtual clock covers every k; print gpu_schedule_predicted", mode="insert_after"),
        Patch("P13 compile-only or dry-run", "    if args.compile_only:\n", "    if args.compile_only or args.dry_run:\n",
              "--dry-run reuses the synthetic-step block"),
        Patch("P14 synthetic micro count", "        for micro in range(grad_accum_steps):\n",
              "        for micro in range(grad_accum_steps * _GPU_SHARES):\n", "k * shares micro-batches per step on this process"),
        Patch("P15 loss seed 1/(k*shares)", "backward_scaled(loss, grad_accum_steps)",
              "backward_scaled(loss, grad_accum_steps * _GPU_SHARES)",
              "the accumulated gradient is the mean over k * shares micro-batches = the all-reduce AVG of per-rank means", count=2),
        Patch("P16 dry-run tail", COMPILE_ONLY_TAIL_ANCHOR, COMPILE_ONLY_TAIL_REPLACEMENT, "3 synthetic steps + schedule print"),
        Patch("P17 stream loader through the data-world wrapper",
              "train_loader = make_train_stream_loader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ, device, grad_accum_steps)",
              "train_loader = _gpu_make_loader(make_train_stream_loader, tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ, device, grad_accum_steps)",
              "shares generators under pseudo-ranks (rank*shares+s, data_world)", count=2),
        Patch("P18 packed fallback through the wrapper",
              "            train_loader = make_masked_packed_loader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ, device)\n",
              "            train_loader = _gpu_make_loader(make_masked_packed_loader, tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ, device)\n",
              "same for the masked packed loader"),
        Patch("P19 make_dataloader fallback through the wrapper",
              '            train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ, "train", device)\n',
              '            train_loader = _gpu_make_loader(make_dataloader, tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ, "train", device)\n',
              "same for prepare.make_dataloader"),
        Patch("P20 per-step micro count + loader k", "        last_loss = None\n        step_k = accum_k\n",
              "        last_loss = None\n        step_k = accum_k\n        micro_n = step_k * _GPU_SHARES\n"
              "        _gpu_loader_set_k(train_loader, step_k)\n",
              "k * shares micro-batches this step; the share cursor learns this step's k before its first in-step fetch"),
        Patch("P21 micro loop", "        for micro in range(step_k):\n", "        for micro in range(micro_n):\n", "loop over k * shares"),
        Patch("P22 async arm on the last micro", "            if async_sync is not None and micro == step_k - 1:\n",
              "            if async_sync is not None and micro == micro_n - 1:\n", "all-reduce armed on the process's last micro"),
        Patch("P23 fetch guard", "            if not LATE_FETCH or micro < step_k - 1:\n",
              "            if not LATE_FETCH or micro < micro_n - 1:\n", "LATE_FETCH guard over k * shares"),
        Patch("P24 virtual clock record", "        dt = t_end - t0\n",
              "        if _GPU_VCLOCK is not None:\n            _GPU_VCLOCK.record(step, step_k)\n",
              "after the step: add T[step_k] to the virtual charged clock (steps 0-4 excluded inside record)", mode="insert_after"),
        Patch("P25 charged read", "        charged = time.monotonic() - budget_started\n",
              "        charged_wall = time.monotonic() - budget_started\n"
              "        charged = _GPU_VCLOCK.charged if _GPU_VCLOCK is not None else charged_wall\n",
              "THE clock replacement: level_next, stop, insurance, MTP stage and accum phase all read `charged`"),
        Patch("P26 stop rule", STOP_ANCHOR, STOP_REPLACEMENT,
              "step_time_estimate over the virtual step times; --stop-step pins the count; --wall-cap"),
        Patch("P27 level switch log", "            accum_got = mtp_got.pop() if ACCUM_ON else 0\n",
              "            if next_level != _GPU_LEVEL_LAST[0]:\n"
              "                _GPU_LEVEL_SWITCHES.append([next_level, step + 1, round(charged, 1)])\n"
              "                _GPU_LEVEL_LAST[0] = next_level\n",
              "record cooldown level switches for the results line", mode="insert_after"),
        Patch("P28 step line wall tag",
              'f"tok/s {tok_per_sec:,.0f} | charged {charged:.0f}s | next_level {next_level}{phases}",\n',
              'f"tok/s {tok_per_sec:,.0f} | charged {charged:.0f}s | next_level {next_level}{phases}{_gpu_wall_tag(charged_wall)}",\n',
              "the step line's `charged` is the virtual clock (diffable against a chip log); wall seconds appended"),
        Patch("P29 accum seeds 1/(k*shares)",
              "    sources = {k: loss_seed(loss, k) for k in (ACCUM_K0, ACCUM_K1, ACCUM_K_FINAL) if k is not None}\n",
              "    sources = {k: loss_seed(loss, k * _GPU_SHARES) for k in (ACCUM_K0, ACCUM_K1, ACCUM_K_FINAL) if k is not None}\n",
              "FF_ACCUM_SCHED backward seeds over k * shares micro-batches"),
        Patch("P30 scoring", EVAL_ANCHOR, EVAL_REPLACEMENT, "8-rank eval emulation; causality check optional"),
        Patch("P31 results line", '        print("ff_summary: " + json.dumps(summary, sort_keys=True), flush=True)\n',
              "        _gpu_write_results(args, summary, public_eval, causal, _gpu_eval_seconds, device, config)\n",
              "one JSON line per run under a file lock", mode="insert_after"),
    ]


def build(source: str, source_path: str) -> Tuple[str, List[Dict[str, object]]]:
    """Apply every patch in order. Returns (generated text, per-patch report)."""
    patches = build_patches(source, source_path)
    embedded = _embed_module(HERE / "vclock.py") + _embed_module(HERE / "datasplit.py")
    clash = sorted({a or b for a, b in _top_level_names(embedded)} & {a or b for a, b in _top_level_names(source)})
    if clash:
        raise PatchError(f"embedded module names clash with the source's top level: {clash}")
    text = source
    report = []
    for p in patches:
        before = len(text)
        text = p.apply(text)
        report.append({"name": p.name, "mode": p.mode, "count": p.count, "delta_chars": len(text) - before, "why": p.why})
    compile(text, "train_gpu.py", "exec")   # syntax check without importing torch
    return text, report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true", help="do not write; fail if --out differs from the fresh build")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    src_path = Path(a.source)
    source = src_path.read_text(encoding="utf-8")
    try:
        rel = str(src_path.resolve().relative_to(REPO))
    except ValueError:
        rel = str(src_path)
    text, report = build(source, rel)
    out = Path(a.out)
    if a.check:
        current = out.read_text(encoding="utf-8") if out.exists() else ""
        if current != text:
            print(f"{out} is stale (differs from a fresh build of {rel})", file=sys.stderr)
            return 1
        print(f"{out} is up to date ({len(text):,} bytes, {len(report)} patches)")
        return 0
    out.write_text(text, encoding="utf-8", newline="\n")
    if not a.quiet:
        for r in report:
            print(f"  {r['name']:<45s} {r['mode']:<13s} x{r['count']}  {r['delta_chars']:+7d} chars")
        print(f"wrote {out} ({len(text):,} bytes) from {rel} sha256 {sha256_text(source)[:16]}; "
              f"generated sha256 {sha256_text(text)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
