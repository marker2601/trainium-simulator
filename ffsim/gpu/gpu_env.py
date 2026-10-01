"""Which FF_* / NEURON_* environment variables the GPU proxy refuses, and why (decided by reading the
K60 train.py; each entry cites the code it rests on). Pure python; used by fleet.py (strip) and
rendered into train_gpu.py (refuse on sight at import).

Verdicts
--------
* refuse  : train_gpu.py exits at import; fleet.strip removes it and reports the reason. A knob that only
            means something on Neuron (compiler flags, runtime queue depth, NKI kernels, stochastic
            rounding), a diagnostic that exists to time or trace the Neuron dispatch (adds synchronize()
            calls; the GPU answers no timing question), or a harness knob that would change the run
            length behind the virtual clock's back.
            Flag-like knobs count only when ON: train.py's _env_flag reads "", "0", "false", "False" as
            off, and the chip's queue env sets several of these to 0 explicitly (FF_COMPILE_SYNC=0,
            FF_XU_QUEUE=0 ...). An off value is inert: fleet strips it, train_gpu.py tolerates it.
* strip   : harmless on the GPU but meaningless (queue-runner timer, the compile env the CLI replaced):
            fleet strips it and says so; train_gpu.py tolerates it.
* allowed : everything that decides what the model learns (init, optimizer, loss, softcap, MTP,
            FF_ATTN_SRC, EMA, cooldown, FF_TIME_TARGET = the virtual clock's denominator) and the dispatch
            mechanisms that are plain torch (views/stacks/fused buffers, FF_KV_SDPA = SDPA with one K/V
            head broadcast via enable_gqa, FF_ACC_IN_GRAPH = accumulate inside the compiled backward).
* auto-off at world_size 1 (train_gpu.py prints and records it): FF_MUON_ZERO / FF_MUON_ZERO2 /
  FF_ADAMW_ZERO assert torch.distributed world > 1 (train.py ~2601, ~2879, ~3206); rank r updates slot r
  and an all_gather hands the rest over. On one process rank 0 owns every row, so the unsharded eager
  path is the same update (fp32 master weights; the forward reads them through .to(bf16) either way).
  FF_MUON_SHARD / FF_MUON_SCHED already gate themselves on world > 1 (~3212, ~3339).

Not a prefix rule for FF_*: an unknown FF_ knob is allowed (the file's own asserts police it).
NEURON_* is refused whatever its value unless allowlisted (data / tokenizer / output locations prepare.py reads).
"""

from typing import Dict, List, Mapping, Tuple

OFF_VALUES = ("", "0", "false", "False")

# name -> reason; refused when the value is ON (see OFF_VALUES), except VALUE_ALWAYS names (any value)
REFUSED: Dict[str, str] = {
    "FF_CC_FLAGS": "neuronx-cc flags (train.py copies it into NEURON_CC_FLAGS at import)",
    "NEURON_CC_FLAGS": "neuronx-cc flags",
    "FF_XU_QUEUE": "sets NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS (Neuron runtime queue depth)",
    "FF_BF16_PARAMS": "bf16 master weights need NEURON_RT_STOCHASTIC_ROUNDING_EN; CUDA rounds to nearest and the "
                      "cooldown tail would stop training (train.py's own comment); refused, not downgraded",
    "FF_NKI_ATTN": "NKI attention kernel (nkilib attention_cte / attention_bwd), Neuron only",
    "FF_COMPILE_OPT": "torch.compile(optimizer.step, backend=\"neuron\") is hard-coded to the neuron backend",
    "FF_PROFILE_TRACE": "torch_neuronx.profiling NeuronConfig trace (diagnostic; falls back to a CPU op trace)",
    "FF_PROFILE_STACK": "profiler stack grouping of neuron::copy call sites (needs FF_PROFILE_TRACE)",
    "FF_PROFILE": "per-phase synchronize() timing (fb/ar/opt); a timing question the GPU does not answer",
    "FF_PHASE_PROFILE": "forward-phase timing (synchronize per phase); timing question",
    "FF_HOST_TIMES": "host dispatch vs device wait split (Neuron eager-drain diagnostic; adds synchronize calls)",
    "FF_COMPILE_SYNC": "synchronize() after every compiled step to time the compiled graph; timing only",
    "FF_DISPATCH_TRACE": "Neuron dispatch trace at step 7 (cpu_to_neuron copies); diagnostic",
    "FF_SYNC_PROBE": "Neuron queue-drain probe at step 7 (_sync_probe); diagnostic, all ranks",
    "FF_SYNC_PROBE2": "Neuron queue-drain probe at step 7 (_sync_probe2); diagnostic",
    "NEURON_COMPETITION_R1_NKI_RELU2": "the example NKI relu**2 kernel, Neuron only",
    "NEURON_COMPETITION_R1_NUM_STEPS": "run length is the virtual clock's (or --stop-step)",
    "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS": "wall-clock cap of the steps schedule; no wall clock drives the proxy",
}
# refused whatever the value (a string, not a flag)
VALUE_ALWAYS: Tuple[str, ...] = ("FF_CC_FLAGS", "NEURON_CC_FLAGS", "NEURON_COMPETITION_R1_NUM_STEPS",
                                 "NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS")

# name -> reason; fleet strips, train_gpu.py tolerates
STRIPPED: Dict[str, str] = {
    "FF_TIMEOUT": "queue-runner kill timer; not read by train.py",
    "NEURON_COMPETITION_R1_COMPILE": "compile is a CLI decision on the GPU (--compile inductor|none); the trainer no longer reads it",
    "NEURON_COMPETITION_R1_STEP_LOG": "the official runner's step-boundary log; no runner is watching",
}

# prefix -> reason (any name starting with the prefix, any value)
REFUSED_PREFIXES: Dict[str, str] = {
    "NEURON_RT_": "Neuron runtime setting",
    "NEURON_CC_": "neuronx-cc setting",
    "NEURON_FRAMEWORK_": "Neuron framework setting",
    "NEURON_COMPILE_": "Neuron compile cache setting",
    "NEURONX_": "neuronx setting",
    "TORCH_NEURONX_": "torch_neuronx setting",
    "XLA_": "XLA setting (Neuron's torch-xla path)",
}

# NEURON_COMPETITION_R1_* names prepare.py reads for data / tokenizer / output locations: allowed
ALLOWED_NEURON: Tuple[str, ...] = (
    "NEURON_COMPETITION_R1_CACHE_DIR",
    "NEURON_COMPETITION_R1_TOKENIZER_PATH",
    "NEURON_COMPETITION_R1_TOKENIZER_URL",
    "NEURON_COMPETITION_R1_TOKENIZER_SHA256",
    "NEURON_COMPETITION_R1_TOKENIZER_PKL_SHA256",
    "NEURON_COMPETITION_R1_TOKEN_BYTES_SHA256",
    "NEURON_COMPETITION_R1_OUT_DIR",
    "NEURON_COMPETITION_R1_EVAL_PUBLIC",
    "NEURON_COMPETITION_R1_EVAL_TOKENS",
)

# name -> note (allowed; the note goes into PATCH_NOTES.md)
ALLOWED_WITH_NOTE: Dict[str, str] = {
    "FF_TIME_TARGET": "KEPT: the virtual clock's denominator (progress = virtual charged / FF_TIME_TARGET)",
    "FF_KV_SDPA": "plain SDPA with one K/V head broadcast over the query heads (enable_gqa); not Neuron-specific",
    "FF_ACC_IN_GRAPH": "accumulates inside the compiled backward (needs --compile inductor); same math as eager "
                       "accumulation. --dry-run turns it off (eager on CPU); a real run with --compile none must "
                       "set FF_ACC_IN_GRAPH=0 explicitly",
    "FF_MUON_ZERO": "auto-off at world_size 1 (asserts world > 1); same update unsharded",
    "FF_MUON_ZERO2": "auto-off at world_size 1 (asserts world > 1); same update unsharded",
    "FF_ADAMW_ZERO": "auto-off at world_size 1 (raises 'needs world > 1'); same update unsharded",
    "FF_MUON_SHARD": "self-gated on world > 1; plain Newton-Schulz on one process",
    "FF_MUON_SCHED": "self-gated on world > 1; FF_MUON_SCHED_OP_US / _TFLOPS only shape the multi-rank plan",
    "FF_ASYNC_SYNC": "inert on one process (no all-reduce); nccl async all-reduce under torchrun",
    "FF_GRAD_BF16": "bf16 all-reduce: inert on one process, so a Trainium-only rounding divergence",
    "FF_ASYNC_LOOP": "host-sync cadence: switches fire only at hs steps on the chip and on the GPU alike "
                     "(walk_schedule models it); keep the chip's value",
    "FF_STAGE_LOADER": "H2D staging of make_stream_dataloader's rows; same rows and order",
    "FF_LATE_FETCH": "fetch order of the next micro-batch; same rows and order",
    "FF_EMA": "EMA / insurance levels follow the virtual clock's cooldown level; K62a (EMA off) is a live knob",
    "FF_SCHEDULE": "steps mode is allowed (no clock-keyed switches); the virtual clock applies to time mode",
}


def is_off(value: str) -> bool:
    return value in OFF_VALUES


def classify(name: str, value: str = "1") -> Tuple[str, str]:
    """('', '') when allowed; ('refuse', reason) when train_gpu.py must exit; ('strip', reason) when the
    fleet removes it and the trainer tolerates it. A refused flag-like knob set to an off value is 'strip'."""
    if name in REFUSED:
        if name in VALUE_ALWAYS or not is_off(value):
            return "refuse", REFUSED[name]
        return "strip", REFUSED[name] + " (set to an off value: inert)"
    if name in STRIPPED:
        return "strip", STRIPPED[name]
    if name.startswith(("NEURON_", "NEURONX_", "TORCH_NEURONX_", "XLA_")):
        if name in ALLOWED_NEURON:
            return "", ""
        for prefix, reason in REFUSED_PREFIXES.items():
            if name.startswith(prefix):
                return "refuse", reason
        return "refuse", "NEURON_* variable outside the prepare.py allowlist (%s)" % ", ".join(ALLOWED_NEURON)
    return "", ""


def refused_in(env: Mapping[str, str]) -> Dict[str, str]:
    """{name: reason} for every variable train_gpu.py refuses in env."""
    out = {}
    for k in sorted(env):
        verdict, why = classify(k, env[k])
        if verdict == "refuse":
            out[k] = why
    return out


def strip(env: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(kept, removed): removed maps each stripped or refused name to its reason. fleet.py calls this and
    prints `removed` into the run's log; train_gpu.py refuses what is left rather than ignoring it."""
    removed = {}
    for k in sorted(env):
        verdict, why = classify(k, env[k])
        if verdict:
            removed[k] = ("%s: %s" % (verdict, why))
    kept = {k: v for k, v in env.items() if k not in removed}
    return kept, removed


def refuse(env: Mapping[str, str], what: str = "train_gpu.py") -> None:
    """Raise SystemExit naming every refused variable and its reason (what train_gpu.py does at import)."""
    bad = refused_in(env)
    if bad:
        lines = ["%s refuses the Neuron-only / diagnostic environment it was given (unset them, or let "
                 "fleet.py strip them):" % what]
        lines += ["  %s=%r: %s" % (k, env.get(k), why) for k, why in bad.items()]
        raise SystemExit("\n".join(lines))


def parse_env_file(text: str) -> Dict[str, str]:
    """KEY=VALUE lines (queue/base.env, a run's `overrides` file); '#' comments and blanks skipped."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def notes_table() -> List[Tuple[str, str, str]]:
    """(name, verdict, reason) rows for PATCH_NOTES.md."""
    rows = [(k, "refused" + (" (any value)" if k in VALUE_ALWAYS else " when on"), v) for k, v in REFUSED.items()]
    rows += [(k, "stripped", v) for k, v in STRIPPED.items()]
    rows += [(p + "*", "refused (any value)", v) for p, v in REFUSED_PREFIXES.items()]
    rows += [(k, "allowed", "prepare.py location / harness knob") for k in ALLOWED_NEURON]
    rows += [(k, "allowed", v) for k, v in ALLOWED_WITH_NOTE.items()]
    return rows
