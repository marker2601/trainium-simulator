# train_gpu.py patch notes (GPU proxy trainer, route 2) — 29 Sep 2026

`ffsim/gpu/train_gpu.py` is GENERATED: `python -m ffsim.gpu.make_train_gpu` applies an ordered list of
32 textual patches to `recipes/K60/train.py` (sha256 `ae396432c31fe831…`).
Every patch has an anchor that must match exactly `count` times or the patcher raises `PatchError` naming
the patch; `--check` fails when the committed file is stale. Re-run it on a K61/K62 file with `--source`.
Nothing in this directory has run on a GPU yet (torch is not installed locally): what is verified is
listed under "Verified here", what is not under "NOT verified".

Files: `make_train_gpu.py` (patcher), `train_gpu.py` (output), `vclock.py` (virtual clock + schedule walk +
chip-log parser, pure python, embedded verbatim), `datasplit.py` (share-cursor arithmetic, embedded),
`gpu_env.py` (env verdict table, rendered into the trainer; `strip(env) -> (kept, removed)` for fleet.py),
`tests/test_ffsim_gpu_trainer.py` (28 tests).

## 1. What the patches do

The K60 file already runs on CUDA (`detect_device_type -> cuda`, nccl, `torch.compile(backend="inductor")`)
and on CPU (gloo). What it cannot do on a GPU is mean the same thing as on Trainium: in `FF_SCHEDULE=time`
every switch is keyed on the CHARGED clock (`charged = time.monotonic() - budget_started`, reset after
step 4): `accum_phase_of(step+1, charged/TT)` (k1 until 6 %, k2 until 20 % of `FF_TIME_TARGET`),
`mtp_stage_of(charged/TT)` (8 stages), `cooldown_level(charged/TT)` (19 levels; the EMA is only saved
above level 3), the insurance save, and the stop (`charged + step_time_estimate(...) + 5 >= TT`).
`FF_MTP` and `FF_ACCUM_SCHED` assert `FF_SCHEDULE=time`, so a fixed-step schedule is not an option.

**P25 is the whole trick**: the single line `charged = time.monotonic() - budget_started` becomes
`charged = _GPU_VCLOCK.charged` where `_GPU_VCLOCK.charged = sum over completed steps >= 5 of T[k(step)]`,
T = Trainium seconds per step of that step's accumulation phase (P24 records `T[step_k]` after each step;
steps 0-4 are excluded inside `VirtualClock.record`, exactly as the real clock excludes them). Every consumer
of `charged` (level, stop, insurance, MTP stage, accum phase, the `charged Ns` in the step line and in the
switch lines) is untouched and now reads the virtual clock; P26 only routes `step_time_estimate` over the
virtual step times so the stop rule fires on the same step as on the chip. `| wall Ns` is appended to the
step line (P28) so the GPU's real timing is still visible; `dt` stays the wall step time.

| patch | mode | x | what / why |
|---|---|---|---|
| P00 header | prepend | 1 | provenance: source path + sha256 at the top of the generated file |
| P01 import block + env refusal | replace | 1 | no neuronx-cc flags; refuse Neuron-only env on sight; --dry-run env defaults; env snapshot for the results line |
| P02 bf16 stochastic rounding refusal | replace | 1 | FF_BF16_PARAMS needs stochastic rounding (Neuron runtime): refused, never downgraded |
| P03 no XU queue env | replace | 1 | NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS is a Neuron runtime queue setting |
| P04 device cuda|cpu | replace | 1 | refuse neuron; cuda when available else cpu |
| P05 sharded-optimizer auto-off at world 1 | insert_before | 1 | FF_MUON_ZERO/ZERO2/ADAMW_ZERO assert world > 1; on one process the unsharded path is the same update |
| P06 runtime block | insert_before | 1 | virtual clock + data-split loader + scoring + results, embedded before main() |
| P07 --device | replace | 1 | --device alias, neuron not a choice |
| P08 --compile inductor|none | replace | 1 | compile is a CLI decision on the GPU (normalised to the bool the loop reads) |
| P09 GPU args + normalise | replace | 1 | add the GPU flags; build the virtual clock; refuse --nki-relu2; --stop-step; --dry-run forces cpu/eager |
| P10 after init_runtime | insert_after | 1 | data_world / shares, fp32 matmul precision, determinism, the banner |
| P11 world tokens per micro from data_world | replace | 1 | grad_accum_steps = the Trainium k (FF_TOTAL_BATCH over the emulated world), not 32 on one GPU |
| P12 grad-accum check + predicted schedule | insert_after | 1 | virtual clock covers every k; print gpu_schedule_predicted |
| P13 compile-only or dry-run | replace | 1 | --dry-run reuses the synthetic-step block |
| P14 synthetic micro count | replace | 1 | k * shares micro-batches per step on this process |
| P15 loss seed 1/(k*shares) | replace | 2 | the accumulated gradient is the mean over k * shares micro-batches = the all-reduce AVG of per-rank means |
| P16 dry-run tail | replace | 1 | 3 synthetic steps + schedule print |
| P17 stream loader through the data-world wrapper | replace | 2 | shares generators under pseudo-ranks (rank*shares+s, data_world) |
| P18 packed fallback through the wrapper | replace | 1 | same for the masked packed loader |
| P19 make_dataloader fallback through the wrapper | replace | 1 | same for prepare.make_dataloader |
| P20 per-step micro count + loader k | replace | 1 | k * shares micro-batches this step; the share cursor learns this step's k before its first in-step fetch |
| P21 micro loop | replace | 1 | loop over k * shares |
| P22 async arm on the last micro | replace | 1 | all-reduce armed on the process's last micro |
| P23 fetch guard | replace | 1 | LATE_FETCH guard over k * shares |
| P24 virtual clock record | insert_after | 1 | after the step: add T[step_k] to the virtual charged clock (steps 0-4 excluded inside record) |
| P25 charged read | replace | 1 | THE clock replacement: level_next, stop, insurance, MTP stage and accum phase all read `charged` |
| P26 stop rule | replace | 1 | step_time_estimate over the virtual step times; --stop-step pins the count; --wall-cap |
| P27 level switch log | insert_after | 1 | record cooldown level switches for the results line |
| P28 step line wall tag | replace | 1 | the step line's `charged` is the virtual clock (diffable against a chip log); wall seconds appended |
| P29 accum seeds 1/(k*shares) | replace | 1 | FF_ACCUM_SCHED backward seeds over k * shares micro-batches |
| P30 scoring | replace | 1 | 8-rank eval emulation; causality check optional |
| P31 results line | insert_after | 1 | one JSON line per run under a file lock |

## 2. The virtual clock: sources of T and what they reproduce

| source | flag | what it reproduces |
|---|---|---|
| per-phase MEANS of a chip log (`phase_seconds_from_log`: charged span between the `FF_ACCUM_SCHED phase` switch lines / steps, pooled by k; the last segment ends at `training_seconds_before_save`) | `--step-seconds-from-log research/sim-data/chipC/C40_cold_K59/train.log` | that run's switch steps within ±2 and its exact step count (C40: max switch diff 2, steps 2357 = 2357, virtual charged 1787.1 = chip 1787.1) |
| the shipped default = C40 means k1 0.29684 / k2 0.51016 / k4 0.93414 | (none) | same as above for the K60 lineage on chip C at TT 1793 |
| a fixed triple | `--step-seconds 0.30,0.56,0.93` (ascending k of FF_ACCUM_SCHED) or `k1=..,k2=..,k4=..` | whatever it is; the task's example triple ends at 2317 steps (k2 0.56 is 10 % above chip C's 0.51) |
| ffsim step-time model medians | `--step-seconds-from-model research/sim-data/models.pkl:M14:C` (needs the ffsim package importable; predicts k1 0.2949 / k2 0.5069 / k4 0.9249 for K60) | medians run 0.3-1 % under the means (`MEDIAN_TO_MEAN_OVERHEAD`): switches 3-11 steps late, stop +12 steps (0.5 %) on C40 |
| `--steps N` (fleet.py's contract) | rescale every phase by ONE factor (bisection over the walk) so the clock-driven run ends at exactly N, then pin the stop at N | both arms of a pair at the same step count with every switch at the same charged FRACTION (6 %, 20 %, 40 % …); C40 means -> 2361 steps is scale 0.99865 |
| `--stop-step N` | pin the count, clock unscaled | the UNscaled switch steps with a forced length |

C41_cold_K60 was not harvested (chipC_runs.csv lists it with 1511 steps and no log), so C40_cold_K59
(same lineage, TT 1793, 2357 steps) is the anchor; K60 = K59 + FF_ATTN_SRC with +0.06 % step cost (S60),
so the K60 boundaries on chip C are those of C40 within a step. The C41 rehearsal count (2361) is what
`--steps 2361` reproduces.

`gpu_schedule_predicted:` (a JSON line) is printed by every run before step 0 and by `--dry-run`, using
the trainer's OWN `cooldown_level / mtp_stage_of / accum_phase_of / accum_k_of / step_time_estimate`
(`_gpu_fns()`), so the walk cannot drift from the loop. The results line records the ACTUAL switches
(`switches.accum / mtp / levels`) for the diff.

## 3. The 8-way data split on fewer processes (`--data-world 8`, default)

Trainium: world 8; `prepare._document_batches("train", rank, 8, 128)` gives rank r every 8th row group;
`make_stream_dataloader` keeps `FF_STREAM_MIX=8` BOS-joined streams per rank (row i of a micro-batch comes
from stream i % 8); one optimizer step at phase k takes k consecutive micro-batches of `FF_MB=8` rows from
every rank; `AsyncGradSync` / `sync_gradients` all-reduce with `ReduceOp.AVG` the per-rank means of k
micro-batches (`loss_seed(loss, k)` = ones/k) -> the mean over 8k micro-batches.

GPU: `shares = data_world / world_size` generators per process (P17-P19 wrap every loader factory), generator
s built AND primed under the pseudo-rank `(rank*shares + s, data_world)` (`_gpu_pseudo_rank` sets
RANK/LOCAL_RANK/WORLD_SIZE for `get_dist_info()`, which the generator reads on its first `next()`); micro j of a
step comes from share `j // k` (`ShareCursor`; P20 tells it the step's k right after the prefetched micro 0,
which is share 0 whatever k is). Every batch is copied into ONE fixed device pair so the compiled graph sees
the same tensor objects (the chip's rule). The step runs `k * shares` micro-batches (P14/P21-P23) with seeds
`1/(k*shares)` (P15, P29), so the accumulated gradient IS the Trainium mean (fp32 summation order aside;
`datasplit.grad_scale_matches`). `world_tokens_per_micro` uses `data_world` (P11) so `grad_accum_steps` stays
the Trainium k (=4, as `FF_ACCUM_SCHED`'s final K requires) and `accum_micro_done`, `total_train_tokens`, the
Muon momentum ramp and the log's token counts are the chip's. With `torchrun --nproc_per_node=8` shares = 1 and
the code is the plain per-rank path; with 2 or 4 GPUs shares = 4 or 2.

Verified here: `emulated_fetch_order` == `trainium_step_rows` per step (same rows, share by share, each share's
generator consumed in order) for shares 1/2/4/8 over a k = 4/1/2/4 phase sequence.

## 4. Environment verdicts (`gpu_env.py`; rendered into train_gpu.py, refused at import)

`refuse` = train_gpu.py exits at import naming the variable and the reason; fleet.py's `strip` removes it and
logs it. Flag-like knobs count only when ON (`_env_flag` reads "", 0, false, False as off; the chip queue env
sets `FF_COMPILE_SYNC=0` etc., which is inert and tolerated). `strip` = harmless but meaningless: fleet strips,
the trainer tolerates and lists it in the banner. Sharded-optimizer flags are AUTO-OFF at world 1 (P05):
`FF_MUON_ZERO`, `FF_MUON_ZERO2` (`assert ... world > 1`, train.py ~2879), `FF_ADAMW_ZERO` (`raise ... needs
world > 1`, ~2601); `FF_MUON_SHARD` / `FF_MUON_SCHED` gate themselves (~3212, ~3339). On one process rank 0 owns
every row, the fp32 master weights are all local and the forward reads them through `.to(bf16)` either way,
so the unsharded eager path is the same update; the results line records `sharding_disabled`.
`--compile none` with `FF_ACC_IN_GRAPH=1` (the K60 default) accumulates eagerly instead (AccumulateGrad
`p.grad += g`, the same fp32 sum in the same order) and records `acc_in_graph_disabled: true`.

| variable | verdict | reason |
|---|---|---|
| `FF_CC_FLAGS` | refused (any value) | neuronx-cc flags (train.py copies it into NEURON_CC_FLAGS at import) |
| `NEURON_CC_FLAGS` | refused (any value) | neuronx-cc flags |
| `FF_XU_QUEUE` | refused when on | sets NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS (Neuron runtime queue depth) |
| `FF_BF16_PARAMS` | refused when on | bf16 master weights need NEURON_RT_STOCHASTIC_ROUNDING_EN; CUDA rounds to nearest and the cooldown tail would stop training (train.py's own comment); refused, not downgraded |
| `FF_NKI_ATTN` | refused when on | NKI attention kernel (nkilib attention_cte / attention_bwd), Neuron only |
| `FF_COMPILE_OPT` | refused when on | torch.compile(optimizer.step, backend="neuron") is hard-coded to the neuron backend |
| `FF_PROFILE_TRACE` | refused when on | torch_neuronx.profiling NeuronConfig trace (diagnostic; falls back to a CPU op trace) |
| `FF_PROFILE_STACK` | refused when on | profiler stack grouping of neuron::copy call sites (needs FF_PROFILE_TRACE) |
| `FF_PROFILE` | refused when on | per-phase synchronize() timing (fb/ar/opt); a timing question the GPU does not answer |
| `FF_PHASE_PROFILE` | refused when on | forward-phase timing (synchronize per phase); timing question |
| `FF_HOST_TIMES` | refused when on | host dispatch vs device wait split (Neuron eager-drain diagnostic; adds synchronize calls) |
| `FF_COMPILE_SYNC` | refused when on | synchronize() after every compiled step to time the compiled graph; timing only |
| `FF_DISPATCH_TRACE` | refused when on | Neuron dispatch trace at step 7 (cpu_to_neuron copies); diagnostic |
| `FF_SYNC_PROBE` | refused when on | Neuron queue-drain probe at step 7 (_sync_probe); diagnostic, all ranks |
| `FF_SYNC_PROBE2` | refused when on | Neuron queue-drain probe at step 7 (_sync_probe2); diagnostic |
| `NEURON_COMPETITION_R1_NKI_RELU2` | refused when on | the example NKI relu**2 kernel, Neuron only |
| `NEURON_COMPETITION_R1_NUM_STEPS` | refused (any value) | run length is the virtual clock's (or --stop-step) |
| `NEURON_COMPETITION_R1_MAX_TRAIN_SECONDS` | refused (any value) | wall-clock cap of the steps schedule; no wall clock drives the proxy |
| `FF_TIMEOUT` | stripped | queue-runner kill timer; not read by train.py |
| `NEURON_COMPETITION_R1_COMPILE` | stripped | compile is a CLI decision on the GPU (--compile inductor/none); the trainer no longer reads it |
| `NEURON_COMPETITION_R1_STEP_LOG` | stripped | the official runner's step-boundary log; no runner is watching |
| `NEURON_RT_*` | refused (any value) | Neuron runtime setting |
| `NEURON_CC_*` | refused (any value) | neuronx-cc setting |
| `NEURON_FRAMEWORK_*` | refused (any value) | Neuron framework setting |
| `NEURON_COMPILE_*` | refused (any value) | Neuron compile cache setting |
| `NEURONX_*` | refused (any value) | neuronx setting |
| `TORCH_NEURONX_*` | refused (any value) | torch_neuronx setting |
| `XLA_*` | refused (any value) | XLA setting (Neuron's torch-xla path) |
| `NEURON_COMPETITION_R1_CACHE_DIR` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_TOKENIZER_PATH` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_TOKENIZER_URL` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_TOKENIZER_SHA256` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_TOKENIZER_PKL_SHA256` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_TOKEN_BYTES_SHA256` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_OUT_DIR` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_EVAL_PUBLIC` | allowed | prepare.py location / harness knob |
| `NEURON_COMPETITION_R1_EVAL_TOKENS` | allowed | prepare.py location / harness knob |
| `FF_TIME_TARGET` | allowed | KEPT: the virtual clock's denominator (progress = virtual charged / FF_TIME_TARGET) |
| `FF_KV_SDPA` | allowed | plain SDPA with one K/V head broadcast over the query heads (enable_gqa); not Neuron-specific |
| `FF_ACC_IN_GRAPH` | allowed | accumulates inside the compiled backward (needs --compile inductor); same math as eager accumulation. --dry-run turns it off (eager on CPU); a real run with --compile none must set FF_ACC_IN_GRAPH=0 explicitly |
| `FF_MUON_ZERO` | allowed | auto-off at world_size 1 (asserts world > 1); same update unsharded |
| `FF_MUON_ZERO2` | allowed | auto-off at world_size 1 (asserts world > 1); same update unsharded |
| `FF_ADAMW_ZERO` | allowed | auto-off at world_size 1 (raises 'needs world > 1'); same update unsharded |
| `FF_MUON_SHARD` | allowed | self-gated on world > 1; plain Newton-Schulz on one process |
| `FF_MUON_SCHED` | allowed | self-gated on world > 1; FF_MUON_SCHED_OP_US / _TFLOPS only shape the multi-rank plan |
| `FF_ASYNC_SYNC` | allowed | inert on one process (no all-reduce); nccl async all-reduce under torchrun |
| `FF_GRAD_BF16` | allowed | bf16 all-reduce: inert on one process, so a Trainium-only rounding divergence |
| `FF_ASYNC_LOOP` | allowed | host-sync cadence: switches fire only at hs steps on the chip and on the GPU alike (walk_schedule models it); keep the chip's value |
| `FF_STAGE_LOADER` | allowed | H2D staging of make_stream_dataloader's rows; same rows and order |
| `FF_LATE_FETCH` | allowed | fetch order of the next micro-batch; same rows and order |
| `FF_EMA` | allowed | EMA / insurance levels follow the virtual clock's cooldown level; K62a (EMA off) is a live knob |
| `FF_SCHEDULE` | allowed | steps mode is allowed (no clock-keyed switches); the virtual clock applies to time mode |

## 5. Results line (`--results FILE`, one JSON object per run, `fcntl`/`msvcrt` exclusive lock)

`status, label, arm, pair, seed, steps, steps_target, stop_pinned, clock_scale, final_train_loss (smoothed),
val_bpb, eval_num_tokens, eval_num_bytes, eval_denominator_ok (== 8,387,440 bytes for 2,097,152 tokens on
public_val with 8 emulated ranks, the number the chip eval reports), eval_ranks_actual / eval_ranks_how,
eval_seconds, causality_passed, train_wall_seconds, wall_total_seconds, median_step_seconds_wall, gpu, device,
torch, cuda, host, clock, step_seconds, step_seconds_source, virtual_charged_final, time_target, schedule,
data_world, world_size, shares, compile, f32_matmul, deterministic, sharding_disabled, acc_in_graph_disabled,
switches{accum, mtp, levels}, final_level, saved_weights, checkpoint, num_params, total_tokens,
grad_accum_steps, micro_per_step, model_config, env (FF_* / NEURON_COMPETITION* snapshot at import),
train_gpu_sha256 (of the running file), source_train_sha256, source_train_path, patcher_version, finished_utc`.
Also printed as `gpu_summary: {...}` after the original `ff_summary:` line.

Scoring: `prepare.evaluate_bpb` on `public_val`, `--eval-tokens 2097152` (the original default 4*524288),
`--eval-emulate-ranks 8`: eight pseudo-ranks in sequence, each every 8th row group and 262,144 tokens, nats and
bytes summed as the all-reduce does. A `torchrun` launch whose live world differs from the emulated count is
refused (it would shard by the live world and score a different token set). `--eval-batch-size` defaults to
`FF_MB`; 262,144 is a multiple of 8*1024 so the token set is the same as at batch 1.

## 6. Left untouched (checked by reading, and by the tests' `count == 1` assertions)

`GPT.__init__` / `init_weights` (RNG order), `MuonAdamW` (every path), `_FusedChunkedCE`, `_Relu2Fn`,
`_LeakySq` (plain torch ops: `torch.maximum`, `relu`, `square`; no Neuron op), `_AccSeed`, softcap / MTP head
(`mtp_weight_table`, `mtp_stage_of`), `FF_ATTN_SRC`, `FF_KEY_OFFSET`, the EMA (`ema_update`, levels), the
cooldown arithmetic, the loaders' row order. The only device branches in the file are `relu_squared` and
`attend` (NKI kernels gated on `device.type == "neuron"`; their env is refused) and `_DispatchTrace`
(diagnostic, refused). `compile_backend = "neuron" if device_type == "neuron" else "inductor"` is left as is
(`neuron` can no longer be selected).

## 7. Known divergences that remain (both arms of a GPU pair share them)

1. bf16 rounding of CUDA kernels vs neuronx-cc (`--auto-cast=matmult`): `--f32-matmul medium` (default) is
   the closest analogue for the few fp32 matmuls; `highest` measures the difference.
2. Attention kernel: SDPA (flash / mem-efficient, `enable_gqa` for the single K/V head) vs Neuron's lowering.
3. Accumulation order: 8 ranks x k then AVG vs `k*shares` sequential fp32 adds on one GPU.
4. `FF_GRAD_BF16` (bf16 all-reduce) is inert on one process; `FF_ASYNC_SYNC` likewise.
5. Compile: `inductor` (default) vs the chip's `neuron` backend; `--compile none` is eager. Adopt one per
   sweep; `gpu_schedule_predicted` and the step line are the same either way.
6. Nondeterministic CUDA kernels (`scatter_add` in the MTP head): `--deterministic` (warn_only) + the seed
   repeat in the protocol's noise term 1.
7. `FF_MUON_ZERO2` hands non-owned Muon rows to other ranks rounded to bf16 on the chip; on one process every
   row is fp32 (the forward casts to bf16 anyway, so identical values; the OPTIMIZER's non-owned copies are not
   read on the chip either).

## 8. Self-check list for the first GPU box (commands and expected output)

Prerequisites on the box: the ORGANIZER's newer `prepare.py` (the one with `TRAIN_STARTUP_STEPS_EXCLUDED = 5`,
`TRAIN_STARTUP_ALLOWANCE_CAP_SECONDS = 900`, `record_step_boundary`; the repo's 2 Sep `prepare.py` lacks them and
the import fails; the 20 Sep working copy of the organiser's prepare.py is that file, 57 changed
lines vs the repo's, evaluate_bpb identical), the tokenizer artifact, the 5 train shards + the public_val shard
in `NEURON_COMPETITION_R1_CACHE_DIR`, torch >= 2.3 with CUDA, tiktoken, pyarrow.

```sh
# 0. the patcher is deterministic and the file is current (no torch needed)
python -m ffsim.gpu.make_train_gpu --check            # "... is up to date (…, 32 patches)"
python -m pytest tests/test_ffsim_gpu_trainer.py -q   # 28 passed

# 1. import + env refusal (expect a SystemExit listing FF_CC_FLAGS)
FF_CC_FLAGS="-O3" python ffsim/gpu/train_gpu.py --dry-run ; echo "rc=$?"   # rc=1, message names FF_CC_FLAGS

# 2. dry-run: tiny model (FF_DEPTH=3 x 32, head 64, FF_ATTN_SRC=1:2), 3 synthetic steps on CPU, schedule print
python ffsim/gpu/train_gpu.py --dry-run --compare-log research/sim-data/chipC/C40_cold_K59/train.log
#   expect: "train_gpu.py 2026-09-29.1: … clock virtual: {'k1': 0.29684, 'k2': 0.51016, 'k4': 0.93414} …
#            data_world 8 = 1 process(es) x 8 share(s) | compile none | … sharding auto-off ['FF_MUON_ZERO',
#            'FF_MUON_ZERO2', 'FF_ADAMW_ZERO']"
#           "train_gpu.py: 32 micro-batches per optimizer step on this process at k=4 (8 share(s) x k)"
#           "dry-run: 3 synthetic optimizer steps on cpu, losses after steps 2-3: [~9.x, ~9.x]" (finite, falling)
#           "dry-run schedule: 2357 steps, virtual charged 1787.1s, stop by clock, final level 1"
#           "  FF_ACCUM_SCHED phase 1: 1 micro-batches per update from step 20 (charged 14s)"
#           "  FF_ACCUM_SCHED phase 2: 2 … from step 336 (charged 108s)"   "phase 3: 4 … from step 828 (charged 359s)"
#           "  FF_MTP stage 1/8 from step 275 … stage 4/8 from step 828 … stage 8/8 from step 1308"
#           "  cooldown level 19 from step 1212 … level 3 from step 2182 <- EMA no longer saved above this level …
#            level 1 from step 2303"
#           the comparison table: max |switch diff| = 2, steps diff = 0, unmatched = 0
#   the same with --steps 2361: "rescaled x0.99865 and pinned at 2361 steps", table steps diff = +4 vs C40 (expected)

# 3. eval token set (gate 0 of the protocol): a 30-step run scored on the same tokens as the chip eval
python ffsim/gpu/train_gpu.py --device cuda --compile none --data-world 8 --steps 30 --eval-emulate-ranks 8 \
    --results runs/smoke.jsonl --label smoke --out-dir runs/smoke
#   expect in gpu_summary: "eval_num_tokens": 2097152, "eval_num_bytes": 8387440, "eval_denominator_ok": true,
#   "steps": 30, "stop_pinned": true, "clock_scale": ~78 (the rescale of a 30-step run is meaningless; use
#   --stop-step 30 for smoke tests to keep the clock unscaled), "acc_in_graph_disabled": true, "sharding_disabled": [...]

# 4. the two compile modes agree (protocol §7 divergence 5): 50 steps eager vs inductor, same seed
python ffsim/gpu/train_gpu.py --compile none     --stop-step 50 --no-eval --label eager50   --out-dir runs/e50
python ffsim/gpu/train_gpu.py --compile inductor --stop-step 50 --no-eval --label induct50  --out-dir runs/i50
#   diff the step lines: loss agrees to ~1e-4 by step 50 (adopt inductor for the sweep only if it does)

# 5. one full K60-length run (2361 steps): timing and the switch diff against the chip log
python ffsim/gpu/train_gpu.py --device cuda --compile inductor --data-world 8 --steps 2361 \
    --results runs/k60.jsonl --label K60_s73 --arm control --pair k60_self --out-dir runs/k60
#   expect: FF_ACCUM_SCHED / FF_MTP / next_level lines at the steps of gpu_schedule_predicted (±0);
#   "final checkpoint: trained weights (last applied level 1 …)"; val_bpb within ~0.01 of the C41 rehearsal
#   0.95910 (absolute level is a smoke test only: deltas are what transfer); train_wall_seconds is the cost basis.

# 6. calibration before any verdict (protocol §2/§4): the pairs in research/sim-data/validation-pairs.json,
#    both arms at the same --steps, same seed, via fleet.py; sign agreement >= 80 % and MAE <= 0.0003.
```

## 9. Verified here (Windows, python 3.11, numpy + pytest, NO torch)

- `py_compile` of `train_gpu.py`, `make_train_gpu.py`, `vclock.py`, `datasplit.py`, `gpu_env.py`; an AST scan
  of every inserted function and `main()` finds no undefined name.
- 28 tests: patch application counts, loud failure on a missing / surplus anchor, re-patching the output refused,
  committed file byte-identical to a fresh build, provenance sha, embedded modules verbatim, untouched Functions /
  optimizer / schedule functions present exactly once, the clock replacement exactly once with every consumer
  intact; the virtual clock vs C40_cold_K59: accum phases 20/336/828 exact, MTP stages within 1, cooldown levels
  within 2, 2357 steps exact, charged 1787.1; medians drift documented (max 11 steps, +12 at the end); rescale to
  2361/2340/2314/2200 exact with switches at the same charged fractions; env verdicts on the real chip C queue
  env (FF_CC_FLAGS / FF_HOST_TIMES / NEURON_RT_* refused; FF_COMPILE_SYNC=0, FF_TIMEOUT tolerated);
  share-cursor arithmetic vs an explicit N-rank split for shares 1/2/4/8.

## 10. NOT verified (needs a GPU box, or at least torch)

- That the K60 file's mechanisms run on one process with the three ZERO flags off and `ACC_IN_GRAPH=0`
  (FF_OPT_FUSE=2, FF_MUON_VIEWS/STACK, FF_MUON_SCHED plan on world 1, FF_SCALAR_CONSTS=2, FF_TENSOR_LR): the
  chip always ran them at world 8. The dry-run is the first test of that.
- `torch.compile(backend="inductor")` of `AccGraph` (`torch.func.functional_call` inside the compiled graph with
  `_AccSeed`) and of the fused CE head on CUDA; the `--compile none` path is the fallback.
- SDPA with `enable_gqa=True` (`FF_KV_SDPA`, one K/V head) on the installed torch version (needs >= 2.5).
- The data-world loader against real shards (only the index arithmetic is tested): row-for-row equality with an
  explicit 8-process run is protocol gate 0's job (`test_gpu_fork_cpu.py` in the 20 Sep working copy did this for
  the K31 fork; port it once torch is available).
- `evaluate_bpb` emulation byte count (8,387,440) — a recorded chip number, asserted at runtime, not here.
- Memory: 124 M params, 32 micro-batches of 8x1024 per step, fp32 master + Muon buffers; fits an L40S/A10G
  (24 GB) comfortably in eager bf16; inductor's graph for the whole model + backward is the unknown.
- The `--wall-cap` abort path writes no results line (the exception propagates); fleet.py should treat a
  missing row as aborted.
- `msvcrt` locking branch (Windows only; the box is Linux and uses `fcntl`).
