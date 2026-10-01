# ffsim route 2: the GPU proxy protocol (Phase 2 edition, 29 Sep 2026)

**Status: nothing has been run on a GPU under this protocol.** The only GPU work the campaign has
done is the 20-21 Sep proxy (`train_gpu.py`, a fork of the K31-era trainer, 22 runs, ~18.5 GPU-hours,
~$35 of the $200 GPU allowance), which was **rejected under its own pre-registered rule** on 21 Sep:
the U-net verdict flipped sign (GPU -0.0020 at seed 50 vs a +0.031 Trainium loss) and the
value-embedding effect came out 1.4x too large (-0.0130 vs the corrected target [-0.0095, -0.0003]);
its g6e.xlarge instance was **stopped, not terminated**, so it
still carried an EBS volume. That failure is the reason section 3 exists: no GPU verdict influences a
chip slot until the proxy has reproduced Trainium pairs the campaign already paid for.

This document supersedes the 20 Sep `PROXY_PROTOCOL.md` (not published). What changed: the base
recipe is K60 (not K31), the trainer is the submission `train.py` itself (not a fork), the schedule is
driven by a **virtual charged clock** (not `--steps`), the calibration targets are the 99 paired
Trainium comparisons in `research/sim-data/validation-pairs.json` (not five hand-picked anchors), the
step-elasticity is 0.057 at ~2,300 steps (not 0.095 at ~900), and every number is recomputed.

Companion files: `ffsim/README.md` (route 1, the statistical surrogate), `ffsim/CONTRACT.md`,
`docs/simulator-spec.md` (Part 2, route 2), `ffsim/gpu/README.md` (quick start).

---

## 1. Purpose, and what transfers

**What transfers: sample efficiency per optimizer step.** "Does change X reach a lower val_bpb after
the same number of optimizer steps, on the same 256 rows per step, with the same LR / batch / MTP
schedule?" is a property of the model and the data, and the same `train.py` computes the same function
on CUDA (it already detects `cuda`, uses `nccl`, `inductor`, and bf16 compute; section 4).

**What does not transfer: anything about time.** Trainium step time is a property of the Neuron
compiler and the chip (the campaign's probes: attention arithmetic nearly free, projections dominate;
removing one elementwise pass over the 4C MLP tensor was worth 1.8%; `torch.maximum` cost 5.9%; QKV
fusion was *slower*; a 96-parameter gate cost 7.6%). An L40S has none of those properties. So the GPU
may never answer "is X faster", "is X worth its step cost", or "how many steps will the 30-minute
budget reach". Those come from ffsim's step-time model or an S60 screen on the chip, and the GPU run is
*told* its step count, never allowed to measure one.

**Why calibration is mandatory.** The 21 Sep gate proved that a clean CPU-equivalence test (223
assertions, identical losses and gradients across twelve flag combinations) and a clean token-set proof
(gate 0) do not guarantee that a *delta* transfers: the U-net residual-stream change moved val_bpb
+0.031 on Trainium and -0.0020 on the GPU. Four of five signs matched; the fifth was the structural
one. The proxy's scope is therefore whatever families it has reproduced, and nothing else (section 6).

---

## 2. The base recipe: K60, driven by a virtual charged clock

### 2.1 K60 as scored

| | |
|---|---|
| file | `recipes/K60/train.py` (sha `ae396432`, 237,050 B, < 262,144 B limit) |
| lineage | K59 (the K57T recipe baked on M12 + `FF_LEAKY_RELU2=0.35 FF_LEAKY_FORM=fnm`) baked on M14 + `FF_ATTN_SRC=5:6,7,8` (layers 6, 7, 8 reuse layer 5's attention source, "AS-a") |
| official | **0.9655** (29 Sep 19:50 UTC, best; #10 cut 0.9651) |
| rehearsal | C41_cold_K60 on chip C: **0.95910 @ 2,361 steps**, startup 497.3 s, charged 1,790.5 s, `FF_TIME_TARGET=1793`; offset to official +0.0064 |
| queue arm | `C:1639_R_m14asa_s73` 0.95995 @ 2,315 steps at `FF_TIME_TARGET=1760` |
| params | `num_params=124,256,278` (chipC/1639_R_m14asa_s73/train.log, the K60 code) |
| flops/token | `flops_per_token=7.581204e+08` (same log line; the trainer's own fwd+bwd estimate; whether it counts the two extra MTP heads is not stated, so the MFU bracket in section 5 absorbs it) |

The K59 knobs are baked as `_env_*` defaults in the file's first ~700 lines; the full 150-knob
effective environment is `ffsim/examples/recipe-K60.json`. The ones that decide what the model learns
and when every switch fires:

```
# geometry / data           # optimizer                      # schedule (all keyed on charged time)
FF_DEPTH=9                  FF_ADAMW_LR_SCALE=1.4142         FF_TIME_TARGET=1793   FF_WARMUP=20
FF_ASPECT_RATIO=113         FF_MATRIX_LR_SCALE=2             FF_ACCUM_SCHED=1:0.06,2:0.2,4
FF_HEAD_DIM=256             FF_EMB_LR=0.3  FF_UNEMB_LR=0.006 FF_ACCUM_LR=0.5,0.7071  FF_ACCUM_FROM=20
FF_KV_HEADS=1 FF_KV_SDPA=1  FF_SCALAR_LR=0.2 FF_WD=0.02      FF_COOLDOWN_FRAC=0.6  FF_COOLDOWN_SHAPE=linear
FF_MB=8  FF_TRAIN_SEQ=1024  FF_EMB_WD=0.001 FF_WD_SCHED=0    FF_COOLDOWN_FLOOR=1
FF_TOTAL_BATCH=262144       FF_MOM_PEAK=0.95 FF_MOM_RAMP=300 FF_MTP=1 FF_MTP_W=1,0.5,0.25
FF_STREAM_MIX=8             FF_MUON_BETA2=0.9 FF_NS_STEPS=5  FF_MTP_PHASES=0.2,0.45 FF_MTP_LEVELS=4
FF_STREAM_ROWS=1            FF_SCALAR_BETA2=0.95             FF_EMA=1 FF_EMA_EVERY=32 FF_EMA_EVAL=0
FF_MASK_BOS_TARGET=1        FF_TENSOR_LR=1 FF_LAMBDA_0D=1    FF_INSURANCE_AT=9
# model                                                      # Neuron-only (inert or refused on CUDA)
FF_LEAKY_RELU2=0.35 FF_LEAKY_FORM=fnm FF_LEAKY_INPLACE=1     FF_CC_FLAGS="--model-type=transformer -O3"
FF_ATTN_SRC=5:6,7,8   FF_QK_GAIN=1.44  FF_SOFTCAP=15         NEURON_COMPETITION_R1_COMPILE=1 FF_ACC_IN_GRAPH=1
FF_VALUE_EMBED=1  FF_KEY_OFFSET=1 FF_KEY_OFFSET_RAD=0.3      FF_XU_QUEUE=32 FF_ASYNC_SYNC=1 FF_ASYNC_OPT=1
FF_FUSED_CE=8  FF_X0_INIT=0.05 FF_WTE_STD=0.8 FF_HEAD_STD=0.001   FF_MUON_ZERO=1 FF_MUON_ZERO2=1 FF_ADAMW_ZERO=1
FF_ROPE_BASE=100000  FF_MLP_MULT=4  FF_UNET=0 FF_LIT=0 FF_BIGRAM=0  FF_MUON_SCHED=1 FF_OPT_FUSE=2 FF_SCALAR_CONSTS=2
FF_SEED=73  (confirmation seeds 73, 58, 67)                  FF_BF16_PARAMS=0 (so NEURON_RT_STOCHASTIC_ROUNDING_EN=0)
```

### 2.2 Why `--steps` cannot drive K60, and what the virtual clock does instead

The file has a fixed-step mode (`FF_SCHEDULE=steps`, `FF_STEP_TOTAL`, cooldown from
`lr_multiplier_steps(step)`), but K60's two most important mechanisms refuse it:

```
train.py:907    assert SCHEDULE == "time", "FF_MTP stages follow charged time: needs FF_SCHEDULE=time"
train.py:1003   assert SCHEDULE == "time", "FF_ACCUM_SCHED: the small-batch phase ends on charged time: needs FF_SCHEDULE=time"
```

and in time mode *every* switch is a function of `charged / FF_TIME_TARGET` read after the previous
step (main loop, lines 4337-4368): `cooldown_level(progress)`, `mtp_stage_of(progress)`,
`accum_phase_of(step+1, progress)`, the insurance checkpoint, the EMA save level, and the stop rule
`charged + next_step_estimate + 5 s >= TIME_TARGET`. A GPU run with a real clock would fire all of them
at GPU step numbers (an L40S at 30-45% MFU takes ~3x the chip's k4 step time) and stop at ~800 steps.
Editing the file to add a step mode would fork it again, which is how the 20 Sep proxy acquired the
`FF_LIT` asymmetry (its divergence 0) unnoticed.

**The virtual clock.** The driver leaves `train.py` unmodified and replaces the charged clock the loop
reads with a step-indexed function:

```
virtual_charged(after step s) = sum over completed steps i in [5, s] of T_C(phase of step i)
T_C = the chip-C step time of the step's accumulation phase, one fixed triple (k1, k2, k4) per lineage
```

Steps 0-4 stay outside the clock (`TRAIN_STARTUP_STEPS_EXCLUDED = 5`, exactly as on the chip);
`FF_TIME_TARGET` stays 1793; the stop is pinned to the reference step count (`--stop-step`), not left to
the reserve rule. Every clock-keyed switch then fires at the **same step number as on Trainium**, and
the run ends at the same total step count, so a GPU delta is a delta over the same optimizer
trajectory as the Trainium delta. Equal steps *and* equal switch points; equal steps alone is not
enough, because a k1->k2->k4 phase boundary that moves by 20 steps changes the token count of the run.

**The triple.** The driver's triple is never typed in. `python -m ffsim.gpu.fleet schedule --write`
derives it from a harvested chip-C log's own switch lines (`vclock.phase_seconds_from_log`: the charged
span between consecutive `FF_ACCUM_SCHED phase` lines divided by the steps in between, per k, the
warm-up pooled with k4), walks it, and writes the result to `ffsim/gpu/configs/schedule-K60-TT1793.json`,
which is what `fleet.DEFAULT_STEP_SECONDS` reads and `plan` prints. The reference is `C41_cold_K60`
itself; **its `train.log` is not harvested** (`research/sim-data/chipC` has no `C41_cold_K60`: the
monitor read only `eval.log` / `steps.log` / `timing.txt` over SSM), so the table is **provisional on
`C40_cold_K59`** (K59 on the same master M14, 2,357 steps at TT 1793): k1 **0.296835 s**, k2
**0.510163 s**, k4 **0.934132 s**. AS-a leaves the step time unchanged (chip D `1552_R_m14asa_s73`
k4 1.0362 vs the chip-D LK35 s73 run 1.0359, +0.03%; chip C `1639_R_m14asa_s73` k1 0.2963 / k2 0.5057
from its truncated log), so the C40 ratios k1 : k2 : k4 are K60's, and only the ratios matter once
`train_gpu.py --steps 2361` rescales the clock (below). After `python -m ffsim harvest --chip C` brings
C41 in, re-run `python -m ffsim.gpu.fleet schedule --write` (it picks C41 over C40 by itself);
`tests/test_ffsim_gpu_fleet.py` regenerates the JSON from the reference log and pins this section and
`README.md` to it, so a stale number here fails the suite. The seven full LK35 chip-C runs (medians k1
0.2922, k2 0.5042, k4 0.9271 in `runs.jsonl`) are an older master: their own log means sit 0.4-0.8%
below C40's and the C40 triple walked at their TT 1760 lands 11-13 steps late on their switches, which
is why the K60 triple comes from an M14 log and not from "LK35 medians x 1.0057" as earlier drafts of
this section had it. For any other lineage use ffsim's `StepTimeModel.predict(code_version, knobs,
chip="C")` (`--step-seconds models.pkl`); a lineage with a "cost unknown" note needs an S60 screen
before its triple is believed.

**Check of the walk against real logs** (`tests/test_ffsim_gpu_trainer.py`, `tests/test_ffsim_gpu_fleet.py`;
the walk is `vclock.walk_schedule`, the same code `train_gpu.py --dry-run` runs with the trainer's own
`cooldown_level` / `mtp_stage_of` / `accum_phase_of`). From each log's OWN per-phase means the walk
reproduces that log's switch steps within 2 and its step count exactly; the 8 MTP stages and the 19
cooldown levels are out of sample (only the three accum switch lines and the final charged seconds
enter the means):

| log (chip C) | TT | steps | phase 2 / 3 logged | max \|walk - log\| over 3 accum + 8 MTP + 19 levels | stop |
|---|---|---|---|---|---|
| `C40_cold_K59` (M14, K59) | 1793 | 2,357 | 336 / 828 | 2 | exact |
| `0816_R_g7lk35_s73` | 1760 | 2,326 | 333 / 820 | 1 | exact |
| `0913_R_g7lk35_s58` | 1760 | 2,327 | 332 / 818 | 2 | exact |
| `0945_R_g7lk35_s67` | 1760 | 2,325 | 330 / 817 | 1 | exact |

What is NOT checked yet: the K60 triple against K60's own run (C41 not harvested, above). The K60 table
below is `fleet.schedule_table` = `vclock.rescale_to_steps` on the C40 triple (one factor,
x0.998653, so the clock-driven stop lands on C41's 2,361) with the stop then pinned there: exactly
what `train_gpu.py --clock virtual --steps 2361` does. `python -m ffsim.gpu.fleet schedule` prints it;
the numbers here are copied from the JSON and pinned by the test:

| event | step (virtual charged) |
|---|---|
| phase 1 (k=1, LR x 0.5) | 20 (14.0 s) |
| phase 1 -> 2 (k=2, LR x 0.7071) | **336** (107.7 s; 0.06 x 1793 = 107.6) |
| phase 2 -> 3 (k=4) = MTP stage 4 | **829** (358.8 s; 0.2 x 1793 = 358.6) |
| MTP stages 1, 2, 3 | 276, 477, 653 |
| MTP stages 5, 6, 7, 8 | 949, 1069, 1190, 1310 |
| cooldown level 19 | **1214** (718.0 s; 0.4 x 1793 = 717.2) |
| EMA save level <= 3 | 2185 |
| level 1 (floor) | 2306 |
| stop (pinned, `--steps 2361`) | **2361** (1,788.0 s virtual charged; C41 measured 2,361 at 1,790.5 s) |

The raw C40 triple free-runs to 2,357 (K59's own count); the rescale to 2,361 (+0.17%) moves no switch
by more than one step, and the 4-decimal triple `fleet.build_command` passes on the command line walks
to the same table. Every candidate on the K60 base gets this pinned schedule.

**Tokens per K60 run** (steps 0-19 at k=4 in warm-up, then 316 k1 steps, 493 k2 steps, 1,532 k4
steps; from the JSON's `steps_by_k`):

```
k1:   316 x  65,536 =  20,709,376
k2:   493 x 131,072 =  64,618,496
k4: 1,552 x 262,144 = 406,847,488      (1,532 + the 20 warm-up steps)
total 2,361 steps = 492,175,360 tokens   (79.5% of the 618,921,984 an all-k4 run would consume)
FLOPs = 7.581204e8 x 492,175,360 = 3.731e17
```

**Modes.** (a) *Equal-steps screening*: every arm uses the base's pinned schedule; the GPU delta is a
sample-efficiency delta, and for a knob that changes step time it is converted to an equal-time
estimate with ffsim's predicted step count and `k = 0.057 bpb per ln(steps)` (contract: 1% of steps ~
0.00057 bpb at ~2,300 steps). (b) *Replay*: each arm runs the schedule its own Trainium run actually
followed (switch steps and stop step parsed from that run's `train.log` by `ffsim.parse_logs`; the
virtual clock is then only a fallback for the events the log did not print). Calibration uses (b)
wherever the log exists, so the raw Trainium delta is the target and no slope is assumed; (a) with the
conversion is the cross-check, and `verdicts` prints both.

---

## 3. The calibration gate

### 3.1 The pairs

`research/sim-data/validation-pairs.json`: 99 pairs, 73 in the surrogate's task list, all full runs
(no screens), `delta = treatment - control` in rehearsal bpb (public shard, 2,097,152 tokens),
**negative is better**. The equal-step column is `delta_raw + 0.057 x ln(steps_T / steps_C)` (the
20 Sep correction with today's slope); it is what mode (a) must reproduce, while mode (b) reproduces
`delta_raw` directly. "Decisive" = |delta| >= 0.0003 on the column the mode uses (the smallest effect
a three-seed pair resolves; README "Trust boundaries").

**Decisive, same-chip, chip C unless marked (the sign gate, 15 pairs):**

| family | pair (seed) | treatment / control run (steps) | raw | equal-step |
|---|---|---|---|---|
| attention-source reuse `FF_ATTN_SRC=5:6,7,8` on LK35 (K60 vs K59 recipe) | s73 | 1639_R_m14asa (2315) / 0816_R_g7lk35 (2326) | -0.00181 | -0.00208 |
| | s67 | 1711_R_m14asa (2326) / 0945_R_g7lk35 (2325) | -0.00136 | -0.00134 |
| | s58 | 1823_R_m14asa (2313) / 0913_R_g7lk35 (2327) | -0.00106 | -0.00140 |
| | D s73 (D-only check) | 1552_R_m14asa (2155) / 1202_R_g7lk35 (2153) | -0.00173 | -0.00168 |
| leaky slope 0.35 fnm vs relu^2 (K57 base) | s73 | 0816_R_g7lk35 (2326) / 0626_R_g7 (2314) | -0.00134 | -0.00105 |
| | s58 | 0913_R_g7lk35 (2327) / 0054_R_g6acc (2310) | -0.00170 | -0.00128 |
| | s67 | 0945_R_g7lk35 (2325) / 0202_R_g6acc (2313) | -0.00221 | -0.00192 |
| leaky slope 0.35 vs 0.5 (same code M12) | s73 | 0816_R_g7lk35 (2326) / 0222_R_g7lkm (2340) | -0.00010 | -0.00044 |
| | s58 | 0913_R_g7lk35 (2327) / 0529_R_g7lkm (2314) | -0.00125 | -0.00093 |
| | s67 | 0945_R_g7lk35 (2325) / 0745_R_g7lkm (2341) | -0.00034 | -0.00073 |
| Muon momentum peak (K57 base) | 0.97, s73 | 1759_R_g7mp97 (2311) / 0626_R_g7 (2314) | +0.00198 | +0.00191 |
| | 0.93, s73 | 2017_R_g7mp93 (2309) / 0626_R_g7 (2314) | +0.00054 | +0.00042 |
| seed luck on LK35 (vs s73 0.96176) | s281 | 1220_R_g7lk35_s281 (2325) / 0816 (2326) | +0.00214 | +0.00216 |
| | s283 | 1254 (2329) / 0816 (2326) | +0.00070 | +0.00077 |
| | s293 | 1325 (2324) / 0816 (2326) | +0.00105 | +0.00100 |
| | s307 | 1356 (2322) / 0816 (2326) | +0.00157 | +0.00147 |

(The seed rows are four pairs; the leaky-vs-relu^2 rows also exist against the RF control, same code
M12: raw -0.00026 / -0.00043 / -0.00120, equal-step -0.00121 / -0.00111 / -0.00205, seeds 73/58/67.
Count 3 + 3 + 3 + 2 + 4 = 15 sign-gate pairs; the D pair is reported, not gated, because D is judged
only against D.)

**Null at equal steps (the custom-Function check, 3 pairs):** `FF_RELU2_FN=1` (relu^2 with the
backward written out, mathematically identical) vs autograd relu^2, K57 base: raw -0.00109 / -0.00127 /
-0.00101 (seeds 73/58/67, steps +2.2 / +1.9 / +2.0%), **equal-step +0.00015 / -0.00017 / +0.00014**.
The whole Trainium win is step time; at equal steps the GPU must see zero. (`FF_ACC_IN_GRAPH=1` is the
same story, -0.00033 / -0.00077 / -0.00072 raw -> -0.00008 / -0.00010 / -0.00005 equal-step, but it is
a Neuron graph-capture flag and has no GPU pair.)

**Ties (must stay inside the band, 10 pairs):** `FF_WD_SCHED=1` +0.00009 / -0.00003 / +0.00006 (s73/
58/67); asymmetric softcap 16.5 on LK35 -0.00022 / +0.00006 (s73/58); leaky 0.25 vs 0.35 +0.00033 /
+0.00017 raw, +0.00026 / +0.00027 equal-step (s73/58); `FF_QK_GAIN=1.6` -0.00028 / +0.00077 / -0.00024
(s58/73/67, the seeds disagree on Trainium, so the GPU is not asked to pick a sign).

**Reported, not gated:** `FF_WARMUP=40` s58 raw +0.00066 at -1.65% steps -> equal-step **-0.00029**:
the correction flips its sign, which is exactly why the raw number is not a mode-(a) target. XSA
(`FF_XSA_LAYERS=all`, s73) raw +0.00338 at -9.1% steps -> -0.00207 equal-step, and layers 0-3 raw
+0.00122 at -3.4% -> -0.00076: an attention-structure change that *wins* per step and loses per second;
it is the scope probe for attention work (section 6) and its correction is a 9.6% extrapolation of a
local slope, so its interval is wide. Single-seed marginals (`FF_MATRIX_LR_SCALE=2.3` +0.00045,
`FF_MUON_BETA2=0.8` +0.00042, `FF_SOFTCAP=18` +0.00043, all M7/M9 bases, one seed) are optional
sign-only rows.

### 3.2 The run list (the gate: 35 runs, 38 with the optional marginals)

All runs: the current `train.py`, mode (b) schedules from the pair's own logs (TT 1760 queue runs:
phase switches ~332 / ~818, stop at the run's own step count), 8-way data-world emulation, eval on the
first 2,097,152 public-shard tokens with 8 emulated ranks, `FF_SKIP_SAVE=1`, eager.

| arm (env on top of K60) | seeds | n | role |
|---|---|---|---|
| `base` = K60 with `FF_ATTN_SRC=` (the K59/LK35 recipe on M14 code) | 73, 58, 67 | 3 | control for AS-a, LK0.5, LK0.25, softcap; treatment for LK-vs-relu^2 |
| `as_a` = K60 as scored | 73, 58, 67 | 3 | anchor: attention-source reuse |
| `relu2` = base + `FF_LEAKY_RELU2=0` | 73, 58, 67 | 3 | K57-recipe control for LK, RF, MOM_PEAK, WD_SCHED, QK16 |
| `rf` = relu2 + `FF_RELU2_FN=1` | 73, 58, 67 | 3 | equal-step null (custom Function numerics) |
| `lk05` = base + `FF_LEAKY_RELU2=0.5` | 73, 58, 67 | 3 | anchor: slope |
| `mp97`, `mp93` = relu2 + `FF_MOM_PEAK=0.97` / `0.93` | 73 | 2 | anchors: optimizer losses |
| `base` seeds 281, 283, 293, 307 | | 4 | seed-luck transfer (does the *same* seed lose on both platforms?) |
| `wds1` = relu2 + `FF_WD_SCHED=1` | 73, 58, 67 | 3 | tie |
| `sc165` = base + `FF_SOFTCAP_A=16.5` | 73, 58 | 2 | tie |
| `lk25` = base + `FF_LEAKY_RELU2=0.25` | 73, 58 | 2 | tie |
| `qk16` = relu2 + `FF_QK_GAIN=1.6` | 58, 73, 67 | 3 | tie (seeds disagree on Trainium) |
| `xsa_all` = relu2 + `FF_XSA_LAYERS=all` | 73 | 1 | scope probe, reported |
| `wu40` = relu2 + `FF_WARMUP=40` | 58 | 1 | reported (sign flips under correction) |
| `repeat` = base s73 again on the same GPU; `repeat2` on another GPU | 73 | 2 | nulls (kernel nondeterminism; device term, only on a multi-GPU box) |
| optional: `mlr23`, `mb08`, `sc18` | 58, 73, 58 | 3 | single-seed sign-only |

The **pre-screen** (section 5) is the first four of these at the K60 schedule (2,361 steps): `base`
s73, `as_a` s73, `relu2` s73, `repeat` s73.

### 3.3 The acceptance rule

`python -m ffsim.gpu verdicts --results runs/gate.jsonl` evaluates, in order:

| # | gate | rule |
|---|---|---|
| 0 | token set | every scored run consumed 2,097,152 tokens / 8,387,440 bytes (the byte count depends only on which tokens were scored). Fail = exit 1, nothing else is read. |
| 1 | noise floor | worst run-to-run null \|delta\| <= 0.0006 (the Trainium seed sd; the 21 Sep nulls were 0.00011 same GPU, 0.00051 across GPUs). Above it, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)` and re-measure before reading on. |
| 2 | absolute level | `base` s73 within 0.01 of the K59 lineage's chip-C level at its schedule (C40_cold_K59 0.96092 @ 2357; queue 0816 0.96176 @ 2326) and `as_a` s73 within 0.01 of C41 (0.95910 @ 2361 at TT 1793) or 0.95995 (queue, TT 1760). Beyond 0.03 = structural: check data-world 8, eval ranks 8, `FF_LAMBDA_0D=1`. |
| 3 | **sign agreement >= 80%** on the 15 decisive pairs (>= 12 of 15), mode (b) against `delta_raw`; each family must contribute at least one agreeing seed (a family that is 0 for 3 fails the gate on its own even if the total passes). |
| 4 | scale | slope through the origin of GPU delta on Trainium delta over the 15 pairs within **[0.5, 2.0]**, with the correlation printed; the 21 Sep proxy's value-embedding row (1.4x, outside its CI) would have failed a CI test but not this band, which is why the band is paired with gate 3 and gate 5. |
| 5 | nulls and ties | the 3 RF rows and the 10 tie rows within max(0.0006, 3 x null sd) for at least 11 of 13. A proxy that invents 0.001 effects where Trainium saw none cannot screen. |
| 6 | per-family notes | printed, not gated: mean and sd of the per-seed GPU deltas next to Trainium's, the seed-luck correlation (4 seeds), the XSA / warm-up rows, the MDE at n = 1, 2, 3, 6 seed pairs from the measured paired sd. |

**Three noise terms, with today's numbers.**

1. Trainium seed sd **0.0006 bpb** at 2M tokens (LK35 chip C: s73 0.96176, s58 0.96189, s67 0.96153,
   new seeds 0.96246-0.96390; contract). Effects under 0.0003 need 3+ seeds on either platform.
2. The route-1 surrogate's per-run residual sd **~0.0003** (the fitted `sigma`), so a same-seed
   Trainium pair carries noise sd ~0.00042 and a perfect model scores ~0.00034 MAE on single-run pairs
   (README, validation). That is the floor any per-pair agreement is judged against: a GPU delta that
   differs from Trainium's by 0.0004 on one seed is *consistent*, not wrong.
3. GPU seed sd and GPU paired-delta sd: **unknown until measured**. The 21 Sep proxy measured a null of
   0.00011 (same GPU) / 0.00051 (across GPUs) and an MDE of 0.0011 at any seed count on the K31 recipe;
   nothing about the K60 recipe on the current code is known. `verdicts` derives the MDE from the
   per-arm seed spread and prints it; a verdict below the MDE is not a verdict.

**If rejected:** stop, spend nothing on screening, and check in this order: gate 0, the nulls (a noisy
null is noise, not infidelity), gate 2, then which *family* failed. A family failure narrows the scope
(section 6) rather than rejecting the proxy for the families that passed, but a proxy with fewer than
two passing families is not worth the screening budget.

---

## 4. Numerical divergences, current code

The 20 Sep fork's CPU suite proved function equality for that fork; the current file is used directly,
so the question is only what the CUDA path of *this* file does differently from the Neuron path. Each
row says why it is acceptable for a **ranking** (delta) even where it is not for an absolute level.

| # | divergence | ranking argument |
|---|---|---|
| 1 | **Data sharding.** `make_stream_dataloader` calls `prepare._document_batches("train", rank, world_size, 128)`: rank r gets every world_size-th row group, then `FF_STREAM_MIX=8` BOS-joined streams per rank. Trainium: world 8 (64 streams). One GPU: world 1. | Removed, not tolerated: the driver runs 8 pseudo-rank loaders and feeds micro-batch j of the 32 from pseudo-rank j // 4, so the 256 rows of an optimizer step are the same rows in the same per-stream order. The 20 Sep fork verified this row for row on CPU; it must be re-verified for the current loader (`tests/test_ffsim_gpu_*.py`, no torch: compare row ids from a fake `prepare`). |
| 2 | **Accumulation.** Trainium: 8 ranks x 4 micro-batches, AVG all-reduce; `FF_MUON_ZERO2` / `FF_ADAMW_ZERO` shard the optimizer and hand updated rows over as bf16. One GPU: 32 micro-batches into one fp32 `.grad`, no sharding. | Same mean gradient by construction (equal micro-batch token counts, mean of means = global mean); fp32 summation order only. The bf16 row hand-over is equivalent in the forward because the forward consumes bf16 weights either way (compute dtype bf16, row 3). Present identically in both arms. |
| 3 | **Matmul precision.** `neuronx-cc` auto-casts fp32 matmuls to bf16; `compute_dtype_for(cuda)` is bf16 when supported (L40S: yes), so the model's own casts match, and the remaining nominally-fp32 matmuls (0-dim lambdas, `FF_TENSOR_LR`, the fp32 optimizer math) run in true fp32 on CUDA. | `torch.set_float32_matmul_precision("medium")` (bf16 internally) is the closest analogue; TF32 stays off (`--tf32` is a separate context). Exposure is small with `FF_LAMBDA_0D=1`; `highest` measures it. Both arms share it. |
| 4 | **Attention kernel.** CUDA SDPA (flash / mem-efficient) vs Neuron's lowering; `FF_KV_SDPA=1 FF_KV_HEADS=1` (one KV head), `FF_KEY_OFFSET`, `FF_QK_GAIN`, and `FF_ATTN_SRC` reuse. | Same mathematics, different tiling. Unlike the 20 Sep gate, an attention-path change **is** an anchor here (AS-a, 3 chip-C seeds), and XSA is the structural probe. Head geometry beyond that is still unvalidated. |
| 5 | **Custom autograd Functions.** Six in the file: `_Relu2Fn`, `_LeakySq` (the fnm form), `_FusedChunkedCE` (`FF_FUSED_CE=8`), `_AccSeed`, and two NKI-only ones (`_ReluSquaredNKI`, `_NkiAttention`, off in K60). | The Python backward runs unchanged on CUDA. The RF null (row 3.1) tests `_Relu2Fn` against autograd, the LK anchors test `_LeakySq`; `_FusedChunkedCE` and `_AccSeed` are in every arm and cancel. |
| 6 | **Stochastic rounding.** With `FF_BF16_PARAMS` off the trainer sets `NEURON_RT_STOCHASTIC_ROUNDING_EN=0` itself (train.py:27-33), and `FF_MUON_ZERO2=1` asserts it (train.py:1115). | No SR on Trainium for this recipe, none on CUDA: no divergence. (The 20 Sep "refuse `FF_BF16_PARAMS`" rule still applies to any candidate that turns it on.) |
| 7 | **Compile.** Trainium: `NEURON_COMPETITION_R1_COMPILE=1`, backend `neuron`, `FF_ACC_IN_GRAPH=1`. CUDA: backend `inductor` (train.py:4006) or eager. | Eager first. `inductor` only after a 50-step A/B of `base` s73 eager vs compiled agrees to ~1e-4 in the loss trace, and it starts a new context (rows are never differenced across contexts). |
| 8 | **Schedule.** Trainium switches fire on a jittering clock (phase 3 at 817 / 818 / 820 in three LK35 runs); the virtual clock pins one step per event. | The pinned table is inside the chip's own jitter (section 2.2); <= 3 steps of 2,361 at a boundary is 0.13% of one phase. Both arms share the table. |
| 9 | **Performance-only knobs.** `FF_ASYNC_SYNC/OPT`, `FF_MUON_SCHED`, `FF_OPT_FUSE=2`, `FF_SCALAR_CONSTS=2`, `FF_MUON_STACK/SCALAR_STACK/VIEWS`, `FF_XU_QUEUE=32` (Neuron runtime only). | Designed bit-identical (memory: fp32 device consts are bit-identical); the driver keeps them as baked and the pre-screen's `base` vs `repeat` null would expose any CUDA-specific nondeterminism they introduce. |
| 10 | **EMA / insurance / kill-safety.** `FF_EMA=1` every 32, decay 0.99, EMA weights saved only above level 3; the final save at level 1 is the trained weights; `FF_EMA_EVAL=0`; insurance checkpoint = a save. | Inert for the scored object (C41 ended at level 1, trained weights). `FF_SKIP_SAVE=1` on the GPU; nothing else changes. |
| 11 | **Eval sharding.** `prepare.evaluate_bpb` shards by the live world size. | 8 emulated ranks, single process; gate 0's byte count proves the token set; a multi-process eval is refused (it would score a different token set while recording "8"). |
| 12 | **Absolute bpb.** Residual bf16 / kernel differences remain after 1-11. | Treated as a smoke test only (gate 2); the deltas are what transfer, and only within the families that passed gate 3. The U-net flip is the standing example of a residual that did not cancel. |

---

## 5. Cost model, recomputed for K60

**FLOPs per run** = `flops_per_token` x tokens = 7.581204e8 x 492,175,360 = **3.731e17** (section 2.2
token count, `configs/schedule-K60-TT1793.json`; the K31 gate's runs were 1.57e17, so a K60 run is 2.4x the work).

**L40S**: dense bf16 tensor-core peak **181.05 TFLOP/s** (the 20 Sep doc's first draft used 362, NVIDIA's
2:4-sparsity figure; nothing here is sparse). At 100% MFU a run is 3.731e17 / 1.8105e14 = 2,061 s =
34.3 min. Eval is 2,097,152 x 7.58e8 / 3 = 5.3e14 FLOPs, seconds of compute; budget 4 min for it at
eval batch 1 with 8 emulated ranks, plus 3 min start-up (data, tokenizer, first steps).

| MFU | train | + eval 4 min + start 3 min |
|---|---|---|
| 20% (eager, 768-wide models landed here on 21 Sep) | 171.4 min | 2.97 h |
| **30%** | 114.3 min | **2.02 h** |
| 40% | 85.7 min | 1.55 h |
| **45%** | 76.2 min | **1.39 h** |

The 30-45% bracket is the realistic eager range for a 124M-parameter, 1,024-wide, 9-layer model with
32 micro-batches of 8,192 tokens per step (each micro-batch is 6.2 TFLOP fwd+bwd, ~34 ms at peak, so
launch overhead is a few percent, not the 21 Sep 20%). **Measure one run and re-cost from its median
step time before committing to the rest.**

**Prices** (`aws pricing get-products`, read-only, 29 Sep 2026, us-west-2 on-demand Linux, shared
tenancy; the 20 Sep quotes were $1.861 and $10.491, unchanged):

| instance | $/h | GPUs | $/GPU-hour | launchable under today's quota |
|---|---|---|---|---|
| g6e.xlarge | 1.861 | 1 | 1.861 | yes: 64 G/VT on-demand vCPUs in us-west-2 = up to **16** at once (4 vCPU each); us-east-1 has 8 vCPUs = 2 |
| g6e.12xlarge | 10.49264 | 4 | 2.623 | yes: 48 vCPUs, one at a time |
| g6e.48xlarge | 30.13118 | 8 | 3.766 | **no**: 192 vCPUs > 64 (`get-service-quota` L-DB2E81BA: us-west-2 = 64, the 20 Sep request for 96 is CASE_CLOSED; us-east-1 = 8, its request still CASE_OPENED) |

Per GPU-hour the single-GPU box is cheapest; the bigger boxes buy wall-clock at 1.41x / 2.02x the
price. The cheap way to buy wall-clock is several g6e.xlarge in parallel (same $/GPU-hour, one data
setup each).

**Sets** (waves = ceil(runs / GPUs), + 0.5 h one-time setup on the box):

| set | instance | @30% MFU | @45% MFU |
|---|---|---|---|
| pre-screen, 4 runs | g6e.xlarge, sequential | 8.6 h, **$16.0** | 6.0 h, **$11.3** |
| pre-screen, 4 runs | g6e.12xlarge | 2.5 h, $26.5 | 1.9 h, $19.8 |
| gate, 38 runs | g6e.xlarge, sequential | 77 h, **$143.9** | 53 h, **$99.0** |
| gate, 38 runs | 16 x g6e.xlarge in parallel (3 waves) | 6.6 h wall, $158 (38 x 2.02 h + 16 x 0.5 h setup) | 4.7 h wall, $113 |
| gate, 38 runs | g6e.12xlarge (10 waves) | 20.7 h, $217.4 | 14.4 h, $150.7 |
| gate, 38 runs | g6e.48xlarge (5 waves; needs a quota increase) | 10.6 h, $319.6 | 7.4 h, $224.0 |

**Marginal cost on a warm g6e.xlarge**: one run **$2.58-3.76**; a three-seed verdict whose base seeds
already exist (3 new runs) **$7.7-11.3**; a six-seed verdict **$15.5-22.6**; a three-seed verdict with a
fresh base (6 runs) $15.5-22.6. On g6e.12xlarge multiply by 1.41, on g6e.48xlarge by 2.02.

**Against a fixed budget** (ours was about $165 at the time): the pre-screen plus the 38-run gate fit
on g6e.xlarge only at the 40%+ end ($110-160 in total); at 30% MFU trim in this order: the 3 optional marginals, then `qk16` (3), then
`lk25` (2) -> 30 runs, $78-119, and the gate keeps all 15 decisive pairs. Screening after acceptance
needs a fresh authorisation and is priced per verdict above. Spot is 60-70% cheaper and reclaimable
mid-run: wrong for the gate (every run must finish), right for a long screening sweep.

---

## 6. How a GPU verdict is used

1. **It chooses which candidates get a three-seed Trainium confirmation (seeds 73, 58, 67). Nothing
   else.** A GPU result never becomes a submission, never sets a rehearsal, never orders a chip queue on
   its own. It is one more input to the README decision rule: a candidate the surrogate puts at rule 2
   (a tie inside 0.0003) or rule 3 (`EXTRAP` / never varied) can be promoted to a chip confirmation by a
   GPU equal-step delta <= -0.0003 at >= 3 seed pairs, or dropped by one >= +0.0003; nothing below the
   printed MDE moves anything.
2. **Scope is per family.** The accepted proxy is labelled with the families it reproduced (activation
   form and slope, attention-source reuse, optimizer momentum, seed luck if gate 6's correlation holds)
   and every verdict quotes that label. A head-geometry, GQA, windowing or residual-stream change is an
   *unvalidated extrapolation* until a Trainium pair in that family has been reproduced; the XSA probe
   is the first such data point for attention structure and is reported with its 9.6% step-correction
   caveat.
3. **Time is never asked.** A candidate that changes step time gets its step count from ffsim's
   step-time model (or an S60 screen with a same-batch control) and the GPU delta is converted with
   k = 0.057; a "cost unknown" lineage is screened on the chip first (README rule 3).
4. **The chip confirmation is the verdict.** One pair rejects a configuration, never a family; chip D
   only against chip D; effects under 0.0003 need all three seeds; a 120-step screen is not quality
   data. The GPU changes the order of the queue, not these rules.

---

## 7. Timeline

**Phase 1 (closes 1 Oct 06:59 UTC; chip C block ends 30 Sep 11:30 UTC): the proxy cannot affect it.**
From a standing start the pre-screen alone is 2-9 h of GPU time and the gate 5-80 h; a verdict would
then need a 3-seed chip confirmation (~2 h of chip C), a cold rehearsal (~38 min) and an upload slot.
The last chip-C hours are already committed to the K62 rehearsals, and no GPU verdict is trusted before
the gate. Nothing in this directory may take a chip slot before the close.

**Phase 2** (from `research/deadline-checklist-20260927.md` and `BRIEFING.md`): winners notified by
8 Oct 06:59 UTC; opens 7 Oct; closes **5 Nov 07:59 UTC**; trn2.48xlarge in us-east-2 (16 chips), a 4-hour
budget, 3 submissions per week, scored 50% val_bpb + 50% CORE (60-minute CORE timeout), an
`inference.py` wrapper required; Phase-2 credits are separate.

The plan, in order:

1. **1-6 Oct, before any Phase-2 chip exists: run the pre-screen and the gate at K60.** The 99
   Trainium pairs exist now and will not be re-run; K60 is the last recipe with three-seed pairs in
   four families. This is the only window in which the proxy can be validated against paid-for data.
   Outcome: a scoped acceptance (or a rejection) of the CUDA path of the current file, plus the measured
   GPU noise terms.
2. **7-10 Oct: re-anchor.** Phase 2 is 16 chips x 4 h = 256x the chip-seconds of Phase 1; the base
   recipe, the step count and the slope `d bpb / d ln steps` all change, so the K60 calibration carries
   only the *families* it validated, not the numbers. The first 3-5 Phase-2 Trainium pairs (a seed pair,
   an activation pair, an attention pair) re-run gates 2-6 at the new scale before the proxy screens
   anything there.
3. **Scale is the constraint in Phase 2, not price per hour.** A full-scale Phase-2 run of this model
   would be ~256 x 0.49B = ~126B tokens = ~9.5e19 FLOPs = 364 h on one L40S at 40% MFU (45 h on eight;
   ~$1,370 on a g6e.48xlarge, which today's quota cannot launch). The GPU proxy in Phase 2 is therefore
   a **reduced-schedule** instrument (a virtual clock over a shortened target, e.g. the first 15-20% of
   the Phase-2 schedule, or a narrower model), and a reduced schedule is a new proxy that needs its own
   gate: the campaign's 120-step and value-embedding sign inversions show that a short run can rank
   late payers backwards. Budget it as a separate line, not as "screening".
4. **Throughout:** the surrogate (route 1) stays the cheap first filter; the GPU is the second filter for
   candidates the surrogate cannot order; the chip confirms. Any change to `train.py` re-runs the
   pre-screen's `base` vs `repeat` null and the RF null before the new file's GPU verdicts count.
