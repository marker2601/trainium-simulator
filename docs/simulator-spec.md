# Trainium run simulator: design brief (29 Sep 2026)

> The brief the simulator was built from, edited for publication (private file references replaced,
> times in CDT). The built system and its measured accuracy are in `docs/SIMULATOR.md`.

Goal: predict the official val_bpb of a candidate `train.py` without spending Trainium time. Then use cheaper hardware
(GPUs) to search recipes, and spend Trainium only on confirmation.

A score is the product of two independent things. Model them separately.

```
official_bpb = quality(recipe, steps, seed)  +  offset(eval shard)
steps        = 1790 s / step_time(code, hardware, runtime)
```

## Part 1: step-time model (how fast Trainium runs a given code)

- **Target:** median seconds per step for each batch phase (k1 = 65k, k2 = 131k, k4 = 262k tokens), per code version.
  Accuracy needed is ±0.3%, because 1% of steps ≈ 0.00057 bpb.
- **Data we already have:**
  - every run's `train.log` on chips C and D: `median_step_seconds_by_k`, per-step `dt`, `steps`;
  - the S60 screens with `FF_HOST_TIMES=1`, which give the host dispatch vs device wait split;
  - the NEFF compile logs.
- **Inputs to model:**
  - model shape: depth, width, heads, MLP multiplier;
  - op counts per block, forward and backward, from `torch.compile` graph dumps;
  - elementwise passes over big tensors;
  - collective count and bytes;
  - eager-optimizer op count;
  - Neuron runtime version.
- **Hard lessons the model must reproduce:**
  - Removing one redundant elementwise pass over the 4C-wide MLP tensor gave 1.8% (RF).
  - `torch.maximum` was slower than a 5-op formula (M11c 5.9% vs M10b 3.4%).
  - QKV fusion was 3.2% slower despite fewer ops.
  - A 96-parameter gate cost 7.6%.
  - Chip D (runtime 2.33.10) is 12% slower at k4 in full runs, but only 1% slower in 60-step screens.

  These show the compiler is not linear in op count. Start with a regression on measured screens, not a first-principles model.
- **Validation:** leave-one-code-version-out error ≤ 0.5% on k4 step time.

## Part 2: quality model (loss per step for a recipe)

- **Target:** val_bpb on the public shard, first 2M tokens, for a given recipe, number of optimizer steps and seed.
- **Data:**
  - `research/experiments.csv`, about 400 runs;
  - the campaign ledger's result lines (private; its run rows are `research/sim-data/monitor-runs.csv`);
  - per-run `train.log` loss curves, one loss line every 10 steps;
  - the 20M-token re-evals (`eval20.log`).
- **Two routes:**
  1. **Statistical surrogate.** Fit bpb from recipe knobs + log(steps) + seed effect, with era fixed effects. (Built as `ffsim/quality.py`.)
  2. **GPU proxy.** Run the same `train.py` model code on a GPU for a fixed number of steps (not a fixed time), then compare the loss at matched steps.
     - Numerics differ: Neuron bf16 rounding, compiled kernels, stochastic rounding off. So the proxy must be calibrated per recipe family.
     - **Our 21 Sep proxy failed validation:** the U-net sign flipped between GPU and Trainium.
     - Every decision must therefore be checked against known Trainium pairs first.
- **Noise floor:** seed sd ≈ 0.0006 bpb (LK0.35 family: new seeds 0.9624–0.9639 vs s73 0.96176). Any recipe effect under ~0.0003 needs 3+ seeds on either platform.
- **Validation:** predict the sign and size of the last 20 paired Trainium comparisons (LK vs RF, slopes 0.25/0.35/0.5, M13/M14 arms, softcap, WD_SCHED, MOM_PEAK). The target is ≥ 80% sign agreement and ≤ 0.0003 mean absolute error.

## Part 3: offset (public-shard rehearsal → official leaderboard)

- **Measured:** +0.0061, +0.0063, +0.0072, +0.0069, +0.0067, +0.0062 (K51 → K59). Mean ≈ +0.0066, sd ≈ 0.0004.
- **Mostly text difficulty.** The first 2M public tokens are easier than typical text: the same model scores +0.0070 worse on 20M public tokens.
- **Model:** a constant, plus upload noise with sd ≈ 0.0004.

## What it would buy, and what it would not

- **Buys:** cheap screening of many recipes in parallel on GPUs, with Trainium used only for the final confirmation and rehearsal.
- **Does not buy:**
  - Trainium throughput itself. Our gap analysis (`GAP-ANALYSIS.md`) puts the gap to the top 10 in kernel throughput, which a GPU proxy cannot measure.
  - Anything before the Phase 1 close. A validated simulator is days of work. Part 1 needs profiling data, and Part 2 needs GPU calibration runs.

## Inputs, as shipped in this release

- `research/experiments.csv` (chips A/B run table), `research/sim-data/monitor-runs.csv` (ledger run rows,
  free-text notes removed), `research/sim-data/runs.jsonl` (the merged dataset the models are fitted on).
- `research/sim-data/validation-pairs.json` (99 same-seed Trainium pairs) and `official-uploads.csv`
  (rehearsal vs official score for every scored upload up to 29 Sep).
- Three reference chip-C run dirs and the queue log under `research/sim-data/chipC/`, and the K60 cold rehearsal log under
  `research/rehearsal-logs/`. The full harvest (about 430 run dirs) is not published: the raw logs carry
  host details.
- Recipes: `recipes/K60/train.py` (the GPU proxy's source) and `recipes/K82s4/train.py` (the best recipe we publish).


## Data updates (appended as results land; newest first)

Campaign-internal exports (per-chip run tables, a paired-run table and a knob summary) were also given to
the builder; they are not part of the release.

### 29 Sep, 10:20 PM CDT - GPU proxy (route 2) validated on its first four runs
- Equal-step L40S runs of the unmodified K60 train.py (virtual Trainium clock, 2314 steps, eager, 8-rank data/eval
  emulation): **K60 s73 0.959952 vs Trainium 0.959948**; attention-source effect **-0.00187 vs -0.00181**; LeakyReLU 0.35 vs
  relu^2 **-0.00114 vs -0.00105 (equal-step)**. Cooldown 0.7: GPU -0.00012, chips +0.00009 / -0.00037 (tie).
- Throughput ~104k tok/s per L40S, ~1.35 h per run; harness fixes needed: PYTHONPATH=~/ff, the organizer prepare.py
  (sha 5807c46e) instead of the repo-root starter copy, FF_LEAKY_FORM=abs for relu^2 arms on K60.
- Consequence: per-step quality questions can now be answered on GPUs in parallel; Trainium is needed only for step time.

### 29 Sep, 12:50 PM CDT
- **K59 official 0.9671** (rehearsal C40 0.96092, offset +0.0062).
- **Attention-source reuse (FF_ATTN_SRC=5:6,7,8)**: same-seed gains -0.00173 (chip D s73), -0.00181 (chip C s73),
  -0.00136 (chip C s67). S60 step cost +0.06% (k4 0.9274 vs 0.9253-0.9282). Variant 4:5,6,7,8: D s73 -0.00086, S60 +0.7%.
- **Step-time facts for Part 1 (S60, chip C, k4 median s):** C0 0.925-0.930; AS-a 0.9274; AS-b 0.9329; POOL3 0.9352;
  LANES3 0.9729 (+4.7%, 18 scalars + ~35 elementwise ops); LANES2 0.9596; ORTHO 0.931; RF (relu^2 custom backward) 0.9135
  vs 0.9306; LK fnm 0.916; M13 rope Fn 0.921, rms_norm Fn 0.9226, QK-fold 0.9315, all three 0.9162 (full run only +0.5%).
- **Chip D (organizer AMI, Neuron runtime 2.33.10)**: full-run k4 1.036 s vs chip C 0.928 (+12%), S60 only ~1% slower.
- **Quality facts for Part 2:** LeakyReLU slope 0.25/0.35/0.5 at s73: 0.96209/0.96176/0.96186 (s58: 0.96207/0.96189/0.96314);
  asymmetric softcap 16.5*tanh(z/15): C s73 -0.00022, C s58 +0.00006, D s73 -0.00036 raw; CFC init 1.0: +0.08 at 60 steps.
- **Seed spread (LK0.35, chip C, TT 1760):** s73 0.96176, s58 0.96189, s67 0.96153 vs new seeds 0.96246-0.96390.
