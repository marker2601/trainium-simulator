# The Trainium simulator (`ffsim`)

`ffsim` predicts the official val_bpb of a candidate `train.py` recipe without spending Trainium time. It has two
routes that answer different questions.

| | route 1: statistical surrogate | route 2: GPU proxy |
|---|---|---|
| question | "what will this recipe score, and is it worth a chip run?" | "does this change learn more per step?" |
| runs on | any CPU, numpy only | one NVIDIA GPU per run (we used L40S on g6e) |
| cost | ~1.5 ms per candidate; 282 candidates in about 1 s | ~1.35 h per 2,314-step run (about $2.5 on a g6e.xlarge) |
| code | `ffsim/*.py` | `ffsim/gpu/` |
| trusted for | ranking knob changes the data has seen | per-step quality in families it has reproduced |
| not trusted for | new mechanisms, new kernels, untried values | step time, which always comes from the chip |

Both rest on the same decomposition of a score:

```
official_bpb = quality(recipe, steps, seed) + offset(eval shard)
steps        = the FF_ACCUM_SCHED walk over (FF_TIME_TARGET - 5.5 s reserve)
               at the step time per batch phase that (code, chip) runs at
```

Speed and quality are modelled separately because the competition scores a fixed wall-clock budget. A change that
helps per step but costs 1% of step time has to earn about 0.00057 bpb just to break even.

## Route 1: the statistical surrogate

### The data

`research/sim-data/runs.jsonl` holds 1,206 run records from four Trainium chips: 928 full runs and 278 short
screens. They were merged from three sources: the chips A/B run table (`research/experiments.csv`), the campaign
ledger's run rows (`monitor-runs.csv`) and the harvested chip logs. Each record carries:

- the effective `FF_*` environment;
- steps, charged and startup seconds;
- median step time per batch phase (k1/k2/k4);
- the loss curve and the bpb on the first 2M public tokens.

`validation-pairs.json` holds 99 same-seed Trainium pairs used to score the quality model.
`official-uploads.csv` holds every scored upload up to 29 Sep with its rehearsal.

### The three models

1. **Step time** (`steptime.py`). Median k1/k2/k4 step seconds per code lineage, with knob features (depth,
   width, batch, fused paths) on top. A mechanism that no fitted run carries is charged a +3% prior, and the output
   says "cost unknown". The compiler is not linear in op count:
   - one fewer elementwise pass over the 4C MLP tensor saved 1.8%;
   - `torch.maximum` was slower than a 5-op formula;
   - QKV fusion was 3.2% slower despite having fewer ops;
   - a 96-parameter gate cost 7.6%.

   So this is a regression on measured runs, not a first-principles cost model.
2. **The step walk** (`simulate.py`). This replays `train.py`'s charged-clock logic:
   - warm-up at the full batch until step 20;
   - each accumulation phase until its fraction of the time target;
   - the cooldown;
   - a stop when the next step would cross target - 5.5 s. The 5.5 s is the measured median reserve over 136
     completed runs.

   Median step times are scaled by 1.0057, the measured median-to-mean overhead. This walk reproduces the
   K54b rehearsal's phase switches to within 6-10 steps and its total to within 0.3%.
3. **Quality** (`quality.py`). A ridge regression of the 2M-token bpb on knob features, ln(steps), seed, chip and
   era fixed effects. It is fitted on 131 full chip C/D runs whose complete environment is known; the residual sd
   is 0.0003. It reports `support` for each candidate: whether every changed value was seen in the data
   (`ok` / `interp`) or not (`EXTRAP` / `NEVER-VARIED`).
4. **Offset** (`offset.py`). Rehearsal to official. Seven chip-C uploads (K51-K60) give +0.0061 to +0.0072:
   mean +0.00653, sd 0.00042.

`simulate` and `search` run all four models as a Monte Carlo over chip jitter, seed luck, coefficient uncertainty,
residuals and the offset. When candidates are compared, the draws are shared (common random numbers), so the
noise cancels wherever two recipes agree. The paired difference `d vs base` and `P(< base)` are therefore real
confidence statements, not sign indicators.

### How well it is calibrated

The design targets were: leave-one-version-out step-time error within 0.5%, at least 80% sign agreement and an
MAE of 0.0003 or less on paired comparisons, and an offset sd of about 0.0004.

| check | result | verdict |
|---|---|---|
| step time, leave one lineage out, k4 | 0.545% run-weighted on the 161 runs whose mechanisms other lineages also ran; 0.95% over all 189; runs with a novel mechanism are off by 1-14% | missed (0.43% lineage-averaged); `validate` prints `passes: False` |
| quality, leave one pair out, 99 pairs | sign agreement 84.8%, MAE 0.00039. Decisive pairs (\|effect\| >= 0.0003): 97% / 0.00046 | sign met; MAE target missed, at the pair noise floor (about 0.00034). `ffsim validate` prints `'mae': False`; that is expected |
| offset, 7 points | mean +0.00653, sd 0.00042 | met |
| anchor K59, seed 73 | predicted 2,369 +- 9 steps, 0.96082 rehearsal, 0.9674 official; measured 2,357, 0.96092, 0.9671 | inside 1 sd on bpb and official; steps 1.3 sd high |
| anchor K60, seed 73 | predicted 2,365 +- 9 steps, 0.95888, 0.9654; measured 2,361, 0.95910, 0.9655 | inside 1 sd |
| 15 pairs that finished after the fit | sign 53%, MAE 0.00104 | a noise-floor detector, not an oracle, on genuinely new mechanisms |

The last row is the honest one. The surrogate tells you which knob moves are inside the noise. It cannot tell you
whether a new mechanism will work. In our hands its main use was to stop us from spending chip time: none of the
282 local candidates around K60 cleared the "confirm on Trainium" bar (`d vs base` <= -0.0003 with support).
Every later win came from mechanisms the data had never seen: the row pool, the EMA blend and the prewarm. The
full validation reports are in `research/sim-data/*-validation.md` and
[`simulator-report-20260929.md`](simulator-report-20260929.md).

### Using it

```bash
pip install -r requirements.txt

# fit the models from the shipped dataset (writes research/sim-data/models.pkl, a local file; ~2 s)
python -m ffsim fit --runs research/sim-data/runs.jsonl --pairs research/sim-data/validation-pairs.json \
    --uploads research/sim-data/official-uploads.csv --out research/sim-data/models.pkl
python -m ffsim validate                              # the calibration reports above

# one recipe; omit --seed to marginalise over seeds
python -m ffsim simulate --recipe ffsim/examples/recipe-K60.json --n 2000 --seed 73
python -m ffsim simulate --recipe my-candidate.json --vs ffsim/examples/recipe-K60.json   # paired

# every single and double knob change around K60: md/csv/json tables plus a "confirm next" list
python -m ffsim search --space ffsim/examples/space-k60-local.json --n-sims 1000 --top 20 \
    --out out/search-k60-local
```

`search` marginalises over training seeds, so its K60 base reads about 0.001 worse (official ~0.9665) than the
seed-73 `simulate` result (0.9654). Writing to `out/` (git-ignored) keeps the shipped `research/sim-data/search-*`
outputs unchanged.

A recipe is a JSON file: a code version, a chip, a time target and the `FF_*` knobs (see `ffsim/examples/`). A
search space adds `dims` and `max_changes`. Read every result with the decision rule in `ffsim/README.md`
("Trust boundaries"):

- confirm on chip only `support ok` candidates whose `d vs base` is -0.0003 or better;
- everything smaller is a tie;
- anything `EXTRAP`, `NEVER-VARIED` or "cost unknown" needs a speed screen first.

`P(beat best)` compares against `--best`, which defaults to 0.9655: K60, the best official score when the
simulator was built. Pass `--best 0.96136` (K82s4) to compare against the final best.

The release ships the outputs of these three searches (`research/sim-data/search-*`). `models.pkl` is not shipped:
it is a pickle, so build it yourself with `fit` and do not load one from someone else.

## Route 2: the GPU proxy

The GPU proxy runs the actual competition `train.py` on a CUDA GPU (or CPU). It is driven so that the GPU follows
the same optimizer trajectory as the chip:

- **The fork is generated, not hand-edited.** `ffsim/gpu/make_train_gpu.py` applies 32 checked textual patches to
  `recipes/K60/train.py` and writes `ffsim/gpu/train_gpu.py`. `python -m ffsim gpu make-train --check` verifies
  that the committed fork is byte-identical to a fresh build.
- **Virtual charged clock** (`vclock.py`). Every clock-keyed switch fires at the same step number as on chip C:
  the k1 -> k2 -> k4 accumulation phases, the eight MTP stages, the cooldown levels, the EMA save level and the
  stop. Real GPU time is ignored. The per-phase step seconds come from a chip-C log
  (`configs/schedule-K60-TT1793.json`, generated from `research/sim-data/chipC/C40_cold_K59/train.log`).
- **Data-world emulation.** The trainer reproduces the official 8-rank sharding of the organiser's data loader on
  one device, so each step sees the same tokens.
- **Calibration before use** (`calibrate.py`, `PROTOCOL.md`). No GPU verdict may influence a chip run until the
  proxy has reproduced Trainium pairs the campaign already paid for. An earlier proxy (20-21 Sep, a different fork)
  was rejected under exactly this rule when a U-net verdict flipped sign.

### What it measured (29-30 Sep 2026)

| comparison, seed 73, equal steps | GPU (L40S) | Trainium chip C |
|---|---|---|
| K60 recipe, absolute val_bpb at 2,314 steps | 0.959952 | 0.959948 |
| attention-source reuse (AS-a) | -0.00187 | -0.00181 |
| LeakyReLU(0.35)^2 vs relu^2 | -0.00114 | -0.00105 |
| cooldown 0.7 | -0.00012 | +0.00009 / -0.00037 (tie) |
| split AdamW cooldown 0.4 | +0.00045 | +0.0006 (official upload) |

What we learned operating it:

- **Default CUDA mode is not bit-repeatable.** The same seed and config on two boxes differed by 0.00056, because
  flash-attention's backward is nondeterministic. `torch.use_deterministic_algorithms(True, warn_only=False)`
  made runs identical at every logged step across boxes, at 76k tokens/s instead of 104k. The shipped
  `--deterministic` flag uses `warn_only=True`. For paired verdicts, switch it to `False`.
- **Hardware can flip the warm-up path.** Seed 58 takes the good path on the chip and a spike path on the GPU
  (see `FINDINGS.md`, section 4). Screen seeds on the GPU first, and void a pair when its step-10 loss sits on a
  different path from its control.
- **The proxy measures per-step quality only.** A GPU win that costs chip step time still has to pay for it at
  0.00057 bpb per 1%. Our GPU waves on the K63 base found nothing that beat K63, which matched the chips.
- **Cost.** About $185 of GPU time by midday 30 Sep, and a few more boxes ran after that.

### Using it

You need an NVIDIA GPU, PyTorch with CUDA, and the organiser's kit for the data loader. `prepare.py` and the
tokenizer are not redistributed here: copy them from
[github.com/aws-neuron/trainium-frontier](https://github.com/aws-neuron/trainium-frontier) into the repo root.

```bash
pip install -r requirements-gpu.txt
python -m ffsim gpu make-train --check                         # the fork matches recipes/K60/train.py
python -m ffsim gpu schedule                                   # the pinned K60 switch table
python -m ffsim gpu plan --config ffsim/gpu/configs/gate-k60.json --quote-prices    # runs, GPU-hours, dollars
```

`ffsim/gpu/launch.sh` brings one EC2 GPU box up and down. It reads `AWS_PROFILE`, and every subcommand that spends
or destroys needs `--yes`. `fleet.py` runs a config's run list, and `calibrate` scores the results against the
Trainium pairs. The gates are in `ffsim/gpu/PROTOCOL.md`, the operations in `RUNBOOK.md`, and what the fork
changes in `PATCH_NOTES.md`. Those three are design-time documents, published as written apart from account
details.

To point the proxy at a newer recipe, run `python -m ffsim.gpu.make_train_gpu --source recipes/K82s4/train.py
--out /tmp/train_gpu_k82.py`. All 32 patches apply to K82s4. The patcher refuses any anchor it cannot find
exactly, so with other recipes a failing patch tells you which part moved. The virtual clock's defaults are
K60's (time target 1793 s, cooldown 0.6). Before trusting the switch steps for K82s4, give it K82s4's schedule
(1795 s, cooldown 0.7) and a step-time triple from a chip log of that recipe.

## Collecting your own chip data

`ffsim/harvest.py` copies run directories off Trainium instances over AWS SSM. It is read-only on the chip. Name
your chips through the environment: `FFSIM_AWS_PROFILE`, optionally `FFSIM_AWS_ACCOUNT` (an identity check), and
`FFSIM_CHIP_C_REGION` / `FFSIM_CHIP_C_INSTANCE` (likewise for D). `build-dataset` then turns run directories into
`runs.jsonl`. Three reference run directories ship in `research/sim-data/chipC/` so that the parsers, the
schedule generator and the tests have real logs to work on.
