# FrontierForge: a Trainium run simulator and our Phase 1 recipe

**Try the simulator UI locally:** `pip install -r space/requirements.txt` then `python space/app.py`, and open http://127.0.0.1:7860 (Gradio UI + API)


This is what team FrontierForge built for Phase 1 of the AWS Trainium Frontier challenge (15 Sep - 1 Oct 2026):

- **`ffsim`**, a simulator that predicts what a `train.py` recipe will score on Trainium without spending chip
  time. It has a numpy surrogate fitted on about 1,200 real chip runs, and a GPU proxy that replays the Trainium
  schedule step for step.
- **The exact-oracle method**: a deterministic chip rehearsal plus a near-constant offset predicts the official
  score. Over the 13 uploads since K50 the offset averaged +0.0067 (sd 0.0004, range +0.0061 to +0.0074); the
  largest miss against a written projection was 0.0006 (K73s6, a salt-selected draw).
- **Our best recipe** (`recipes/K82s4/train.py`) and what we learned getting there: what worked, what lost, and
  why the top 10 was out of reach on recipe changes alone.

Our best scored upload was **0.9617** (K77a). Our last uploads, K82s4 (projected 0.9611-0.9619) and a planned
final upload of the best rehearsed shuffle salt, had not been scored when this was written. We finished outside
the top 10: at our last check (about 11 PM CDT on 30 Sep), #10 was 0.9555 and #1 was 0.9328.

The challenge, its rules, the data pipeline and the baseline are the organiser's:
[github.com/aws-neuron/trainium-frontier](https://github.com/aws-neuron/trainium-frontier). None of the organiser's
kit files are redistributed here.

## Results

| date (CDT) | upload | official val_bpb | what it added |
|---|---|---|---|
| 15 Sep | F2 | 1.1463 | first upload |
| 21 Sep | K40 | 0.9953 | first score under 1.0 |
| 26 Sep | K50 | 0.9760 | multi-token prediction, batch warm-up |
| 29 Sep | K60 | 0.9655 | attention-source reuse |
| 30 Sep | K70a | 0.9625 | row-pool data sampling (-0.0023 on chip, -0.0022 officially) |
| 30 Sep | K77a | **0.9617** | salt 4 + EMA blend 0.6 |
| 30 Sep | K82s4 | not yet scored | + EMA prewarm (rehearsal 0.954472, projected 0.9611-0.9619) |

Every scored upload, with its rehearsal and offset, is in
[`results/official-scores.csv`](results/official-scores.csv). The story behind them is in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

## How the simulator works

```
official_bpb = quality(recipe, steps, seed) + offset(eval shard)
steps        = the batch-schedule walk over the charged-time budget at the step time (code, chip) runs at
```

Speed and quality are modelled separately because the challenge scores a fixed wall-clock budget. A change that
costs 1% of step time has to buy about 0.00057 bpb per step just to break even.

- **Route 1, the surrogate** (`ffsim/*.py`, numpy only). It has a step-time model per code lineage, a replay of
  `train.py`'s charged-clock schedule, a ridge-regression quality model fitted on 131 full chip runs, and a
  rehearsal-to-official offset model. It runs a Monte Carlo with common random numbers, so paired comparisons are
  honest about noise.
  - **Calibration:** K60 was predicted inside 1 sd on everything (2,365 +- 9 steps vs 2,361 measured; 0.9654 vs
    0.9655 official). K59 was inside 1 sd on bpb and official, but its steps came out 1.3 sd high (2,369 vs
    2,357). Sign agreement was 85% on 99 held-out Trainium pairs. The leave-one-pair-out MAE (0.0004) misses
    its 0.0003 target, and the step-time model misses its 0.5% target (0.545%), so `ffsim validate` prints
    `passes: False` and `'mae': False`. Both misses are expected and documented in `docs/SIMULATOR.md`. The
    fitted offset is +0.00653 (sd 0.0004) over the seven chip-C uploads K51-K60.
  - **Limit:** new mechanisms are outside what it can know. On pairs run after the fit it was a noise-floor
    detector, not an oracle.
- **Route 2, the GPU proxy** (`ffsim/gpu/`). It is a generated CUDA fork of the real `train.py`, driven by a
  *virtual charged clock*, so every batch-phase, MTP-stage and cooldown switch fires at the same step as on the
  chip. At equal steps it reproduced Trainium closely:
  - K60 seed 73: 0.959952 on an L40S vs 0.959948 on chip C;
  - attention-source effect: -0.00187 vs -0.00181;
  - LeakyReLU^2 vs relu^2: -0.00114 vs -0.00105.

Details, calibration tables and the caveats are in [`docs/SIMULATOR.md`](docs/SIMULATOR.md).

## Quick start

Python 3.11 and numpy are all route 1 needs.

```bash
pip install -r requirements.txt
python -m pytest -q tests/                      # 255 pass, 5 skip (they need the full private chip harvest)

# fit the three models from the shipped dataset (~2 s), then check them
python -m ffsim fit --runs research/sim-data/runs.jsonl --pairs research/sim-data/validation-pairs.json \
    --uploads research/sim-data/official-uploads.csv --out research/sim-data/models.pkl
python -m ffsim validate

# predict one recipe, then rank every single and double knob change around it
python -m ffsim simulate --recipe ffsim/examples/recipe-K60.json --n 2000 --seed 73
python -m ffsim search --space ffsim/examples/space-k60-local.json --n-sims 1000 --top 20 \
    --out out/search-k60-local
```

`search` marginalises over training seeds, so its K60 base comes out about 0.001 worse (official ~0.9665) than
the seed-73 `simulate` figure (0.9654). The shipped outputs of the three searches are in
`research/sim-data/search-*`; writing to `out/` (git-ignored) leaves them untouched.

The GPU proxy additionally needs CUDA PyTorch (`requirements-gpu.txt`), plus the organiser's `prepare.py` and
tokenizer copied into the repo root (see [`docs/SIMULATOR.md`](docs/SIMULATOR.md#route-2-the-gpu-proxy)).

## Reproducing the recipe on Trainium

1. Set up the organiser's kit on a trn2 instance as its README describes.
2. Replace its `train.py` with [`recipes/K82s4/train.py`](recipes/K82s4/train.py). Every setting is a default in
   the file: run it with no environment overrides, seed 73.
3. Run the kit's training and public-shard evaluation. A cold run on our chips gave val_bpb 0.954472 at 2,388
   steps on the first 2M public tokens. Add about +0.0066 to +0.0070 for the private-shard score
   ([`docs/EXACT-ORACLE.md`](docs/EXACT-ORACLE.md)). That is the final-day range; the simulator's offset model,
   fitted on the seven chip-C uploads K51-K60, gives +0.00653.

`recipes/K60/train.py` is the 29 Sep recipe (official 0.9655). The GPU proxy is generated from it.

## Key findings

- **Steps were the main currency.** Near our operating point, `delta_bpb ~= 0.063 * ln(compute ratio)`: 1% more
  steps is worth about 0.00057. Most of our descent from 1.146 to 0.962 came from making the same model run more
  steps in 30 minutes: 912 steps on 18 Sep, about 2,390 on 30 Sep.
- **Row-pool data sampling was the best late lever: -0.0023 on chip, -0.0022 officially** (0.9647 -> 0.9625). It shuffles
  rows within a 256-batch pool, starting after the warm-up.
- **EMA blend 0.6 + EMA prewarm** gained about 0.0005, plus 15-30 steps that the first EMA kernel compile would
  otherwise eat inside the charged clock.
- **Seeds are a lottery with a known shape.** About 40% of seeds take a warm-up "spike" path in steps 3-10 that
  costs +0.003 to +0.0045 for good. Any change that touches steps 0-19 re-rolls it. Shuffle salts re-draw only the
  data order (sd about 0.0004) and partly transfer to the private shard.
- **What lost:** bigger models (depth 10 +0.004, 2 KV heads +0.006) and a smaller, faster one (depth 8 +0.005).
  Also: a late k=8 batch, split AdamW cooldown, AdamW LR retunes, NoPE and dropping MLPs. Scalar retunes won 0 of
  61 knob keys.
- **The gap to the top 10 was kernel throughput, not recipe.** From our best official 0.9617, reaching #10
  (0.9555) needed about +10% effective compute, and #1 (0.9328) about +58%. We ran about 285k tokens/s at
  roughly 30% MFU.

## What we would do next

Write NKI kernels for attention, the fused LeakyReLU^2 MLP and the optimizer updates (+15-30% tokens/s would be
worth 0.009-0.017). Then grow the model into the freed time, keeping the data-efficiency stack on top. Price every
kernel with a 60-step chip screen and the exchange rate before building it, and confirm it with a cold rehearsal.
More in [`docs/GAP-ANALYSIS.md`](docs/GAP-ANALYSIS.md).

## Layout

| path | what |
|---|---|
| `ffsim/` | the simulator: route 1 (surrogate), `gpu/` (route 2 proxy), `cloud/` (optional AWS batch runner), `examples/` |
| `tests/` | about 260 tests; fixture logs in `tests/fixtures/ffsim/` |
| `research/sim-data/` | the fitted dataset (`runs.jsonl`), 99 validation pairs, the upload table, validation reports, search outputs, three reference chip-C run dirs |
| `research/experiments.csv` | the chips A/B run table (an input to `build-dataset`) |
| `research/rehearsal-logs/` | the K60 cold rehearsal log (2,361 steps, 0.95910; official 0.9655) |
| `recipes/` | `K82s4/train.py` (the best recipe we publish) and `K60/train.py` (GPU proxy source), byte-identical to the uploaded files |
| `results/official-scores.csv` | every scored upload: rehearsal, official score, offset |
| `docs/` | `FINDINGS.md`, `SIMULATOR.md`, `EXACT-ORACLE.md`, `GAP-ANALYSIS.md`, and three campaign-time documents (simulator design brief, integration report, first-principles note) |

The `ffsim/` READMEs and `docs/simulator-report-20260929.md` are working documents from the campaign. They are
published as written, apart from account details and paths, so their timestamps are in UTC.

## Licence

Our own code and documents are under the MIT licence ([`LICENSE`](LICENSE)).

`recipes/*/train.py` and `ffsim/gpu/train_gpu.py` are modified versions of the organiser's Apache-2.0 baseline.
They stay under Apache-2.0 ([`recipes/LICENSE-APACHE-2.0`](recipes/LICENSE-APACHE-2.0)); see [`NOTICE`](NOTICE).
