---
title: FrontierForge Trainium Simulator
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Predict Trainium val_bpb of a training recipe
---

# FrontierForge Trainium simulator

Predict what a `train.py` recipe from the AWS Trainium Frontier challenge would score (val_bpb) without a chip
run, and convert throughput gains into score. This Space is a front end for **ffsim** route 1, the numpy surrogate
team FrontierForge built during Phase 1 (15 Sep - 1 Oct 2026). Full code, data and write-ups: https://github.com/Marker2601/trainium-simulator.

## Tabs

- **Predict a recipe**: start from K82s4 (our best published recipe, anchored to its measured rehearsal
  0.954472 @ 2,388 steps) or K60 (the simulator's calibration recipe), change schedule / LR / mechanism knobs, and
  get predicted steps, step time per batch phase, local-chip val_bpb and the official-equivalent (local + offset
  +0.00653) with p10-p90 bands, plus a paired comparison against the default and a support / trust verdict.
  Inputs are checked first: impossible or far-out values return an error, and a recipe train.py would refuse is
  reported as invalid instead of being scored.
- **Speed -> score calculator**: `delta_bpb = 0.063 x ln(compute ratio)` both ways (throughput gain -> score,
  target score -> throughput needed), next to the fitted quality model's own (flatter) step slope and the
  0.00057-per-1% rule of thumb.
- **About**: how it works and how accurate it is.

## Accuracy, in one paragraph

Measured error on genuinely new changes is about 0.001 bpb (15 post-fit pairs: MAE 0.00104, sign 53%); held out on
99 historical pairs it scores 85% sign agreement and MAE 0.00039, at the pair noise floor. Differences at the
+-0.0003 level are a coin flip: read them as ties. The p10-p90 bands are the simulator's own spread and do not
include that out-of-sample error. It is calibrated to single-chip Trainium (trn2) runs of one recipe family
(depth 9, width 1024, 262k-token batches, ~1,760-1,795 s budget). **Model size is not modelled:** depth, width, head
dim, KV heads and MLP width never varied in the fitted runs, so any change to them gets a flat +3% "cost unknown"
step-time charge (whatever the direction) and an unfitted quality prior.

## API

Every button is an endpoint (`/predict`, `/predict_overrides`, `/speed_to_score`, `/score_to_speed`):

```python
from gradio_client import Client
c = Client("jaggu6408/trainium-simulator")
c.predict("K82s4", "FF_COOLDOWN_FRAC=0.6 FF_MATRIX_LR_SCALE=2.3", 73, "C", 2000, api_name="/predict_overrides")
c.predict(0.9617, 10, api_name="/speed_to_score")
c.predict(0.9617, 0.9555, api_name="/score_to_speed")
```

`predict_overrides(base, overrides, seed, chip, n_draws)`: base `K82s4` or `K60`; overrides as `KEY=VALUE` pairs or a
JSON object (`FF_SEED` there is ignored); seed is one of the 26 seeds in the fitted data (51, 53, 58, 61, 62, 67, 71,
73, 77, ... 307), or < 0 for the average over seeds (any other seed also gives the average, with a note); chip `C`
or `D` (`H` is treated as `C`). Returns the full prediction as JSON, `{"error": ...}` for bad input, or
`"invalid": true` with no numbers for a recipe train.py would refuse.

Reference scores, as of 30 Sep 2026: our best official 0.9617 (K77a; K82s4 was uploaded but not yet scored);
leaderboard #10 0.9555 at about 11 PM CDT.

## What is in this folder

| path | what |
|---|---|
| `app.py` | the Gradio app; fits the three models at start-up (~0.1 s), no pickle shipped or loaded |
| `ffsim/` | the route-1 simulator package, copied from the release (`gpu/`, `cloud/`, `harvest.py` and `CONTRACT.md` left out) |
| `research/sim-data/` | `runs.jsonl` (1,206 chip run records; the unused `FF_CODE_DIR` path knob removed), `validation-pairs.json` (99 pairs), `official-uploads.csv` |
| `recipes/recipe-K82s4.json` | K82s4's knob values (from `recipes/K82s4/train.py` in the release) |
| `LICENSE` | MIT, as in the release |

The CLI works here too, for `fit`, `simulate`, `search` and `report` (the `harvest`, `gpu` and `cloud` commands need
modules this Space does not include). It fits on first use and caches the fit as a pickle at `--models`, so point
that at a scratch path rather than into the repo:

```
python -m ffsim simulate --recipe recipes/recipe-K82s4.json --seed 73 --models /tmp/ffsim-models.pkl
```

That is the raw, unanchored model: it does not know K82s4's post-fit levers and predicts ~0.9592 vs the measured
0.9545.

## Credits and licence

Built by the **FrontierForge team**. Simulator code, data and this app: MIT (`LICENSE`). The recipe `train.py` files
of the full release are modified versions of the organiser's Apache-2.0 baseline and remain Apache-2.0; they are not
included in this Space (the `LICENSE` exception note refers to them). The challenge kit and data belong to the
organiser ([github.com/aws-neuron/trainium-frontier](https://github.com/aws-neuron/trainium-frontier)) and are not
redistributed.
