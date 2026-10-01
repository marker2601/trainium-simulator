# Step-time model (Part 1): fit and validation on runs.jsonl (29 Sep 2026)

Module: `ffsim/steptime.py` (`StepTimeModel`), tests: `tests/test_ffsim_steptime.py` (34 pass, 5 of them on this
dataset). Data: `research/sim-data/runs.jsonl` (1188 records), chips C and D, every record with a `step_time.k4`
median: 275 records. Reproduce every number here with the scratch ladder (`StepTimeModel().fit(records)`,
`.loo_report("k4")`, `.loro_report("k4")`, `.coefficient_table("k4")`, `.predict(...)`, `steps_from_step_time(...)`);
the tests `test_real_*` pin the headline numbers.

Revised 29 Sep after review (F2 = S1 / S2 / F10, then DF-8): the headline leave-one-lineage-out number is given in
both averaging conventions and `passes` is judged run-weighted, where it **fails** (0.545% > 0.5%);
`MEDIAN_TO_MEAN_OVERHEAD` is documented as the K59-era chip-C three-phase value with the population bias table; a
mechanism no fitted run carries is charged a +3% prior instead of 0; no "full run" median includes a `log_incomplete`
partial log and the 96-parameter gate pair is quoted against its single true control (DF-8, sections 4-5).
`models.pkl` was refitted (the fitted coefficients did not change) and every number below was re-derived from
`runs.jsonl` on the second pass.

## 1. Plain statement

- Leave-one-lineage-out on k4, 13 lineages, scored on the held-out runs whose mechanisms another lineage measured
  ("reachable": 161 of 189), in BOTH averaging conventions (review F2 / S1: the earlier text quoted the lineage
  average as if it were the mean over the 161 runs): **run-weighted over the 161 runs 0.545% mean absolute error (60
  of them above 0.5%; full runs 0.50% over 120, screens 0.67% over 41), RMSE 0.73%, which MISSES the spec's 0.5%
  target**;
  averaged per lineage (the mean of the 13 per-lineage means) 0.43%, under the target only because the small lineages
  (M4 n=3 0.13%, M13 n=2 0.11%, M10 n=2 0.20%) cancel the two big ones (M9 n=44 0.51%, M12 n=30 0.66%: the K59
  lineage itself misses the target held out, see section 3). For the
  M9-M15 family (the same graph with flag-gated mechanisms; 91 reachable runs) it is 0.33% by lineage / 0.51% by run.
  `loo_report()` returns `mean_abs_pct_reachable_by_lineage` and `mean_abs_pct_reachable_by_run` and judges `passes`
  on the run-weighted one (the convention `rmse_pct` always used): **`passes` is False on this dataset**. Within known
  lineages the leave-one-run-out error is 0.42% (RMSE 0.80%, run-weighted over 189 runs): that is the chip-jitter /
  screen-batch floor, not model error.
- It **cannot predict the cost of a mechanism no fitted run carries**. 28 of the 189 held-out runs are of that kind
  (XSA gate, abs/max leaky forms, RF backward, the M13 custom Functions, attention-source reuse, lanes, pooling,
  one-hot CE, fused-CE chunking, and the chip-D screen interaction), and their leave-one-lineage-out errors are 1-14%.
  The model names them in `predict()["notes"]` ("feature(s) no fitted run carries, cost unknown"), widens `sd_k4`, and
  (review F10) **charges a +3% prior per unknown knob mechanism (`NOVEL_PRIOR_LOG`)** instead of centring it on 0:
  centred on 0, the simulator's symmetric jitter factor `1 + s z` on the widened sd (E[1 / (1 + s z)] > 1) handed a
  recipe with an unknown mechanism MORE steps than its base (FF_FUSE_QKV=1 / FF_VALUE_EMBED=3 / FF_DEPTH=12: 2390.6 vs
  2385.4 at an identical k4). With the prior, K59 + FF_FUSE_QKV=1 predicts k4 0.9537 s (x1.030, sd 0.029 s) and 2299
  steps at C40's clock vs 2368 for K59; the contract's measured costs are +3.2% (QKV fusion) and +7.6% (the gate), so
  +3% is the low end, the note says it was charged, and one S60 screen replaces it with a fitted coefficient.
  Counting the novel runs, the leave-one-lineage-out error is 1.02% by lineage / 0.95% by run (the LOO scores novel
  runs by the regression alone, without the prior): **that is the honest number for "a code version with a genuinely
  new mechanism"**, and the reason the spec asked for a regression on measured screens first: one S60 screen of the
  new mechanism turns it from unpredictable into ~0.5%.
- Pre-M9 lineages (M3-M8 = real code changes, not flags) are predicted from their neighbours to 0.13-0.49% except
  **M6 (1.6%)**: the G3 -> G4 recipe change (MUON_ZERO2/ADAMW_ZERO/EMA 4 -> 32) is collinear with the M6 -> M7 code change
  in this data, so the lineage effect absorbs it and M6 cannot be told apart from M7 by knobs alone.
- Two of the four contract lessons are reproduced from the data by the fitted multipliers (RF -2.0%; M11c max +6.3% vs
  M10b abs +3.6% on the k4 median = the contract's step-count -5.9% / -3.4%). **QKV fusion is in no chip C/D run** and
  the **96-parameter gate exists only in the excluded `code` era** (direct pair +7.65%): neither is a fitted
  coefficient; asking for either returns the "cost unknown" note.
- k1 / k2 medians are noisier (few steps, forced 10-step phases in screens): leave-one-lineage-out 1.6% / 1.3%
  reachable. They are 6% / 14% of the charged clock, so their error moves the step count by ~0.1-0.2%.
- Steps from step time: the phase walk (`steps_from_step_time`) on a run's own medians over-counts its steps because a
  median under-counts the mean step. `MEDIAN_TO_MEAN_OVERHEAD = 0.0057` is the **K59-era chip-C three-phase value**,
  calibrated on the twelve hand-launched rehearsals (review S2); over the 129 chip C/D full runs the raw walk is
  +0.78% +- 0.33 and the corrected one +0.20% +- 0.33 (MAE 0.30%), with a schedule/chip bias on top of the +-0.3%
  per-run jitter floor: chip C three-phase +0.15%, two-phase `2:0.2,4` +0.46%, `2:0.12,4` +0.59%, chip D -0.33%
  (section 6). K59 predicted from `predict("M12", K59 env, "C")`: **2368 steps at C40's clock vs 2357 measured
  (+0.47%)** and **2324 at the queue budget vs the seven LK0.35 seeds 2322-2329 (mean abs 0.08%)**. C40 itself ran
  0.45% slower per step than the seven seeds of the same recipe on the same chip; the model sits on the seeds.

## 2. Data used

| set | records | note |
|---|---|---|
| chip C/D with `step_time.k4` | 275 | 191 full runs, 84 screens |
| excluded: `code_version == "code"` | 79 | the default `/root/ff-claude/code/` dir, re-deployed as M1..M9; 71 driver-era runs are a depth-12 x head-128 model at k4 1.84-2.7 s (dataset-qa.md section 8: "an unknown era, not one era") |
| excluded from the k4 fit: `n_k4 < 10` | 7 | 5-step probes (`0220_G_AF`, `1505_G_E2`, `1503_G_E1`) and partial logs (`1639_R_m14asa_s73` n=1, `0640_R_g3mpl2_s58` n=1, `0052_R_g5cd65_s58` n=1, `1454_R_g5cd65_s58` n=8); 4 of them keep a usable k1/k2 |
| **fitted, k4** | **189** | 13 lineages M3..M15 (`M7a` has one partial run, dropped); k1 fit 154, k2 fit 178 |
| of which `log_incomplete` partial logs with `n_k4 >= 10` | 2 | `D:1630_R_lk35sc165_s73` (63 k4 lines, no ff_summary, k4 1.0387) and `C:1340_R_g5mtp50_s58` (142 lines): kept in the fit (they are measurements), but left out of every direct "full run" median quoted in sections 4-6 |

Lineage keys: the chips' code dirs are flag-gated bakes of a few `train_ff.py` files (known-facts `code_dirs`, QA
section 8), so `code_k12off` = `M12` = `K59`, `code_k57` = `code_k56` = `code_m9*` = `M9`, `code_k5x` -> M5..M8,
`code_m10b/off/xa/xl` -> M10, `code_m11c` -> M11 (`LINEAGE_ALIASES`). What a dir bakes in that a screen's `overrides`
does not state is applied by `CODE_DIR_KNOBS` (M1-M5 without `FF_MUON_ZERO`/`FF_ACC_IN_GRAPH`, M6-M8 and the K54c bakes
without `FF_ACC_IN_GRAPH`, `code_m10b` abs form, `code_m11c` max form, `code_m10xa/xl` XSA layer lists, `code_k52*` EMA
every 8). Features are centred on the K57-recipe reference (relu^2, accumulation in graph, zero-init, EMA every 32,
fused CE 8), so a K57-recipe control contributes zero on every knob feature and K59 = M12 + `leaky_fnm`.

Why the shipped module needed this: fitted as shipped (code-dir keys, K59-centred defaults that read every K57-recipe
control as a leaky run, no RF / ATTN_SRC / LANES / custom-Function features, `code` kept as one lineage) its
leave-one-code-version-out error on this dataset was **4.23% mean, 25.2% RMSE** (the `code` placeholder fell back to
`code_m11c` by string similarity, -44%), and its leave-one-run-out error 2.18% (RMSE 4.67%).

## 3. Leave-one-lineage-out ladder (k4, target <= 0.5%)

"reachable" = held-out runs whose every non-zero feature is carried by at least one training-fold run; "novel" = runs
carrying a mechanism (or the chip-D screen interaction) no other lineage ran. Every mean-absolute column is given as
**by lineage / by run**: "by lineage" is the mean of the per-lineage means (each of the 13 lineages counts once,
so M4's 3 runs weigh as much as M9's 48), "by run" is over the held-out runs themselves (the convention the RMSE column
always used). `passes` judges the run-weighted reachable error. Ridge lambda 1e-3, weights n/(n+36) unless stated.
Rows A and B0 are the pre-lineage-grouping builds and are not reproducible from the current module: their by-run
numbers were never computed and are left blank.

| step | fit n | fit rmse % | LOO mean abs %, all runs (lineage / run) | LOO mean abs %, reachable (lineage / **run**) | LOO rmse %, reachable | max abs % | novel runs | reachable runs > 0.5% | M9-M15 reachable % (lineage / run, n) | passes (reachable by run <= 0.5) |
|---|---|---|---|---|---|---|---|---|---|---|
| A shipped module (code-dir keys, K59-centred defaults, `code` fitted) | 275 | 3.32 | 4.23 / - | - | - (25.2 all) | 64.3 | - | - | - | no |
| B0 code-dir keys (no lineage grouping), new features/fill | 189 | 0.32 | 1.16 / - | 0.75 / - | 1.23 | 11.30 | 19/189 | - | 1.79 / - | no |
| **B lineage grouping + baked knobs + full features (default)** | 189 | 0.36 | 1.02 / 0.95 | **0.43 / 0.54** | 0.73 | 13.86 | 28/189 | 60/161 | 0.33 / 0.51 (91) | **no** |
| C1 fewer features: 'mechanisms' (drops shape/mtp/value-embed columns) | 189 | 0.36 | 1.02 / 0.95 | 0.43 / 0.54 | 0.73 | 13.86 | 28/189 | 60/161 | 0.33 / 0.51 (91) | no |
| C2 fewer features: 'core' (9 mechanisms) | 189 | 0.46 | 1.16 / 1.09 | 0.74 / 0.78 | 1.08 | 13.34 | 16/189 | 93/173 | 0.86 / 0.86 (103) | no |
| C3 fewer features: 'minimal' (4 mechanisms) | 189 | 0.89 | 1.40 / 1.93 | 1.35 / 1.82 | 2.61 | 12.92 | 7/189 | 114/182 | 1.97 / 2.56 (112) | no |
| D1 weighting: equal weights (n_half=0) | 189 | 0.38 | 1.00 / 0.90 | 0.41 / 0.49 | 0.69 | 13.54 | 28/189 | 52/161 | 0.31 / 0.40 (91) | yes (by 0.01) |
| D2 weighting: n_half=400 | 189 | 0.31 | 1.07 / 1.11 | 0.47 / 0.71 | 0.91 | 14.16 | 28/189 | 95/161 | 0.41 / 0.81 (91) | no |
| D3 weighting: w ~ n_k4 (n_half=1e6) | 189 | 0.60 | 1.34 / 1.86 | 0.89 / 1.73 | 2.33 | 9.33 | 28/189 | 116/161 | 1.15 / 2.50 (91) | no |
| E1 residual kNN k=3, bandwidth 0.5 | 189 | 0.36 | 1.02 / 0.96 | 0.42 / 0.55 | 0.73 | 13.86 | 28/189 | 66/161 | 0.33 / 0.54 (91) | no |
| E2 residual kNN k=5, bandwidth 1.0 | 189 | 0.36 | 1.03 / 0.96 | 0.42 / 0.54 | 0.72 | 13.86 | 28/189 | 61/161 | 0.34 / 0.53 (91) | no |
| F ridge lambda=0.1 | 189 | 0.39 | 1.01 / 0.96 | 0.45 / 0.60 | 0.84 | 11.13 | 28/189 | 69/161 | 0.41 / 0.65 (91) | no |

Reading: **the 0.5% target is met only in the lineage-averaged convention (B: 0.43%); run-weighted over the same 161
runs it is 0.54% (full runs 0.50% over 120, screens 0.67% over 41) and misses**, because the two biggest lineages are the hard ones (M9 0.51% over 44 runs, M12 0.66%
over 30) and the by-lineage mean lets the 2-3-run lineages (M4, M10, M13 at 0.1-0.2%) cancel them. The run-weighted
number is the one a user of `predict()` experiences, so `passes` is judged on it. Equal weights (D1) would pass by
0.01 (0.49%, 52/161 above 0.5%); that difference is inside the +-0.3% jitter floor and picking the weighting by the
threshold would be tuning on the test, so the default stays n/(n+36) and the model is reported as missing the target.
Fewer knob features makes it worse in both conventions (the dropped mechanisms become unexplained residuals inside the
fold, not "novel"). Per-lineage grouping is what turned 0.75% into 0.43% by lineage (B0 -> B). Weighting by n_k4
outright (D3) is worse because the 40-step screens then carry no information and the single-run mechanisms are fitted
from full runs only. The residual nearest-neighbour term changes nothing at the second decimal (the ridge residuals
are already at the jitter floor), so it stays off by default (`knn_k=0`). Leave-one-run-out (B, run-weighted, 189
runs): mean abs 0.42%, RMSE 0.80%, max 5.9% (the max is M11c: leaving out the only max-form run makes `leaky_max`
unidentifiable).

### Per lineage (step B)

Per-lineage means are what the "by lineage" column averages; "reachable" counts the runs that enter the run-weighted
0.54%. The two conventions differ because M9 and M12 hold 74 of the 161 reachable runs and sit at 0.51% / 0.66%.

| held-out lineage | runs | reachable runs | predicted from | mean abs % (all) | mean abs % (reachable) | bias % | max abs % | novel runs | novel features | worst run (err %) |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 7 | 7 | M4 | 0.25 | 0.25 | +0.25 | 0.36 | 0 | - | C:1420_R_k50cd050_s67 (+0.36) |
| M4 | 3 | 3 | M5 | 0.13 | 0.13 | -0.13 | 0.16 | 0 | - | C:1805_R_g0ko_s58 (-0.16) |
| M5 | 12 | 12 | M6 | 0.44 | 0.44 | -0.36 | 0.83 | 0 | - | C:1955_R_g1ko3_s58 (-0.83) |
| M6 | 14 | 14 | M7 | 1.61 | 1.61 | -1.61 | 2.46 | 0 | - | C:0356_R_g3ptp_s58 (-2.46) |
| M7 | 31 | 31 | M8 | 0.32 | 0.32 | -0.10 | 0.76 | 0 | - | C:1415_G_KM (+0.76) |
| M8 | 3 | 3 | M9 | 0.49 | 0.49 | +0.37 | 0.66 | 0 | - | C:1521_G_F3 (+0.66) |
| M9 | 48 | 44 | M10 | 0.63 | 0.51 | -0.63 | 3.97 | 4 | ce_onehot, log2_fused_ce, mtp_loss_next | C:1456_G_CF16 (-3.97, novel) |
| M10 | 7 | 2 | M11 | 4.04 | 0.20 | -4.04 | 9.30 | 5 | leaky_abs, xsa_layers | C:1544_G_XA (-9.30, novel) |
| M11 | 8 | 7 | M12 | 0.92 | 0.22 | -0.65 | 5.85 | 1 | leaky_max | C:1939_R_g7lrm_s73 (-5.85, novel) |
| M12 | 37 | 30 | M13 | 1.52 | 0.66 | +0.73 | 13.86 | 7 | chip_D_x_screen, relu2_fn | D:1154_G_RF (+13.86, novel) |
| M13 | 7 | 2 | M14 | 0.64 | 0.11 | +0.51 | 1.41 | 5 | qk_gain_fold, rmsnorm_fn, rope_fn | C:0606_R_g7lks123_s73 (+1.41, novel) |
| M14 | 7 | 3 | M15 | 0.49 | 0.33 | +0.31 | 1.44 | 4 | attn_src_layers, pool_last | D:1552_R_m14asa_s73 (+1.44, novel) |
| M15 | 5 | 3 | M14 | 1.79 | 0.30 | -1.79 | 4.69 | 2 | lanes_extra | C:1619_G_LANES3 (-4.69, novel) |
| **by lineage / by run** | 189 | 161 | | **1.02 / 0.95** | **0.43 / 0.54** | | | 28 | | |

M12 reachable 0.66% (30 runs) is the K59 lineage predicted from M13 and it does not meet 0.5% held out: its 16 chip-C
full runs are at 0.39% (signed -0.83..+0.25%), its 6 chip-C screens at 0.75% (+0.41..+1.23%; the 28 Sep screen batch,
`1443_G_CC0/1500_G_CC1` at 0.937 vs 0.931 the next day, pulls the screen term), and its 8 chip-D full runs (the seven
LK0.35 seeds and the partial-log `1630_R_lk35sc165_s73`) at 1.13% (-1.87..-0.29%): with M12 held out, `chip_D` rests
on the single remaining chip-D full run (`D:1552_R_m14asa_s73`, 1.0362, the fastest D value) and under-predicts the
seven seeds. An earlier draft's "the M12 full runs are within 0.2-0.5%" was true of the chip-C runs only. M9 reachable
0.51% (44 runs: 29 full 0.41%, 15 screens 0.72%) is the same batch offset (`C:1443_G_CC0` -1.39%). Screens carry a
per-batch offset of about +-0.6% that the model treats as noise; their within-batch contrasts are what identify the
single-run mechanisms.

## 4. The four hard lessons, from the data itself

Direct pairs are k4 medians from `runs.jsonl` (same chip, same kind of run); "model" is the fitted multiplier relative
to the M9 reference on chip C (`predict("M9", {}, "C")["k4"]` = 0.9292 s).

| lesson (contract) | direct pair in runs.jsonl | fitted model | reproduced? |
|---|---|---|---|
| RF: one elementwise pass fewer over the 4C MLP tensor, -1.8% | RF full runs 0.9113 (n=3) vs K57-recipe `code_m9` seeds 0.9290 (n=16): **-1.91%**; rehearsal C39 0.9098 vs C36/C37 0.9288: -2.05%; S60 C 0.9135 vs 0.9310: -1.88%; S60 D 0.9189 vs 0.9410: -2.35% | `relu2_fn` **-2.02%** (6 runs); RF vs M9 reference -1.97% | yes, from 6 runs on two chips |
| M11c torch.maximum +5.9% vs M10b abs form +3.4% | M11c 0.9874 vs G7 s73 0.9282: **+6.38%** (steps 2177 vs 2314 = -5.9%); M10b 0.9623 vs 0.9282: **+3.67%** (steps 2235 vs 2314 = -3.4%) | `leaky_max` +6.30% (1 run), `leaky_abs` +4.03% (1 run); vs the M9 reference **+6.26% / +3.56%** | yes, but each form is ONE run: the coefficient is that run, not a validated effect (LOO error when held out: 5.85% / 3.48%) |
| QKV fusion +3.2% despite fewer ops | **no chip C/D run sets FF_FUSE_QKV** (known-facts: pre-K44 era, 25 Sep re-audit, code version not recorded); chips A/B carry no step times | `fuse_qkv` support 0: `predict()` returns "cost unknown: fuse_qkv" and charges the +3% prior (`NOVEL_PRIOR_LOG`) | **no** - not in the data |
| 96-parameter gate +7.6% | only in the excluded `code` era: `0427_F68_ve3_K26` (FF_VALUE_EMBED=3) 1.9904 vs its single true control `C7_cold_K26` 1.8489: **+7.65%**. The other two K26 runs are arms, not controls (`0403_v12_early_ema_K26` 1.8374 is an early-EMA probe, `0507_F69_unet_K26` 1.8974 the unet arm); an earlier draft quoted "the K26 controls median 1.8489", which is C7 alone by coincidence of the median | `value_embed_gate` support 0 (the `code` era is excluded from the fit): "cost unknown", the +3% prior (vs +7.65% measured) | data yes (one pair, one control), model **no** by design |

Also from the data: XSA gate all 9 layers +9.49% full / +10.03% S60 (model `xsa_layers` +9.81%, 4 runs); XSA on 4
layers +3.15% (0-3, full) / +3.41% (5-8, S60); MUON_ZERO -3.12% (Z1/Z2 vs I2/E1/I3 on M6; model `muon_zero_off`
+3.01%, 25 runs); FF_ACC_IN_GRAPH -2.14% (ACC1/ACC2 vs AC0/AC1 on M9; model `acc_in_graph_off` +1.27%, 76 runs,
diluted by the M7 full runs where it is collinear with the lineage change); M13 custom Functions all three -0.93%
full / -1.2% S60 (model rope -0.86%, rms -0.69%, QK-fold +0.26%, 3 runs each); attention-source reuse AS-a +0.07% S60
/ -0.95% on chip D full (model -0.40% x 3/9 layers, 3 runs: inside the noise); LANES 2/3 +3.1% / +4.7% (model
`lanes_extra` +2.39% per lane, 2 runs); leaky fnm: -0.22% (slope 0.35, 7 runs), -0.74% (0.5, 3 runs), -0.4% (0.25)
in full runs but -1.62% in the one S60 pair (model `leaky_fnm` -0.40%, 46 runs, dominated by the full runs).

## 5. Chip D vs chip C

| quantity | direct from runs.jsonl | fitted model |
|---|---|---|
| full-run k4 ratio D/C | LK0.35 seeds: D median 1.0461 (n=7, 1.0359-1.0526) / C median 0.9270 (n=7, 0.9260-0.9279) = **1.1285**; s73 pair 1.0359/0.9279 = 1.116. The eighth chip-D `code_k12off` full run, `1630_R_lk35sc165_s73`, is a `log_incomplete` partial log (63 k4 lines, no ff_summary, k4 1.0387) and is left out of every median here; with it the D median would read 1.04545 (n=8: the value `dataset-qa.md`'s chip-D table prints as 1.0454, because that table summarises every harvested full run) | `chip_D` +12.76% -> K59 on D 1.0441 s, ratio **1.1276** |
| 60-step screen k4 ratio D/C | K57-recipe C0: 0.9410 / 0.9310 = 1.0107; RF: 0.9189 / 0.9135 = 1.0059 | `chip_D + chip_D_x_screen` = +12.76% - 10.49% -> ratio **1.0094** (2 D screens) |
| k1 / k2 ratio D/C (full) | the same seven LK0.35 seeds per chip: k1 0.2900 / 0.2927 = 0.991; k2 0.5085 / 0.5043 = 1.008 (an earlier draft compared the 17 chip-C `code_k12off` full runs of mixed recipe, k1 0.2922 / k2 0.5033, with the 8 chip-D runs including the partial log, 0.2898 / 0.5080: 0.99 / 1.009) | k1 0.994, k2 1.009 |

The D slowdown is k4-specific (the k4 phase accumulates 4 micro-batches; k1/k2 are unchanged), and a 60-step screen
on D sees only ~1% of it. Holding out M12 (the only lineage with D screens) makes `chip_D_x_screen` novel: the model
then predicts a D screen as a D full run (+11-14%). Rule kept: chip D is judged against chip D's own controls.

## 6. K59 step count

`predict("M12", K59 env, "C")` (K59 = K57T recipe + FF_LEAKY_RELU2=0.35 fnm): k1 0.2921 s, k2 0.5038 s, **k4 0.9259 s**
(sd_k4 0.0033 s = 0.36%), no notes (in-lineage, every feature supported). Measured: the seven LK0.35 queue seeds k4
0.9260-0.9279 (median 0.9270, -0.12% vs the prediction); C40 rehearsal 0.9312 (+0.57% vs the prediction: C40 is the
slow run, 0.45% above the seeds of its own recipe on the same chip).

| budget | charged s / FF_TIME_TARGET | steps: raw medians | steps: with MEDIAN_TO_MEAN_OVERHEAD (0.57%) | measured |
|---|---|---|---|---|
| C40 (timing.txt) | 1790.1 / 1793 | 2385 | **2371** | 2357 (+0.6%) |
| C40 (log clock at the last save) | 1787.1 / 1793 | 2382 | **2368** | 2357 (+0.47%) |
| queue LK0.35 seeds | 1754.2-1754.8 / 1760 | 2338 | **2324** | 2322-2329, mean abs error **0.08%** over the 7 seeds |
| C40 from C40's own medians (step walk only) | 1787.1 / 1793 | 2368 (+0.47%) | 2354 (-0.13%) | 2357 |

On the twelve hand-launched rehearsals the corrected walk is within 0.3% (C36 2356 vs 2359, C39 2409 vs 2412, K54b
C32 2349 vs 2353; K51/K52 on `2:0.12,4` +0.59%); the K59 residual at C40's budget is C40's own step time. That is
where `MEDIAN_TO_MEAN_OVERHEAD = 0.0057` comes from, and it is the K59-era chip-C three-phase value, not a universal
constant (review S2: the earlier text said "+0.47% everywhere" and "right to 0.13%", which hold for those twelve runs
only). Walked from each run's own medians, the population looks like this (chip C/D full runs of a fitted lineage
with n_k4 >= 100; mean +- sd of predicted / measured - 1):

| population | runs | raw medians % | with MEDIAN_TO_MEAN_OVERHEAD % | MAE % | worst abs % |
|---|---|---|---|---|---|
| all chip C/D full runs | 129 | +0.78 +- 0.33 | +0.20 +- 0.33 | 0.30 | 2.16 (`C:1100_R_g7wu40_s58`) |
| chip C, `1:0.06,2:0.2,4` (the K57-K59 schedule) | 85 | +0.73 +- 0.29 | +0.15 +- 0.29 | 0.23 | 2.16 |
| chip C, `2:0.2,4` | 25 | +1.04 +- 0.26 | +0.46 +- 0.25 | 0.49 | 0.65 |
| chip C, `2:0.12,4` (K50-K52 era) | 4 | +1.16 +- 0.04 | +0.59 +- 0.05 | 0.59 | 0.63 |
| chip C, `2:0.3,4` | 3 | +0.78 +- 0.10 | +0.21 +- 0.10 | 0.21 | 0.31 |
| chip C, other schedules (`1:0.06,2:0.3,4` x2, `1:0.08,2:0.2,4`, `1:0.06,4`) | 4 | +0.76..+1.30 | +0.18..+0.68 | - | 0.68 |
| chip D, `1:0.06,2:0.2,4` | 8 | +0.27 +- 0.24 | -0.33 +- 0.24 | 0.33 | 0.74 |
| the 12 hand-launched rehearsals C28..C40 | 12 | +0.56 +- 0.29 | -0.03 +- 0.29 | 0.22 | 0.59 |

So the constant removes the chip-C three-phase bias (+0.73% -> +0.15%) but leaves a +0.3..0.6% over-count on the
two-phase schedules (a k4-only run has fewer, longer phases, so the slow-step tail is a larger share of it) and turns
chip D into a -0.33% under-count; the +-0.3% sd is the per-run jitter floor no constant removes, and one queue run
(`1100_R_g7wu40_s58`, warm-up 40) is +2.2% either way. Use `overhead_frac=MEDIAN_TO_MEAN_OVERHEAD` for K59-family
step counts on chip C and read the table for anything else (a schedule/chip-dependent overhead would need the
simulator, which owns the schedule, to pass it; not done here). Same-recipe siblings: K57 (M9 reference) 0.9292 s ->
2358 steps (C36 2359, C37 2359); K58-RF 0.9109 s -> 2403 (C39 2412, -0.4%); K58-LK0.5 0.9259 s -> 2368 (C38 2380,
-0.5%: the model does not separate slope 0.5 from 0.35, see section 7).

## 7. Which knobs carry the step-time signal

Fitted k4 multipliers (chip C, relative to the K57-recipe reference), with the number of fitted runs carrying the
feature. A support of 1-3 means the coefficient IS those runs; 0 means the feature never varied in chips C/D and the
model cannot speak to it.

| feature | multiplier % (k4) | k1 % | k2 % | fitted runs carrying it | meaning |
|---|---|---|---|---|---|
| chip_D | +12.76 | -0.64 | +0.86 | 11 | chip D (runtime 2.33.10) vs chip C, full run |
| screen | +0.57 | +0.05 | +0.46 | 59 | 60/120-step screen (forced phases, FF_HOST_TIMES=1) vs full run, chip C |
| chip_D_x_screen | -10.49 | +1.12 | -0.00 | 2 | extra chip-D effect in a screen |
| xsa_layers | +9.81 | +9.52 | +10.07 | 4 | XSA gated blocks / depth; 'all' = 1.0 |
| leaky_max | +6.30 | +4.39 | +6.01 | 1 | leaky relu^2, torch.maximum form (M11c) |
| leaky_abs | +4.03 | +3.28 | +3.89 | 1 | leaky relu^2, abs form (M10b) |
| muon_zero_off | +3.01 | +3.51 | +5.62 | 25 | FF_MUON_ZERO off |
| lanes_extra | +2.39 | +3.39 | +2.47 | 2 | FF_LANES - 1 |
| relu2_fn | -2.02 | -1.42 | -1.61 | 6 | FF_RELU2_FN custom relu^2 backward |
| ce_onehot | +1.33 | +1.24 | - | 1 | FF_CE_ONEHOT != 0 |
| acc_in_graph_off | +1.27 | -1.79 | +3.09 | 76 | FF_ACC_IN_GRAPH == 0 (S60 pair says +2.1%; diluted by M7-era collinearity) |
| log2_fused_ce | +1.24 | +2.65 | +1.41 | 2 | log2(FF_FUSED_CE / 8): F16 +2.7%, F4 +0.2% in S60 |
| rope_fn | -0.86 | -1.51 | -0.02 | 3 | FF_ROPE_FN custom Function |
| pool_last | +0.86 | +1.27 | +1.73 | 1 | FF_POOL_LAST > 0 |
| rmsnorm_fn | -0.69 | -1.25 | -1.27 | 3 | FF_RMSNORM_FN custom Function |
| ema_off | -0.45 | -0.42 | -0.43 | 33 | FF_EMA == 0 (screens) |
| leaky_fnm | -0.40 | -0.38 | -0.42 | 46 | leaky relu^2 fn/fnm form, any slope (K59) |
| attn_src_layers | -0.40 | -1.85 | +1.29 | 3 | blocks reading a shared attention source / depth (AS-a = 3/9): noise-level |
| qk_gain_fold | +0.26 | +1.76 | +0.78 | 3 | FF_QK_GAIN_FOLD custom Function |
| mtp_loss_next | -0.16 | -0.24 | - | 1 | FF_MTP_LOSS_NEXT |
| ema_rate | +0.05 | +1.36 | +0.08 | 35 | 32 / FF_EMA_EVERY - 1 |
| fuse_qkv, mtp_off, mtp_2, value_embed_off, value_embed_gate, depth_rel, log_aspect, log2_mb, fused_ce_off, log2_kv_heads, log2_head_dim, log2_total_batch, chip_AB, chip_AB_x_screen | not identifiable (0) | - | - | 0 | never varied in the fitted chip C/D runs (model shape is fixed at depth 9 x 113 x head 256 x MB 8 outside the excluded `code` era) |
| lineage baselines ver:m3..m15 | M3 +1.38, M4 +1.63, M5 +1.51, M6 +1.16, M7 -0.72, M8 -0.81, M9 -0.62, M10 -1.06, M11 -0.66, M12 -0.57, M13 -0.50, M14 -0.54, M15 -0.13 | | | 3-48 | after the knobs, M9..M15 sit within 0.5% of each other (the same graph); M3-M5 vs M6 vs M7/M8 are the real code changes of the pre-G7 era |

Signal, in order of size: chip D (+12.8% full, +0.9% screen); XSA gating (+9.8% all layers, ~+3.5% on four); the
leaky forms (max +6.3%, abs +4.0%); MUON_ZERO (-3.0% when on); lanes (+2.4% per lane); RF (-2.0%); CE one-hot,
accumulation in graph, fused-CE chunk count, pooling (0.9-1.3%); the M13 custom Functions (-0.7 to -0.9% each,
QK-fold +0.3%); EMA on/off (0.45%); leaky fnm (-0.4%). Zero-cost by the data (not features): softcap, QK gain,
warm-up, LR scales, WD schedule, momentum peaks, MTP phases/weights, ortho init, CFC init, MUON_SCHED_OP_US, rope
base, stochastic rounding (each within +-0.3% of its control in full runs or S60 pairs).

Recipe knobs that carry NO step-time signal are the quality model's business; the K59-lineage search space
(`FF_LEAKY_RELU2` slope, softcap, WD_SCHED, MOM_PEAK, ATTN_SRC layers) is flat in step time to within the jitter
floor except `FF_ATTN_SRC` on four layers (+0.7% S60, one run) and any lanes/pool/XSA/leaky-form change.

## 8. What the model can and cannot predict

Can:
- k4 (and k1/k2) medians for any K59-era recipe on chip C or D, full run or S60 screen, when every mechanism it uses
  is among the 21 supported features above (18 knob features plus chip D, screen and their interaction): 0.33% by
  lineage / 0.51% by run for a new lineage of the M9-M15 family (0.43% / 0.545% over all 13 lineages: the spec's 0.5%
  is met only in the lineage-averaged convention), 0.42% within a lineage.
- Step counts through the phase walk to ~0.5-0.7% for a chip-C three-phase recipe: +0.15% +- 0.3 from the corrected
  walk (a +0.3..0.6% over-count on two-phase schedules, -0.3% on chip D) plus the k4 prediction error.
- The sign and size of RF, XSA, the leaky forms, MUON_ZERO, ACC_IN_GRAPH, lanes, fused-CE chunking, EMA on/off.

Cannot:
- The cost of a mechanism no C/D run carried: QKV fusion, the value-embedding gate, FF_MTP modes, model-shape changes
  (depth, width, head dim, micro-batch, GQA, total batch), chips A/B, and anything not yet screened. It says so, and
  charges +3% per such knob mechanism (`NOVEL_PRIOR_LOG`) rather than 0, so a search cannot reward the unknown.
- Better than the screen-batch offset (+-0.6%) from a single S60 screen: a new mechanism's cost needs a same-batch
  control, which the campaign's S60 pairs provide; the model's single-run coefficients (support 1-3) inherit that noise.
- A pre-M9 lineage's real code change from its knobs (M6: 1.6%).
- The slope dependence of the fnm leaky form (0.25 / 0.35 / 0.5 are one feature, -0.4%): the full-run data disagree
  with the S60 pair (-0.2..-0.7% vs -1.6%) and the model follows the 46 full runs.
- Anything about the pre-26 Sep `code` era (depth-12 model, no phase schedule): excluded, not modelled.

## 9. Files

- `ffsim/steptime.py`: `LINEAGE_ALIASES`, `CODE_DIR_KNOBS`, `EXCLUDE_VERSIONS`, `MIN_N`, the extra mechanism
  features, `effective_env()`, `StepTimeModel(features=, min_n=, exclude_versions=, knn_k=, n_half=)`,
  `unsupported_features()`, `support()` (consumed by `ffsim.search.support_check`), `coefficient_table()`,
  `loo_report()` with the reachable/novel split in both conventions (`mean_abs_pct[_reachable]_by_run`,
  `..._by_lineage`, `n_reachable`, `n_reachable_above_target`, `judged_on`; the unsuffixed keys are run-weighted),
  `NOVEL_PRIOR_LOG` (+3% per unknown mechanism in `predict()` / `explain()`). Public API unchanged: `fit / predict / explain / loo_report /
  loro_report / known_versions / nearest_version / steps_from_step_time / simulate_phases / parse_accum_sched /
  knob_features / structural_features / version_key / version_similarity / record_weight / features_from_code`.
- `tests/test_ffsim_steptime.py`: 34 tests; `test_real_*` (5) pin the fit size, the RF / max / abs / XSA / chip-D
  multipliers, the K59 step counts, the reachable LOO in both conventions on this dataset (0.43% by lineage,
  0.545% by run, `passes` False) and, since DF-8, the direct medians of sections 4-5 (chip-D `code_k12off` full-run
  k4 1.0461 over n=7 with the partial log excluded, 1.04545 with it; the seven chip-C seeds 0.9270; F68 vs
  `C7_cold_K26` +7.65%); `test_unsupported_mechanism_is_flagged_and_charged_the_prior` pins the +3% prior.
