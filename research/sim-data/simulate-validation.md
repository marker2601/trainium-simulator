# Simulator (`ffsim/simulate.py`) validation after the review fixes (29 Sep 2026)

Module: `ffsim/simulate.py`; tests: `tests/test_ffsim_simulate.py` (29 pass; the three `search.evaluate` / report tests pass
`validate=False` because `search.py`'s train.py rule 565 now refuses `FF_RELU2_FN=1` next to the stub base's
`FF_LEAKY_RELU2=0.35`, which the stubs treat as a pure speed change). Models: `research/sim-data/models.pkl`
as fitted by the quality/step-time owners at **2026-09-29T20:33:06Z** (1188 records; the pickle was refitted by them
while this was written, so every table below was recomputed on that one pickle; **none of the fixes touches a model**,
only the Monte Carlo that consumes them). Every number here is reproducible with `python -m ffsim validate` (anchor
check) and `simulate_pair(cand, K59, *models, n=4000, rng=0, seed=...)` on the recipes named; the "old noise" column
is the same pickle scored through a proxy that exposes only `predict` / `steps_slope`, which is exactly what the old
simulator used.

## 1. What changed

| finding | fix in `simulate.py` |
|---|---|
| F1 (blocker): P(< base) / P(a beats b) were sign indicators. One shared `z_quality` scaled by each recipe's marginal sd made the quality model's parameter uncertainty and its per-run residual cancel between recipes (paired sd ~ \|sd_a - sd_b\| ~ 0). | Quality noise is now three components with the right correlation: (a) seed luck + unseen seed/era/chip prior = one shared draw scaled per recipe (cancels in a pair, as it does in a same-seed confirmation); (b) coefficient uncertainty = one `beta ~ N(beta, cov)` per draw (`Noise.param_seed`), evaluated on each recipe's `QualityModel.design_row`, so it cancels only where the rows agree; (c) residual `sigma` drawn independently per recipe and draw (`residual_draw`, keyed on `recipe_key` = code version, chip, training seed, knobs, time target, never the name, so an identical recipe reproduces the base's own draw and P(base < base) stays exactly 0.5). The three add up to exactly `predict()`'s sd. A model with only `sigma` gets an independent residual of that size; a model with neither keeps its sd shared and the SimResult notes say `P(< base) is a sign indicator`. `summarise_batch` rows gain `delta_sd_vs_base` (the paired sd). |
| F3 (major): steps +1.2% high on the K59 anchor (2385 vs 2357). | Every phase median is scaled by `1 + MEDIAN_TO_MEAN_OVERHEAD` (0.0057, imported from `ffsim.steptime`, local fallback) before the step walk (`walk_step_times`; `overhead_frac` argument on `simulate*`, default on). `SAVE_RESERVE_S` 3.0 -> **5.5 s**, the measured step-clock reserve (section 2). `PHASE_RATIO_TO_K4` (used only when a step-time model gives k4 alone) 0.325/0.56 -> **0.316/0.545**, the measured chip C full-run ratios (C32..C40 k1/k4 0.306-0.318, k2/k4 0.544-0.547; C40 0.3154/0.5444), because the old values were a calibration that absorbed exactly the bias now applied explicitly. The contract cross-check walks the same overhead-scaled medians. Reported `step_time` stays the model's medians. |
| F4 (minor): `steps_slope` was called with four positional arguments, which `QualityModel.steps_slope(steps)` rejects, so the fallback returned the slope at STEPS_REF (2300). | `steps_slope(steps_mean)` is tried first, then the four-argument form, then the no-argument form. Fitted model: -0.05100 at 2300 vs -0.04992 at 2369 (0.00002 bpb per 1% of steps, small, now right). |

## 2. Step-clock reserve, from `runs.jsonl`

`time_target - charged_seconds` (charged = the log clock at the last step) over the completed chip C/D full runs with an
`ff_summary` bpb and > 1500 steps: n = 138, median **5.5 s**, p10 5.1, p90 5.9; excluding the two aborted runs
(`1639_R_m14asa_s73` 1514 s, `1630_R_lk35sc165_s73` 800 s) n = 136, mean 5.51. Rehearsals: C40 1793 -> 1787.1 (5.9),
C39 5.8, C38 5.9, C37 5.3, C36 1790.5 -> 1784.9 (5.6), C35 5.9, C33 5.0, C32 5.1; the K59 queue seeds 1760 -> 1754.3
(5.7). CONTRACT's "charged ~1790" for K59 is the harness clock (timing.txt 1790.1), not the step clock.

## 3. K59 step count (fitted medians k1 0.2921 / k2 0.5038 / k4 0.9259 s)

| reserve | overhead | steps @ FF_TIME_TARGET 1793 | vs C40 2357 | steps @ 1760 | vs the seven LK0.35 seeds 2322-2329 (median 2325) |
|---|---|---|---|---|---|
| 3.0 s | none (before) | 2385 | +1.19% | 2341 | +0.69% |
| 3.0 s | 0.57% | 2371 | +0.59% | 2327 | +0.09% |
| **5.5 s** | **0.57% (after)** | **2369** | **+0.51%** | **2324** | **-0.04%** |
| 5.7 / 5.9 s | 0.57% | 2368 | +0.47% | 2324 | -0.04% |

The remaining +0.51% at C40's budget is C40's own step time: its k4 median was 0.9312 s, 0.57% above the fitted
0.9259 (the seven seeds of the same recipe on the same chip: 0.9260-0.9279). C40 walked from **its own** medians
(0.2937 / 0.5069 / 0.9312) with the overhead and the 5.5 s reserve gives **2355** (measured 2357, -0.08%); pinned by
`test_steps_reproduce_k59_anchor`. The knob-blind anchor stub (k4 0.928, measured ratios) now walks to 2362 at 1793.

## 4. Anchor check (`python -m ffsim validate`, K59 seed 73, n = 2000, rng 0)

| | steps | bpb_2m | official | P(official < 0.9671) |
|---|---|---|---|---|
| before | 2385 +- 9 | 0.96047 +- 0.00038 | 0.96704 +- 0.00059 | 0.54 |
| **after** | **2369 +- 9** | **0.96082 +- 0.00037** | **0.96738 +- 0.00061** | **0.32** |
| measured | 2357 | 0.96092 (C40) | 0.9671 | |

bpb_2m error vs C40: -0.00045 before, **-0.00010** after (the quality model at the true 2357 steps gives 0.96106,
+0.00014; the slope -0.0499 x ln(2369/2357) = -0.00025 accounts for the rest). Marginal over seeds (n = 4000): steps
2368.7 +- 8.5, bpb_2m 0.96189 +- 0.00073, official 0.96845 +- 0.00088, P(beat 0.9671) 0.063 (before, same pickle:
2385.1 +- 8.6, 0.96155 / 0.96811 / 0.119). The seed-73 P(beat best) of 0.32 is the honest reading of a recipe whose
rehearsal + offset sits 0.0003 above the best score: K59's official 0.9671 was on the good side of its own offset.

## 5. Pair probabilities vs K59 (pickle 20:33:06Z, n = 4000, rng 0, seed marginal; "analytic" = the quality model's own predict_delta form, sqrt(d cov d + 2 sigma^2), sigma 0.0003)

| candidate | delta (bpb) | old noise: sd / P(< K59) | new noise: sd / P(< K59) | analytic sd / P |
|---|---|---|---|---|
| FF_LEAKY_RELU2=0.3 | -0.00002 | 0.000002 / 1.000 | 0.00043 / 0.53 | 0.00043 / 0.53 |
| FF_WD_SCHED=1 | -0.00004 | 0.000023 / 0.98 | 0.00045 / 0.54 | 0.00045 / 0.54 |
| FF_COOLDOWN_FRAC=0.7 | -0.00002 | 0.000032 / 0.78 | 0.00047 / 0.52 | 0.00047 / 0.52 |
| FF_COOLDOWN_FRAC=0.7 + FF_WD_SCHED=1 | -0.00008 | 0.000054 / 0.91 | 0.00049 / 0.56 | 0.00049 / 0.56 |
| FF_COOLDOWN_FRAC=0.5 | +0.00004 | 0.000032 / 0.20 | 0.00047 / 0.47 | 0.00047 / 0.48 |
| K59 itself | 0 | 0 / 0.500 | 0 / 0.500 (by construction) | 0.00042 / 0.50 |

(Re-run 29 Sep, last pass: the Monte Carlo deltas move by +-0.00001 between re-runs, their MC standard error at
n = 4000 is 0.000007; the P values and paired sds are unchanged to the digits shown.)

Seed 73 (seed effect fitted, so no shared seed-luck term): the same to +-0.01 (CD0.7+WD1 0.00050 / 0.56; old noise
0.00009 / 0.78). On the 20:29:27Z pickle the review was written against, the cooldown deltas were larger
(CD0.7 -0.00022, CD0.7+WD1 -0.00027, CD0.5 -0.00017) and the old noise gave P 0.99-1.00 where the analytic P was
0.66 / 0.69 / 0.62; the new noise reproduced those too (0.65 / 0.69 / 0.63). The pair noise floor sqrt(2) sigma =
0.00042 is what the builders' pair validation reports (median pred_sd of same-seed pairs 0.00047).

## 6. End-to-end search after the fix (`python -m ffsim search --space ffsim/examples/space-k59-local.json --n-sims 1000 --gen local`, pickle 20:33:06Z, 282 candidates, 37 in support, 2.6 s; reproduced unchanged on the last pass)

```
base K59: 2369 steps, bpb_2m 0.96191, official 0.9684 [0.9674, 0.9696], P(beat 0.9671) 0.05
  # official    d.base P<base Pbest steps    sup  changes
  1   0.9684  -0.00009   0.58  0.08  2369 EXTRAP  FF_COOLDOWN_FRAC=0.7 FF_WD_SCHED=1
  2   0.9684  -0.00007   0.57  0.07  2369     ok  FF_WD_SCHED=1
  3   0.9684  -0.00007   0.57  0.07  2369     ok  FF_LEAKY_RELU2=0.3 FF_WD_SCHED=0.5
  4   0.9684  -0.00006   0.58  0.07  2369     ok  FF_LEAKY_RELU2=0.3 FF_WD_SCHED=1
  5   0.9684  -0.00006   0.56  0.07  2369 EXTRAP  FF_COOLDOWN_FRAC=0.7 FF_LEAKY_RELU2=0.3
  6   0.9684  -0.00006   0.54  0.07  2369 EXTRAP  FF_COOLDOWN_FRAC=0.7 FF_WD_SCHED=0.5
  7   0.9684  -0.00005   0.55  0.06  2369     ok  FF_LEAKY_RELU2=0.3
  8   0.9684  -0.00005   0.54  0.06  2369 EXTRAP  FF_LEAKY_RELU2=0.25 FF_WD_SCHED=1
  9   0.9684  -0.00003   0.53  0.07  2369 EXTRAP  FF_COOLDOWN_FRAC=0.5 FF_WD_SCHED=1
 10   0.9684  -0.00003   0.51  0.37  2369 EXTRAP  FF_SCALAR_LR=0.16 FF_WD_SCHED=1
confirm next (seeds 73, 58, 67): FF_WD_SCHED=1; FF_LEAKY_RELU2=0.3 FF_WD_SCHED=0.5; FF_LEAKY_RELU2=0.3 FF_WD_SCHED=1
```

(The support flags and the cooldown deltas belong to the 20:33:06Z pickle: on the 20:29:27Z one 91 of 282 were in
support and FF_COOLDOWN_FRAC=0.7 was `ok`; the review's `FF_COOLDOWN_FRAC=0.7 FF_WD_SCHED=1  d.base -0.00028  P<base
0.99` was that pickle through the old noise.) Every P(< base) now sits at 0.51-0.58 for deltas of 0.00003-0.00009
against a paired sd of 0.00043-0.00049 (`delta_sd_vs_base` in the JSON/CSV rows): the surrogate's honest statement is
that the local K59 neighbourhood is flat within the pair noise floor, and a confirmation needs the 3-seed protocol, as
the campaign rules already say. `confirm next` still ranks by expected delta, but its probabilities are now real.

## 7. Limits kept on purpose

- The unseen era/chip prior (`extra` from `design_row` beyond the seed spread) is treated as shared between recipes.
  Exact for the search (every candidate shares the base's code version and chip); for a candidate on another chip or
  lineage it cancels a prior it should not, i.e. that P(< base) is still slightly overconfident. Splitting `extra` by
  source needs `QualityModel.design_row` to return the components.
- The residual is keyed on the recipe: a candidate that changes a knob the surrogate never saw gets its own residual
  (P(< base) ~ 0.5 +- MC noise, paired sd 0.00042), the identical recipe gets P = 0.5 exactly.
- `steptime-validation.md` section 6 still quotes the simulator's old raw-median 2385 in its table (that file is the
  step-time model's; its own numbers, 2371 / 2368 / 2324 with the overhead, agree with this walk).
- Not done here because the files belong to other modules: `report.py` line 153 still says "time_target − 3 s save
  reserve" in the legend and its confirmations list prints P(< base) without the paired sd (`delta_sd_vs_base` is in
  every row, so it is a one-line print); `cli.py`'s `_anchor_check` prints the K59 prediction beside the measured
  line but does not compute the step / bpb error vs C40 (2357 / 0.96092) and the seed median (2325 @ 1760) or flag a
  miss > 0.5%.
