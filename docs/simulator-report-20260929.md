# ffsim: Trainium run simulator, integration report (29 Sep 2026, 6:30 PM CDT)

> Campaign-time report, published as written except for account details and private file references.
> Section and log times are UTC as logged (UTC = CDT + 5 h). What happened after this report, including
> the GPU proxy (route 2) actually running, is in `docs/SIMULATOR.md` and `docs/FINDINGS.md`.

Spec: `docs/simulator-spec.md`. Contract: `ffsim/CONTRACT.md`, `ffsim/schema.py`. How to run: `ffsim/README.md`.
Fit used throughout: `research/sim-data/models.pkl`, `fitted_utc 2026-09-29T23:16:45Z`, 1,206 records, 99 validation pairs,
26 upload rows (7 chip-C calibration points). Every number below was produced by the commands in section 9 on this fit;
the search tables are `research/sim-data/search-{k60-local,k59-local,k59-wide}.{md,csv,json}`.

## 0. New facts applied today (after the 16:20 UTC data extraction)

Verified against the campaign ledger (private; entries 16:40 to 22:45 UTC) and, for the numbers, against the
chips' own `eval.log` / `steps.log` / `timing.txt` (read over SSM at 23:10 UTC, read-only; nothing was written on the chips
outside `/tmp`):

- **K60 scored 0.9655 = best official** (19:50 UTC). K60 = K59 + M14 attention-source reuse `FF_ATTN_SRC=5:6,7,8`
  (`ae396432`, 237,050 B). Rehearsal C41_cold_K60: 0.9591028698885415 @2361 steps (startup 497.3 s, charged 1790.5 s);
  offset +0.0064. Same-seed AS-a pairs 4/4: C s73 -0.00181, C s67 -0.00136, C s58 -0.00106, D s73 -0.00173 (mean -0.0015).
- Stacking AdamW-LR -16% (`FF_ADAMW_LR_SCALE=1.189 FF_SCALAR_LR=0.2379`) on K60 hurt: +0.00082 (s73), +0.00095 (s67); K61
  dropped. Closed today: XSA, CFC init 1.0, last-layer pooling, two-lane residual (+4.7% step), orthogonal init, M13
  rope/norm Functions, softcap 16.5 (~0 on LK0.35; +0.00068 on K60 s73). Top-10 cut 0.9651. K62a = K60 + EMA off + TT 1795 is
  rehearsing (C42, started 22:53 UTC).

Applied: K60 is the 7th row of `research/sim-data/official-uploads.csv`; the 18 chip C/D full runs that finished after
the 16:45 UTC harvest are rows of `research/sim-data/monitor-runs.csv` (real dir names, exact `val_bpb`); `build-dataset`
and `fit` were re-run; the simulator's default `best_official` is 0.9655 (`ffsim/simulate.py BEST_OFFICIAL`, `cli.py --best`,
`report.py` fall-backs, README); `ffsim/examples/recipe-K60.json` and `space-k60-local.json` exist and the K60 space is the
primary search base (runs.jsonl carries two **full** runs with `FF_ATTN_SRC`, C:1639_R_m14asa_s73 and D:1552_R_m14asa_s73,
both with chip logs, so the quality model has the mechanism); `space-k59-wide.json` is the random 1-3-knob space.

One thing did NOT happen: the 18 post-harvest runs are **monitor-only** records (per-run overrides, `knobs_complete`
False), and `ffsim/quality.py` drops such records from the fit by design (`require_effective_knobs`: their base recipe is
not known from a monitor line). So the quality and step-time fits are **numerically identical** to the 16:45 UTC corpus
(131 fitted runs, `no_effective_knobs` 37 -> 55). Only the offset model changed (6 -> 7 points). To bring the 18 runs into
the fit, harvest chips C and D again before the blocks end (30 Sep 11:30 UTC): `python -m ffsim harvest --chip C` and
`--chip D`, then `build-dataset`, `fit`. Section 5 uses the 18 runs the other way round: as an out-of-sample test.

## 1. What was built

`ffsim/` (pure numpy + stdlib, Python 3.11; 171 tests in `tests/test_ffsim_*.py`):

| module | what it does |
|---|---|
| `harvest.py` | pulls run dirs off chips C/D over SSM (read-only on chip) into `research/sim-data/chip{C,D}/` |
| `parse_logs.py`, `parse_csv.py`, `parse_monitor.py`, `dataset.py` | train.log / eval.log / overrides / runner.log, experiments.csv (chips A/B), the monitor table -> one `RunRecord` per (chip, run), merged by source precedence -> `runs.jsonl` |
| `steptime.py` (Part 1) | per-lineage k1/k2/k4 medians with knob features and a novel-mechanism prior; leave-one-lineage-out report |
| `quality.py` (Part 2) | ridge regression of bpb_2m on knob features + ln(steps) + seed + chip + era fixed effects; leave-one-pair-out (LOPO) and leave-one-knob-value-out (LOKVO) validation on `validation-pairs.json` |
| `offset.py` (Part 3) | rehearsal -> official offset from the scored chip-C uploads, predictive sd, chi-square interval |
| `simulate.py` | step walk over the `FF_ACCUM_SCHED` phases (5.5 s step-clock reserve, 1.0057 median->mean overhead), Monte Carlo over chip jitter / seed / coefficient / residual / offset -> `SimResult`; `simulate_pair` with common random numbers |
| `search.py`, `report.py`, `cli.py` | candidate spaces (local / oat / grid / random), train.py startup-assert validator, era-aware support flags, ranking, markdown/CSV/JSON, launch lines with the real chip-C queue job env |
| `cloud/` | Dockerfile, `run_batch.sh`, `ec2_launch.sh` (numpy-only t3.large, self-terminating; no GPU anywhere) |

Route 2 of the spec (GPU proxy) is out of scope: the 21 Sep proxy failed validation and nothing here runs a GPU.

## 2. Data harvested (counts)

- Chip corpus (16:45 UTC pull): chip C 346 run dirs / 339 train.log / 231 eval.log / 18 eval20.log (11.8 MB of train.log,
  `_meta/` with base.env, runner.log, code dirs); chip D 11 runs. Contract inventory said 341/334 and 10: the surplus is
  the runs created between the inventory and the pull (`dataset-qa.md` section 1).
- `runs.jsonl`: **1,206 records** = 1,406 source records (experiments.csv 348, monitor 708, chip_log 350) merged by
  (chip, run) (200 co-sourced). Chips A 563 / B 247 / C 349 / D 19 / unknown 28. Full runs 928 (847 with bpb_2m, 739 with
  steps, 191 with k4 step time, 124 with the complete effective env), screens 278.
- The 18 post-harvest rows (monitor-only, values from the chips' eval.log): chip C C41_cold_K60, 1711_R_m14asa_s67,
  1823_R_m14asa_s58, 1900/1931_R_k60alr84_s73/s67, 2002/2039_R_m14asd_s73/s67, 2110_R_k60sc165_s73, 2144/2221_R_g7lk35_s311/s313;
  chip D 1705_R_m14asb_s73, 1742/1928/2000_R_lk35alr84_s97/s113/s131, 1814_R_m14asc_s73, 1851_R_m14asd_s73, 2032_R_m14asd_s97,
  2104_R_m14asa_s97. (2221 s313 0.96221 and D 2032 AS-d s97 0.96793 were not yet in the ledger.)
- Quality fit: 131 full chip C/D runs with a chip log (eras M3 7, M4 4, M5 11, M6 9, M7 29, M9+ 71; residual sigma 0.0003;
  79 knobs varied). Dropped: screens 278, no bpb 81, no steps 113, short 8, no code version 464 (chips A/B), unknown era 76,
  monitor-only 55.
- Pairs: 99 same-seed pairs in `validation-pairs.json` (73 on the builder-defined task list, 33 of those in the families
  the spec names; 62 recipe pairs, 10 seed pairs, 8 prior-only pairs, 10 rehearsal pairs).
- Uploads: 26 rows, 23 calibration points (K54b rejected for size, K44_rerun and F6_dup organiser-side reruns excluded);
  the chip-C subset used by the model is now 7 points (K51, K53, K54b_min, K56, K57, K59, K60).

## 3. Validation against the spec's three targets

| target (spec) | what the shipped fit scores | verdict |
|---|---|---|
| Part 1: leave-one-code-version-out k4 error <= 0.5% | 13 lineages, 189 held-out runs. **Run-weighted over the 161 "reachable" runs (every mechanism seen in another lineage): 0.545% (60 of 161 runs above 0.5%; RMSE 0.73%)**; lineage-averaged 0.428%; over all 189 runs 0.95% by run / 1.02% by lineage; the 28 "novel" runs (RF backward, abs/max leaky forms, XSA, M13 Functions, attention-source reuse, lanes, pooling, one-hot / chunked CE, chip-D screen) are 1-14% wrong when held out. `passes: False`. | **missed** (met only on the lineage-averaged reading) |
| Part 2: sign agreement >= 80% and MAE <= 0.0003 on the paired comparisons | Leave-one-pair-out, all 99 pairs: **84.8% sign, MAE 0.00039** (RMSE 0.00055, 97% inside 2 sd; decisive 64 pairs: 96.9% / 0.00046). Task list 73: 84.9% / 0.00038; the 54 supported recipe pairs (seed pairs and prior-only pairs excluded): 81.5% / 0.00031, cluster-bootstrap 90% interval 72-90% (37% of draws below 80%); spec-named families 33: 81.8% / 0.00035. Leave-one-knob-VALUE-out (predicting a value never run): 83.6% / 0.00060 on the 73. | sign **met within noise**; MAE **missed** (the per-pair noise floor is ~0.00034 for single-run pairs) |
| Part 3: offset with sd ~0.0004 | Seven chip-C uploads K51..K60: +0.0061, +0.0063, +0.0072, +0.0069, +0.0067, +0.0062, +0.0064: **mean +0.00653, sd 0.00042** (95% interval on the sd 0.00027-0.00092; predictive sd 0.00045). All 23 calibration rows since F2: +0.00637, sd 0.00064. | **met** |

Details: `research/sim-data/steptime-validation.md`, `quality-validation.md`, `simulate-validation.md`, `dataset-qa.md`.

Anchor checks (Monte Carlo, n = 2000, rng 0, seed 73; measured values in brackets):

| recipe | steps | bpb_2m | official | P(< 0.9655) |
|---|---|---|---|---|
| K59 (M12) | 2369 +- 9 [C40: 2357] | 0.96082 +- 0.00037 [0.96092] | 0.96736 +- 0.00058 [0.9671] | 0.00 |
| K60 (M14, + `FF_ATTN_SRC=5:6,7,8`) | 2365 +- 9 [C41: 2361] | 0.95888 +- 0.00042 [0.95910] | 0.96542 +- 0.00061 [0.9655] | 0.54 |
| K60 marginal over seeds | 2365 +- 9 | 0.95996 +- 0.00076 | 0.96650 +- 0.00087 | 0.13 (P(< 0.9651) 0.05, P(< 0.9671) 0.75) |
| K62a = K60 + EMA off + TT 1795 (prospective; C42 running) | 2378 +- 9 | 0.95861 +- 0.00044 | 0.96514 +- 0.00062 | 0.71; vs K60 -0.00029 +- 0.00043, P(K62a < K60) 0.75 |

The K60 anchor is in sample for the mechanism (the AS-a coefficient comes from the two harvested AS-a runs) but the
rehearsal itself (C41, TT 1793) is not in the fit. The K62a line is a genuine forecast: the quality model has never seen
`FF_EMA=0` in a full run (`NEVER-VARIED`), so its -0.00029 is the step gain the step-time model attributes to EMA off (k4
0.9207 vs 0.9249 s, from the S60 screens) plus 2 s of TT; the campaign's own estimate was ~-0.0004. Check it against
C42_cold_K62a when it lands (~23:45 UTC).

## 4. What the first searches say

All three searches: 1,000 draws per candidate, rng 0, `--best 0.9655`. `d vs base` is the analytic paired difference with
its sd `sqrt(d cov d + 2 sigma^2)`; `P(< base)` is the matching probability (0.5 = no information; -0.0003 gives ~0.75).
Support: `ok` = every changed value seen in >= 2 M9+ runs; `interp` = between seen values; `weak` = one run, or none in the
M9+ era; `interaction` = the two knobs were never co-observed; `EXTRAP` = outside the seen range. P(beat 0.9671) columns
come from re-scoring the same spaces with `--best 0.9671` (scratch copies, identical rows).

### 4a. `space-k60-local` (primary): K60 + every single/double change over 8 knobs, 282 candidates, 42 in support

Base K60: 2364 steps, bpb_2m 0.95997, official 0.9665 [p10 0.9654, p90 0.9676], P(beat 0.9655) 0.13, P(beat 0.9671) 0.74.

| # | change(s) | official (mean) | d vs base | P(< base) | P(beat 0.9655) | P(beat 0.9671) | support |
|---|---|---|---|---|---|---|---|
| 1 | FF_LEAKY_RELU2=0.3 FF_WD_SCHED=1 | 0.96643 | -0.00007 +- 0.00045 | 0.57 | 0.15 | 0.77 | interp / ok(n=3) |
| 2 | FF_COOLDOWN_FRAC=0.7 FF_WD_SCHED=1 | 0.96643 | -0.00007 +- 0.00047 | 0.58 | 0.15 | 0.77 | interaction (cooldown 0.7: one M3 run) |
| 3 | FF_LEAKY_RELU2=0.3 FF_COOLDOWN_FRAC=0.7 | 0.96645 | -0.00005 | 0.54 | 0.15 | 0.77 | weak |
| 4 | FF_LEAKY_RELU2=0.3 FF_WD_SCHED=0.5 | 0.96645 | -0.00005 +- 0.00043 | 0.56 | 0.14 | 0.78 | interp |
| 6 | FF_WD_SCHED=1 | 0.96646 | -0.00005 +- 0.00044 | 0.55 | 0.15 | 0.78 | ok(n=3) |
| 8 | FF_LEAKY_RELU2=0.3 | 0.96648 | -0.00003 +- 0.00043 | 0.53 | 0.14 | 0.77 | interp |
| 10 | FF_WD_SCHED=0.5 | 0.96648 | -0.00002 | 0.51 | 0.13 | 0.77 | interp |

Nothing reaches the README's rule 1 (`support ok` and `d vs base` <= -0.0003). The whole in-support top is a rule-2 tie:
the surrogate's held-out MAE (0.00038) is larger than any difference in the table.

### 4b. `space-k59-local` (the previous base, same 8 dims): 282 candidates, 42 in support

Base K59: 2369 steps, bpb_2m 0.96188, official 0.9684 [0.9673, 0.9694], P(beat 0.9655) 0.00, P(beat 0.9671) 0.06.
The ordering is identical to 4a (the knob effects are additive in the model): FF_LEAKY_RELU2=0.3 FF_WD_SCHED=1 -0.00007,
FF_WD_SCHED=1 -0.00005, FF_LEAKY_RELU2=0.3 -0.00003; P(beat 0.9671) 0.06-0.08 for every row. The search cannot reach K60
from this space because `FF_ATTN_SRC` is not one of its dims (it needs the M14 code; that is what `space-k60-local` is for).

### 4c. `space-k59-wide` (random, 1-3 changes over 16 knobs with M9+ support): 401 candidates (215 triple, 156 double, 29 single), 11 in support, `--workers 8`, 5.0 s

| # | change(s) | official | d vs base | P(< base) | P(beat 0.9655) | P(beat 0.9671) | support |
|---|---|---|---|---|---|---|---|
| 1 | FF_SOFTCAP=13 FF_SOFTCAP_A=16.5 | 0.96786 | -0.00055 +- 0.00152 | 0.65 | 0.07 | 0.33 | EXTRAP (softcap 13 never run; 15 and 18 only) |
| 3 | FF_LEAKY_RELU2=0.3 FF_SOFTCAP_A=16.5 FF_WD_SCHED=2 | 0.96800 | -0.00041 +- 0.00055 | 0.78 | 0.01 | 0.16 | interaction; WD_SCHED=2 weak (1 run) |
| 5 | FF_SOFTCAP_A=16.5 FF_WD_SCHED=1 | 0.96808 | -0.00034 +- 0.00050 | 0.76 | 0.00 | 0.13 | interaction (never co-observed) |
| 6 | FF_LEAKY_RELU2=0.3 FF_SOFTCAP_A=16.5 | 0.96810 | -0.00032 +- 0.00048 | 0.75 | 0.00 | 0.12 | interp + ok(n=3) = in support |
| 9 | FF_SCALAR_LR=0.1664 FF_SOFTCAP_A=16.5 | 0.96810 | -0.00031 +- 0.00301 | 0.55 | 0.19 | 0.17 | EXTRAP (scalar LR: one M7 run; sd 6x) |
| 10 | FF_SOFTCAP_A=16.5 | 0.96812 | -0.00029 +- 0.00047 | 0.73 | 0.00 | 0.12 | ok(n=3) |
| - | FF_WD_SCHED=1 | 0.96837 | -0.00005 | 0.54 | 0.00 | 0.07 | ok(n=3) |

Every row above the noise floor carries `FF_SOFTCAP_A=16.5`. Its coefficient (-0.0002) rests on the three softcap pairs in the
fit (C s73 -0.00022, C s58 +0.00006, D s73 -0.00036). The fourth pair, run on the K60 base after the fit's data
(2110_R_k60sc165_s73: **+0.00068**), moves the four-pair mean to +0.0000, and the campaign closed the knob. Row 6 crosses
rule 1 on paper; with that pair it is a tie. The other 2-3-knob combinations add nothing beyond additivity (the model has
no interaction terms, and 226 of the 401 candidates pair knobs never co-observed).

### 4d. Worth a 3-seed Trainium confirmation?

**None of the searched candidates earns a three-seed confirmation** (rule 1) on this fit; every in-support gain is inside the
0.0003 pair-noise floor or contradicted by a post-fit measurement. Ranked by how informative a Trainium run would be:

1. **K62a rehearsal (C42, already running)**: the surrogate's forecast is -0.00029 +- 0.00043 vs K60, official 0.9651 +- 0.0006,
   P(beat 0.9655) 0.71. This is the one prospective test the fit gets tonight; if C42 lands near 0.9588 the step-side of
   the model is right, if it lands at 0.9591 EMA-off bought nothing.
2. **FF_WD_SCHED=1 on the K60 recipe** (rule 2, single knob, `ok(n=3)`, -0.00005 +- 0.00044, consistent sign on both bases):
   the only in-support single-knob candidate. Worth one seed-73 probe at most, read as a probe; three seeds would cost
   three chip-hours to resolve an expected 0.00005.
3. **FF_LEAKY_RELU2=0.3** (interp between the measured 0.25 / 0.35 / 0.5, -0.00003): the slope line search already put the
   optimum at 0.35 (0.25: +0.00033 / +0.00017); the surrogate's bowl agrees; not worth chip time.
4. The K60 **seed hunt** the campaign queued (20 seeds on C, 20 on D) is the better use of the chips than any knob above:
   a fresh K60 seed simulates to official 0.9665 +- 0.0009 (P(< 0.9655) 0.13, P(< 0.9651) 0.05); with the seed sd 0.0006 the
   best of 20 draws sits ~1.9 sd below the mean, i.e. about where s73 already is (predicted 0.95888, measured 0.95910), so
   the hunt is expected to find at most one seed as good as s73, and s73's edge is partly the selection luck the meta-model
   warned about.

Do not queue: anything with `FF_COOLDOWN_FRAC` 0.5 / 0.7 (one M3-era run each, 0 in M9+), `FF_SOFTCAP` 13 (never run),
`FF_SCALAR_LR` 0.1664 (one M7 run, prior-only sd 0.003), any AdamW-LR move (section 5: five post-fit pairs, the model is
prior-only there and the campaign closed it).

## 5. Out-of-sample check on the 15 pairs that finished after the harvest

The 18 post-harvest runs form 15 same-seed pairs the fit never saw. Each treatment was scored in the candidate convention
(control's complete env + the treatment's overrides, at the treatment's steps/seed/chip/code version) with
`QualityModel.predict_delta` (an ad-hoc script, not included):

| pair | observed | predicted +- sd | sign |
|---|---|---|---|
| C41 K60 vs C40 K59 (cold rehearsals, TT 1793) | -0.00182 | -0.00210 +- 0.00049 | ok |
| AS-a s67 (C) | -0.00136 | -0.00204 +- 0.00049 | ok |
| AS-a s58 (C) | -0.00106 | -0.00171 +- 0.00049 | ok |
| AS-a s97 (D) | +0.00115 | -0.00099 +- 0.00049 | wrong (2109 vs 2150 steps: the +0.00102 step term is in the prediction; the rest is the seed / a bad run) |
| AS-b 4:5,6,7,8 s73 (D) | -0.00086 | -0.00229 +- 0.00049 | ok (the feature is "ATTN_SRC non-empty": variants are not distinguished) |
| K60 + ALR s73 (C) vs AS-a | +0.00081 | +0.00167 +- 0.00261 | ok (prior-only) |
| K60 + ALR s67 (C) vs AS-a | +0.00095 | +0.00224 +- 0.00261 | ok (prior-only) |
| ALR s97 (D) vs LK0.35 | -0.00078 | +0.00173 +- 0.00261 | wrong |
| ALR s113 (D) vs LK0.35 | -0.00009 | +0.00181 +- 0.00261 | wrong |
| ALR s131 (D) vs LK0.35 | +0.00044 | +0.00213 +- 0.00261 | ok |
| AS-d 5:7,8 s73 (C) vs AS-a | -0.00000 | +0.00015 +- 0.00042 | wrong (tie) |
| AS-d 5:7,8 s67 (C) vs AS-a | +0.00033 | +0.00039 +- 0.00043 | ok |
| AS-d 5:7,8 s73 (D) vs AS-a | -0.00020 | +0.00002 +- 0.00042 | wrong (tie) |
| AS-c 6:7,8 s73 (D) vs AS-a | +0.00076 | -0.00007 +- 0.00042 | wrong |
| K60 + softcap 16.5 s73 (C) vs AS-a | +0.00068 | -0.00022 +- 0.00047 | wrong |

**15 pairs: sign 53% (8/15), MAE 0.00104; decisive (|observed| >= 0.0003) 12 pairs: 67%, MAE 0.00111.** Read by family:
the attention-source mechanism is predicted with the right sign 4 of 5 times but ~0.0005 too large (the coefficient is
two runs, both s73, both -0.0018); the five AdamW-LR pairs are prior-only (the sd 0.0026 says so) and the model's +0.002
is wrong in size (observed mean +0.0003); the five sub-variant / softcap-on-K60 pairs are inside the noise floor for the
model (|predicted| <= 0.0002 for four of them) and would be scored "tie" by the README's decision rule. This is the number to
hold against the 84.8% LOPO headline: LOPO measures replicate prediction of knob values still in the fit; on mechanisms
and values that are genuinely new the surrogate is a noise-floor detector, not an oracle.

## 6. Honest limits

1. The quality MAE target (0.0003) is missed at every level (0.00038-0.00039 LOPO; 0.00060 leave-one-knob-value-out), and
   the sign target is met by a margin smaller than its sampling uncertainty (cluster-bootstrap 90% interval 72-90%; one
   control run, C:0626_R_g7_s73, anchors 12 of the 54 supported pairs). In sample the same pairs give 89% / 0.00022.
2. The step-time target is missed run-weighted (0.545%) and only met lineage-averaged (0.43%); the K59 lineage itself
   (M12) is 0.66% when held out. A mechanism no fitted run carries is charged a +3% prior and its step count is a guess.
3. The surrogate only sees knobs that varied in full chip C/D runs with a chip log; `FF_ACC_IN_GRAPH`, MTP on/off and the
   pre-M9 base-env moves are era-confounded (no coefficient), `FF_EMA`, `FF_MUON_*` and most of the 143 knobs never varied
   (`NEVER-VARIED`), and 191 of the 282 local candidates and 131 of the 401 wide ones are EXTRAP. There are no interaction
   terms: a 2-3-knob candidate is the sum of its parts.
4. The 18 post-harvest runs are not in the fit (monitor-only rows are held out by contract); the AS-a coefficient rests on
   two runs; the post-fit pairs score 53% sign / MAE 0.00104 (section 5). The fit must be re-done after a final harvest.
5. The offset is 7 points (sd known to a factor ~2: 0.00027-0.00092) and calibrated on K5x-shaped recipes rehearsed on
   chip C at ~2000-2360 steps; K60's +0.0064 sits inside the band. It may not transfer to a mechanism whose step time
   reacts differently to the organiser's runtime (chip D is 7-12% slower at k4 in full runs, ~1% in screens).
6. Chip D candidates are judged only against D's own controls; the D pairs above include one 2109-step run (AS-a s97)
   whose +0.00115 is mostly the step deficit.
7. `P(beat best)` for the K60 base is 0.13 marginal over seeds and 0.54 at seed 73: the model does not say a better seed
   is likely, and it cannot see anything the campaign has not already run.
8. Route 2 (GPU calibration) does not exist; nothing here can score a new kernel or a new architecture without an S60
   screen (step time) and a full run (quality).

## 7. Stale statements left for their owners (not edited here)

> Campaign-time housekeeping, kept as written; the operational status it describes is stale. In this release the
> harvested-chip test below is skipped, because the full chip harvest is not shipped.

- `ffsim/CONTRACT.md` line 35 lists six scored uploads; K60 makes seven (mean +0.00653, sd 0.00042). `ffsim/offset.py`
  carries the seventh (`CONTRACT_SUBMISSIONS`); the contract text was not changed (frozen).
- `research/sim-data/dataset-qa.md` quotes 690 monitor rows / 1,188 records; now 708 / 1,206 (the 18 rows are listed in
  section 2 above and in `qa-metrics.json`'s note). `quality-validation.md` still says "37 no_effective_knobs" (now 55);
  its fitted numbers are unchanged.
- `ffsim/quality.py` line 148 calls the K59 reference recipe "the current best, official 0.9671"; K59 remains the
  reference point of the design (x = 0), the score description is historical.
- `tests/test_ffsim_parsers.py::test_harvested_chip_dirs_parse` fails as before on C:0036_F34_b262k_d14_s50 (a depth-14
  run that finished 436 steps < the test's 500-step floor); pre-existing, unrelated to today's changes.

## 8. Tests

`python -m pytest -q tests/test_ffsim_*.py`: **170 passed, 1 failed** (28.8 s); the failure is the pre-existing one in
section 7. Pins updated today for the data that legitimately moved: `tests/test_ffsim_offset.py` and
`tests/test_ffsim_quality.py::test_load_official_uploads_real_table` (7 offset points: mean 0.00653, sd 0.00042, se 0.00016,
predictive 0.00045, CI 0.00027-0.00092), `tests/test_ffsim_simulate.py` (the `format_simresult` default now reads
`simulate.BEST_OFFICIAL`), `research/sim-data/qa-metrics.json` (reconciled monitor csv rows 690 -> 708).

## 9. Exact commands

Local (Windows Git Bash / PowerShell / Linux; Python 3.11 + numpy; run from the repo root):

```bash
# 0. (before 30 Sep 11:30 UTC) re-harvest the chips: pulls only new / unfinished run dirs (read-only on chip)
python -m ffsim harvest --chip C
python -m ffsim harvest --chip D

# 1. dataset, fit, validation reports (K59 anchor check included)
python -m ffsim build-dataset --experiments research/experiments.csv \
    --monitor research/sim-data/monitor-runs.csv --chips-dir research/sim-data --out research/sim-data/runs.jsonl
python -m ffsim fit --runs research/sim-data/runs.jsonl --pairs research/sim-data/validation-pairs.json \
    --uploads research/sim-data/official-uploads.csv --out research/sim-data/models.pkl
python -m ffsim validate

# 2. one recipe (seed 73 = that seed's effect; omit --seed = marginal over seeds); paired comparison
python -m ffsim simulate --recipe ffsim/examples/recipe-K60.json --n 2000 --seed 73
python -m ffsim simulate --recipe ffsim/examples/recipe-K59.json --n 2000
python -m ffsim simulate --recipe cand.json --vs ffsim/examples/recipe-K60.json --n 4000 --seed 73

# 3. searches (md/csv/json under research/sim-data/)
python -m ffsim search --space ffsim/examples/space-k60-local.json --n-sims 1000 --top 20 --out research/sim-data/search-k60-local
python -m ffsim search --space ffsim/examples/space-k59-local.json --n-sims 1000 --top 20 --out research/sim-data/search-k59-local
python -m ffsim search --space ffsim/examples/space-k59-wide.json --gen random --n-random 400 --n-sims 1000 --top 20 \
    --workers 8 --out research/sim-data/search-k59-wide
python -m ffsim report --results research/sim-data/search-k60-local.json --top 40      # re-render with another top-K

# 4. tests
python -m pytest -q tests/test_ffsim_*.py
```

Timings on this laptop: build-dataset 11.7 s, fit 0.2 s (+1.6 s of reports), validate 1.8 s, simulate < 1 s, the
282-candidate local searches 0.4 s compute / 1.1 s wall, the 401-candidate wide search 5.0 s wall with 8 workers.

Cloud (`ffsim/cloud/README.md`; numpy-only t3.large, self-terminating, < $0.10 per run, no GPU):

```bash
# local Docker
docker build -f ffsim/cloud/Dockerfile -t ffsim .
docker run --rm -v "$PWD/research/sim-data:/work/research/sim-data" ffsim \
    ffsim/examples/space-k60-local.json local 1000 20 4        # space gen n_sims top workers

# EC2: dry run (read-only checks + the exact commands), then execute
bash ffsim/cloud/ec2_launch.sh --profile "$AWS_PROFILE" --region us-east-1 --bucket MY-BUCKET \
    --instance-profile <INSTANCE_PROFILE> --spot --space ffsim/examples/space-k60-local.json --gen local
bash ffsim/cloud/ec2_launch.sh --profile "$AWS_PROFILE" --region us-east-1 --bucket MY-BUCKET \
    --instance-profile <INSTANCE_PROFILE> --spot --space ffsim/examples/space-k60-local.json --gen local --yes
aws --profile "$AWS_PROFILE" --region us-east-1 s3 sync s3://MY-BUCKET/ffsim/<stamp>/results/ research/sim-data/cloud-<stamp>/
```

`ec2_launch.sh --yes` tars `ffsim/` + `research/sim-data/`, uploads it, presigns a 2-hour URL, and launches a user-data
script that installs numpy, runs `run_batch.sh` (fit, then search), uploads the results and shuts the box down. On
Windows Git Bash the user-data path is converted with `cygpath -m` (tested against the native aws.exe).

## 10. Files written or changed in this integration pass

New: `ffsim/examples/recipe-K60.json`, `ffsim/examples/space-k60-local.json`, `ffsim/examples/space-k59-wide.json`,
`research/sim-data/search-k60-local.{md,csv,json}`, `research/sim-data/search-k59-wide.{md,csv,json}`, this report.
Changed: `research/sim-data/official-uploads.csv` (+K60), `research/sim-data/monitor-runs.csv` (+18 rows),
`research/sim-data/qa-metrics.json` (row count + note), `research/sim-data/runs.jsonl` and `models.pkl` (rebuilt),
`research/sim-data/search-k59-local.{md,csv,json}` (re-run), `ffsim/simulate.py` (`BEST_OFFICIAL` 0.9655),
`ffsim/cli.py` (`--best` default 0.9655; anchor line prints the best in force), `ffsim/report.py` (fall-backs read
`BEST_OFFICIAL`), `ffsim/offset.py` (`CONTRACT_SUBMISSIONS` + K60; notes), `ffsim/README.md` (quick start, table, validation
bullets, worked example, speed, outputs), `tests/test_ffsim_offset.py`, `tests/test_ffsim_quality.py`,
`tests/test_ffsim_simulate.py` (pins, see section 8). Nothing committed; nothing written on the chips outside `/tmp`.

## 11. Phase 2 pointer: the GPU proxy track (route 2), integrated 30 Sep 00:40 UTC

Route 2 of `docs/simulator-spec.md` Part 2 is **built, unit-tested and unrun**: `ffsim/gpu/` holds the
equal-steps CUDA fork of the K60 `train.py` (`make_train_gpu.py` -> `train_gpu.py`, virtual charged clock from
`configs/schedule-K60-TT1793.json`: phase 2 at step 336, phase 3 at 829, cooldown at 1214, stop 2361), the fleet
runner, the calibration rule (`PROTOCOL.md` 3.3, frozen rows, `calibrate --freeze/--rule`) and `launch.sh`. It is
reachable from this CLI as `python -m ffsim gpu <plan|schedule|run|report|verdict|from-search|prices|calibrate|make-train|launch> ...`
(lazy imports; numpy + stdlib here, torch only on the box). Nothing has run on a GPU. Read-only facts: us-west-2
on-demand G/VT quota 64 vCPUs granted (0 in use), the 39-run / 16-pair gate is 90,336 optimizer steps = 115.8 GPU-hours
at the estimated 4.5 s/step (about $217 on a g6e.xlarge, $317 on a g6e.12xlarge, 29 Sep list prices). The 20 Sep proxy
flipped a verdict's sign against Trainium, so no GPU delta may influence a chip slot before `calibrate` accepts.
**The spend decision belongs to the owner**; `launch up --yes` is the only command that spends. Commands, status and
trust boundaries: `ffsim/README.md` "Phase 2: GPU proxy track", `ffsim/gpu/README.md`, `PROTOCOL.md`, `RUNBOOK.md`.

Tests after the integration: `python -m pytest -q tests/test_ffsim_*.py` = **259 passed, 1 failed** (65 s); the failure
is still the pre-existing `test_ffsim_parsers.py::test_harvested_chip_dirs_parse` of section 7 (a 436-step chip-C run
dir, not touched here). New: `tests/test_ffsim_gpu_cli.py` (12 tests: registration, `--config` pass-through, lazy
import in a subprocess, `plan`/`schedule`/`calibrate --freeze`/`verdict` (pending)/`make-train`/`from-search` through
the group, `launch` against a stub bash). Files touched by this pass: `ffsim/cli.py` (the `gpu` group), `ffsim/README.md`
(the Phase 2 section, module map), this section, `tests/test_ffsim_gpu_cli.py`. Nothing committed; nothing launched.
