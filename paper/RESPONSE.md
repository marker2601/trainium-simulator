# Revision 1: response to the three internal reviews

**Paper:** *Rehearse, Don't Guess: Deterministic Rehearsal, a Run Simulator and the Price of a Step on AWS Trainium*
(retitled; the old subtitle was 22 words). **Date:** 1 Oct 2026. **Scope:** text, analysis scripts, generated
numbers, tables and figures, `check_tex.py`, `refs.bib`, `VENUES.md`, the paper CI workflow.

**How to check this revision.**

```bash
python paper/analysis/make_all.py      # regenerates numbers.tex (571 values), 23 tables, 7 figures
python paper/check_tex.py -v           # OK; estimated widths of all 26 tabulars; figure scales
python -m pytest -q tests              # 384 passed, 5 skipped
python paper/make_bundle.py            # Overleaf / arXiv bundle (7 figures)
```

Every number below comes from `paper/numbers.tex` after this run.

**What did not change.** Every number in the old paper that the reviewers checked still reproduces (offset +0.00671,
SD 0.00038, slope 0.971, frozen MAE 0.00035, LOPO 84.8% / 82.8%). What changed is what the paper claims about
them, the baselines and sensitivities next to them, and how they are presented.

**What is still not done (cannot be done without the chip or a TeX installation).** The PDF has still not been
compiled on this machine. No new chip or GPU experiments were run. The final-day E/G/H run records are still not in
`runs.jsonl`. These are listed under "Remaining weaknesses" at the end.

Legend: **Done** = changed in this revision. **Partly** = addressed as far as the released data allow.
**Not done** = needs new experiments, data or owner input, with the reason.

---

## Reviewer 1 (verdict: weak reject for MLSys; major revision for TMLR; accept for a workshop)

### Major points

**R1.1 The offset depends on the chip, not only on the text.** **Done.**
- The oracle model is now `y_i = r_i + mu + gamma_c(i) + eps_i` (Eq. offset): `mu` is the text part and `gamma_c` is a
  per-chip part, explained as (official-host speed minus rehearsal-chip speed) times the step price.
- New table `tab:offset-chips` gives the offset per chip: C +0.00652 (n=8), H +0.00684 (n=2), G +0.00712 (n=3). The
  within-chip SD is 0.00034, against 0.00038 pooled.
- A leave-one-out comparison favours the text+chip reading: MAE 0.00028 with the same-chip mean against 0.00036 with
  the pooled mean (n=13).
- Frozen-fit errors are now reported by chip (C -0.04, H +0.31, G +0.59 x1e-3), with a sign test (6 of 7 positive,
  one-sided p=0.06).
- "Near-constant" and "a property of the text" were removed from the abstract and C1. The text-gap evidence is now
  labelled n=1.

**R1.2 The oracle cannot resolve the late levers; 12/12 is weak; the projection method is unspecified.** **Done.**
- New subsection "Resolution". One upload is predicted to +-0.0008 (95%). A difference between two uploads has SD
  sqrt(2)s = 0.00053, so the oracle resolves about 0.0010 at 95% (0.00048 SD on the same chip).
- The text now says the oracle could not have measured the 0.0003-0.0006 late levers. Its role was to translate chip
  pairs onto the official scale.
- 12/12 is now reported against 11.4 expected, with a note that t(2, 0.975) = 4.3 at a burn-in of three.
- New `tab:coverage` gives coverage at 50/80/95% for burn-ins of 3 and 5 (7/12, 11/12, 12/12 and 6/10, 9/10, 10/10).
- The written projections for K82s4 and K82s7 used a final-day offset of +0.0069 to +0.0070, not the frozen chip-C
  fit, which misses them by +0.00036 and +0.00049. The exact rule was not recorded, so their closeness is called
  anecdote, not a test.

**R1.3 Determinism rests on one comparison, and the step count is not deterministic.** **Done (text); not done
(replicated reruns need chip time).**
- `sec:determinism` now separates "same trajectory given the same steps" (one pair, 1.041525 vs 1.041520, "repeatable
  to 1e-5, not bit-for-bit, n=1") from "same number of steps".
- Step-count jitter is now measured from the seed sweeps (same recipe and chip; the seed does not change step time):
  SD 2.2-6.2 steps, range 6-22 steps. Across chips the spread is 3.5%.
- Replicated cold reruns of the final recipe on one chip and across chips are listed as follow-up E-A.

**R1.4 The surrogate slope is not independent evidence; k has no uncertainty; two rates are used.** **Done.**
- One price is used throughout: kappa = 0.057 bpb per unit ln(steps) (rho = 0.00057 per 1%). The long-run anchor,
  kappa = 0.063, is used only as the upper end of a sensitivity range.
- The gap is now 10-11% to #10 and 58-65% to #1. Both kappa values are shown as columns in `tab:gap` and as two
  curves in the exchange figure.
- The surrogate's slope is stated as not independent, because its prior mean is kappa itself.
- The anchor is described as a one-point calibration, not a fit.
- The curvature prior (c = 0.026) and the 0.095-0.115 slope at ~900 steps are shown. The curvature implies 0.069 at
  the anchor's mid-point, if the anchor ran at K44's step count.
- The repository does not record how many comparisons the per-step rate rests on, so no CI is invented; the paper
  says so. A boxed definition sets where the constant is used.

**R1.5 No baseline for the quality model; "decisive" is defined by the outcome; sigma sits on the floor.** **Done.**
- Three baselines, scored with the same leave-one-pair-out protocol and a cluster bootstrap, are added to
  `tab:sim-calibration`:
  - a steps + era + chip + seed model with every knob effect switched off: 66.7% [51-79%] on recipe pairs, MAE 0.71e-3;
  - a leave-one-out majority sign: 62.1%;
  - the zero predictor: MAE 0.63e-3.
- The full surrogate scores 82.8% [73-89%], MAE 0.37e-3. A paired bootstrap gives +6 to +29 points of sign agreement
  and -0.63 to -0.17e-3 of MAE in its favour.
- Forward in time, the reviewer's suspicion is confirmed and now stated. Supported vs prior-only pairs score 89% vs
  83% (28 Sep cutoff) and 91% vs 93% (29 Sep), so the knob features add little there.
- Sign agreement on prediction-decisive pairs (|predicted| >= 0.0003) is added: 92.7% on 55 pairs (steps-only:
  75.0% on 36).
- The unfloored sigma is 0.00025, with a noise floor of 0.00028. The text no longer says the MAE is "at the noise
  floor".

**R1.6 The post-fit result cannot be reproduced; 53% is chance; "negative screening saved chip time" is
counterfactual.** **Done.**
- `paper/analysis/simulator.py` now scores the 15 pairs from their per-pair values (new `tab:postfit`): 8/15, Wilson
  30-75%, MAE 1.04e-3.
- The zero predictor (MAE 0.75e-3) beats the surrogate on these pairs, and the paper now says so.
- The per-pair predictions are still the report's (doc-quoted), because re-predicting them needs the release fit,
  which is not reconstructed.
- The "saved chip time" and "the surrogate anticipated" claims were removed. The text now says the screened-out
  candidates were never run, so this is not evidence that they were bad.

**R1.7 The GPU-proxy claim is below its own noise.** **Done.**
- The claim was removed from the abstract.
- In the body, only 2 of the 4 comparisons count as informative (equal-step chip pairs, max diff 0.00009). Cooldown
  0.7 (the mean of two chip pairs of opposite sign) and split AdamW (an official delta at different steps) are marked
  as not informative.
- The paper states that the docs do not record which mode each run used, and that box-to-box noise (0.00056) exceeds
  the agreement.
- The proxy is now called a "feasibility demonstration, consistent in sign and magnitude". Replicate GPU runs are
  listed as follow-up E-E.

**R1.8 C3 rests on n=1 at a selected seed, and the late records are not released.** **Partly.**
- Every lever now has a noise SD and a |Delta|/SD column (`tab:levers`).
- Doc-only numbers carry a dagger in the text and tables, and new appendix `tab:docfacts` lists all of them with
  their sources (generated).
- The winner's curse on seed 73 is stated in the findings and the limitations, and the chip/official agreement for
  the row pool is called "what determinism predicts, not independent evidence".
- The reproducibility statement now says the E/G/H numbers are "reported, not recomputable".
- **Not done:** multi-seed or multi-salt replications (no chip time) and harvesting the E/G/H runs (the raw logs are
  not in the public tree). Both are follow-ups E-B and E-G.

**R1.9 "The gap is kernel throughput" is untested.** **Done (framing); not done (equal-step depth test).**
- The top-level section was removed. It is now `sec:gap`, "Pricing the gap in compute", inside the findings. Its
  conclusion is titled "A hypothesis, not a finding" and names the alternative (better recipes or data handling)
  and the same-curve assumption.
- The capacity claim is softened: depth 10 lost while running fewer steps; whether it lost *only* because of its
  steps is untested. The GPU-proxy equal-step depth test is follow-up E-D.

**R1.10 Missing related work on leaderboard reuse and selection, AlgoPerf, spikes and nondeterminism.** **Done.**
- Added and cited: Dwork et al. 2015, Blum & Hardt 2015, Roelofs et al. 2019, Cawley & Talbot 2010, Dahl et al. 2023
  (AlgoPerf), Takase et al. (COLM 2025), Molybog et al. 2023, and Pham et al. (ASE 2020).
- New paragraph "Leaderboards, adaptive reuse and selection", which ties these to the K65 tests, the salt lotteries
  and the possible shard overlap.
- New paragraph "What is and is not new" in the introduction: the contribution is the measurement.
- Muon and modded-nanoGPT have no paper, so they remain web citations.

### Minor points

| # | Point | Response |
|---|---|---|
| 1 | Abstract says "fitted on 1,206" | **Done**: "fitted on 131 fully specified runs" |
| 2 | Compute understated | **Done**: about 580 trn2 instance-hours in the dataset (928 full runs, median 1,754 s charged + 380 s start-up, and 278 screens; medians imputed), plus E/G/H runs and about $185 of GPU time. The funding split is an owner TODO |
| 3 | Prewarm is not a late discovery | **Done**: "A regression, not a discovery". The prewarm was in K53, was lost when K63 turned the EMA off, and K77a re-enabled the EMA without it |
| 4 | Prewarm step counts are single runs | **Done**: jitter of 6-22 steps quoted. The chip-G drop (23 steps) is several times the jitter; prewarm vs no-EMA (+7 on G, -2 on H) is inside it |
| 5 | K70a normalisation | **Done**: formula given; new `tab:oracle-sens` shows the results with K70a as used, raw and excluded (SD 0.38/0.54/0.39, frozen MAE 0.35/0.53/0.40, 12/12, 11/12, 11/11). It is called post hoc because no time-stamped projection exists |
| 6 | Rounding | **Done**: the +-0.00005 publication rounding is stated. Offsets are now rounded half-up from exact decimals, so K38 +0.0054, K53 +0.0061 and the range +0.0061 match the CSV |
| 7 | Seed-path independence and the post-hoc threshold | **Done**: 1 seed is shared across sweeps; a distinct-seed count gives 10/25 (Wilson 23-59%). The threshold is called post hoc, and the MTP-weighted loss is explained |
| 8 | Cooldown 0.7 comparisons differ | **Done**: explained in `sec:levers` and the `tab:gpu` caption. The K63-K60 confound is stated |
| 9 | Pipeline stress test is weak; "self-improving" | **Done**: C4 is renamed "Released artifacts and a guarded contribution pipeline" and moved to the end, the weakness is stated explicitly, and the mechanics are moved to an appendix |
| 10 | Acknowledgements; double-blind | **Done**: see "Decisions recorded" below |
| 11 | Never compiled; wide tables; unreferenced labels | **Partly**: tables are narrowed and an estimated-width check is added (all 26 tabulars <= 462 pt of 469.75 pt). All labels are referenced (0 warnings). Compilation is still pending (no TeX here); CI now prints overfull boxes |
| 12 | "Fit" vs calibration; curvature | **Done** (see R1.4) |
| 13 | Slower-host rerun "is what this rate predicts" | **Done**: "consistent with rho (n=1)" |
| 14 | Figures without uncertainty | **Done**: the duplicate levers figure is dropped, and noise SD and \|Delta\|/SD are added to `tab:levers` |
| 15 | Define effective compute | **Done**: optimizer steps at a fixed recipe and batch schedule. The anchor ran at a constant 262k batch, before the batch warm-up existed |

---

## Reviewer 2 (verdict: major revision, presentation)

### Major points

**R2.1 Seven tables run past the text width.** **Done (by estimate).**
- Every generated table now uses `@{}` edges, `p{}` columns for long text and smaller fonts where needed.
- `check_tex.py` now estimates every tabular's width with the Computer Modern metrics shipped with matplotlib and
  fails above the text width. Widest: `tab:recipe` 462 pt, `sim_other` 458 pt, `levers` 449 pt (text width 469.75 pt).
- These remain estimates until the PDF is compiled.

**R2.2 Never compiled.** **Not done (no TeX installation on this machine; installing one was out of scope).**
- CI (`paper.yml`) still compiles the PDF. It now also prints `Overfull \hbox`, `undefined` and `Float too large`
  lines and uploads `main.log` with the PDF.
- `make_bundle.py` produces the Overleaf zip. Compile once and read the log before submitting.

**R2.3 Figure text at 4-5 pt.** **Done.**
- All seven figures are now drawn by `paper/analysis` at their printed width (6.5 in or less) with 7.5-8.5 pt text,
  and placed at natural size.
- `check_tex.py` reports the scale: 1.00 for six figures and 1.17 for the score history (text enlarged, not shrunk).
- The paper no longer uses `docs/figures/paper/`. The in-figure key blocks are in the captions and the ledger.

**R2.4 Contribution framing inconsistent.** **Done.**
- C1 oracle, C2 simulator, C3 accounting (with the price of a step) and C4 artifacts are used in the same order in
  the abstract, the introduction, the section titles ("C1: ...") and the conclusion.
- The price of a step is now its own subsection (3.3) with a boxed definition, ahead of the oracle.

**R2.5 Notation clashes.** **Done.**
- kappa is the price and k is the number of micro-batches.
- t_res is the stop reserve and s_j are the prior SDs.
- sigma_off is the offset residual SD and sigma_q the surrogate residual SD.
- varepsilon is used everywhere.
- y is the official score in both equations, and rho is defined in the box.
- New appendix notation table (`tab:notation`).

**R2.6 Three "prices".** **Done**: one constant, a boxed definition, and a figure, text and tables that agree (the
late change is 0.9% steps everywhere). `fig:exchange` is referenced from sections 3.3 and 7.8.

**R2.7 "Early" means two different sets.** **Done**: "chip-C calibration uploads (K51-K60)" everywhere, and the
period table row is renamed.

**R2.8 The prewarm contradicts the ledger.** **Done** (see R1 minor 3).

**R2.9 Front-matter blockers; length.** **Partly.**
- New `\ifanonymous` switch: it replaces the author block, hides the acknowledgements and swaps the repository URL.
- The author names are still an owner TODO.
- Moved to appendices: the temporal table, the seed table, the guard table, the pipeline mechanics, the GPU build
  details and the LOPO figure.
- The levers figure (a duplicate) and the gap section are gone.
- The main text is about 10,000 words with 17 floats: right for arXiv and TMLR. An MLSys version would still need
  about half the main text cut.

### Minor points

| Point | Response |
|---|---|
| Abstract overloaded and opaque | **Done**: rewritten around takeaways, with about ten numbers; "honest failure mode" is gone; the GPU proxy is out of the abstract |
| Precision varies | **Done**: the best score is 0.96136 in text and figures; tables show official scores to 4 decimals, with the two 7-decimal finals in captions; rehearsals have 5 decimals in the ledger and 6 in the frozen table (all 6-digit) |
| Mixed units | **Done**: absolute scores are in bpb; offsets, errors and effects in tables are in 1e-3 bpb, stated in each caption; the K60 absolute row in `tab:gpu` is marked "(bpb)"; nats are explained |
| Hard-coded numbers | **Done** for data numbers: 2,288/1,788 s, 28%, 0.0005, 13.x nats, 4.3, the projections, the late-lever range, the GPU cooldown pairs and the priors (-0.057, 0.026, 0.003, 0.004, 0.0006, the 0.0003 floor) are all `\val`. Design constants (the 0.0003 decision threshold, 2,300 steps, recipe settings) stay literal |
| Internal jargon | **Done**: the ledger's shorthand is spelled out in the appendix table (`PLAIN` map in `oracle.py`); family names are plain; code eras are defined; "c_proj" is now "MLP output projection" |
| `tab:features` truncated | **Done**: full encodings, wrapped |
| `tab:sim-calibration` misuses columns; the 0.9540 row | **Done**: the quality table now has consistent columns; step-time, end-to-end and offset rows moved to `tab:sim-other` with proper columns; the 0.9540 row was removed |
| Chip naming | **Done**: chip = separate instance; A-D are in the dataset; E, G, H produced only final-day rehearsals; F does not appear |
| `fig:steps` mixes batch regimes | **Done**: regimes are shaded and labelled; the caption says to compare within a regime and a chip; the figure is in the appendix |
| Colour-only captions | **Done**: chip markers, line styles (dashed spike path), hatched slow chip; captions refer to markers and styles |
| Score-history annotation | **Done**: the figure is redrawn and shows K82s4 0.96136 |
| "One closed-form solve" | **Done**: the actual procedure is described (MAP solve, active set, three empirical-Bayes sigma updates, floor, BLUP seeds); 2,300 is explained where first used |
| `\texttt` in a section title | **Done**: `\texorpdfstring`; `check_tex.py` now warns on this |
| placeins and float pages | **Partly**: fewer main-text floats per section; `[tbp]`; still to be checked in the compiled PDF |
| `\knob` cannot break | **Done**: `\knob` is now a url-style command that breaks after `, : = _`; `check_tex.py` errors if it appears in a caption or title |
| Dead source | **Done**: `\bpb` and `inputenc` removed; the refs.bib header fixed; all labels referenced |
| Compute statement | **Done** (see R1 minor 2) |
| "(doc)" invisible in the PDF | **Done**: daggers in text and tables, plus appendix `tab:docfacts` (generated) |
| Long subtitle | **Done** |
| Related work length; dated claim | **Partly**: condensed to about 640 words; the Trainium claim is dated ("a search on 1 October 2026") |

---

## Reviewer 3 (verdict: major revision; reproducibility verified; compliance and disclosure problems)

### Major points

**R3.1 The evaluation shard and "did not tune on validation data".** **Done.**
- The rules are now quoted ("pinned validation shard, computed over ~20M tokens"), and the paper says they do not
  state whether that shard is disjoint from the public validation data.
- New paragraph "The evaluation text": the offset size is consistent with the official text being like the first
  ~20M public tokens. If the shards overlap, seed and salt selection was selection on part of the evaluation set.
- All three sentences were reworded: "private shard" and "hidden metric" are gone, "Nothing in this paper uses or
  infers the private validation data" is gone, and the ethics paragraph describes exactly what was scored and
  selected.

**R3.2 The contest rules are neither cited nor checked.** **Done.**
- New `refs.bib` entry `trainiumfrontier2026rules` (Devpost rules page, accessed 1 Oct).
- The setting now gives the 30-min budget excluding start-up and compilation (rules), and the 262,144-byte limit as
  "enforced by the portal, not in the published rules".
- It gives the Round 1 deadline (11:59 PM PT, 30 Sep). K82s7 at 1:41 AM CDT = 11:41 PM PT was inside it.
- It quotes "one entry per team" and notes that the portal scored repeated uploads, and that the rules do not say
  best or last (K82s4 vs K82s7 differ by 0.00003).
- The dead kit URL footnote was removed.
- `VENUES.md` was corrected: the terms are public; the finalist paper, the NeurIPS 6-12 Dec event, the winners list
  and the absence of any publication restriction are recorded.

**R3.3 The K70a rehearsal is model-adjusted.** **Done** (see R1 minor 5).
- No timestamp exists for the normalised value, so it is now treated as post hoc.
- The protocol now names K70a as the one exception to "exact-file cold rehearsal".
- The row-pool "predicted to within 0.0001" claim was removed.

**R3.4 Salt-selected uploads are mislabelled.** **Done.**
- Marks are now `s` (salt chosen by a lottery: K73s4, K73s6, K82s7), `i` (salt 4 inherited: K77a, K82s4) and `n`
  (normalised: K70a).
- The "excluding them MAE 0.00023" sentence was replaced by group MAEs: direct 0.00053, inherited 0.00040, none
  0.00003 (n=2, one normalised).
- The confound between chip G and selection is stated.
- The lotteries are listed: 13 seeds (K57 recipe), 4 salts (K73), and a final-recipe salt lottery whose size is not
  in the released files.

**R3.5 The compute disclosure is wrong.** **Done** (instance-hours computed from `runs.jsonl`; credits cited). Who
paid is an owner TODO in a LaTeX comment.

**R3.6 The throughput story is misattributed.** **Done.**
- New `tab:steps`: step changes between uploads rehearsed on the same chip, grouped by kind. Dispatch: K43->K46 +8.9%
  (chip B). Model size: depth 10->9 +8.2% (chip A). Batch schedule: K46->K50 +10.8%. Mixed FLOPs and dispatch: K24->K38
  +67%.
- The text now restricts "dispatch" to the measured same-recipe step-time wins and drops the cross-chip "same final
  batch size" claim.

**R3.7 The most transferable findings are not in the release.** **Partly.**
- The reproducibility statement is narrowed ("reported, not recomputable"), and every such number is listed in
  `tab:docfacts` and daggered.
- **Not done:** harvesting the 30 Sep - 1 Oct E/G/H runs into `runs.jsonl`. The logs are not in the public tree and
  would need the harvest tooling on the instances (follow-up E-G).

**R3.8 Prospective claims are not archived.** **Done.**
- These analyses are renamed "pseudo-prospective (leakage-free retrospective) simulations".
- The text says only three written projections are archived.
- The frozen offset is described as the simulator's chip-C fit, whose last upload was scored on 29 Sep (before any
  final-day upload). No commit hash is cited, because the release history starts after the close.

**R3.9 The "honest failure" number cannot be reproduced.** **Partly.**
- The scoring (sign, Wilson CI, MAE, zero baseline) now runs in `simulator.py` from the per-pair values.
- The predictions themselves are still the report's, because re-predicting needs the release-time fit (follow-up
  E-C).
- The temporal hold-out's bias towards supported pairs is now stated.

### Minor points

| Point | Response |
|---|---|
| Rounding +0.0060 vs +0.0061 | **Done**: half-up from exact decimals (`offset_str`) |
| Determinism overstated | **Done** (see R1.3) |
| 84.8% includes noise pairs | **Done**: the abstract and C2 quote 82.8% on recipe pairs |
| GPU "four chip effects" | **Done**: removed from the abstract; the official-delta row is marked not informative |
| "No other team's results" | **Done**: "rank-indexed scores from the public leaderboard snapshot, and one public write-up by another participant is cited" |
| Acknowledgements and citation | **Done**: see "Decisions recorded" below; Ritwika Kancharla's write-up is cited (`kancharla2026trainiumofthought`) |
| VENUES.md terms "not public" | **Done** (see R3.2) |
| LeakyReLU, three sizes | **Done**: the three comparisons are explained (campaign log on an earlier base, a chip-C equal-step pair, official K59-K57) |
| Levers "chip pair" with n="--" | **Done**: relabelled "campaign log", with a dagger and "no run identifiers"; the batch warm-up's step equivalent is n/a (it would double-count) |
| Prewarm history | **Done** |
| Winner's-curse s_eta | **Done**: lambda is now a range of 0.46-0.75 over s_eta choices (within-chip, final-day, since-K50, 0.0002); "shrink by a quarter to a half" |
| "Nine submissions" | **Done**: the claim was removed |
| The 0.9540 row | **Done**: removed |
| refs.bib header; archival versions; access dates | **Done**: header fixed; izsak (EMNLP 2021), bouthillier (MLSys 2021), zhuang (MLSys 2022), brandon (NeurIPS 2024), hu (COLM 2024) and dao (ICLR 2024) moved to their venues; @misc entries already carry "Accessed 2026-10-01" notes |
| Stale "255 passing" | **Done**: README badge, OUTLINE and VENUES now say 384 passed, 5 skipped |
| Unreferenced labels | **Done**: 0 warnings |
| Data licence | **Done**: the reproducibility statement says code, data and documents are MIT (as the README already says), recipes Apache-2.0 and contributions CC-BY-4.0 |
| Salt files and hashes | **Partly**: the text says each salt is its own file with its own hash. **Not done**: publishing the K82s7 hash next to K82s4's in `recipes/README.md`, because the K82s7 file is not in the public tree (owner to add) |

---

## Decisions recorded

1. **Acknowledgements and the "no personal names" rule.** The workflow's computed task said "never personal names of
   other people", but the owner's own request asked to thank Shiv and Danish for helping, and Ritwika for her paper
   (she did not help). The owner's request takes precedence. The acknowledgement now reads: "We thank Danish Ali and
   Shiv for their help during the campaign, and Ritwika Kancharla, whose public write-up of her own approach to the
   contest [cite] we learned from." It does not say she helped. The names appear only in the acknowledgements, which
   `\anonymoustrue` hides. **Owner to confirm:** the spelling "Danish Ali", Shiv's surname, and that all three agree
   to be named.
2. **Ritwika Kancharla's write-up** was found on her public portfolio: *Trainium of Thought: Model and Kernel
   Co-Design under a 30-Minute Training Budget*. It is cited in related work as an example of another participant's
   write-up. None of its scores or ranks are reported.
3. **Default build = public (arXiv) version** (`\anonymousfalse`), so that the acknowledgements print. Set
   `\anonymoustrue` for MLSys or TMLR.
4. **One price**, kappa = 0.057, with 0.063 as the sensitivity bound. The per-step rate is the operating-point
   measurement; the long run sits at a lower step count, where the curvature makes the slope steeper.

---

## Remaining weaknesses an editor would still flag

1. **No compiled PDF yet.** Layout, float placement and overfull boxes are estimated, not seen.
2. **The late findings are n=1 at a selected seed and salt** (row pool, EMA blend, output-projection LR, prewarm), and
   their raw records are not released. Only attention-source reuse is replicated across seeds.
3. **Determinism rests on one pair**, and the offset's chip component rests on 2-3 uploads per chip.
4. **The price of a step has no confidence interval.** One rate has an unrecorded n, and the other is one run.
5. **The surrogate's value is limited to replicate prediction.** Forward in time its knob features add little, and on
   new mechanisms it loses to the zero predictor.
6. **The GPU proxy has two informative pairs**, run in an unrecorded mode.
7. **Possible overlap of the official shard with the public validation text**, which would make the seed and salt
   selection selection on the evaluation set.
8. **Novelty is modest for an MLSys main track**: one team, one contest, one hardware type.
9. **Owner inputs are still needed**: author names, acknowledgement confirmation, the compute-funding split, the
   K82s7 hash.

## What would most strengthen a follow-up paper (Nov-Dec 2026)

Pre-register each in the repository before running it. Ordered by value per chip-hour.

- **E-A, determinism and jitter (about 6 chip-hours).** Run five cold reruns of K82s4 on one chip and one on each of
  three other chips. Report the variance of steps and of bpb, separating trajectory determinism from step-count
  jitter. This directly answers R1.3.
- **E-B, multi-seed and multi-salt replication of the late levers (about 30-40 chip-hours).**
  - Row pool, EMA blend and prewarm, and the output-projection LR, each at 3 seeds x 2 salts on one chip, with a
    same-chip control.
  - Turns the n=1 findings into effects with CIs, and tests the winner's curse on seed 73.
- **E-C, rebuild the release fit and re-predict the 15 post-fit pairs** (CPU only). Makes the "new mechanisms" row
  fully recomputable, and adds a nearest-neighbour baseline.
- **E-D, equal-step capacity test on the GPU proxy (about $30).** Depth 10 vs 9 and 2 vs 1 KV heads at equal steps,
  in deterministic mode with replicates. Tests whether capacity lost only because of steps (R1.9).
- **E-E, GPU-proxy replicates (about $40).** Three deterministic-mode replicates of each parity pair, giving the
  proxy's own noise floor.
- **E-F, a multi-point price-of-a-step curve.** Train the final recipe at 5 budgets (e.g. 0.8-1.3x) on one chip,
  plus the GPU proxy at equal recipe. Gives kappa(S) with a CI and replaces the single anchor.
- **E-G, release the final-day data.** Harvest the chip E/G/H runs (30 Sep - 1 Oct) into `runs.jsonl` (host details
  stripped), so that the row-pool pair, the salt lotteries and the test bench become recomputable.
- **E-H, step-normalised rehearsals.** Rehearse with a fixed step count (or log the per-step clock) to estimate
  gamma_c at the source, and test the text+chip offset model prospectively with archived, time-stamped projections.
- **E-I, a mechanism check for the row pool.** Within-batch stream adjacency, duplicate-document rate and
  gradient-noise scale, with and without the pool.
- **E-J, one NKI kernel, priced before it is built.** A short chip screen for its step time, kappa for the price, and a
  cold rehearsal to confirm. This is the only direct test of the kernel hypothesis.
