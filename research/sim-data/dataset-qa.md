# runs.jsonl QA (29 Sep 2026)

Built by `python -m ffsim build-dataset` from `research/experiments.csv` (chips A/B), `research/sim-data/monitor-runs.csv` (LLM-extracted monitor table, reconciled) and the harvested chip dirs `research/sim-data/chipC/<run>/`, `chipD/<run>/` (with `_meta/queue/base.env`, `_meta/queue/runner.log` on C and `_meta/base.env`, `_meta/runner.log` on D). Merge by run_id = `chip:run`; precedence chip_log > monitor > experiments.csv per field, except `bpb_2m`, where experiments.csv outranks the monitor (DF-3, 29 Sep: the csv's `raw_bpb` is the exact machine value, the monitor's copy is LLM-extracted and rounded to 5-6 dp; `ffsim/dataset.py` `CSV_EXACT_FIELDS`). QA script: `qa.py` (not published) (this file's numbers are recomputed from runs.jsonl). Sections 2 and 4 were recomputed on 29 Sep (review findings DF-1 / DF-7) by `qa_metrics.py` (not published), which parses the chip dirs directly with `ffsim.parse_logs.parse_run_dir` and writes the machine-readable lists to `research/sim-data/qa-metrics.json`.

## 1. Counts

- records: **1188**; by chip: ? 28, A 563, B 247, C 339, D 11
- by source combination: `monitor` 490, `experiments.csv` 332, `chip_log+monitor` 184, `chip_log` 166, `monitor+experiments.csv` 16
- records carrying each source: chip_log 350, experiments.csv 348, monitor 690
- is_screen by chip: ?: 28 full / 0 screens, A: 426 full / 137 screens, B: 234 full / 13 screens, C: 213 full / 126 screens, D: 9 full / 2 screens
- harvested dirs vs CONTRACT.md: `chipC/INDEX.json` has 346 run dirs / 339 train.log (eval.log 231, eval20.log 18) and `chipD/INDEX.json` 11 runs, against the contract's on-chip inventory of 341 / 334 (231 / 18) and 10. The surplus is 6 runs launched after that inventory, not over- or under-harvesting (`chipC/_meta/HARVEST.json`: n_train_log_on_chip 339 = n_train_log_local 339, missing_on_disk []; `chipD/INDEX.json`: missing_runs []): chip C `1619_G_LANES3`, `1627_G_LANES2`, `1635_G_ORTHO`, `1637_G_C1` (S60 screens, code_k15off, launched 16:19-16:37 UTC by `t_launch`, so the contract's inventory was taken between 16:11 UTC, when its 341st dir `1611_G_C0` started, and 16:19 UTC) and `1639_R_m14asa_s73` (launched 16:39 UTC, at step 600 when the corpus was built at 16:45 UTC); chip D `1630_R_lk35sc165_s73` (launched 16:30 UTC, after the 16:27 UTC pull-1 inventory in `chipD/HARVEST_NOTES.md` found 10 dirs; 150 step lines at the 16:50 UTC pull). None of the six has an eval.log, which is why the 231 / 18 eval counts match the contract exactly. All six are in runs.jsonl (C 339 records = the 339 train.log dirs; D 11): 4 screens plus 2 `log_incomplete` runs whose final steps/bpb come from the monitor (sections 2, 3 and 5). `runs_created_after_snapshot` in `chipC/INDEX.json` is [] because it is relative to the corpus build, and all 346 dirs already existed at the 16:43 UTC probe (`_meta/harvest_log.txt`: run_dirs=346 train_logs=339).

### by chip x code_version (all records; screens in parentheses)

| chip | code_version | full | screens |
|---|---|---|---|
| ? | (none) | 28 | 0 |
| A | (none) | 370 | 117 |
| A | M1 | 0 | 3 |
| A | M2 | 13 | 3 |
| A | M3 | 9 | 0 |
| A | M4 | 7 | 0 |
| A | M5 | 11 | 0 |
| A | M6 | 10 | 6 |
| A | M7 | 4 | 8 |
| A | M7a | 1 | 0 |
| A | ad7f75a5 | 1 | 0 |
| B | (none) | 222 | 9 |
| B | M2 | 7 | 2 |
| B | M3 | 4 | 0 |
| B | ad7f75a5 | 1 | 2 |
| C | M12 | 3 | 0 |
| C | M3 | 7 | 0 |
| C | M4 | 4 | 0 |
| C | M5 | 11 | 0 |
| C | M6 | 1 | 0 |
| C | M7 | 2 | 0 |
| C | M8 | 1 | 0 |
| C | M9 | 3 | 0 |
| C | code | 88 | 62 |
| C | code_k12off | 17 | 8 |
| C | code_k13off | 1 | 6 |
| C | code_k14off | 1 | 6 |
| C | code_k15off | 0 | 5 |
| C | code_k52 | 0 | 1 |
| C | code_k52m6 | 0 | 5 |
| C | code_k53 | 0 | 1 |
| C | code_k54b | 0 | 1 |
| C | code_k54bmin | 0 | 3 |
| C | code_k54cmin | 0 | 2 |
| C | code_k56 | 0 | 4 |
| C | code_k57 | 0 | 7 |
| C | code_m10 | 2 | 0 |
| C | code_m10b | 1 | 0 |
| C | code_m10lr | 0 | 1 |
| C | code_m10off | 0 | 2 |
| C | code_m10xa | 0 | 1 |
| C | code_m10xl | 0 | 1 |
| C | code_m11 | 7 | 0 |
| C | code_m11c | 1 | 0 |
| C | code_m6 | 8 | 0 |
| C | code_m7 | 28 | 1 |
| C | code_m7a | 1 | 0 |
| C | code_m9 | 26 | 0 |
| C | code_m9acc1 | 0 | 2 |
| C | code_m9ln | 0 | 1 |
| C | code_m9off | 0 | 4 |
| C | code_m9oh | 0 | 2 |
| D | code_k12off | 8 | 2 |
| D | code_k14off | 1 | 0 |

### full runs (is_screen = False): field coverage

| chip | full runs | bpb_2m | steps | step_time.k4 | step_time.k1 | knobs_complete | bpb_20m | bpb+steps+k4+complete | loss_curve | seed | date_utc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ? | 28 | 5 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 28 |
| A | 426 | 390 | 320 | 0 | 0 | 0 | 0 | 0 | 0 | 419 | 269 |
| B | 234 | 220 | 178 | 0 | 0 | 0 | 0 | 0 | 0 | 234 | 103 |
| C | 213 | 205 | 210 | 182 | 92 | 115 | 18 | 111 | 210 | 212 | 213 |
| D | 9 | 9 | 9 | 9 | 9 | 9 | 0 | 9 | 9 | 9 | 9 |
| all | 910 | 829 | 721 | 191 | 101 | 124 | 18 | 120 | 219 | 900 | 622 |

`knobs_complete` is True only for queue runs on chips C/D (the `overrides` file written by the queue runner is base.env + job env verbatim) with no contradiction against the train.log config echo. experiments.csv rows (A/B) carry only the per-run overrides (never complete); monitor rows carry `knob_changes` only.

- env provenance of the chip_log records (`env_source=` in notes): driver 184, queue 132, hand 34
- runs whose overrides were recovered from runner.log START lines: **0** (every dir without an `overrides` file is a hand-launched rehearsal / warm run with `launch-command.txt` = `torchrun train.py`, absent from runner.log; their env is baked into the submission train.py and was recovered from the log's config echo instead)
- runs where the env file contradicts the config echo (`env_mismatch=`): **0**

- base.env vintage: chip C `_meta/queue/ls.txt` gives base.env mtime 1790422479 = 2026-09-26 11:34:39 UTC; the first queue START is 26 Sep 11:36:41 UTC, so base.env was NOT edited during the queue era (all 123 chip C queue runs; chip D base.env is byte-identical to chip C). The K5x recipe moves (cooldown 0.45 -> 0.60, EMA_EVERY 4 -> 32, FF_ACCUM_SCHED, LK) happened in the per-job START overrides, not in base.env. The depth-12 / cooldown-0.35 era predates the queue (driver/suite launches that never read base.env: `env_source=driver`).
- chip C runner.log START lists vs the runs' `overrides` files (last value wins): **0** disagreements

## 2. Monitor vs chip-log disagreements (chips C/D)

**`monitor-runs.csv` is the log-reconciled table, not an independent extraction (DF-1).** Before this comparison was made, the reconciliation (`monitor-reconcile-report.md`, method step 4) overwrote every row that names a harvested run dir, or whose printed bpb matched exactly one harvested `eval.log`, with the log's `val_bpb`, `eval20.log`, `ff_summary.steps`, seed and `code.sha256` lineage: **254** of the 690 rows carry the `log-confirmed` note (+4 `log-linked`), 206 of them were renamed from a `desc:`/queue-id label to the run dir, **53** step counts were bumped by one (the monitor's `@N` quotes the last logged step line; `ff_summary.steps` is N+1), **4** value disagreements were resolved to the log (K20_warm_K19 steps 840 -> 842, C3_cold_K19 840 -> 822, 1318_R_k50acc1_s67 chip A -> C and seed 58 -> 67) and 1 rounding-only difference (0427_F68_ve3_K26). The reconciled-table numbers below therefore only show that the reconciliation was applied consistently; they are NOT evidence that the monitor agrees with the logs. The independent number is the raw-extraction comparison that follows. Lists: `qa-metrics.json` `monitor_vs_chip_log` (chip-log side parsed straight from the dirs with `parse_run_dir`, matched by exact run name and chip).

Reconciled `monitor-runs.csv` vs the harvested dirs:

- run_ids present in both: **184**; with a bpb_2m on both sides: 140
- bpb_2m disagreeing by > 1e-5: 0 (pre-patched, see above); bpb_20m: 0; seed: 0 of 183
- steps disagreeing: 2 of 183 -> C:1639_R_m14asa_s73: monitor 2315 vs log 601; D:1630_R_lk35sc165_s73: monitor 2160 vs log 1401 (both `log_incomplete`: the monitor reports the finished run and wins in the merge)
- monitor rows on chips C/D with no harvested dir: 0 (`desc:` pseudo-runs and older runs deleted from the chip)

Raw extractions `monitor-runs.a.csv` / `monitor-runs.b.csv` (never patched) vs the same dirs, matched by exact run name on chips C/D (the independent agreement figure):

- a: **23** rows match a harvested dir (22 with bpb on both sides): bpb_2m > 1e-5 disagreements **0**; steps compared on 21: **10** have the monitor one below `ff_summary.steps` (C:0253_R_g7rf_s73 2364 vs 2365, C:0222_R_g7lkm_s73 2339 vs 2340, 0745_R_g7lkm_s67, 0637_R_g7rf_s58, 0714_R_g7rf_s67, 0529_R_g7lkm_s58, 2017_R_g7mp93_s73, 1759_R_g7mp97_s73, 0127_R_g2zero_s58, 1805_R_g0ko_s58), 0 other; seed disagreements 0 of 23
- b: **18** match (18 with bpb): bpb disagreements **0**; steps compared on 17: **6** off-by-one (0253_R_g7rf_s73, 0222_R_g7lkm_s73, 2017_R_g7mp93_s73, 1759_R_g7mp97_s73, 0127_R_g2zero_s58, 1805_R_g0ko_s58), 0 other; seed 0 of 18
- only 23 / 18 raw rows can be matched by name because the extractors labelled most 26-29 Sep runs by description or queue id (the 206 renames above). Ignoring the chip label adds the 15-19 Sep dirs that sit on the inherited chipC volume but ran on chip A: a 43 matched (42 bpb, 0 disagreements; 38 steps: the same 10 off-by-one + 2 other = K20_warm_K19 840 vs 842, C3_cold_K19 840 vs 822), b 39 (38 bpb, 0; 33 steps: 6 off-by-one + 1 other = K20_warm_K19)
- reading: wherever a raw extraction and a log can be compared, bpb agrees to 1e-5 (42 + 38 runs, 0 disagreements), while the monitor's step count is one below `ff_summary.steps` on about half of the 26-29 Sep queue runs (10/21, 6/17) and off by 2-18 steps on two early rehearsals. Monitor-only records (chips A/B, no log) inherit that step bias; 1% of steps is ~0.00057 bpb, one step at ~2300 is ~0.000025, so the -1 is negligible for quality but the rehearsal-era errors are not.
- **step-count convention (DF-2, 29 Sep).** `steps` in runs.jsonl = optimizer steps completed = train.log `ff_summary` `steps` = last logged step index + 1 (`parse_logs`: summary steps, else last index + 1; e.g. chipC/0626_R_g7_s73 last line `step 02313`, ff_summary 2314, record 2314; C40_cold_K59 last line `step 02356`, record 2357). This is the convention the contract's `@` numbers use for the rehearsals (C40 @2357) and the reconciled monitor-runs.csv follows it (0626_R_g7_s73 2314). Two sources use the LAST-INDEX convention instead, one below the count: (a) **experiments.csv chip A/B `steps`** (the 120-step profile rows A:0519_P_prof_d9, A:0556_P_prof_d9_shard, A:0835_P_baseline, A:1256_P_base3, A:1301_P_copt carry 119 where monitor-runs.csv says 120), so A/B records and any cross-chip step comparison against C/D records are off by one step (0.04% at 2300 steps, 0.051*ln(2314/2313) = 2e-5 bpb: negligible for the surrogate, but do not treat a 1-step A/B-vs-C/D gap as real); (b) the queue-era monitor text (`@2313` for 0626_R_g7_s73) and, until 29 Sep, validation-pairs.json, whose `treatment_steps` / `control_steps` were rewritten to the runs.jsonl value for every arm that resolves by `chip:run` (172 arm values in 86 pairs, all +1; `step_diff_pct` re-rounded for 7 pairs); the 10 rehearsal pairs already matched and the 3 chip-A wildcard pairs (`A:*_R_...`, monitor-only records at 2291/2142/2209/2089) were left as the monitor reported them, which is also what their runs.jsonl records carry. `quality.pair_validation` reads steps from the records, not from the pair file, so no fitted number moved.

## 3. Runs with missing eval (chips C/D, not screens)

| run_id | steps (log) | charged s | tag | status | reason |
|---|---|---|---|---|---|
| C:0003_F32_b262k_d12_lr2_s51 |  |  |  | train_exit=1 00:34:58 |  |
| C:0601_G1_d13w768 |  |  |  |  |  |
| C:0640_R_g3mpl2_s58 | 551 | 303.0000 | log_incomplete |  | train.log has no ff_summary and no eval: still running at harvest at step 550 |
| C:1340_R_g5mtp50_s58 | 2081 | 1545.0000 | log_incomplete | train_exit=1 14:06:56 | train.log has no ff_summary and no eval: ended at step 2080 |
| C:1454_R_g5cd65_s58 | 901 | 432.0000 | log_incomplete | train_exit=1 15:02:52 | train.log has no ff_summary and no eval: ended at step 900 |
| C:C2_stall_K1 | 891 | 1181.0000 | log_incomplete | train_exit=1 13:10:28 | train.log has no ff_summary and no eval: ended at step 890 |
| C:DUP-0155_F61_muon15 | 661 | 1232.0000 | log_incomplete |  | train.log has no ff_summary and no eval: still running at harvest at step 660 |
| C:KILLED-0546_F70_ve3unet_K26 |  |  |  |  |  |

Screens on C/D without an eval (by design, never quality points): 64. Still-running-at-harvest runs (re-pull with `python -m ffsim harvest --chip C|D` after they finish): C:1639_R_m14asa_s73, D:1630_R_lk35sc165_s73, C:0640_R_g3mpl2_s58, C:DUP-0155_F61_muon15

## 4. Duplicate run_ids and semantic duplicates

- duplicate run_ids within a source (would be silently merged): experiments.csv: 0, monitor: 0, chip_log: 0
- duplicate run_ids in runs.jsonl: **0**
- monitor `desc:` pseudo-run rows (no run-dir name): 333 (A 218, B 88, ? 27; 226 with a bpb_2m). Rule (DF-7): a `desc:` record is a semantic duplicate when its bpb_2m agrees with an experiments.csv record's bpb_2m (= `raw_bpb`) to 5 decimal places (|a - b| <= 5e-6) on the same chip, or on any chip when the `desc:` record's chip is `?`. Under that rule **128** distinct `desc:` rows match (169 (desc, csv) pairs over 138 csv rows; strict `round(x, 5)` equality gives the same 128) = the same run counted twice under two run_ids. The two other readings differ, which is why the number did not reproduce: same chip only (the four `?` rows excluded) 124; any chip 130 (the 28 extra cross-chip pairs are 5-dp coincidences or mislabelled chips and are listed separately as `cross_chip_only_pairs`). Every one of the 128 has at least one match whose seed agrees or whose csv seed is unknown (3 VE-stream pairs); 6 of the 169 pairs are 5-dp coincidences between different seeds (e.g. B:desc:d9_WD0_B_s58 = 0.98874 = B:0230_R_rope1k_s67, whose true twin is B:0909_R_d9wd0_s58) and 11 `desc:` rows match more than one csv row, so a pair-level consumer should keep `seed_agrees != false`; 39 of the 169 pairs are screens (bpb > 1.5). Examples: A:desc:ASYNC_LOOP_base_A = A:0519_P_prof_d9; A:desc:ASYNC_LOOP_K8_A = A:0519_P_prof_d9; A:desc:ASYNC_LOOP_K32_A = A:0519_P_prof_d9; ?:desc:shape_d9x768 = A:1459_S_d9w768_s58; ?:desc:shape_d9x1024_current = B:1933_R_cd060_s58; ?:desc:shape_d7x768 = B:1531_S_d7w768_s58; ?:desc:shape_d6x1280 = B:1454_S_d6w1024_s58; B:desc:warmup10_s58 = B:0306_R_warm10_s58 ...
  These are NOT merged (different run_id). The full list (`desc_run_ids`, and the `pairs` with both sides' bpb, chip, seed and `seed_agrees`) is in `research/sim-data/qa-metrics.json` `semantic_duplicates`; `tests/test_ffsim_dataset_qa.py` recomputes it from runs.jsonl. A quality fit over `full_runs()` should drop the listed `desc:` run_ids. `ffsim/quality.py` as fitted never uses them (monitor-only rows have no effective knobs and are held out: `dropped['no_effective_knobs']`), so shipping the list changes neither the fit nor `quality-validation.md`.
- records with chip `?` (monitor rows whose chip the log never stated; all `desc:` A/B-era summaries): **28**, 5 with bpb_2m. Kept as-is; they violate `schema.CHIPS` and must be filtered by consumers (`chip in CHIPS`).
- chip C dirs renamed on the chip with a DUP-/KILLED- prefix (aborted duplicates of a queued name): C:DUP-0155_F61_muon15, C:KILLED-0546_F70_ve3unet_K26; both are `log_incomplete` and carry no eval.

## 5. K59-like runs (LK slope 0.35, fnm) with their k1/k2/k4 medians

| run_id | seed | code_version | TT | steps | bpb_2m | k1 | k2 | k4 | n_k4 | kind | beyond the K59 recipe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C:0816_R_g7lk35_s73 | 73 | code_k12off | 1760 | 2326 | 0.96176 | 0.2901 | 0.5038 | 0.9279 | 1526 | strict |  |
| C:0913_R_g7lk35_s58 | 58 | code_k12off | 1760 | 2327 | 0.96189 | 0.2910 | 0.5043 | 0.9264 | 1529 | strict |  |
| C:0945_R_g7lk35_s67 | 67 | code_k12off | 1760 | 2325 | 0.96153 | 0.2930 | 0.5031 | 0.9271 | 1528 | strict |  |
| C:1220_R_g7lk35_s281 | 281 | code_k12off | 1760 | 2325 | 0.96390 | 0.2927 | 0.5055 | 0.9260 | 1530 | strict |  |
| C:1254_R_g7lk35_s283 | 283 | code_k12off | 1760 | 2329 | 0.96246 | 0.2922 | 0.5032 | 0.9261 | 1530 | strict |  |
| C:1325_R_g7lk35_s293 | 293 | code_k12off | 1760 | 2324 | 0.96281 | 0.2931 | 0.5049 | 0.9270 | 1528 | strict |  |
| C:1356_R_g7lk35_s307 | 307 | code_k12off | 1760 | 2322 | 0.96333 | 0.2934 | 0.5049 | 0.9273 | 1527 | strict |  |
| C:1427_R_g7lk35sc165_s73 | 73 | code_k12off | 1760 | 2323 | 0.96154 | 0.2922 | 0.5036 | 0.9282 | 1526 | variant | FF_SOFTCAP_A, FF_SOFTCAP_B |
| C:1539_R_g7lk35sc165_s58 | 58 | code_k12off | 1760 | 2321 | 0.96195 | 0.2923 | 0.5047 | 0.9285 | 1525 | variant | FF_SOFTCAP_A, FF_SOFTCAP_B |
| C:1639_R_m14asa_s73 | 73 | code_k14off | 1760 | 2315 | 0.95995 | 0.2963 | 0.5057 | 0.9315 | 1 | variant | FF_ATTN_SRC |
| C:C40_cold_K59 | 73 | M12 | 1793 | 2357 | 0.96092 | 0.2937 | 0.5069 | 0.9312 | 1549 | rehearsal |  |
| D:1202_R_g7lk35_s73 | 73 | code_k12off | 1760 | 2153 | 0.96644 | 0.2897 | 0.5132 | 1.0359 | 1361 | strict |  |
| D:1240_R_g7lk35_s97 | 97 | code_k12off | 1760 | 2150 | 0.96942 | 0.2900 | 0.5035 | 1.0526 | 1350 | strict |  |
| D:1312_R_g7lk35_s113 | 113 | code_k12off | 1760 | 2153 | 0.96871 | 0.2908 | 0.5042 | 1.0524 | 1354 | strict |  |
| D:1344_R_g7lk35_s127 | 127 | code_k12off | 1760 | 2149 | 0.96817 | 0.2921 | 0.5075 | 1.0467 | 1355 | strict |  |
| D:1416_R_g7lk35_s131 | 131 | code_k12off | 1760 | 2152 | 0.96746 | 0.2876 | 0.5098 | 1.0448 | 1356 | strict |  |
| D:1448_R_g7lk35_s137 | 137 | code_k12off | 1760 | 2147 | 0.96800 | 0.2932 | 0.5085 | 1.0461 | 1355 | strict |  |
| D:1520_R_g7lk35_s139 | 139 | code_k12off | 1760 | 2150 | 0.96768 | 0.2883 | 0.5105 | 1.0408 | 1359 | strict |  |
| D:1552_R_m14asa_s73 | 73 | code_k14off | 1760 | 2155 | 0.96471 | 0.2929 | 0.5129 | 1.0362 | 1369 | variant | FF_ATTN_SRC |
| D:1630_R_lk35sc165_s73 | 73 | code_k12off | 1760 | 2160 | 0.96608 | 0.2896 | 0.5053 | 1.0387 | 63 | variant | FF_SOFTCAP_A, FF_SOFTCAP_B |

`strict` = the K59 queue recipe exactly (LK035_K59 extras on the G7 arm, code_k12off, no further knobs), evaluated. The rehearsals C40_cold_K59 (and C41 when it lands) ran the submission train.py (env baked in, TT 1793). C:1639_R_m14asa_s73 and D:1630_R_lk35sc165_s73 were still running at harvest: bpb/steps come from the monitor, their step_time from the partial log (n_k4 = 1 and 63: weight by n_k4 or re-harvest).

### medians over the strict K59-recipe runs

| chip | TT | n | seeds | k1 median | k2 median | k4 median | steps median | bpb_2m median |
|---|---|---|---|---|---|---|---|---|
| C | 1760 | 7 | 73,58,67,281,283,293,307 | 0.2927 | 0.5043 | 0.9270 | 2325 | 0.96246 |
| D | 1760 | 7 | 73,97,113,127,131,137,139 | 0.2900 | 0.5085 | 1.0461 | 2150 | 0.96800 |

Anchor C40_cold_K59: bpb_2m 0.960920 @ 2357 steps, k1/k2/k4 0.2937/0.5069/0.9312 s, charged 1787.1 s (timing.txt 1790.1), startup 474.84 s (timing.txt 518.0: the harness clock includes process start; the log's step-0 dt is the compile).
Anchor C36_cold_K57: bpb_2m 0.961927 @ 2359 steps, k4 0.9278 s.

## 6. k4 medians per code_version (full runs; screens listed separately)

### chip C, full runs

| code_version | lineage tag | n | k4 min | p25 | k4 median | p75 | k4 max | k1 median | k2 median | steps median |
|---|---|---|---|---|---|---|---|---|---|---|
| code |  | 58 | 0.9923 | 1.8401 | 1.8479 | 1.9921 | 2.7152 |  | 0.5886 | 908 |
| code_m7 | M7 | 28 | 0.9354 | 0.9384 | 0.9413 | 0.9421 | 0.9442 | 0.2887 | 0.5143 | 2288 |
| code_m9 | M9 | 26 | 0.9239 | 0.9281 | 0.9287 | 0.9299 | 0.9316 | 0.2940 | 0.5061 | 2308 |
| code_k12off | M12 | 17 | 0.9108 | 0.9221 | 0.9261 | 0.9273 | 0.9311 | 0.2922 | 0.5033 | 2326 |
| M5 |  | 11 | 0.9870 | 0.9915 | 0.9941 | 0.9952 | 0.9977 |  | 0.5666 | 2001 |
| code_m6 | M6 | 8 | 0.9488 | 0.9600 | 0.9611 | 0.9635 | 0.9683 | 0.3078 | 0.5336 | 2106 |
| M3 |  | 7 | 0.9910 | 0.9915 | 0.9921 | 0.9927 | 0.9931 | 0.3407 | 0.5653 | 1903 |
| code_m11 | M11 | 7 | 0.9232 | 0.9283 | 0.9289 | 0.9308 | 0.9313 | 0.2940 | 0.5056 | 2309 |
| M12 |  | 3 | 0.9098 | 0.9155 | 0.9211 | 0.9262 | 0.9312 | 0.2918 | 0.5040 | 2380 |
| M4 |  | 3 | 0.9925 | 0.9935 | 0.9945 | 0.9947 | 0.9949 |  | 0.5664 | 2003 |
| M9 |  | 3 | 0.9277 | 0.9277 | 0.9278 | 0.9288 | 0.9299 | 0.2946 | 0.5062 | 2359 |
| M7 |  | 2 | 0.9356 | 0.9372 | 0.9387 | 0.9403 | 0.9419 | 0.2866 | 0.5128 | 2274 |
| code_m10 | M10 | 2 | 0.9574 | 0.9721 | 0.9869 | 1.0016 | 1.0163 | 0.3110 | 0.5368 | 2169 |
| M6 |  | 1 | 0.9599 | 0.9599 | 0.9599 | 0.9599 | 0.9599 |  | 0.5319 | 2142 |
| M8 |  | 1 | 0.9422 | 0.9422 | 0.9422 | 0.9422 | 0.9422 | 0.2887 | 0.5132 | 2337 |
| code_k13off | M13 | 1 | 0.9135 | 0.9135 | 0.9135 | 0.9135 | 0.9135 | 0.2913 | 0.5018 | 2352 |
| code_k14off | M14 | 1 | 0.9315 | 0.9315 | 0.9315 | 0.9315 | 0.9315 | 0.2963 | 0.5057 | 2315 |
| code_m10b | M10b | 1 | 0.9623 | 0.9623 | 0.9623 | 0.9623 | 0.9623 | 0.3015 | 0.5213 | 2235 |
| code_m11c | M11c | 1 | 0.9874 | 0.9874 | 0.9874 | 0.9874 | 0.9874 | 0.3077 | 0.5363 | 2177 |
| code_m7a |  | 1 | 0.9613 | 0.9613 | 0.9613 | 0.9613 | 0.9613 |  | 0.5355 | 551 |

### chip C, screens (S60 / suite probes; 10-step forced phases, medians from ff_summary)

| code_version | lineage tag | n | k4 min | p25 | k4 median | p75 | k4 max | k1 median | k2 median | steps median |
|---|---|---|---|---|---|---|---|---|---|---|
| code |  | 21 | 0.9969 | 1.0061 | 1.9829 | 2.0150 | 2.4136 |  |  | 30 |
| code_k12off | M12 | 8 | 0.9135 | 0.9170 | 0.9257 | 0.9308 | 0.9390 | 0.2903 | 0.5022 | 60 |
| code_k57 | M9 | 7 | 0.9266 | 0.9317 | 0.9368 | 0.9382 | 0.9625 | 0.2953 | 0.5128 | 60 |
| code_k13off | M13 | 6 | 0.9162 | 0.9214 | 0.9244 | 0.9263 | 0.9315 | 0.2917 | 0.5051 | 60 |
| code_k14off | M14 | 6 | 0.9253 | 0.9276 | 0.9294 | 0.9324 | 0.9352 | 0.2935 | 0.5054 | 60 |
| code_k15off | M15 | 5 | 0.9296 | 0.9296 | 0.9310 | 0.9596 | 0.9729 | 0.2948 | 0.5123 | 60 |
| code_k52m6 | M6 | 5 | 0.9608 | 0.9624 | 0.9911 | 0.9926 | 0.9976 |  | 0.5624 | 60 |
| code_k56 | M9 | 4 | 0.9304 | 0.9329 | 0.9339 | 0.9343 | 0.9351 | 0.2953 | 0.5091 | 60 |
| code_m9off | M9 | 4 | 0.9428 | 0.9454 | 0.9469 | 0.9479 | 0.9487 | 0.2889 |  | 60 |
| code_k54bmin | M7 | 2 | 0.9379 | 0.9604 | 0.9829 | 1.0054 | 1.0279 | 0.2887 |  | 32 |
| code_k54cmin | M8 | 2 | 0.9402 | 0.9403 | 0.9404 | 0.9404 | 0.9405 | 0.2868 |  | 60 |
| code_m10off | M10 | 2 | 0.9297 | 0.9300 | 0.9304 | 0.9307 | 0.9310 | 0.2928 | 0.5047 | 60 |
| code_m9acc1 | M9 | 2 | 0.9272 | 0.9275 | 0.9279 | 0.9283 | 0.9286 | 0.2916 |  | 60 |
| code_k52 | M5 | 1 | 0.9911 | 0.9911 | 0.9911 | 0.9911 | 0.9911 |  | 0.5615 | 60 |
| code_k53 | M6 | 1 | 1.0083 | 1.0083 | 1.0083 | 1.0083 | 1.0083 |  |  | 5 |
| code_k54b | M7 | 1 | 0.9390 | 0.9390 | 0.9390 | 0.9390 | 0.9390 | 0.2855 |  | 60 |
| code_m10xa | M10 | 1 | 1.0237 | 1.0237 | 1.0237 | 1.0237 | 1.0237 | 0.3207 | 0.5557 | 60 |
| code_m10xl | M10 | 1 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.9621 | 0.3031 | 0.5248 | 60 |
| code_m7 | M7 | 1 | 0.9416 | 0.9416 | 0.9416 | 0.9416 | 0.9416 | 0.2882 |  | 101 |
| code_m9ln | M9 | 1 | 0.9449 | 0.9449 | 0.9449 | 0.9449 | 0.9449 | 0.2882 |  | 60 |
| code_m9oh | M9 | 1 | 0.9590 | 0.9590 | 0.9590 | 0.9590 | 0.9590 | 0.2925 |  | 60 |

### chip D, full runs

| code_version | lineage tag | n | k4 min | p25 | k4 median | p75 | k4 max | k1 median | k2 median | steps median |
|---|---|---|---|---|---|---|---|---|---|---|
| code_k12off | M12 | 8 | 1.0359 | 1.0403 | 1.0454 | 1.0481 | 1.0526 | 0.2898 | 0.5080 | 2151 |
| code_k14off | M14 | 1 | 1.0362 | 1.0362 | 1.0362 | 1.0362 | 1.0362 | 0.2929 | 0.5129 | 2155 |

### chip D, screens

| code_version | lineage tag | n | k4 min | p25 | k4 median | p75 | k4 max | k1 median | k2 median | steps median |
|---|---|---|---|---|---|---|---|---|---|---|
| code_k12off | M12 | 2 | 0.9189 | 0.9244 | 0.9300 | 0.9355 | 0.9410 | 0.2914 | 0.5063 | 60 |

Contract anchors (chip C full-run k4 medians): LK0.35 on code_k12off n=7 median 0.9270 s (contract 0.928); RF on code_k12off n=3 median 0.9113 s (contract 0.911); G7/K57 recipe on code_m9/code_k57 n=26 median 0.9287 s (contract 0.928).

## 7. Seed sd among same-recipe full runs

Same recipe = same chip, same code_version, identical knobs apart from FF_SEED. A/B: experiments.csv rows inside the era window only (`in_window`), keyed on their overrides; the A/B base recipe still moved inside the window without a knob change (the D-family seed tail spans several days), so A/B raw sds are upper bounds - the campaign's own A/B noise figure is the `norm_bpb` seed sd. Raw sd is on the raw 2M bpb; the step-adjusted sd removes the step-count jitter with d bpb / d ln steps = -0.057 (same-recipe use only; groups with median steps > 1500). Groups with >= 3 seeds; one run per seed (the earliest).

| chip | code_version | family/recipe | knobs (abridged) | n seeds | seeds | mean bpb_2m | sd raw | steps | sd step-adj | sd norm_bpb (csv) |
|---|---|---|---|---|---|---|---|---|---|---|
| B |  | D |  | 47 | 57 58 59 64 65 66 67 68 69 70 71 72 73 92 93 94 95 96 97 98 99 100 101 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 170 171 172 173 | 0.98697 | 0.00250 | 1550-1590 | 0.00246 | 0.00246 |
| A |  | D |  | 38 | 55 56 58 60 61 62 63 67 81 82 83 84 85 86 87 88 89 90 91 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 132 | 0.98820 | 0.00374 | 1526-1575 | 0.00348 | 0.00344 |
| C | code_m9 | K57 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, TIME_TARGET= | 16 | 51 53 58 61 62 67 71 73 77 79 83 89 101 103 107 109 | 0.96419 | 0.00059 | 2304-2326 | 0.00062 |  |
| C | code_k12off | LK0.35 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, LEAKY_RELU2= | 7 | 58 67 73 281 283 293 307 | 0.96253 | 0.00087 | 2322-2329 | 0.00085 |  |
| D | code_k12off | LK0.35 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, LEAKY_RELU2= | 7 | 73 97 113 127 131 137 139 | 0.96798 | 0.00095 | 2147-2153 | 0.00093 |  |
| A |  | R | COOLDOWN_FRAC=0.60 | 5 | 62 63 67 131 132 | 0.98591 | 0.00152 | 1551-1557 | 0.00155 | 0.00156 |
| C | code_m7 | RAMP3 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, TIME_TARGET= | 3 | 53 58 67 | 0.96510 | 0.00088 | 2278-2287 | 0.00076 |  |
| A |  | R | ADAMW_LR_SCALE=1.0, COOLDOWN_FRAC=0.50, MATRIX_LR_SCALE=1.6 | 3 | 62 63 67 | 0.98676 | 0.00081 | 1555-1579 | 0.00118 | 0.00125 |
| B |  | R | ADAMW_LR_SCALE=1.0, COOLDOWN_FRAC=0.50, MATRIX_LR_SCALE=1.6 | 3 | 58 71 73 | 0.98474 | 0.00011 | 1570-1583 | 0.00029 | 0.00033 |
| B |  | R | COOLDOWN_FRAC=0.60 | 3 | 58 71 73 | 0.98493 | 0.00014 | 1569-1571 | 0.00010 | 0.00010 |
| C | code_k12off | LK0.5 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, LEAKY_RELU2= | 3 | 58 67 73 | 0.96229 | 0.00074 | 2314-2341 | 0.00036 |  |
| C | code_k12off | RF | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, RELU2_FN=1,  | 3 | 58 67 73 | 0.96235 | 0.00036 | 2355-2365 | 0.00032 |  |
| C | code_m7 | KO0.5 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, TIME_TARGET= | 3 | 53 58 67 | 0.96470 | 0.00079 | 2284-2290 | 0.00074 |  |
| C | code_m11 | WD_SCHED=1 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, TIME_TARGET= | 3 | 58 67 73 | 0.96352 | 0.00030 | 2299-2309 | 0.00034 |  |
| C | code_m9 | QK1.6 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, TIME_TARGET= | 3 | 58 67 73 | 0.96356 | 0.00029 | 2308-2313 | 0.00022 |  |
| C | code_m7 | KO0.3 | ACCUM_SCHED=1:0.06,2:0.2,4, COOLDOWN_FRAC=0.60, TIME_TARGET= | 3 | 53 58 67 | 0.96462 | 0.00079 | 2282-2300 | 0.00059 |  |

Headline groups:

- LK0.35 (K59 recipe) seeds on chip C, TT 1760, code_k12off: n = 7 seeds [58, 67, 73, 281, 283, 293, 307], mean 0.96253, **sd raw 0.00087**, sd step-adjusted 0.00085; runs 0816_R_g7lk35_s73, 0913_R_g7lk35_s58, 0945_R_g7lk35_s67, 1220_R_g7lk35_s281, 1254_R_g7lk35_s283, 1325_R_g7lk35_s293, 1356_R_g7lk35_s307
- G7 / K57 recipe seeds on chip C, TT 1760 (code_m9 / code_k57, no LK, no RF): n = 16 seeds [51, 53, 58, 61, 62, 67, 71, 73, 77, 79, 83, 89, 101, 103, 107, 109], mean 0.96419, **sd raw 0.00059**, sd step-adjusted 0.00062; runs 0054_R_g6acc_s58, 0131_R_g6acc_s53, 0202_R_g6acc_s67, 0359_R_g7_s51, 0436_R_g7_s61, 0523_R_g7_s62, 0554_R_g7_s71, 0626_R_g7_s73, 0657_R_g7_s77, 1308_R_g7_s101, 1339_R_g7_s103, 1410_R_g7_s107, 1502_R_g7_s109, 2324_R_g7_s79, 2355_R_g7_s83, 0026_R_g7_s89
- LK0.35 seeds on chip D, TT 1760, code_k12off: n = 7 seeds [73, 97, 113, 127, 131, 137, 139], mean 0.96798, **sd raw 0.00095**, sd step-adjusted 0.00093; runs 1202_R_g7lk35_s73, 1240_R_g7lk35_s97, 1312_R_g7lk35_s113, 1344_R_g7lk35_s127, 1416_R_g7lk35_s131, 1448_R_g7lk35_s137, 1520_R_g7lk35_s139

Contract: seed sd ~0.0006 at 2M tokens; effects < 0.0003 need 3+ seeds.

## 8. code_version naming and lineage

| code_version (chip dir) | lineage tag (from co-sourced monitor rows) | records |
|---|---|---|
| code_k12off | M12 | 35 |
| code_k13off | M13 | 7 |
| code_k14off | M14 | 8 |
| code_k15off | M15 | 5 |
| code_k52 | M5 | 1 |
| code_k52m6 | M6 | 5 |
| code_k53 | M6 | 1 |
| code_k54b | M7 | 1 |
| code_k54bmin | M7 | 3 |
| code_k54cmin | M8 | 2 |
| code_k56 | M9 | 4 |
| code_k57 | M9 | 7 |
| code_m10 | M10 | 2 |
| code_m10b | M10b | 1 |
| code_m10lr | M10 | 1 |
| code_m10off | M10 | 2 |
| code_m10xa | M10 | 1 |
| code_m10xl | M10 | 1 |
| code_m11 | M11 | 7 |
| code_m11c | M11c | 1 |
| code_m6 | M6 | 8 |
| code_m7 | M7 | 29 |
| code_m7a | (no monitor row) | 1 |
| code_m9 | M9 | 26 |
| code_m9acc1 | M9 | 2 |
| code_m9ln | M9 | 1 |
| code_m9off | M9 | 4 |
| code_m9oh | M9 | 2 |

- chip_log records whose code_version is the placeholder `code` (ran /root/ff-claude/code/ with no FF_CODE_DIR and no monitor row naming the M-version): **150** (83 evaluated full runs), dates 2026-09-15 .. 2026-09-26. The default dir was re-deployed many times (M1..M9); only one such dir has a code.sha256 (0501_R_g2ps_s58 = the current code/, ac95e51e). Their lineage can only be dated from the monitor's deployment timeline; downstream should treat `code` as an unknown era, not one era.
- naming mix in code_version: chip_log wins with the chip dir name (`code_k12off`, `code_m9`, ...) while monitor-only runs carry `M12`, `M9`; `steptime.version_key` maps `code_k12off` -> `k12` and `M12` -> `m12` (different keys, similarity 0.67) and `quality.parse_version` gives ('K',12,'off') vs ('M',12,''): the same lineage is split in two eras unless consumers map via the `lineage:` tag. The dir names are more precise for step time (code_m10xa / code_m10xl / code_m9acc1 are distinct files with distinct shas); the M-name is the quality era.

## 9. Other checks

- time_target of evaluated C/D full runs: 1710.0: 53, 1760.0: 149, 1788.0: 2, 1790.5: 6, 1793.0: 4
- charged_seconds of evaluated C/D full runs: min 246.0, median 1754.3, max 1787.7 (log clock at the last save; timing.txt charged_s is ~3 s higher, it includes the save)
- step_time source for chip_log records: summary 193, lines 157
- step_time.k4 outside (0.2, 3.0) s: 0 []
- bpb_2m outside (0.9, 1.8): 58 records, 57 of them screens (the untrained-weights placeholder eval, e.g. 1.7048, and trace/profile probes); non-screens: [('C:1339_v1_early_ema', 2.9995)] (a 700-step early-EMA probe from the driver era: not a quality point although not flagged as a screen)
- chip C queue runs flagged is_screen: 8 = 7 60-step host-timing probes queued with NEURON_COMPETITION_R1_NUM_STEPS=60 (screens by design; their eval is the placeholder ~2.45) + ['0052_R_g5cd65_s58'] (crashed at step ~100, no eval, flagged by the steps <= 200 rule; harmless: no bpb)
- loss_curve last step >= steps: 0 []
- pre-existing test failure (not caused by this build): `tests/test_ffsim_parsers.py::test_harvested_chip_dirs_parse` asserts steps > 500 for every evaluated full run; `C:0036_F34_b262k_d14_s50` (depth 14, b262k, driver era) legitimately finished 436 steps in 1709 charged s.

## 10. Files touched by the build

- `ffsim/parse_logs.py`: `find_base_env` also looks in `_meta/queue/` (where the chip C harvest put the queue dir); runner.log parsing (`find_runner_log`, `parse_runner_log`, `load_runner_log`) and START-line recovery of a missing `overrides`; env provenance (`env_source=queue|driver|hand|unknown`) deciding whether base.env is layered; `model_config={...}` JSON captured into `TrainLog.model_config`; `echo_env()` + reconciliation of knobs against the config echo (`knobs_from_echo=`, `env_mismatch=`, knobs_complete=False on a mismatch); `log_incomplete` tag; echo-filled values are spelled as base.env / the runner.log jobs spell them ('0.60', '2.0', '0.5,0.7071').
- `ffsim/dataset.py`: a `log_incomplete` chip_log loses the scalar fields (steps, bpb, charged) to the monitor / csv in the merge (the monitor reports the finished run, e.g. C:1639_R_m14asa_s73 2315 steps vs the 600-step harvested log); `code_dir:` / `lineage:` tags. DF-3 (29 Sep): `CSV_EXACT_FIELDS = ("bpb_2m",)` ranks experiments.csv above the monitor for `bpb_2m` only (`steps` keeps monitor > csv: the csv's steps column is the last logged step index, 119 on the six 120-step profile screens); the rebuild moved 10 chip A/B records (B:1429_Q_adamw1cd050_s58, B:1547_SS_s57, A:0329_X_d9w1024cd045_s55, A:1428_Q_b393_s58, A:1500_Q_combo_s58, A:1906_P_cc_fp8, A:0547_M_backout_s58, A:0622_C_ctrl_s58, A:1325_C_s58, B:0556_M_backout_s58) from the monitor's rounded value to the csv's full-precision `raw_bpb` (largest change 3.4e-6 bpb). Recomputed over all 348 csv rows against runs.jsonl: 0 bpb differences, 0 rows missing, 0 is_screen disagreements, 0 rows using norm_bpb as bpb_2m. `python -m ffsim fit` on the uncorrected and corrected runs.jsonl gives bit-identical models and printed reports (`tests/test_ffsim_parsers.py::test_merge_csv_exact_bpb_beats_monitor`).
- `ffsim/cli.py`: unchanged (build-dataset ran as shipped).

