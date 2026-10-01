# ffsim build contract (29 Sep 2026)

Spec: `docs/simulator-spec.md`. Package: `ffsim/` (pure numpy + stdlib; Python 3.11; no torch/pandas).
Tests: `tests/test_ffsim_*.py` (pytest 8 is installed). Data: `research/sim-data/` (git-tracked evidence; NOT `data/`, which is gitignored).

## Layout and ownership (one owner per file; others only read)

| file | purpose |
|---|---|
| `ffsim/schema.py` | RunRecord / StepTime / Recipe / SimResult (frozen contract, written) |
| `ffsim/dataset.py` | load/save `runs.jsonl`; merge records from the three sources by (chip, run); `effective_knobs(record, base_env)` |
| `ffsim/parse_logs.py` | parse a harvested run dir: train.log (dt per step -> StepTime by k phase, loss curve, steps, charged/startup s, config echo lines), eval.log / eval20.log (val_bpb), overrides, code.sha256 |
| `ffsim/parse_csv.py` | `research/experiments.csv` -> RunRecords (chips A/B, 348 rows, cols FF_* + overrides) |
| `ffsim/parse_monitor.py` | `research/sim-data/monitor-runs.csv` (LLM-extracted table) -> RunRecords; official uploads table |
| `ffsim/harvest.py` | pull run dirs off chips C/D over SSM (read-only on chip) into `research/sim-data/chip{C,D}/<run>/` |
| `ffsim/steptime.py` | Part 1: StepTimeModel.fit(records) / predict(code_version, knobs, chip) -> {k1,k2,k4}; leave-one-code-version-out report |
| `ffsim/quality.py` | Part 2: QualityModel.fit(records) / predict(knobs, steps, code_version, seed=None) -> (mean, sd); paired validation |
| `ffsim/offset.py` | Part 3: OffsetModel (mean, sd from the official table); sample() |
| `ffsim/simulate.py` | steps from step_time + FF_ACCUM_SCHED phases + charged time; Monte Carlo over seed/offset/step-time noise -> SimResult |
| `ffsim/search.py` | candidate space (json), random/LHS/local search over the surrogate, ranking, top-K to confirm on Trainium |
| `ffsim/report.py` | markdown + CSV writers |
| `ffsim/cli.py`, `ffsim/__main__.py` | `python -m ffsim harvest|build-dataset|fit|validate|simulate|search|report` |
| `ffsim/cloud/` | Dockerfile, `run_batch.sh`, `ec2_launch.sh` (numpy-only box), README |
| `ffsim/README.md` | how to run locally / in the cloud; what to trust |

## Facts every module must respect

- Step phases: `FF_ACCUM_SCHED=1:0.06,2:0.2,4` = k1 until 6% of charged time, k2 until 20%, k4 after; `FF_ACCUM_LR=0.5,0.7071`
  scales LR in k1/k2. K57-K59 use this. Warm-up 20 steps. Startup (step 0 compile, ~450-530 s cold) and steps 0-4 are excluded from the
  charged clock; the run stops when charged time reaches FF_TIME_TARGET (K59: target 1793, charged ~1790).
- Anchor: K59 (LK slope 0.35, seed 73, code M12): cold rehearsal C40 = 0.96092 @2357 steps on chip C; official 0.9671. K57: C36 0.96193 @2359, official 0.9686.
- Chip C k4 medians (full runs): LK0.35 s73 0.928 s, RF 0.911 s, K57 0.928 s. Chip D (runtime 2.33.10) k4 1.036 s in full runs (~7-12% slower) but only ~1% slower in 60-step screens.
- Hard step-time lessons to reproduce (Part 1): removing one elementwise pass over the 4C MLP tensor gave 1.8% (RF); torch.maximum (M11c) 5.9% slower vs abs-form 3.4% (M10b); QKV fusion +3.2% slower; a 96-parameter gate +7.6%.
- Quality noise: seed sd ~0.0006 bpb at 2M tokens. Effects < 0.0003 need 3+ seeds. 1% of steps ~ 0.00057 bpb (d bpb / d ln steps ~ 0.057 at ~2300 steps; it was 0.095-0.115 at ~900 steps).
- Offset rehearsal -> official: K51 +0.0063, K53 +0.0061, K54b_min +0.0072, K56 +0.0069, K57 +0.0067, K59 +0.0062 (six scored chip-C uploads; `research/sim-data/official-uploads.csv`), mean +0.0066, sd 0.0004.
- Rules from the campaign: step-normalise bpb ONLY for same-recipe runs, never for a mechanism that changes step time; chip D runs are judged only against chip D's own control (never as controls for C); one pair rejects a configuration, never a family; a 120-step screen predicts early loss only, not final bpb.
- Chip access (read-only harvest): an AWS profile with SSM access to the chip instances (named through the environment; see `ffsim/harvest.py`); chip C (block ended 30 Sep 11:30 UTC, 341 run dirs, 334 train.log, 231 eval.log, 18 eval20.log, 11.8 MB of train.log); chip D (10 runs). Run dirs: `/root/ff-claude/runs/<run>/{train.log,eval.log,eval20.log,overrides,code.sha256,steps.log,full_run.status,t_*}`; queue log `/root/ff-claude/queue/runner.log` (START/END lines with the override list); base env `/root/ff-claude/queue/base.env`; code dirs `/root/ff-claude/code*/train_ff.py`.
  SSM stdout is capped at 24,000 chars per invocation; the instance role cannot list S3 (PutObject untested). The dispatcher gzip+base64-encodes the script, sends it with RunShellScript, polls and prints stdout (`ffsim/harvest.py`).
- train.log step line: `step 00250 | loss 8.2381 | lrm 0.707 | dt 0.57s | tok/s 230,615 | charged 152s | next_level 20`; phase switches are logged as `FF_ACCUM_SCHED phase 1: 2 micro-batches per update (131,072 tokens) from step 20 (charged 20s)` and `phase 2: 4 micro-batches ... from step 610`; MTP stages as `FF_MTP stage k/8: ... from step N`. Noise lines to drop: `OperatorEntry`, `registered at`, `dispatch key`, `new kernel`, `previous kernel`, `operator:`, `W0926 ...torch/distributed/run.py`, `ShardIndexInjection`.
- experiments.csv columns: chip,run,steps,raw_bpb,norm_bpb,in_window,is_screen,FF_* (63 knobs),NEURON_*,overrides. Chips A/B only; run names `HHMM_<family>_<tag>_s<seed>`.
