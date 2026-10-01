# ffsim: the FrontierForge Trainium run simulator

Predict the official val_bpb of a candidate recipe without spending Trainium time, score hundreds of
candidates in seconds, and get back the short list worth confirming on the chip.

```
official_bpb = quality(recipe, steps, seed) + offset(eval shard)
steps        = (FF_TIME_TARGET - 5.5 s step-clock reserve) split over the FF_ACCUM_SCHED phases
               at the per-phase step times the step-time model predicts for (code, chip), each
               median scaled by 1.0057 (the median -> mean step overhead the logged dt hides)
```

> Release note: this is the campaign-time working README, edited only to remove account details and to fix
> paths. Times are UTC as logged. Start with the top-level `README.md` and `docs/SIMULATOR.md`. Step 1 below
> needs the raw campaign tables (`research/experiments.csv`, all harvested chip logs), which are not part of the
> release; `runs.jsonl` ships pre-built, so `fit`, `validate`, `simulate` and `search` work out of the box.

Pure numpy + stdlib, Python 3.11. Design brief: `docs/simulator-spec.md`; frozen data
contract and module ownership: `ffsim/CONTRACT.md`, `ffsim/schema.py`.

## Quick start

```bash
# 0. (only with your own Trainium instances) copy run logs off the chips over SSM. The chips are named through
#    FFSIM_CHIP_<X>_REGION / FFSIM_CHIP_<X>_INSTANCE and FFSIM_AWS_PROFILE (see ffsim/harvest.py). The release
#    ships the already-built research/sim-data/runs.jsonl, so steps 0-1 are optional.
python -m ffsim.harvest --help          # the real options: --chip {C,D} --dest --all --inventory-only ...
python -m ffsim harvest --chip C        # the wrapper passes its arguments through (its own --help shows only `rest`;
                                        #  `python -m ffsim harvest -- --help` reaches the real one)

# 1. experiments.csv + monitor table + harvested train.log/eval.log -> research/sim-data/runs.jsonl
python -m ffsim build-dataset --experiments research/experiments.csv \
    --monitor research/sim-data/monitor-runs.csv --chips-dir research/sim-data --out research/sim-data/runs.jsonl

# 2. fit the three models -> research/sim-data/models.pkl, print the validation reports
python -m ffsim fit --runs research/sim-data/runs.jsonl --pairs research/sim-data/validation-pairs.json \
    --uploads research/sim-data/official-uploads.csv --out research/sim-data/models.pkl

# 3. the reports again (LOO step-time error, pair sign agreement / MAE, offset mean/sd, K59 anchor check).
#    Read `passes` / `pass` with "Trust boundaries" below. On the shipped fit both are misses, and neither
#    line says so in words: step time `passes: False` (0.545% run-weighted over the 161 "reachable" held-out
#    runs; the 28 novel-mechanism runs are not in that number at all) and quality
#    `pass: {'sign_agreement': True, 'mae': False}` (MAE 0.00039 > 0.0003). The decision rule below says
#    what to act on instead.
python -m ffsim validate

# 4. one recipe (seed 73 = that seed's effect; omit --seed = marginal over seeds). K60 is the current best
#    (official 0.9655, 29 Sep 19:50 UTC); K59 stays the anchor recipe that `fit` / `validate` check.
python -m ffsim simulate --recipe ffsim/examples/recipe-K60.json --n 2000 --seed 73
python -m ffsim simulate --recipe cand.json --vs ffsim/examples/recipe-K60.json     # P(cand < K60), paired

# 5. hundreds of candidates -> research/sim-data/search-<space>.{md,csv,json}. The K60 space is the primary base
#    (runs.jsonl carries the two full FF_ATTN_SRC runs the fit uses); the K59 spaces are kept for comparison.
python -m ffsim search --space ffsim/examples/space-k60-local.json --n-sims 1000 --gen local --top 20
python -m ffsim search --space ffsim/examples/space-k59-local.json --n-sims 1000 --gen local --top 20
python -m ffsim search --space ffsim/examples/space-k59-wide.json --gen random --n-random 400 --n-sims 1000 --top 20 --workers 8

# 6. re-render a saved search with another top-K
python -m ffsim report --results research/sim-data/search-k59-local.json --top 40
```

Before `models.pkl` exists, `--models stub` runs `simulate`, `search` and `validate` with the
**knob-blind anchor stubs** (K59: k4 0.928 s, 0.96092 @ 2357 steps, offset +0.0066 +- 0.0004). They prove
the pipeline runs; every candidate scores the same, so they rank nothing.

Tests: `python -m pytest tests/test_ffsim_simulate.py -q` (stub models; no other ffsim module needed).

## What each number means

| column | meaning |
|---|---|
| `steps` (mean +- sd) | optimizer steps in the charged time. Warm-up at K until step `FF_WARMUP`, then each `FF_ACCUM_SCHED` phase until its fraction of `FF_TIME_TARGET`, then K until the next step would cross `FF_TIME_TARGET - 5.5` (the measured step-clock reserve: the log clock stops a median 5.5 s short of the target over 136 completed chip C/D full runs). Every phase median is scaled by 1 + 0.0057 before the walk (logged medians under-count the mean step), so the K59 anchor walks to 2369 at 1793 s (C40 2357, a 0.5% slow run) and 2324 at the 1760 s queue budget (the seven LK0.35 seeds: 2322-2329); the old raw walk over `- 3 s` gave 2385 (+1.2%). The sd is chip jitter: one multiplicative draw per run (sd 0.4% unless the model says otherwise). 1% of steps ~ 0.00057 bpb. |
| `bpb_2m` (mean +- sd) | predicted public-shard rehearsal bpb (trained weights, first 2,097,152 tokens). The quality model answers at the mean step count; draws move along `d bpb / d ln steps` (about -0.057 at 2,300 steps). sd = the model's sd: seed luck (~0.0006) + fit error when no seed is given, fit error only for a named seed. |
| `official` p10 / p50 / p90 | `bpb_2m` + the rehearsal -> official offset (the seven scored chip-C uploads K51..K60: +0.0061..+0.0072, mean +0.00653, sd 0.00042; the draws use the predictive sd 0.00045, which carries the mean's own uncertainty). Monte Carlo percentiles. |
| `P(beat best)` | fraction of draws with official < the best official score (`--best`, default 0.9655 = K60, scored 29 Sep 19:50 UTC; K59 was 0.9671). |
| `d vs base`, `P(< base)` | paired with the base recipe under **common random numbers**: chip jitter, seed luck, the quality model's coefficient draw (one `beta ~ N(beta, cov)` per draw, evaluated on each recipe's design row) and the upload offset are shared, so they cancel exactly where the two recipes agree; each recipe keeps its own per-run residual (`sigma`, 0.0003 in the shipped fit). The paired sd (`delta_sd_vs_base` in the CSV/JSON rows, `diff sd` from `simulate --vs`) is therefore `sqrt(d cov d + 2 sigma^2)`, the same number the quality model's `predict_delta` reports (0.00042-0.00049 for single- and double-knob changes around K59), and `P(< base)` is the matching probability: a real statement of confidence in `d vs base`, not a sign indicator. Identical recipes give exactly 0.5; a `d vs base` of -0.0003 gives about 0.75. A model that exposes no `sigma`/`cov` (the anchor stubs) cannot be split and its notes say `P(< base) is a sign indicator`. Judge size with `d vs base` against the 0.0003 floor (Trust boundaries). |
| `support` | `ok` when every changed knob value was seen in the fitted runs (or lies between seen values, marked `interp`); `EXTRAP` when the surrogate is guessing (`NEVER-VARIED` = the knob never changed in the data). |

The markdown report ends with **Next Trainium confirmations**: the top in-support candidates by expected
improvement, each with the env overrides for seeds 73, 58, 67. Apply the decision rule in Trust boundaries
before queueing any of them (`d vs base` against the 0.0003 floor). One pair rejects a configuration, never a
family; effects under 0.0003 need all three seeds.

## Trust boundaries

- **The surrogate only sees knobs that varied.** A knob that never changed in `runs.jsonl` has no
  coefficient; a candidate that changes it is flagged `NEVER-VARIED` and its prediction is the base's.
  Values outside the seen range are `EXTRAP`. Neither is a reason to run it; both are a reason to screen.
  The support table is the quality model's own when it exposes one; otherwise `fit` derives it from the
  fitted records (every knob set explicitly in a full run with a bpb, with the values seen) and stores it in
  `models.pkl` (`meta["support"]`); `validate` prints it.
- **Unseen code versions fall back.** The step-time model knows the code versions it was fitted on
  (M9/M12/M13/M14 lineages, K57 bakes, RF/LK forms). For a new `train.py` it falls back to the nearest
  lineage and says so in `notes`; the compiler is not linear in op count (QKV fusion was slower with fewer
  ops; a 96-parameter gate cost 7.6%), so a new kernel needs an S60 screen before its steps are believed.
- **Chip D is not chip C.** D (runtime 2.33.10) is 7-12% slower at k4 in full runs but only ~1% in S60
  screens; simulate with `"chip": "D"` only against D's own data. Step-normalised bpb is valid only for
  same-recipe runs, never for a mechanism that changes step time; the simulator therefore never normalises,
  it predicts steps and quality separately.
- **The 21 Sep GPU proxy failed validation** (the U-net sign flipped between GPU and Trainium). Route 2 of
  the spec is the separate `ffsim/gpu/` track below; nothing in the route-1 modules or `ffsim/cloud` runs a GPU.
- **A 120-step screen predicts early loss only**, and S60 screens are speed checks: screens are never
  quality data points (`is_screen` records are excluded by the quality model).
- **Validation results** (29 Sep 23:16Z fit, 1,206 records: the 1,188 of the 16:45 UTC harvest plus 18 post-harvest
  chip C/D full runs added from the monitor at 23:10 UTC, which the quality fit holds out because a monitor row
  carries per-run overrides only; `fit` and `validate` print them). The spec's targets were
  leave-one-code-version-out k4 step-time error <= 0.5%, >= 80% sign agreement and <= 0.0003 MAE on the
  paired Trainium comparisons, and offset sd ~0.0004. What the shipped fit scores:
  - *Step time*: **`passes: False`**. It is judged (`judged_on`) on `mean_abs_pct_reachable_by_run`, the
    **reachable subset only**: the 161 of 189 held-out runs whose every mechanism some other lineage also
    ran. Run-weighted that error is **0.545%** (60 of the 161 runs above 0.5%; RMSE 0.73%), so the 0.5%
    target is missed. The lineage-averaged `mean_abs_pct_reachable_by_lineage` 0.43% (0.33% inside the
    M9-M15 family, whose run-weighted figure is 0.51%) is under the target only because the 2-3-run lineages
    cancel M9 (0.51%, 44 runs) and M12 (0.66%, 30 runs: the K59 lineage itself misses when held out). Over all
    189 runs the error is 0.95% by run / 1.02% by lineage (`mean_abs_pct`), because the other 28 (`n_novel`)
    carry a mechanism only their own lineage ran: the RF backward, the abs and max leaky forms (the contract's
    M10b / M11c lessons, one run each), XSA gating, the M13 custom Functions, attention-source reuse, lanes,
    pooling, one-hot / chunked CE, and the chip-D screen interaction. Held out, those runs are predicted 1-14%
    wrong, and they are outside every `reachable` number. In the shipped model their lineages are in the fit,
    so RF (-2.0%) and abs / max (+4.0% / +6.3%) are reproduced, but each leaky form is one run. QKV fusion and
    the 96-parameter gate are in no chip C/D run and have no coefficient: a mechanism no fitted run carries is
    charged a +3% prior (`NOVEL_PRIOR_LOG`; the contract's measured costs were +3.2% and +7.6%, so that is the
    low end) and the step-time `notes` say "cost unknown". Such a candidate's step count is unvalidated.
    Details: `research/sim-data/steptime-validation.md`.
  - *Quality*: leave-one-pair-out on the 73-pair task list (`in_task_list`: every clean pair in the harvested
    data; 33 of the 73 are in the families the spec's "last 20 paired comparisons" names, the rest are seed,
    rehearsal, schedule and other pairs from the same data), sign agreement **85%** (target met) and MAE
    **0.00038** (target 0.0003 **missed**). `validate` scores all 99 pairs (the 73 plus the extra seed / chip
    pairs) and prints 84.8% / 0.00039 with `pass: {'sign_agreement': True, 'mae': False}`; the miss is not
    spelled out in words. Recipe pairs only (62): 82% / 0.00034; decisive recipe pairs (|observed| >= 0.0003,
    n = 39): 95% / 0.00040 (all 64 decisive pairs of the 99: 97% / 0.00046). The MAE sits on
    the pair noise floor: the fitted per-run residual sd is 0.0003, so a same-seed pair carries noise sd
    ~0.00042 and even a perfect model would score ~0.00034 MAE on single-run pairs. In-sample the same pairs
    give 89% / 0.00022, which is not the honest number. Details: `research/sim-data/quality-validation.md`.
  - *Offset*: mean +0.00653, sd 0.00042 from the seven K51-K60 chip-C uploads (K60: C41 0.95910 -> 0.9655,
    +0.0064; target met; the sd's 95% interval is 0.00027-0.00092).
  - *Out of sample, after the fit*: the 15 same-seed pairs that finished after the harvest
    (`docs/simulator-report-20260929.md`, section 5) score 53% sign / MAE 0.00104 against the shipped fit. The five
    attention-source pairs are right 4 of 5 but over-predicted by ~0.0005 (the coefficient rests on two runs), the
    five AdamW-LR pairs are prior-only (sd 0.0026) and the five sub-variant / softcap-on-K60 pairs sit inside the
    noise floor (|predicted| <= 0.0002 for four of them). Read `d vs base` with that in mind.
- **Decision rule for a search.** (This replaces "do not act on a search whose `validate` misses the
  targets": the shipped fit misses the MAE target and, run-weighted, the 0.5% step-time target too, and the one
  step-time figure under 0.5% is judged on the reachable subset, so that rule would either discard every search
  or trust a number that excludes the novel-mechanism runs.) Judge each candidate by its
  `support` flag, its step-time `notes`, and `d vs base` against the 0.0003 floor:
  1. **Confirm on Trainium** (seeds 73, 58, 67) only a `support ok` candidate with `d vs base` <= -0.0003 and
     no "cost unknown" note. At that size the surrogate's sign was right 95% of the time held out, and it is
     the smallest effect a three-seed pair resolves.
  2. **Tie**: `support ok` with -0.0003 < `d vs base` < 0. The surrogate cannot order these (its held-out MAE
     is larger than the difference), so the ranking among them is noise. Spend at most one seed-73 probe on
     the top one, and read it as a probe: an effect under 0.0003 needs all three seeds before it counts.
  3. **Screen first**: `EXTRAP` / `NEVER-VARIED`, or a step-time note saying "cost unknown" (a mechanism no
     fitted run carried; held-out error 1-14%, and the +3% the model charges it is a prior, not a measurement).
     One S60 screen with a same-batch control turns the step time into a ~0.5% prediction (screens carry a
     +-0.6% batch offset, so the control is not optional); a 120-step screen says nothing about final bpb.
  4. `P(< base)` is the surrogate's own confidence in the sign of `d vs base` against the pair noise floor
     (see the table above); at the 0.0003 confirmation threshold it is only ~0.75, which is why the three-seed
     protocol exists. Use `d vs base` for size. Chip D candidates are judged only against D's own control. One
     pair rejects a configuration, never a family.

  On the 29 Sep 23:16Z fit the `space-k59-local` and `space-k60-local` searches each have 42 of their 282 candidates
  in support and nothing reaches rule 1: the top row of both, `FF_LEAKY_RELU2=0.3 FF_WD_SCHED=1` at `d vs base`
  -0.00007 (`P(< base)` 0.57-0.58, `support ok`), is rule 2, a tie with the next in-support rows, not a
  confirmation; the runner-up `FF_COOLDOWN_FRAC=0.7 FF_WD_SCHED=1` (also -0.00007) is `EXTRAP`
  (`FF_COOLDOWN_FRAC=0.7` was seen in one M3-era run and the pair was never co-observed): rule 3. The
  401-candidate `space-k59-wide` random search (11 in support) puts `FF_LEAKY_RELU2=0.3 FF_SOFTCAP_A=16.5` at
  -0.00032 and `FF_SOFTCAP_A=16.5` alone at -0.00029 (`support ok`: three M9+ pairs, mean -0.00017). On paper the
  first crosses rule 1, but the fourth softcap pair, run on the K60 base after the fit's data
  (`2110_R_k60sc165_s73`: +0.00068 at seed 73), moves the four-pair mean to +0.0000 and the campaign closed the
  knob: treat both as rule 2 ties. Earlier fits the same day put the best local row at -0.00027; these numbers move
  with every refit, so re-run the search rather than quoting them.

## Search space format

```json
{"name": "k59-local",
 "base": {"name": "K59", "code_version": "M12", "chip": "C", "time_target": 1793, "knobs": {"FF_LEAKY_RELU2": "0.35", "...": "..."}},
 "dims": [{"knob": "FF_LEAKY_RELU2", "values": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
          {"knob": "FF_SOFTCAP", "range": [12, 20], "step": 1},
          {"knob": "code_version", "values": ["M12", "M14"]}],
 "max_changes": 2}
```

`base` may instead be `{"recipe": "recipe-K59.json"}` (path relative to the space file). Dims are FF_*
knobs, or the special dims `code_version`, `chip`, `time_target`. Values are written the way train.py's
`_env_*` readers parse them (`15.0` -> `"15"`, `true` -> `"1"`). Generators (`--gen`): `local` = base +
every single and double change (the default; the example gives 282 candidates), `oat` = one-at-a-time,
`grid` = the full product filtered by `max_changes`, `random` = `--n-random` distinct draws.

Speed: about 1.5 ms per candidate at 1,000 draws (the support tables are computed once per worker; the draws are one
(candidates x draws) numpy matrix), so the 282-candidate local spaces take 0.4 s of compute (the time the report
header prints; ~1.1 s wall with interpreter start-up) and the 401-candidate wide space 5.0 s wall with `--workers 8`
(process start-up dominates; results are identical for any N because the draws are regenerated from `--rng`). The
other steps: `build-dataset` ~12 s, `fit` and `validate` ~2 s each, `simulate` and `report` under 1 s.

The example recipes: `ffsim/examples/recipe-K59.json`, the `_env_*` defaults of
the K59 upload `train.py` (M12 code; not shipped: `recipes/K60/train.py` is K59 + `FF_ATTN_SRC` on M14 code) (K57T recipe baked on M12 + `FF_LEAKY_RELU2=0.35 FF_LEAKY_FORM=fnm`;
official 0.9671, C40 rehearsal 0.96092 @ 2357), and `ffsim/examples/recipe-K60.json`, K59 + `FF_ATTN_SRC=5:6,7,8` on
M14 (`recipes/K60/train.py`; official 0.9655, C41 rehearsal 0.95910 @ 2361; at seed 73 the
simulator gives 2365 +- 9 steps, bpb_2m 0.95888 +- 0.00042, official 0.96542 +- 0.00061, P(< 0.9655) 0.54). The
spaces: `space-k60-local.json` (primary; its `queue` block is the K60 arm the chip C queue ran,
`C:1639_R_m14asa_s73`, so the launch lines carry the real job env), `space-k59-local.json` (the same eight dims
around K59) and `space-k59-wide.json` (16 dims over every knob the fit saw varied in the M9+ era, 1-3 changes,
`--gen random`).

## Local vs cloud

Local (Windows Git Bash / PowerShell / Linux, numpy only):

```bash
python -m ffsim search --space ffsim/examples/space-k60-local.json --n-sims 1000 --gen local --top 20 \
    --out research/sim-data/search-k60-local.md
python -m ffsim search --space ffsim/examples/space-k59-wide.json --gen random --n-random 400 --n-sims 1000 --top 20 \
    --workers 8 --out research/sim-data/search-k59-wide.md
```

Cloud (`ffsim/cloud/`, numpy-only box, < $0.10 per thousand-candidate search, no GPU):

```bash
docker build -f ffsim/cloud/Dockerfile -t ffsim . && docker run --rm -v "$PWD/research/sim-data:/work/research/sim-data" ffsim
bash ffsim/cloud/ec2_launch.sh --profile PROFILE --region us-east-1 --bucket BUCKET --instance-profile ROLE --spot   # dry run
bash ffsim/cloud/ec2_launch.sh ... --yes                                                                          # launch
```

`ec2_launch.sh` is read-only until `--yes`; the instance runs `run_batch.sh` (fit, then search) and terminates
itself. Details and IAM needs: `ffsim/cloud/README.md`.

## Phase 2: GPU proxy track (route 2, `ffsim/gpu/`)

**Status: built, unit-tested, and run on 30 Sep 2026** (first four equal-step runs matched chip C; see
`docs/SIMULATOR.md`). Route 2 is the equal-steps proxy of
`docs/simulator-spec.md` Part 2: `make_train_gpu.py` patches the K60 submission `train.py` into a
CUDA/CPU fork (`train_gpu.py`) that walks the Trainium schedule under a *virtual charged clock* (k1/k2/k4 seconds per
step from `configs/schedule-K60-TT1793.json`, so the accumulation phases, MTP stages, cooldown and EMA levels switch at
the same step numbers as on chip C and the run stops at the same step count); `fleet.py` runs a config's run list one
subprocess per GPU; `calibrate.py` scores the GPU pairs against the Trainium `validation-pairs.json` under the frozen
rule of `ffsim/gpu/PROTOCOL.md` section 3.3; `launch.sh` brings one box up and down. The 20 Sep proxy flipped a
verdict's sign against Trainium, so **no GPU verdict may influence a chip slot before `calibrate` accepts** (exit 0);
until then every GPU number is a calibration measurement, not a prediction. Details: `ffsim/gpu/README.md`
(what to trust), `PROTOCOL.md` (the gates), `RUNBOOK.md` (operations), `PATCH_NOTES.md` (what the fork changes).

Before spending, check your own account: the on-demand G/VT vCPU quota (L-DB2E81BA; a g6e.xlarge needs 4, the
12xlarge 48) and the current Deep Learning AMI (`launch.sh plan` resolves it from the public SSM parameter). The
39-run / 16-pair gate (`configs/gate-k60.json`, 90,336 optimizer steps) is 115.8 GPU-hours at the *estimated*
4.5 s/step: about $217 on one g6e.xlarge (116 h wall) or $317 on a g6e.12xlarge (30 h wall) at the 29 Sep 2026
list prices; the first measured `median_step_seconds` re-costs it (measured: ~1.35 h per 2,314-step run on an
L40S in the default mode, ~1.8 h in strict deterministic mode).

**The spend decision belongs to the owner**: nothing below `plan` runs without `--yes`, `up` is the only subcommand
that spends and `down` the only one that destroys. The exact commands (`python -m ffsim gpu <sub> ...` passes
everything after the sub name to `ffsim.gpu.fleet` / `calibrate` / `make_train_gpu` / `launch.sh` and imports them
lazily, so this CLI never needs torch):

```bash
# 0. free: the pinned K60 walk, the run list, GPU-hours and dollars; then the quota/AMI/price/straggler check
python -m ffsim gpu plan --config ffsim/gpu/configs/gate-k60.json          # --quote-prices skips the pricing lookup
python -m ffsim gpu schedule                                                 # the switch table alone
python -m ffsim gpu launch plan --region us-west-2                           # == bash ffsim/gpu/launch.sh plan
python -m ffsim gpu make-train --check                                       # train_gpu.py == fresh build of K60 train.py
python -m ffsim gpu calibrate --configs ffsim/gpu/configs/gate-k60.json --freeze research/sim-data/gpu-rule.json

# 1. spends (owner decision, PROTOCOL.md section 5): one g6e.xlarge in us-west-2, hard poweroff after --max-hours
python -m ffsim gpu launch up --yes --key <ec2-keypair> --ssh-key ~/.ssh/<keypair>.pem --max-hours 12 --auto-down

# 2. code + data + the organizer's prepare.py on the box (once), then the smoke test and the gate
python -m ffsim gpu launch push --train recipes/K60/train.py
python -m ffsim gpu launch run --config ffsim/gpu/configs/prescreen.json --results runs/prescreen.jsonl
python -m ffsim gpu launch run --config ffsim/gpu/configs/gate-k60.json --results runs/gate.jsonl
python -m ffsim gpu launch status

# 3. results back, the seven gates (exit 0 accepted / 1 rejected / 2 pending), the box gone
python -m ffsim gpu launch fetch --wait --out research/sim-data/gpu/<date>
python -m ffsim gpu report --results research/sim-data/gpu/<date>/gate.jsonl
python -m ffsim gpu verdict --results research/sim-data/gpu/<date>/gate.jsonl --json research/sim-data/gpu-calibration.json
python -m ffsim gpu calibrate --results research/sim-data/gpu/<date>/gate.jsonl --rule research/sim-data/gpu-rule.json
python -m ffsim gpu launch down --yes

# 4. only after acceptance: ffsim search top-K -> a paired candidate config for the same fleet
python -m ffsim gpu from-search --top-k 5 --out ffsim/gpu/configs/candidates-from-ffsim.json
```

Tests: `tests/test_ffsim_gpu_{cli,fleet,launch,trainer}.py` (the launch tests drive `launch.sh` against a fake `aws`;
the trainer tests exercise the pure-python pieces of the fork, torch is not installed locally).

## Outputs

| path | content |
|---|---|
| `research/sim-data/runs.jsonl` | one `RunRecord` per line (see `schema.py`) |
| `research/sim-data/models.pkl` | pickled `{steptime, quality, offset, meta}` |
| `research/sim-data/search-<space>.md` | top-K tables + next Trainium confirmations + legend |
| `research/sim-data/search-<space>.csv` | every candidate, one row each (`rank`, `changes`, `steps_mean`, ..., `in_support`) |
| `research/sim-data/search-<space>.json` | the full result (`report` re-renders from it) |
| `research/sim-data/search-k60-local.*`, `search-k59-local.*`, `search-k59-wide.*` | the three 29 Sep searches on the 23:16Z fit (`docs/simulator-report-20260929.md` reads them) |

## Module map

| module | role |
|---|---|
| `simulate.py` | `steps_scalar` / `steps_from_step_times` (train.py phase semantics), `simulate_batch` (the vectorised core), `simulate`, `simulate_pair` (common random numbers), `Models` bundle, anchor stubs |
| `search.py` | space json, generators, `evaluate` (CRN across candidates, support flags, `--workers`) |
| `report.py` | markdown / CSV / JSON writers, one-screen summaries |
| `cli.py`, `__main__.py` | the subcommands above, plus the `gpu <sub>` group (lazy delegation to `ffsim/gpu/`) |
| `gpu/` | route 2, the equal-steps GPU proxy: `make_train_gpu.py`, `train_gpu.py`, `vclock.py`, `fleet.py`, `calibrate.py`, `launch.sh`, `configs/` (see Phase 2 above) |
| `steptime.py`, `quality.py`, `offset.py`, `dataset.py`, `parse_*.py`, `harvest.py` | the fitted models and the data pipeline (other owners; see `CONTRACT.md`) |
