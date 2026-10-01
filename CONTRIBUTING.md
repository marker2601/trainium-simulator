# Contributing

Thanks for helping. There are two kinds of contribution: **runs** (the numbers from training runs you already
did, so the simulator learns from them) and **code or docs** (ordinary pull requests).

## Contributing runs: help the simulator learn

The simulator's quality model is fitted on run records, and nearly all of them come from one team's chips and one
recipe family. Every run you share widens what it knows. A run it predicts badly is the most valuable kind.

### What a run record is

One JSON object per run, defined by [`contrib/schema.json`](contrib/schema.json):

| field | required | meaning |
|---|---|---|
| `schema_version` | yes | always `"1"` |
| `hardware` | yes | `trn2.3xlarge`, `trn2.48xlarge`, `trn1.2xlarge`, `trn1.32xlarge`, `gpu-proxy` (an `ffsim/gpu` replay) or `other` (then give `hardware_other`, free text) |
| `time_budget_s` | yes | training time budget in seconds (the challenge: 1800) |
| `base_recipe` | no | `K82s4`, `K60`, `K59` or `custom` (default). With a published base, `recipe` lists **only the knobs you changed** |
| `recipe` | yes | `FF_*` knob -> value, the names `train.py` reads (`{}` if you changed nothing). Numbers must be finite and, for the knobs the simulator reads, inside a plausible range (`KNOB_BOUNDS` in `ffsim/contrib.py`, e.g. `FF_DEPTH` 1-64, learning rates above 0 and below 100, fractions in [0, 1]) |
| `code_version` | no | one of the simulator's code lineages `M1` ... `M15`, or `custom`; defaults to the base recipe's |
| `seed` | no | the training seed (`FF_SEED`) |
| `chip_val_bpb` | one of the two | your local rehearsal val_bpb of the trained weights |
| `chip_eval_tokens` | no | tokens that local score was measured on (default 2,097,152: the first 2M public tokens) |
| `official_val_bpb` | one of the two | the official leaderboard score, if the run was uploaded |
| `steps` | yes | optimizer steps completed |
| `step_seconds` | no | mean seconds per optimizer step |
| `date` | no | `YYYY-MM-DD` |
| `framework_notes` | no | SDK / runtime versions, anything unusual (500 characters) |
| `contributor` | no | your GitHub login, to credit you (on an issue it must be the account that opens it) |
| `consent` | yes | must be `true` (see below) |

Example, starting from our best published recipe:

```json
{
 "schema_version": "1",
 "hardware": "trn2.3xlarge",
 "time_budget_s": 1800,
 "base_recipe": "K82s4",
 "recipe": {"FF_COOLDOWN_FRAC": "0.65"},
 "seed": 58,
 "chip_val_bpb": 0.9551,
 "steps": 2390,
 "contributor": "your-github-login",
 "consent": true
}
```

### How to submit

Pick one; they all end in the same reviewed pull request.

1. **The app.** Run `python space/app.py`, open the **Add my runs** tab, fill in the form and press *Check and build
   my submission*. It validates the run, shows you the exact JSON that will be shared and the current model's
   prediction for it, and gives you a pre-filled GitHub issue link. The app sends nothing; you open the link and
   submit the issue yourself. (API: `/contrib_record`.)
2. **The issue form.** Open a [Submit a training run](https://github.com/marker2601/trainium-simulator/issues/new?template=run-submission.yml)
   issue, paste the JSON, tick the consent box and submit.
3. **The command line.** Check a record before you share it, then print its pre-filled issue link:

   ```bash
   python -m ffsim contrib validate my-run.json
   python -m ffsim contrib issue-url my-run.json          # --plain for a template-less issue
   ```

Several runs: one issue per run. Duplicates (the same measurement sent again, whatever the handle or notes) are
recognised by a content hash and refused.

### What is shared, and what is never accepted

- **The issue is public the moment you submit it**, and so is its edit history. Only what is in the JSON record
  goes into the dataset, but everything you type into the issue (including the "Anything else" box) is visible to
  everyone. The data is published under **CC-BY-4.0** (the code stays MIT). Ticking the consent box and setting
  `"consent": true` says you agree to that and that the record holds no secrets or personal data.
- **Rejected automatically:** anything that looks like an AWS access key, account id (12 digits, also written
  1234-5678-9012), ARN, instance or other resource id, GitHub / Hugging Face / API token, JSON web token, private
  key, password assignment, URL with credentials, e-mail address, host IP address (4-part version strings such as
  `neuronx-cc 2.15.128.0` are fine), or a long opaque key-like string, in the record or in the free-text box. The
  checker names the field (by position, if the field name itself is the problem) without repeating the value, and
  **replaces the issue text with a notice** so the value is no longer on the page. It is still in the edit history
  until a maintainer deletes that revision: if it was a real credential, rotate it. The scan is best effort; do not
  rely on it.
- **Credit** is the optional `contributor` field, which must be your own GitHub login when you submit through an
  issue (so nobody can submit in someone else's name). Leave it out to stay anonymous.
- The app and the CLI never send anything: they only build the link.

### What happens after you submit

1. **Check.** A GitHub Action ([`.github/workflows/contrib.yml`](.github/workflows/contrib.yml)) reads the record
   from the issue, runs `python -m ffsim contrib validate` on it, and comments the result: errors to fix, notes, flags
   for the reviewer, and the current model's prediction for your run. Edit the issue to fix anything; the check
   reruns on every edit.
2. **Review and approval.** A maintainer checks that the numbers are plausible for the hardware and recipe and that
   the free-text fields hold nothing private, then adds the `contrib-approved` label. Runs the current model finds
   surprising (|z| > 4) are **flagged, never rejected**: the reviewer may ask you to confirm them. Other flags:
   `extrapolates` (your changes move a knob outside the fitted runs), `shifts_builtin_fit` (adding the run would trip
   the refit guard below), `contributor_cap` (one account already has many runs in the data), `claims_team_lineage`
   (a custom recipe that names one of the team's code lineages).
3. **Pull request.** The label opens a pull request labelled `contrib` that appends the record, exactly as it was
   when the label was added, to [`research/sim-data/contrib/runs-contrib.jsonl`](research/sim-data/contrib/). Its
   branch name carries the record's content hash. **Editing the issue after approval removes the label**, and a
   changed record needs a new approval and gets a new pull request: nothing can change under a reviewer. The pull
   request closes your issue when merged.
4. **Refit.** Merging triggers [`.github/workflows/refit.yml`](.github/workflows/refit.yml): the three models are
   refitted on the built-in data plus the contributions, and a **refit pull request** updates
   [`ffsim/model/params.json`](ffsim/model/params.json) (plain JSON, never a pickle),
   [`contrib/stats.json`](contrib/stats.json) and the README line: number of runs and contributors, hardware
   breakdown, the error on contributed runs before (built-in model only) and after the model learned from them
   (leave-one-out, from five usable runs on), and the guard numbers. **The guard** refuses the refit (the workflow
   fails and opens nothing) when the contributions make the model worse on the team's own runs: the built-in
   in-sample or pair-validation error grows by more than 0.0001 bpb, or a published base recipe's prediction moves by
   more than 0.001 bpb.
5. **Everywhere else.** The CLI and the app in this repository fit on the data at start-up, so they use merged runs
   at once. `ffsim/model/params.json` is a published snapshot of the fit for other tools; the app and the CLI do not
   load it. A hosted copy of the app (for example a Hugging Face Space built from `space/`) picks up merged runs
   when its maintainers redeploy it; no workflow pushes to it.

### How a contribution is used by the model

- Each hardware type and base recipe gets its own small fixed effect (`chip` = `contrib-<hardware>-<base>`, prior
  sd 0.003 bpb), so a systematic difference between your setup and ours is absorbed there rather than in the knob
  effects. Keying it by base recipe too matters for K82s4: its newest levers are outside the fitted data, and the
  app anchors it by about -0.004 bpb; until a (hardware, K82s4) effect has data, the outlier check and the "before"
  error apply that same anchor, so a K82s4 run is not judged against a known bias.
- One account (or the anonymous bucket) adds at most 25 runs to a fit; further runs are stored, not fitted.
- Every reader of the data file skips lines it cannot use (unreadable, duplicated by content hash, non-finite or
  implausible numbers) and says so, so one bad line cannot stop the app, the CLI or a workflow.
- The knob features are read from the base recipe plus your changes. Knobs the feature table does not read are
  stored but do not move the fit (the checker lists them).
- Runs shorter than 500 steps are stored but not fitted (a short screen predicts early loss only). A local score on
  another eval length is stored but not fitted unless an official score is given; an official-only record is fitted
  as `official - offset` and flagged.
- The step-time model is not refitted from contributions (it needs per-phase step times); `step_seconds` is kept
  for a future step-time route.

## For maintainers

- Create the labels `run-submission` (applied by the issue form), `contrib-approved` (your approval: adding it
  opens the pull request) and `contrib` (applied to the generated pull requests). The check also runs for issues
  titled `[run] ...`. Only a `contrib-approved` label added by an account with write access counts.
- Settings -> Actions -> General: keep the default workflow token **read-only** (every workflow declares its own
  `permissions:`) and tick only "Allow GitHub Actions to create and approve pull requests". Do not give the
  Actions token a bypass of `main`'s protection: nothing here pushes to `main`; the refit arrives as a pull request.
  Optionally require actions to be pinned to a full-length commit SHA (they are; Dependabot keeps them current).
- Protect `main` with "Dismiss stale pull request approvals when new commits are pushed" and "Require approval of
  the most recent reviewable push".
- **Merging a contribution changes the published model** (after the refit pull request is merged). Read the flags.
- Pull requests opened with the workflow token do not trigger CI on their own; close and reopen one (or push to
  it) to run CI before merging. CI stays green on `main` between a contribution merge and its refit merge: the
  shipped-model test only compares `params.json` with the data it was fitted on.
- Two pending contribution pull requests both append to the same file; `.gitattributes` marks it `merge=union`, so
  a local merge or rebase combines them. If GitHub reports a conflict, rebase the second one. If the same run was
  merged twice, readers count it once (duplicates are dropped by content hash).
- If the check replaced an issue's text because it found a secret, delete the original revision from the issue's
  edit history and tell the submitter to rotate the credential.
- During a spam wave, use the repository's interaction limits; the check only comments, and nothing reaches a pull
  request without your label.
- Reverting a bad record: delete its line from `research/sim-data/contrib/runs-contrib.jsonl` and merge; the refit
  runs again. To refit by hand: `python -m ffsim contrib refit --readme README.md --sync-space --guard`.
- A hosted Space built from `space/` does not update itself: redeploy it after a refit is merged.

## Contributing code or docs

1. Fork, create a branch, make the change.
2. Run the tests: `pip install -r requirements.txt && python -m pytest -q tests` (on Windows, from Git Bash or
   PowerShell; the tests that drive the shell scripts put Git's `usr/bin` on their own PATH). Changes to `ffsim/` that the app
   uses must be copied to `space/ffsim/` (a test checks the copies are identical). For the app:
   `pip install -r space/requirements.txt && python space/app.py`.
3. If you change the data or the fitting code, refresh the shipped model: `python -m ffsim contrib refit
   --readme README.md --sync-space` (a test checks `ffsim/model/params.json` matches the data).
4. Open a pull request describing what changed and how you checked it. CI (`.github/workflows/ci.yml`) runs the
   test suite and imports the app.

By contributing code you agree it is released under the repository's [MIT licence](LICENSE) (Apache-2.0 for
changes to the files listed in [NOTICE](NOTICE)).
