# ffsim GPU proxy: launch and operations runbook

Owner-facing, step by step. The tools are `ffsim/gpu/launch.sh` (this runbook), `ffsim/gpu/bootstrap_gpu.sh`
(first-boot script the launcher sends as user-data) and `ffsim/gpu/fleet.py` + `train_gpu.py` (the proxy
itself; see `ffsim/gpu/PROTOCOL.md` for what a GPU delta means and the six-gate decision rule).

**Safety model.** `launch.sh` has eight subcommands. `up` is the only one that spends money and `down` the only
one that destroys anything; both refuse without `--yes`. `plan`, `status` and `fetch` are read-only on AWS
(describe-*, get-*, pricing) or talk to the box over SSH. `plan` prints the exact `run-instances` command `up`
would run. Every box is tagged `Name=ffsim-gpu`, so `status` and `down` find it even without the state file
(`~/.ffsim-gpu/<region>.env`). The box powers itself off after `--max-hours` (default 12) no matter what.

## Example quota facts from one account (2026-09-29; re-check your own with `launch.sh plan`)

| fact | value |
|---|---|
| `L-DB2E81BA` Running On-Demand G and VT instances (vCPUs), **us-west-2** | **64** (adjustable; 0 in use) |
| `L-DB2E81BA` same quota, **us-east-1** | **8** |
| `L-3819A6DF` All G and VT Spot Instance Requests (vCPUs), us-west-2 | **8** (spot fits only g6e.xlarge / g6e.2xlarge) |
| Deep Learning AMI (SSM `/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id`, us-west-2) | resolved at plan time (then: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260427", root snapshot 40 GiB gp3 |
| `ffsim-gpu` instances up | none (default VPC) |
| ffsim-gpu security group | not yet created (`up` creates `ffsim-gpu-ssh`, tcp/22 from the caller's IP/32 only) |

Prices at 2026-09-29T23:40Z, us-west-2, Linux on-demand list (pricing API) and lowest-AZ spot (`describe-spot-price-history`):

| type | GPUs | vCPU | on-demand $/h | spot $/h now | fits 64-vCPU quota | $/GPU-hour (od) |
|---|---|---|---|---|---|---|
| g6e.xlarge | 1x L40S 48 GB | 4 | 1.861 | 1.70-1.86 (no real discount) | yes (spot too) | 1.86 |
| g6e.2xlarge | 1x L40S 48 GB | 8 | 2.242 | 1.42-2.24 | yes (spot too) | 2.24 |
| g6e.12xlarge | 4x L40S 48 GB | 48 | 10.493 | 6.10-9.91 | yes (on-demand only) | 2.62 |
| g6e.48xlarge | 8x L40S 48 GB | 192 | 30.131 | 9.6-30 | **no** | 3.77 |
| g5.xlarge | 1x A10G 24 GB | 4 | 1.006 | 0.49-0.51 | yes | 1.01 |
| g5.2xlarge | 1x A10G 24 GB | 8 | 1.212 | 0.52-0.56 | yes | 1.21 |
| g5.12xlarge | 4x A10G 24 GB | 48 | 5.672 | 3.6-5.1 | yes (on-demand only) | 1.42 |
| g5.48xlarge | 8x A10G 24 GB | 192 | 16.288 | 5.5-11 | **no** | 2.04 |

Prices move: `plan` re-fetches them. The default is **g6e.xlarge on-demand** (one L40S, 48 GB: the K60 model at
`FF_MB=8` fits with room, bf16 dense 181 TFLOP/s). A g5 (A10G, 24 GB, ~70 TFLOP/s bf16) is 2.6x slower per
GPU-hour and only cheaper per dollar if `FF_MB` fits in 24 GB; take it only for a long, unhurried screening
sweep. Multi-GPU boxes are faster in wall clock and never cheaper per GPU-hour.

## Step 0: quota check and plan (free)

```bash
bash ffsim/gpu/launch.sh plan --gpu-hours 24                 # us-west-2, g6e.xlarge
bash ffsim/gpu/launch.sh plan --region us-east-1             # shows the 8-vCPU quota: one small box only
```

What `plan` prints: the account; both quota values with vCPUs in use and what the chosen type needs; the AMI
id/name and root size; the cost table for `--gpu-hours` (box-hours = GPU-hours / GPUs per box, on-demand and
spot cost, whether the type fits the quota); the user-data file it rendered (with `--max-hours`, `--auto-down`,
`--grace-min` baked in); the security-group rule `up` would add; any existing `ffsim-gpu` instance; and the
exact `aws ec2 run-instances ...` command. Nothing is launched. If the quota line says `NO (...)`, request an
increase (free, usually granted in minutes to hours):

```bash
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 96 --region us-west-2 --profile "$AWS_PROFILE"
```

## Step 1: spend decision

Decide the GPU-hours first, then the type. Per-run GPU time comes from `python ffsim/gpu/fleet.py plan --configs
<set>.json` (offline, no AWS): it costs each config from the K60 step count (~2361 steps) and the L40S dense
bf16 figure at 20-40 % MFU. Rules of thumb from the 20 Sep proxy protocol (§5, not published), rescaled to K60's 2.6x more steps than K31:
one full equal-step run is ~1.8-3.3 h on one L40S (~$3.4-6.2), so a 12-run calibration set is ~$40-75 and a
6-seed-pair verdict (12 runs) ~$40-75. **Measure one run and re-cost from its median step time before
committing to a set** (`status` shows the log tail; `fleet.py report` gives the step time).

Worst case is bounded by the hard stop: `--max-hours H` x $/h (12 h x $1.861 = $22.33 on the default). Set
`--max-hours` to the planned wall clock plus a fetch margin, never "just in case" large.

## Step 2: up (spends)

```bash
bash ffsim/gpu/launch.sh up --yes --key <ec2-keypair> [--ssh-key ~/.ssh/<keypair>.pem] \
     --max-hours 10 [--auto-down --grace-min 30] [--type g6e.12xlarge] [--spot] [--instance-profile NAME] [--s3 s3://bucket/prefix]
```

- `--key` is an existing EC2 key pair in the region (`aws ec2 describe-key-pairs`); `--ssh-key` its private key
  (default `~/.ssh/<key>.pem`), stored in the state file for `push/run/status/fetch/ssh`.
- `up` refuses if: no `--yes`; no `--key`; an instance tagged `ffsim-gpu` is pending/running; the state file
  exists; the type does not fit the (on-demand or spot) quota. It prints the estimated cost (worst case and
  planned), creates/reuses the `ffsim-gpu-ssh` security group with tcp/22 from your current public IP only,
  writes the user-data, waits 5 s, runs `run-instances` (64 GiB gp3 root, IMDSv2, shutdown-behavior=terminate,
  tags Name=ffsim-gpu / purpose / stamp), waits for `running`, and records id/ip/key/sg in
  `~/.ffsim-gpu/<region>.env`.
- `--instance-profile` is only needed for `--s3` (results sync from the box on halt); nothing else on the box
  calls AWS. `--spot` uses a one-time request with interruption behaviour terminate.
- `--auto-down`: the box powers off `--grace-min` minutes after `fleet.py` exits (a new `run` cancels the
  countdown). Use it with `--s3` or with `fetch --wait` running on your side, otherwise unfetched results die
  with the box.

## Step 3: push (code, data, tokenizer, shards)

```bash
bash ffsim/gpu/launch.sh push [--shards 5] [--train recipes/K60/train.py]
```

Waits for `~/ff/BOOTSTRAP_DONE` (the bootstrap takes 2-5 min after boot: systemd timers, torch/CUDA check,
`pip install pyarrow tiktoken requests`, unattended-upgrades disabled so apt cannot reboot mid-run), prints any
`BOOTSTRAP_WARN` lines, then sends by tar-over-ssh: `ffsim/` (package incl. `ffsim/gpu/` with `fleet.py`,
`train_gpu.py`, `configs/`), the submission `train.py` (`--train`, default K60; copied to `~/ff/train.py` and kept
at its submission path), the organizer's `prepare.py` (unmodified), `artifacts/tokenizer_8192.zip`,
`research/sim-data/{validation-pairs.json,models.pkl,search-*.csv}` and `research/sim-data/chipC/*/train.log`
(the Trainium schedules the gate arms replay).
On the box it runs the organizer's data step from the README, `python prepare.py setup --num-train-shards N`
(`NEURON_COMPETITION_R1_CACHE_DIR=~/ff/cache`, set in `~/ff/env.sh`; the tokenizer artifact is found at
`~/ff/artifacts/tokenizer_8192.zip` relative to `prepare.py`), py-compiles `ffsim/gpu/*.py` and prints `PUSH-OK`.
Five shards (~526 MB) are what the Trainium runs used; keep 5 so the token stream is identical.

## Step 4: run (prescreen, calibration gate, verdict sets)

```bash
bash ffsim/gpu/launch.sh run --config ffsim/gpu/configs/<set>.json [--results runs/<set>.jsonl] [-- <extra fleet.py args>]
bash ffsim/gpu/launch.sh status
```

`run` starts `python ffsim/gpu/fleet.py run --configs ... --results ...` under `nohup` on the box, writes
`~/ff/fleet.pid`, logs to `~/ff/logs/fleet-<stamp>.log`, and writes `~/ff/fleet.done` (the exit code) when it
ends. It refuses if a fleet is already running. Results are one JSON line per finished run in an append-only
file (`--results`, default `runs/<config>.jsonl`; fleet's own manifest sits next to it as
`<config>-manifest.jsonl`). **Resume is done by `launch.sh run`, not by `fleet.py`**: before starting it drops
every label that already has a scored row (`val_bpb` present or `status: ok`) in the results file, writes the
remainder to `runs/<config>-resume-<stamp>.json` and runs that; with nothing left it starts nothing. So an
interrupted set is resumed by running the same command again. `--no-resume` reruns every label (the old rows
stay; `fleet.py report` shows both). Extra `fleet.py run` arguments go after `--`, e.g.
`-- --step-seconds 0.30,0.56,0.93 --per-gpu 1`.

The order that the protocol requires, and what each set is for:

1. **prescreen** (short runs, a few GPU-hours): catches a broken setup at a tenth of the gate's cost; gate 0
   (exact eval token/byte count) and the same-seed null pair are as measurable here as at full length.
2. **calibration gate** (full equal-step runs of known Trainium pairs from `research/sim-data/validation-pairs.json`):
   the GPU proxy failed its 21 Sep sanity check (a U-net verdict flipped sign), so **no GPU verdict may
   influence a chip slot before this gate passes**. `fleet.py verdict` prints the six gates, the sign
   agreement against the Trainium pairs, the null noise floor and the minimum detectable effect; exit 0 = pass,
   1 = rejected (stop spending), 2 = incomplete.
3. **verdict sets** (new levers, >= 6 seed pairs each): only after 2 passes, only for the class of change the
   gate covered, only above the printed MDE.

`status` shows: instance state/ip/type/launch time; bootstrap marker and warnings; the two systemd timers and
their next fire time; whether fleet is running / its exit code; row counts of every `runs/*.jsonl`; GPU
utilisation and memory; disk; the last three log lines.

## Step 5: fetch

```bash
bash ffsim/gpu/launch.sh fetch [--wait] [--out research/sim-data/gpu/<date>]
```

`--wait` polls every 60 s until `fleet.done` exists. It copies `runs/*.jsonl` first (small, the deliverable),
then `logs/` and every `runs/<label>/` log and env file (checkpoints `*.pt` are excluded), `bootstrap.log`,
and writes `fetch-meta.txt` (instance, type, region, time, repo sha). Default target
`research/sim-data/gpu/<YYYYMMDD>/`; run the verdict locally from there. Fetch **before** `down` and before the
hard stop; the root volume dies with the instance.

## Step 6: down (destroys)

```bash
bash ffsim/gpu/launch.sh down --yes
```

Terminates the state-file instance and every non-terminated instance tagged `ffsim-gpu` in the region, waits,
deletes the `ffsim-gpu-ssh` group (kept if still referenced; it costs nothing), clears the state file, then lists
leftover `available` volumes and still-running G instances (both should be empty). Idempotent: with nothing to
terminate it exits 0. Confirm the bill is closed with `plan` (its "existing ffsim-gpu instances" line is empty)
or in the console.

## Failure modes

| symptom | cause | action |
|---|---|---|
| `CUDA out of memory` in the run log | micro-batch too large for the GPU (24 GB A10G, or a wide candidate on the L40S) | lower `FF_MB` in the config **keeping rows per optimizer step identical** (`FF_MB x accumulations` per step unchanged; on Trainium a step is 8 ranks x 4 accumulations x 8 rows); equal-step comparisons are only valid across identical row sets, and fleet.py refuses to difference rows whose context differs |
| `BOOTSTRAP_WARN: nvidia-smi failed` / `torch import ... failed` | driver/CUDA mismatch on the AMI (a newer DLAMI, or a PyTorch build that does not match the driver) | `ssh` in, `nvidia-smi`, `cat ~/ff/python_path`; pin a known AMI with `up --ami ami-...` (the 29 Sep id above) or use `/opt/pytorch/bin/python`; never `apt upgrade` the driver on a rented hour |
| spot instance reclaimed (state `terminated`, `status` shows none) | spot interruption (2-min warning, no checkpoint) | `up` again (on-demand if it recurs), `push`, then the same `run`: `results.jsonl` is append-only and finished labels are skipped, only the run in flight is lost; `fetch` between sets so the file is never only on the box |
| box gone before `fetch` | hard stop (`--max-hours`) or `--auto-down` fired | results were on the root volume; next time set `--max-hours` from the measured step time plus margin, use `fetch --wait`, or `--s3` with an instance profile so the halt script syncs `runs/` and `logs/` out first |
| `push` hangs at "waiting for the bootstrap marker" for > 5 min | box still booting, or ssh blocked | `status` (ssh line says "still booting, or the security group does not allow this IP"); if your public IP changed: `aws ec2 authorize-security-group-ingress --group-id <sg from the state file> --protocol tcp --port 22 --cidr <new-ip>/32` |
| `up` says `REFUSED: ... quota` | vCPU quota short (us-east-1 is 8; spot is 8 in us-west-2) | smaller type, on-demand instead of spot, or request an increase (Step 0) |
| `up` says an instance is already up | a previous box is still running (money) | `status`, then `fetch` and `down --yes`, or reuse it with `push`/`run` |
| `fleet: finished rc=1` | fleet.py failed (config error, missing data) | `ssh` in: `tail -50 ~/ff/logs/fleet-*.log`; `prepare.py setup` output is in the `push` log; re-run `push` if `~/ff/cache` is incomplete |
| `plan` prints `n/a` in the price columns | the pricing API (us-east-1 endpoint) is not permitted for the profile, or offline | prices in the table above are the 29 Sep values; nothing else depends on them |
| `aws: error ... file://` on `up` | Git Bash path conversion | the launcher already passes `cygpath -m` paths and `MSYS_NO_PATHCONV=1`; set `TMPDIR` to a POSIX path (`/c/tmp`) if it is a Windows one |
| the daily apt upgrade | disabled by the bootstrap (it SIGTERMed Trainium runs at 06:00-07:00 UTC) | nothing to do; reboots do not happen |

## Reading the outputs

- `runs/<set>.jsonl`: one row per finished run (`label`, `arm`, `seed`, `steps`, `val_bpb`, eval token and byte
  counts, context signature, timing). Rows are never rewritten; `fleet.py report` groups them by context and
  `fleet.py verdict` differences treatment-control pairs only inside one context.
- `fleet.py verdict` exit code: 0 pass, 1 rejected, 2 incomplete. A pass is scoped to the class of change the
  calibration pairs covered; extending it needs a new Trainium-verified anchor.
- `status` timers line: `ffsim-maxhours.timer: <next fire>` is the hard stop; `ffsim-autodown.timer` polls every
  minute and does nothing unless `--auto-down` was set and `fleet.done` is older than the grace period.
- The cost columns of `plan`: `box_h` = GPU-hours / GPUs of the type; `od_$` / `spot_$` = box_h x price;
  `quota(od)` = whether the type fits the on-demand quota with what is already running.
