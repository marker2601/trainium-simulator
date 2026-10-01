# ffsim in the cloud (numpy-only box)

The surrogate search is pure numpy: scoring a candidate with 1,000 Monte Carlo draws costs a fraction
of a millisecond, so a thousand-candidate search finishes in seconds on a laptop. The cloud path exists
for two reasons only: to run many spaces unattended while the laptop is busy, and to keep a copy of
`research/sim-data/` plus the results in S3. **No GPU is involved.**

## What is NOT here

- **GPU proxy runs (route 2 of `docs/simulator-spec.md`) are not provided.** The 21 Sep
  GPU proxy failed validation (the U-net sign flipped between GPU and Trainium); a GPU calibration
  campaign is out of scope. Everything below evaluates the *statistical surrogate* (route 1) only.
- Nothing here touches the Trainium chips. Harvesting run logs off chips C/D is `python -m ffsim harvest`
  and runs from the laptop.

## Cost

| item | price | a thousand-candidate search |
|---|---|---|
| t3.large on-demand (us-east-1) | ~$0.083 / h | boot + pip + fit + search ~ 5-10 min: **~$0.01** |
| t3.large spot | ~$0.03 / h | **< $0.01** |
| S3 (repo tarball ~15 MB + results < 1 MB) | negligible | < $0.001 |

Even a whole hour on-demand is < $0.10. The `--max-minutes` hard stop (default 60) bounds the worst case;
the instance uses `instance-initiated-shutdown-behavior=terminate`, so `shutdown -h` terminates it and the
root volume is deleted.

## Files

| file | what |
|---|---|
| `Dockerfile` | `python:3.11-slim` + numpy; copies `ffsim/` and `research/sim-data/`; entrypoint `run_batch.sh` |
| `run_batch.sh` | `fit` (when `runs.jsonl` exists) then `search`; uploads results when `AWS_RESULTS_URI` is set |
| `ec2_launch.sh` | prints (and with `--yes` runs) the exact commands to launch a self-terminating t3.large |

## Local Docker

```bash
# from the repo root; research/sim-data/ must exist (runs.jsonl / models.pkl, or empty for the stubs)
docker build -f ffsim/cloud/Dockerfile -t ffsim .
docker run --rm -v "$PWD/research/sim-data:/work/research/sim-data" ffsim \
    ffsim/examples/space-k59-local.json local 1000 20 4      # space gen n_sims top workers
```

Results: `research/sim-data/search-<space>-<stamp>.{md,csv,json}` (+ `-summary.txt`, `-fit.txt`).
Without a fitted `models.pkl` or `runs.jsonl` the batch falls back to the **knob-blind anchor stubs**,
which only prove the pipeline runs: every candidate then scores the same.

## EC2 (one shot, self-terminating)

```bash
# dry run: read-only checks + the exact commands, nothing launched
bash ffsim/cloud/ec2_launch.sh --profile "$AWS_PROFILE" --region us-east-1 --bucket MY-BUCKET \
    --instance-profile ffsim-s3-writer --spot --space ffsim/examples/space-k59-local.json --gen local

# execute
bash ffsim/cloud/ec2_launch.sh ... --yes
```

What `--yes` does, in order: `tar` the repo (`ffsim/` + `research/sim-data/`), `aws s3 cp` it to
`s3://BUCKET/ffsim/<stamp>/repo.tgz`, `aws s3 presign` a 2-hour download URL (so the instance needs no IAM
role to fetch the code), `aws ec2 run-instances` with a user-data script that installs numpy in a venv,
downloads the tarball, runs `run_batch.sh`, uploads the results and shuts down.

Requirements on the `--profile`: `sts:GetCallerIdentity`, `s3:PutObject`/`GetObject` on the bucket,
`ec2:RunInstances` (+ `ec2:DescribeInstanceTypeOfferings`, `ec2:CreateTags`), and `iam:PassRole` for
`--instance-profile`. The instance profile needs `s3:PutObject` on `s3://BUCKET/ffsim/*`. Without
`--instance-profile` the search still runs, but the results exist only in the console output
(`aws ec2 get-console-output --instance-id ... --latest`), which is fine for a quick look at the top-20.

The box has no inbound ports unless you pass `--key-name` and a security group. It uses the account's
default VPC/subnet unless `--subnet-id` / `--security-group-id` are given; it needs outbound internet
(apt, PyPI, S3).

Fetch results:

```bash
aws --profile P --region R s3 sync s3://MY-BUCKET/ffsim/<stamp>/results/ research/sim-data/cloud-<stamp>/
```

## Trust boundary (same as `ffsim/README.md`)

The cloud box runs the same surrogate as the laptop. It cannot see knobs that never varied in the
fitted runs, its step-time model knows only the code versions it was fitted on, and its output is a
ranked list of **what to confirm on Trainium next** (seeds 73, 58, 67), not a score.
