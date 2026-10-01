#!/usr/bin/env bash
# Launch one small EC2 box that runs ffsim/cloud/run_batch.sh and terminates itself.
#
#   bash ffsim/cloud/ec2_launch.sh --profile PROFILE --region REGION --bucket BUCKET [options] [--yes]
#
# Without --yes this script only READS (sts get-caller-identity, ec2 describe-*) and PRINTS the exact
# commands it would run. With --yes it runs them: tar the repo, upload it to S3, presign a download
# URL, launch the instance with a self-terminating user-data script.
#
# Options:
#   --instance-type t3.large      (default) 2 vCPU / 8 GB: a thousand-candidate search takes seconds
#   --spot                        request a spot instance (about a third of the on-demand price)
#   --instance-profile NAME       IAM instance profile with s3:PutObject on the bucket: results are then
#                                 uploaded to s3://BUCKET/ffsim/<stamp>/results/. Without it results stay
#                                 in the instance console output only (aws ec2 get-console-output).
#   --space PATH                  space json inside the repo (default ffsim/examples/space-k59-local.json)
#   --gen local|oat|random|grid   (default local)     --n-sims N (default 1000)   --top K (default 20)
#   --workers N                   (default 2)
#   --subnet-id ID --security-group-id ID --key-name NAME   optional networking / SSH
#   --ami ID                      default: the latest Canonical Ubuntu 24.04 amd64 AMI via the public SSM parameter
#   --max-minutes M               hard self-terminate after M minutes even if the batch hangs (default 60)
#   --stamp S                     S3 key prefix stamp (default UTC time)
#   --yes                         actually run the commands
#   TMPDIR (env)                  where the repo tarball and user-data script are written (default /tmp;
#                                 POSIX form under Git Bash, e.g. /c/tmp)
set -euo pipefail

PROFILE=""; REGION=""; BUCKET=""; INSTANCE_TYPE="t3.large"; SPOT=0; INSTANCE_PROFILE=""
SPACE="ffsim/examples/space-k59-local.json"; GEN="local"; N_SIMS=1000; TOP=20; WORKERS=2
SUBNET=""; SG=""; KEY=""; AMI=""; MAX_MIN=60; STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2;;
    --spot) SPOT=1; shift;;
    --instance-profile) INSTANCE_PROFILE="$2"; shift 2;;
    --space) SPACE="$2"; shift 2;;
    --gen) GEN="$2"; shift 2;;
    --n-sims) N_SIMS="$2"; shift 2;;
    --top) TOP="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --subnet-id) SUBNET="$2"; shift 2;;
    --security-group-id) SG="$2"; shift 2;;
    --key-name) KEY="$2"; shift 2;;
    --ami) AMI="$2"; shift 2;;
    --max-minutes) MAX_MIN="$2"; shift 2;;
    --stamp) STAMP="$2"; shift 2;;
    --yes) YES=1; shift;;
    -h|--help) sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0;;   # the comment block above
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$PROFILE" ] && [ -n "$REGION" ] && [ -n "$BUCKET" ] || { echo "need --profile --region --bucket" >&2; exit 2; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
[ -f "$REPO/$SPACE" ] || { echo "space not found in the repo: $SPACE" >&2; exit 2; }

AWS=(aws --profile "$PROFILE" --region "$REGION")
if [ -z "$AMI" ]; then
  AMI="resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
fi
S3_PREFIX="s3://$BUCKET/ffsim/$STAMP"
TMP="${TMPDIR:-/tmp}"                                   # POSIX form (under Git Bash: /tmp or /c/..., never C:/...)
TARBALL="$TMP/ffsim-repo-$STAMP.tgz"
USERDATA="$TMP/ffsim-userdata-$STAMP.sh"
# The user-data path as the aws CLI must see it. Under Git Bash on Windows aws is a native exe: MSYS rewrites
# bare POSIX path arguments for it ("/tmp/x" -> "C:/<user-temp>/x") but leaves URL-shaped ones alone, so
# "file:///tmp/x" would make aws open a non-existent C:\tmp\x. cygpath -m gives "C:/<user-temp>/x"; where
# cygpath does not exist (Linux/macOS) the POSIX path is used as is.
host_path() { cygpath -m "$1" 2>/dev/null || printf '%s\n' "$1"; }
USERDATA_HOST="$(host_path "$USERDATA")"
RESULTS_URI=""
[ -n "$INSTANCE_PROFILE" ] && RESULTS_URI="$S3_PREFIX/results"

echo "== read-only checks (profile $PROFILE, region $REGION)"
"${AWS[@]}" sts get-caller-identity --output text || { echo "credentials for --profile $PROFILE do not work" >&2; exit 2; }
"${AWS[@]}" ec2 describe-instance-type-offerings --location-type region \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" --query 'InstanceTypeOfferings[0].InstanceType' --output text \
  | grep -q "$INSTANCE_TYPE" || echo "WARNING: $INSTANCE_TYPE not offered in $REGION"
if [ -n "$INSTANCE_PROFILE" ]; then
  "${AWS[@]}" iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --query 'InstanceProfile.Arn' --output text \
    || echo "WARNING: instance profile $INSTANCE_PROFILE not found (results will not be uploaded)"
fi
"${AWS[@]}" s3 ls "s3://$BUCKET" >/dev/null 2>&1 && echo "bucket s3://$BUCKET reachable" || echo "WARNING: cannot list s3://$BUCKET (upload may still work with put-only rights)"

# ---- the user-data script the instance runs on first boot -------------------------------------------
write_userdata() {  # $1 = presigned repo URL
  cat > "$USERDATA" <<EOF
#!/bin/bash
set -uxo pipefail
exec > >(tee /var/log/ffsim-user-data.log | logger -t ffsim) 2>&1
shutdown -h +$MAX_MIN "ffsim hard stop"          # terminates: instance-initiated-shutdown-behavior=terminate
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y python3-venv curl
python3 -m venv /opt/venv && /opt/venv/bin/pip install --quiet "numpy>=2,<3" $( [ -n "$RESULTS_URI" ] && echo awscli )
export PATH=/opt/venv/bin:\$PATH PYTHON=/opt/venv/bin/python AWS_DEFAULT_REGION=$REGION
mkdir -p /opt/ffsim && cd /opt/ffsim
curl -fsSL "$1" -o repo.tgz && tar xzf repo.tgz
export AWS_RESULTS_URI="$RESULTS_URI"
bash ffsim/cloud/run_batch.sh "$SPACE" "$GEN" "$N_SIMS" "$TOP" "$WORKERS"
echo "FFSIM-BATCH-DONE rc=\$?"
[ -n "$RESULTS_URI" ] && aws s3 cp /var/log/ffsim-user-data.log "$RESULTS_URI/user-data.log" || true
shutdown -h now
EOF
}

RUN_ARGS=(ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --count 1
  --instance-initiated-shutdown-behavior terminate
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ffsim-$STAMP},{Key=ffsim,Value=$STAMP}]"
  --user-data "file://$USERDATA_HOST")
[ "$SPOT" = 1 ] && RUN_ARGS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
[ -n "$INSTANCE_PROFILE" ] && RUN_ARGS+=(--iam-instance-profile "Name=$INSTANCE_PROFILE")
[ -n "$SUBNET" ] && RUN_ARGS+=(--subnet-id "$SUBNET")
[ -n "$SG" ] && RUN_ARGS+=(--security-group-ids "$SG")
[ -n "$KEY" ] && RUN_ARGS+=(--key-name "$KEY")

echo
echo "== plan (stamp $STAMP)"
echo "1. tar -C \"$REPO\" -czf $TARBALL ffsim research/sim-data"
echo "2. ${AWS[*]} s3 cp $TARBALL $S3_PREFIX/repo.tgz"
echo "3. URL=\$(${AWS[*]} s3 presign $S3_PREFIX/repo.tgz --expires-in 7200)   # the box downloads without IAM"
echo "4. write $USERDATA (installs numpy in a venv, downloads the repo, runs run_batch.sh $SPACE $GEN $N_SIMS $TOP $WORKERS, uploads to ${RESULTS_URI:-<nowhere: no --instance-profile>}, shuts down)"
echo "5. ${AWS[*]} ${RUN_ARGS[*]}"
echo "then:  ${AWS[*]} ec2 describe-instances --filters Name=tag:ffsim,Values=$STAMP --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text"
[ -n "$RESULTS_URI" ] && echo "       ${AWS[*]} s3 ls $RESULTS_URI/     # then: aws s3 sync $RESULTS_URI/ research/sim-data/cloud-$STAMP/"
echo "       ${AWS[*]} ec2 get-console-output --instance-id <id> --latest --output text | grep -A40 'ffsim search'"
echo "cost: $INSTANCE_TYPE $( [ "$SPOT" = 1 ] && echo spot || echo on-demand ) for about 5-10 minutes: well under \$0.10; hard stop after $MAX_MIN minutes."

if [ "$YES" != 1 ]; then
  echo
  echo "dry run: nothing launched. Re-run with --yes to execute."
  exit 0
fi

echo
echo "== executing"
tar -C "$REPO" -czf "$TARBALL" ffsim research/sim-data
ls -l "$TARBALL"
"${AWS[@]}" s3 cp "$TARBALL" "$S3_PREFIX/repo.tgz"
URL="$("${AWS[@]}" s3 presign "$S3_PREFIX/repo.tgz" --expires-in 7200)"
write_userdata "$URL"
INSTANCE_ID="$("${AWS[@]}" "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)"
echo "launched $INSTANCE_ID ($INSTANCE_TYPE, $REGION). It terminates itself when the batch ends (hard stop $MAX_MIN min)."
echo "watch:   ${AWS[*]} ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[].Instances[].State.Name' --output text"
[ -n "$RESULTS_URI" ] && echo "results: ${AWS[*]} s3 ls $RESULTS_URI/"
echo "console: ${AWS[*]} ec2 get-console-output --instance-id $INSTANCE_ID --latest --output text"
