#!/usr/bin/env bash
# ffsim/gpu/launch.sh -- one GPU box for the equal-step proxy: plan it, bring it up, push the
# code and data, run fleet.py on it, watch it, fetch the results, take it down.
#
#   bash ffsim/gpu/launch.sh plan   [--region R] [--type T] [--gpu-hours H] [--max-hours H] [--spot]
#   bash ffsim/gpu/launch.sh up     --yes --key KEYPAIR [--ssh-key ~/.ssh/KEYPAIR.pem] [--type T] [--spot]
#                                   [--max-hours H] [--auto-down] [--grace-min M] [--instance-profile NAME]
#                                   [--volume-gb G] [--subnet-id ID] [--s3 s3://bucket/prefix]
#   bash ffsim/gpu/launch.sh push   [--shards N] [--train recipes/<K>/train.py]   # code + data + prepare.py setup
#   bash ffsim/gpu/launch.sh run    --config ffsim/gpu/configs/X.json [--results runs/X.jsonl] [--no-resume] [-- fleet args]
#   bash ffsim/gpu/launch.sh status
#   bash ffsim/gpu/launch.sh fetch  [--out research/sim-data/gpu/<date>] [--wait]
#   bash ffsim/gpu/launch.sh down   --yes                       # terminate; idempotent
#   bash ffsim/gpu/launch.sh ssh
#
# Money and destruction: `up` is the only subcommand that spends and `down` the only one that
# destroys, and both refuse without --yes. Everything else is read-only on AWS (describe-*, get-*,
# pricing) or talks to the box over SSH. `plan` prints the exact run-instances command `up` would run.
#
# Defaults: --region us-west-2 (G quota 64 vCPU; us-east-1 has 8), --type g6e.xlarge (1x L40S 48 GB),
# --profile $AWS_PROFILE (else "default"), --max-hours 12 (hard poweroff on the box), --volume-gb 64 gp3.
# State (instance id, ip, key, sg) lives in $FFSIM_GPU_STATE_DIR (default ~/.ffsim-gpu)/<region>.env;
# every subcommand also finds the box by its tag Name=ffsim-gpu, so a lost state file is not fatal.
#
# Windows / Git Bash: aws.exe gets file:// paths through cygpath -m; MSYS path conversion is off for
# every aws call (SSM parameter names start with "/"); ssh timeouts use -o ConnectTimeout (no `timeout`
# binary); the user-data file goes to $TMPDIR (POSIX form, e.g. /c/tmp) or /tmp.
set -euo pipefail

CMD="${1:-}"; [ $# -gt 0 ] && shift
REGION="${FFSIM_GPU_REGION:-us-west-2}"
TYPE="${FFSIM_GPU_TYPE:-g6e.xlarge}"
PROFILE="${FFSIM_GPU_PROFILE:-${AWS_PROFILE:-default}}"
TAG="${FFSIM_GPU_TAG:-ffsim-gpu}"
KEY=""; SSH_KEY=""; INSTANCE_PROFILE=""; SPOT=0; VOLUME_GB=64; MAX_HOURS=12; AUTO_DOWN=0; GRACE_MIN=30
GPU_HOURS=24; YES=0; SHARDS=5; CONFIG=""; RESULTS=""; OUT=""; WAIT=0; SUBNET=""; S3=""; AMI=""; RESUME=1
TRAIN="${FFSIM_GPU_TRAIN:-recipes/K60/train.py}"   # the current best; shipped as ~/ff/train.py
REMOTE_USER=ubuntu
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --train) TRAIN="$2"; shift 2;;
    --no-resume) RESUME=0; shift;;
    --region) REGION="$2"; shift 2;;
    --type) TYPE="$2"; shift 2;;
    --profile) PROFILE="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    --ssh-key) SSH_KEY="$2"; shift 2;;
    --instance-profile) INSTANCE_PROFILE="$2"; shift 2;;
    --spot) SPOT=1; shift;;
    --volume-gb) VOLUME_GB="$2"; shift 2;;
    --max-hours) MAX_HOURS="$2"; shift 2;;
    --auto-down) AUTO_DOWN=1; shift;;
    --grace-min) GRACE_MIN="$2"; shift 2;;
    --gpu-hours) GPU_HOURS="$2"; shift 2;;
    --shards) SHARDS="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --results) RESULTS="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --wait) WAIT=1; shift;;
    --subnet-id) SUBNET="$2"; shift 2;;
    --s3) S3="$2"; shift 2;;
    --ami) AMI="$2"; shift 2;;
    --yes) YES=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STATE_DIR="${FFSIM_GPU_STATE_DIR:-$HOME/.ffsim-gpu}"
STATE="$STATE_DIR/$REGION.env"
TMP="${TMPDIR:-/tmp}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$STATE_DIR"

# ---- helpers -------------------------------------------------------------------------------------
# aws.exe on Windows prints CRLF; the tr strips the CR so ids can be reused as arguments (pipefail keeps aws's status)
aws_ro() { MSYS_NO_PATHCONV=1 AWS_PAGER="" aws --profile "$PROFILE" --region "$REGION" "$@" | tr -d '\r'; }
aws_pricing() { MSYS_NO_PATHCONV=1 AWS_PAGER="" aws --profile "$PROFILE" --region us-east-1 pricing "$@" | tr -d '\r'; }
# the two calls that spend or destroy go through here: they print themselves and need --yes
aws_rw() {
  echo "+ aws --profile $PROFILE --region $REGION $(shq "$@")" >&2
  [ "$YES" = 1 ] || { echo "REFUSED: --yes is required for a call that spends or destroys" >&2; exit 3; }
  MSYS_NO_PATHCONV=1 AWS_PAGER="" aws --profile "$PROFILE" --region "$REGION" "$@" | tr -d '\r'
}
host_path() { cygpath -m "$1" 2>/dev/null || printf '%s\n' "$1"; }
shq() {   # shell-quote a command line for display: plain words as is, anything else in single quotes
  local out="" a
  for a in "$@"; do
    case "$a" in *[!A-Za-z0-9_./:=,@+-]*|"") out="$out '${a//\'/\'\\\'\'}'";; *) out="$out $a";; esac
  done
  printf '%s\n' "${out# }"
}
die() { echo "$*" >&2; exit "${RC:-2}"; }
refuse() { RC=3 die "REFUSED: $*"; }

# instance-type facts (verified 29 Sep 2026 with ec2 describe-instance-types, us-west-2): type vcpu gpus gpu mem_gib
TYPE_TABLE="g6e.xlarge 4 1 L40S 48
g6e.2xlarge 8 1 L40S 48
g6e.12xlarge 48 4 L40S 48
g6e.48xlarge 192 8 L40S 48
g5.xlarge 4 1 A10G 24
g5.2xlarge 8 1 A10G 24
g5.12xlarge 48 4 A10G 24
g5.48xlarge 192 8 A10G 24"
type_fact() { awk -v t="$1" -v f="$2" '$1==t {print $f}' <<<"$TYPE_TABLE"; }
vcpus_of() { local v; v="$(type_fact "$1" 2)"; [ -n "$v" ] && echo "$v" || echo 0; }
gpus_of() { local v; v="$(type_fact "$1" 3)"; [ -n "$v" ] && echo "$v" || echo 1; }

# quota codes: L-DB2E81BA = Running On-Demand G and VT instances (vCPUs); L-3819A6DF = All G and VT Spot Instance Requests
quota_value() { aws_ro service-quotas get-service-quota --service-code ec2 --quota-code "$1" --query 'Quota.Value' --output text 2>/dev/null || echo "?"; }
used_g_vcpus() {
  aws_ro ec2 describe-instances --filters Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)||starts_with(InstanceType,`vt`)].[InstanceType,CpuOptions.CoreCount,CpuOptions.ThreadsPerCore]' \
    --output text 2>/dev/null | awk 'NF==3 {s+=$2*$3} END {print s+0}'
}
find_ami() {   # -> "NAME AMI-ID": newest Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04) via public SSM
  [ -n "$AMI" ] && { echo "--ami $AMI"; return; }
  aws_ro ssm get-parameters-by-path --path /aws/service/deeplearning/ami/x86_64/ --recursive \
    --query "Parameters[?contains(Name,'oss-nvidia-driver-gpu-pytorch') && contains(Name,'ubuntu-22.04') && ends_with(Name,'/latest/ami-id')].[Name,Value]" \
    --output text | sort -V | tail -1
}
ami_root_gb() { aws_ro ec2 describe-images --image-ids "$1" --query 'Images[0].BlockDeviceMappings[0].Ebs.VolumeSize' --output text 2>/dev/null || echo 0; }
od_price() {   # on-demand list $/h for TYPE in REGION (pricing API lives in us-east-1); "n/a" on failure
  local p
  p="$(aws_pricing get-products --service-code AmazonEC2 --filters \
        "Type=TERM_MATCH,Field=instanceType,Value=$1" "Type=TERM_MATCH,Field=regionCode,Value=$REGION" \
        Type=TERM_MATCH,Field=operatingSystem,Value=Linux Type=TERM_MATCH,Field=tenancy,Value=Shared \
        Type=TERM_MATCH,Field=preInstalledSw,Value=NA Type=TERM_MATCH,Field=capacitystatus,Value=Used \
        --query 'PriceList[0]' --output text 2>/dev/null | grep -o '"USD":"[0-9.]*"' | head -1 | cut -d'"' -f4)"
  [ -n "$p" ] && printf '%.4f\n' "$p" || echo "n/a"
}
spot_table() {   # lines "TYPE MIN_SPOT_PRICE" over the AZs of REGION, current prices
  aws_ro ec2 describe-spot-price-history --instance-types "$@" --product-descriptions "Linux/UNIX" \
    --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --query 'SpotPriceHistory[].[InstanceType,SpotPrice]' --output text 2>/dev/null \
    | awk 'NF==2 {if (!($1 in m) || $2+0 < m[$1]) m[$1]=$2+0} END {for (t in m) printf "%s %.4f\n", t, m[t]}'
}
my_ip() { [ -n "${FFSIM_GPU_MYIP:-}" ] && { echo "$FFSIM_GPU_MYIP"; return; }
  curl -s --max-time 10 https://checkip.amazonaws.com 2>/dev/null | tr -d ' \r\n' || true; }
find_instances() {   # lines "ID STATE IP TYPE LAUNCH" for every non-terminated instance tagged Name=$TAG
  aws_ro ec2 describe-instances --filters "Name=tag:Name,Values=$TAG" Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress,InstanceType,LaunchTime]' --output text 2>/dev/null | awk 'NF>=2'
}
load_state() { [ -f "$STATE" ] && . "$STATE" || true; : "${INSTANCE_ID:=}"; : "${IP:=}"; : "${SG_ID:=}"; : "${SSH_KEY:=}"; }
resolve_box() {   # sets INSTANCE_ID and IP from the state file, refreshed from the tag lookup
  load_state
  local line; line="$(find_instances | { grep -m1 running || true; })"
  if [ -n "$line" ]; then INSTANCE_ID="$(cut -f1 <<<"$line")"; IP="$(cut -f3 <<<"$line")"; fi
  [ -n "$INSTANCE_ID" ] && [ -n "$IP" ] && [ "$IP" != None ] || die "no running $TAG instance in $REGION (state: $STATE). Run 'up' first."
}
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4
          -o UserKnownHostsFile="$STATE_DIR/known_hosts" -o LogLevel=ERROR)
ssh_box() { local k=(); [ -n "$SSH_KEY" ] && k=(-i "$SSH_KEY"); ssh "${SSH_OPTS[@]}" ${k[@]+"${k[@]}"} "$REMOTE_USER@$IP" "$@"; }
scp_box() { local k=(); [ -n "$SSH_KEY" ] && k=(-i "$SSH_KEY"); scp "${SSH_OPTS[@]}" ${k[@]+"${k[@]}"} "$@"; }

render_userdata() {   # $1 = output path. bootstrap_gpu.sh with the FFSIM_* knobs prepended (its shebang dropped)
  { echo "#!/bin/bash"
    echo "FFSIM_MAX_HOURS=$MAX_HOURS"; echo "FFSIM_AUTO_DOWN=$AUTO_DOWN"; echo "FFSIM_GRACE_MIN=$GRACE_MIN"
    echo "FFSIM_S3=$S3"; echo "FFSIM_USER=$REMOTE_USER"
    sed '1{/^#!/d;}' "$HERE/bootstrap_gpu.sh"; } > "$1"
}
run_instances_args() {   # $1 = ami, $2 = root GiB, $3 = sg id (or placeholder), $4 = user-data host path
  RUN_ARGS=(ec2 run-instances --image-id "$1" --instance-type "$TYPE" --count 1
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$2,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=purpose,Value=ffsim-equal-step-gpu-proxy},{Key=stamp,Value=$STAMP}]"
    --instance-initiated-shutdown-behavior terminate
    --metadata-options HttpTokens=required,HttpEndpoint=enabled
    --security-group-ids "$3" --user-data "file://$4")
  [ -n "$KEY" ] && RUN_ARGS+=(--key-name "$KEY")
  [ "$SPOT" = 1 ] && RUN_ARGS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  [ -n "$INSTANCE_PROFILE" ] && RUN_ARGS+=(--iam-instance-profile "Name=$INSTANCE_PROFILE")
  [ -n "$SUBNET" ] && RUN_ARGS+=(--subnet-id "$SUBNET")
  return 0
}
cost_table() {   # for GPU_HOURS of GPU time: instance-hours = gpu-hours / gpus per box
  local quota="$1" used="$2" types="g6e.xlarge g6e.2xlarge g6e.12xlarge g6e.48xlarge g5.xlarge g5.2xlarge g5.12xlarge g5.48xlarge"
  local spots; spots="$(spot_table $types)"
  printf '%-13s %4s %4s %5s %9s %9s %8s %10s %10s  %s\n' type gpus vcpu gpu "od_\$/h" "spot_\$/h" "box_h" "od_\$" "spot_\$" "quota(od)"
  for t in $types; do
    local g v od sp fit
    g="$(gpus_of "$t")"; v="$(vcpus_of "$t")"; od="$(od_price "$t")"
    sp="$(awk -v t="$t" '$1==t {print $2}' <<<"$spots")"; [ -n "$sp" ] || sp="n/a"
    if [ "$quota" != "?" ] && awk -v q="$quota" -v u="$used" -v v="$v" 'BEGIN{exit !(u+v<=q)}'; then fit=ok; else fit="NO ($v > $quota-$used)"; fi
    awk -v t="$t" -v g="$g" -v v="$v" -v gpu="$(type_fact "$t" 4)/$(type_fact "$t" 5)G" -v od="$od" -v sp="$sp" -v h="$GPU_HOURS" -v fit="$fit" \
      'BEGIN{bh=h/g; odc=(od=="n/a")?"n/a":sprintf("%.2f",bh*od); spc=(sp=="n/a")?"n/a":sprintf("%.2f",bh*sp);
             printf "%-13s %4d %4d %5s %9s %9s %8.1f %10s %10s  %s\n", t, g, v, gpu, od, sp, bh, odc, spc, fit}'
  done
}

# ---- subcommands ---------------------------------------------------------------------------------
case "$CMD" in

plan)
  command -v aws >/dev/null 2>&1 || die "aws CLI not found"
  echo "== ffsim GPU proxy plan  (read-only; nothing is launched)  $STAMP"
  echo "profile $PROFILE   region $REGION   type $TYPE   tag $TAG   spot $SPOT   max-hours $MAX_HOURS   auto-down $AUTO_DOWN"
  ACCT="$(aws_ro sts get-caller-identity --query Account --output text 2>/dev/null || echo '?')"
  echo "account $ACCT"
  Q="$(quota_value L-DB2E81BA)"; QS="$(quota_value L-3819A6DF)"; USED="$(used_g_vcpus)"; NEED="$(vcpus_of "$TYPE")"
  echo
  echo "== quota (vCPUs): on-demand G/VT L-DB2E81BA = $Q, in use now = $USED, $TYPE needs $NEED; spot G/VT L-3819A6DF = $QS"
  echo "   request more (free): aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 96 --region $REGION"
  echo
  read -r AMI_NAME AMI_ID < <(find_ami) || true
  [ -n "${AMI_ID:-}" ] || { AMI_ID="?"; AMI_NAME="(SSM lookup failed)"; }
  ROOT_GB="$(ami_root_gb "$AMI_ID")"; [ "$ROOT_GB" -gt 0 ] 2>/dev/null || ROOT_GB=0
  [ "$VOLUME_GB" -ge "$ROOT_GB" ] 2>/dev/null || VOLUME_GB="$ROOT_GB"
  echo "== AMI: $AMI_ID  $AMI_NAME  (root snapshot $ROOT_GB GiB; root volume would be $VOLUME_GB GiB gp3)"
  echo
  echo "== cost for $GPU_HOURS GPU-hours in $REGION (on-demand list via the pricing API, spot = lowest AZ right now)"
  cost_table "$Q" "$USED"
  echo "   spot needs quota L-3819A6DF ($QS vCPU) and can be reclaimed mid-run; results.jsonl is append-only, so 'run' again."
  echo "   the hard stop (--max-hours $MAX_HOURS) bounds the worst case at $MAX_HOURS h x \$/h of the chosen type."
  echo
  IPME="$(my_ip)"; [ -n "$IPME" ] || IPME="<your-ip>"
  UD="$TMP/ffsim-gpu-userdata-$STAMP.sh"; render_userdata "$UD"
  echo "== user-data written to $UD ($(wc -l <"$UD") lines: max-hours $MAX_HOURS, auto-down $AUTO_DOWN, grace $GRACE_MIN min)"
  echo "== security group 'up' would ensure: ffsim-gpu-ssh in the default VPC, tcp/22 from $IPME/32 only"
  echo "== existing $TAG instances in $REGION (up refuses if any is pending/running):"
  find_instances | sed 's/^/   /' || true
  echo "== every other non-terminated G/VT instance in $REGION (any tag; stopped ones still bill their EBS volume):"
  aws_ro ec2 describe-instances --filters "Name=instance-type,Values=g*,vt*" Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,LaunchTime,Tags[?Key==`Name`].Value|[0],BlockDeviceMappings[0].Ebs.VolumeId]' \
    --output text 2>/dev/null | awk 'NF>=3' | sed 's/^/   /' || true
  run_instances_args "$AMI_ID" "$VOLUME_GB" "<sg-id-from-up>" "$(host_path "$UD")"
  echo "== the command 'up --yes --key KEYPAIR' would run:"
  echo "   aws --profile $PROFILE --region $REGION $(shq "${RUN_ARGS[@]}")"
  [ -n "$KEY" ] || echo "   (add --key KEYPAIR: an existing EC2 key pair in $REGION; --ssh-key gives its .pem for ssh)"
  echo
  echo "Nothing was launched. 'up --yes' spends; 'down --yes' destroys."
  ;;

up)
  command -v aws >/dev/null 2>&1 || die "aws CLI not found"
  [ "$YES" = 1 ] || refuse "up needs --yes (it spends money). Run 'plan' first."
  [ -n "$KEY" ] || refuse "up needs --key KEYPAIR (an existing EC2 key pair in $REGION)"
  [ -n "$SSH_KEY" ] || SSH_KEY="$HOME/.ssh/$KEY.pem"
  [ -f "$SSH_KEY" ] || echo "WARNING: private key $SSH_KEY not found; push/run/fetch will need --ssh-key" >&2
  RUNNING="$(find_instances | { grep -E 'pending|running' || true; })"
  [ -z "$RUNNING" ] || refuse "an instance tagged Name=$TAG is already up in $REGION:
$RUNNING
Use it (push/run/status) or 'down --yes' first."
  [ ! -f "$STATE" ] || refuse "$STATE exists (instance ${INSTANCE_ID:-?}). 'down --yes' clears it."
  NEED="$(vcpus_of "$TYPE")"
  if [ "$SPOT" = 1 ]; then Q="$(quota_value L-3819A6DF)"; QN="spot G/VT (L-3819A6DF)"; USED=0
  else Q="$(quota_value L-DB2E81BA)"; QN="on-demand G/VT (L-DB2E81BA)"; USED="$(used_g_vcpus)"; fi
  if [ "$Q" != "?" ] && ! awk -v q="$Q" -v u="$USED" -v v="$NEED" 'BEGIN{exit !(u+v<=q)}'; then
    refuse "$TYPE needs $NEED vCPU but $QN quota is $Q with $USED in use in $REGION"
  fi
  read -r AMI_NAME AMI_ID < <(find_ami) || true
  [ -n "${AMI_ID:-}" ] && [ "$AMI_ID" != "?" ] || die "AMI lookup failed"
  ROOT_GB="$(ami_root_gb "$AMI_ID")"; [ "$VOLUME_GB" -ge "$ROOT_GB" ] 2>/dev/null || VOLUME_GB="$ROOT_GB"
  OD="$(od_price "$TYPE")"; SP="$(spot_table "$TYPE" | awk -v t="$TYPE" '$1==t {print $2}')"; [ -n "$SP" ] || SP="n/a"
  echo "== up: $TYPE in $REGION, AMI $AMI_ID ($AMI_NAME), root $VOLUME_GB GiB gp3, key $KEY, spot $SPOT"
  echo "   price: on-demand \$$OD/h, spot now \$$SP/h; hard stop after $MAX_HOURS h"
  awk -v od="$OD" -v sp="$SP" -v h="$MAX_HOURS" -v g="$GPU_HOURS" -v spot="$SPOT" 'BEGIN{
     p=(spot==1 && sp!="n/a")?sp:od; if (p=="n/a") {print "   estimated cost: unknown (pricing API failed)"; exit}
     printf "   ESTIMATED COST: worst case %.2f USD (%s h x %.4f), planned %.2f USD for %s GPU-hours on this box\n", h*p, h, p, g*p, g}'
  IPME="$(my_ip)"; [ -n "$IPME" ] || die "cannot determine the caller's public IP (set FFSIM_GPU_MYIP=a.b.c.d)"
  VPC="$(aws_ro ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
  [ -n "$VPC" ] && [ "$VPC" != None ] || die "no default VPC in $REGION; pass --subnet-id and create the SG by hand"
  SG_ID="$(aws_ro ec2 describe-security-groups --filters Name=group-name,Values=ffsim-gpu-ssh "Name=vpc-id,Values=$VPC" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
  if [ -z "$SG_ID" ] || [ "$SG_ID" = None ]; then
    SG_ID="$(aws_ro ec2 create-security-group --group-name ffsim-gpu-ssh --description "ffsim gpu proxy: ssh from the operator only" --vpc-id "$VPC" --query GroupId --output text)"
  fi
  aws_ro ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$IPME/32" >/dev/null 2>&1 \
    || echo "   (ssh rule for $IPME/32 already present on $SG_ID)"
  echo "   security group $SG_ID: tcp/22 from $IPME/32"
  UD="$TMP/ffsim-gpu-userdata-$STAMP.sh"; render_userdata "$UD"
  run_instances_args "$AMI_ID" "$VOLUME_GB" "$SG_ID" "$(host_path "$UD")"
  echo "   launching in ${FFSIM_GPU_CONFIRM_SLEEP:-5} s (Ctrl-C to abort) ..."; sleep "${FFSIM_GPU_CONFIRM_SLEEP:-5}"
  INSTANCE_ID="$(aws_rw "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)"
  [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != None ] || die "run-instances returned no instance id"
  printf 'INSTANCE_ID=%s\nTYPE=%s\nKEY=%s\nSSH_KEY=%s\nSG_ID=%s\nLAUNCHED=%s\nSPOT=%s\nMAX_HOURS=%s\n' \
    "$INSTANCE_ID" "$TYPE" "$KEY" "$SSH_KEY" "$SG_ID" "$STAMP" "$SPOT" "$MAX_HOURS" > "$STATE"
  echo "   launched $INSTANCE_ID; waiting for running"
  aws_ro ec2 wait instance-running --instance-ids "$INSTANCE_ID"
  IP="$(aws_ro ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
  echo "IP=$IP" >> "$STATE"
  echo "== $INSTANCE_ID running at $IP (state $STATE). The box powers off by itself after $MAX_HOURS h."
  echo "next: bash ffsim/gpu/launch.sh push --region $REGION      # waits for the bootstrap, then code + data"
  ;;

push)
  resolve_box
  echo "== push to $INSTANCE_ID ($IP): waiting for the bootstrap marker (up to 30 min)"
  n=0
  until ssh_box "test -f ~/ff/BOOTSTRAP_DONE" 2>/dev/null; do
    n=$((n+1)); [ "$n" -le 90 ] || die "bootstrap not finished after 30 min: ssh in and read /var/log/ffsim-bootstrap.log"
    sleep 20
  done
  ssh_box 'cat ~/ff/BOOTSTRAP_WARN 2>/dev/null | sed "s/^/   bootstrap WARN: /"; true'
  [ -f "$REPO/$TRAIN" ] || die "submission train.py not found: $TRAIN (pass --train)"
  echo "== code + data (tar over ssh: ffsim/, $TRAIN, prepare.py, tokenizer, validation-pairs.json, models.pkl, search-*.csv, chipC train.logs)"
  ( cd "$REPO" && tar -czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='.state' \
      ffsim "$TRAIN" prepare.py artifacts/tokenizer_8192.zip research/sim-data/validation-pairs.json research/sim-data/models.pkl \
      $(ls research/sim-data/search-*.csv 2>/dev/null) $(ls research/sim-data/chipC/*/train.log 2>/dev/null) ) \
    | ssh_box "mkdir -p ~/ff && tar -xzf - -C ~/ff && cp ~/ff/$(printf '%q' "$TRAIN") ~/ff/train.py && echo '   train.py <- $TRAIN'"
  echo "== remote setup: deps, data shards ($SHARDS), tokenizer (NEURON_COMPETITION_R1_CACHE_DIR=~/ff/cache)"
  ssh_box "SHARDS=$SHARDS bash -s" <<'REMOTE'
set -euo pipefail
cd ~/ff && source ~/ff/env.sh && PY="$(cat ~/ff/python_path)"
echo "--- gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "--- torch"; "$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'devices',torch.cuda.device_count())"
echo "--- deps"; "$PY" -m pip install -q pyarrow tiktoken requests 2>&1 | tail -1 || true
echo "--- prepare.py setup --num-train-shards $SHARDS  (shards + tokenizer into $NEURON_COMPETITION_R1_CACHE_DIR)"
"$PY" prepare.py setup --num-train-shards "$SHARDS" 2>&1 | tail -5
echo "--- syntax check of the proxy"; for f in ffsim/gpu/*.py; do "$PY" -m py_compile "$f" && echo "ok $f"; done
echo "--- df"; df -h / | tail -1
echo "PUSH-OK"
REMOTE
  echo "next: bash ffsim/gpu/launch.sh run --config ffsim/gpu/configs/<set>.json --region $REGION"
  ;;

run)
  [ -n "$CONFIG" ] || die "run needs --config <path relative to ~/ff on the box, e.g. ffsim/gpu/configs/prescreen.json>"
  resolve_box
  [ -n "$RESULTS" ] || RESULTS="runs/$(basename "$CONFIG" .json).jsonl"
  FLEET="${FFSIM_GPU_FLEET:-ffsim/gpu/fleet.py}"
  EXTRA_Q=""; for a in ${EXTRA[@]+"${EXTRA[@]}"}; do EXTRA_Q="$EXTRA_Q $(printf '%q' "$a")"; done
  echo "== run on $INSTANCE_ID ($IP): $FLEET run --configs $CONFIG --results $RESULTS$EXTRA_Q"
  ssh_box "CONFIG=$(printf '%q' "$CONFIG") RESULTS=$(printf '%q' "$RESULTS") FLEET=$(printf '%q' "$FLEET") EXTRA=$(printf '%q' "$EXTRA_Q") STAMP=$STAMP RESUME=$RESUME bash -s" <<'REMOTE'
set -euo pipefail
cd ~/ff && source ~/ff/env.sh && PY="$(cat ~/ff/python_path)"
if [ -f fleet.pid ] && kill -0 "$(cat fleet.pid)" 2>/dev/null; then echo "REFUSED: fleet.py already running (pid $(cat fleet.pid)); wait or kill it"; exit 3; fi
[ -f "$CONFIG" ] || { echo "config not found on the box: $CONFIG (push first?)"; exit 2; }
mkdir -p runs logs; rm -f fleet.done
# resume: results.jsonl is append-only, so a label that already has a scored row is dropped from the config
if [ "$RESUME" = 1 ] && [ -f "$RESULTS" ]; then
  FILTERED="runs/$(basename "$CONFIG" .json)-resume-$STAMP.json"
  "$PY" - "$CONFIG" "$RESULTS" "$FILTERED" <<'PY' || exit 0
import json, sys
cfg_path, res_path, out_path = sys.argv[1:4]
cfg = json.load(open(cfg_path, encoding="utf-8"))
done = set()
for ln in open(res_path, encoding="utf-8"):
    ln = ln.strip()
    if not ln:
        continue
    try:
        row = json.loads(ln)
    except ValueError:
        continue
    if row.get("label") and (row.get("val_bpb") is not None or row.get("status") == "ok"):
        done.add(row["label"])
runs = cfg.get("runs", [])
keep = [r for r in runs if r.get("label") not in done]
skipped = [r.get("label") for r in runs if r.get("label") in done]
if skipped:
    print(f"resume: {len(skipped)} of {len(runs)} labels already scored in {res_path}, skipped: {', '.join(map(str, skipped))}")
if not keep:
    print("resume: every label already has a results row; nothing to run")
    sys.exit(1)
cfg["runs"] = keep
json.dump(cfg, open(out_path, "w", encoding="utf-8"), indent=1)
print(f"resume: running {len(keep)} labels from {out_path}")
PY
  [ -f "$FILTERED" ] && CONFIG="$FILTERED"
fi
LOG="logs/fleet-$STAMP.log"
nohup bash -c "$PY $FLEET run --configs $CONFIG --results $RESULTS $EXTRA; rc=\$?; echo \$rc > fleet.done; echo \"fleet exit \$rc\"" > "$LOG" 2>&1 < /dev/null &
echo $! > fleet.pid
sleep 2
echo "started pid $(cat fleet.pid), log ~/ff/$LOG, results ~/ff/$RESULTS (append-only: finished labels are skipped on a re-run)"
head -20 "$LOG" || true
REMOTE
  echo "watch: bash ffsim/gpu/launch.sh status --region $REGION     fetch when done: bash ffsim/gpu/launch.sh fetch --wait --region $REGION"
  ;;

status)
  load_state
  echo "== $TAG instances in $REGION (state file: ${INSTANCE_ID:-none}):"
  LINES="$(find_instances)"; [ -n "$LINES" ] && echo "$LINES" | sed 's/^/   /' || { echo "   none"; exit 0; }
  LINE="$(grep -m1 running <<<"$LINES" || true)"; [ -n "$LINE" ] || exit 0
  INSTANCE_ID="$(cut -f1 <<<"$LINE")"; IP="$(cut -f3 <<<"$LINE")"
  ssh_box 'bash -s' <<'REMOTE' || echo "   (ssh failed: still booting, or the security group does not allow this IP)"
cd ~/ff 2>/dev/null || { echo "   ~/ff missing: bootstrap not started"; exit 0; }
echo "   uptime: $(uptime -p 2>/dev/null)   bootstrap: $( [ -f BOOTSTRAP_DONE ] && echo "done $(cat BOOTSTRAP_DONE)" || echo pending )$( [ -s BOOTSTRAP_WARN ] && echo "  WARN: $(tr '\n' ';' <BOOTSTRAP_WARN)" )"
echo "   timers: $(systemctl list-timers --all --no-legend 2>/dev/null | grep -E 'ffsim-(maxhours|autodown)' | awk '{print $NF": "$1" "$2" "$3}' | tr '\n' ';')"
if [ -f fleet.pid ] && kill -0 "$(cat fleet.pid)" 2>/dev/null; then echo "   fleet: RUNNING pid $(cat fleet.pid)"; elif [ -f fleet.done ]; then echo "   fleet: finished rc=$(cat fleet.done) at $(date -u -r fleet.done +%FT%TZ)"; else echo "   fleet: not started"; fi
for f in runs/*.jsonl; do [ -f "$f" ] && echo "   $f: $(wc -l <"$f") rows"; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/   gpu /'
echo "   disk: $(df -h / | awk 'NR==2 {print $3" used of "$2" ("$5")"}')"
L="$(ls -t logs/fleet-*.log 2>/dev/null | head -1)"; [ -n "$L" ] && { echo "   log $L:"; tail -3 "$L" | sed 's/^/     /'; }
REMOTE
  ;;

fetch)
  resolve_box
  [ -n "$OUT" ] || OUT="$REPO/research/sim-data/gpu/$(date -u +%Y%m%d)"
  if [ "$WAIT" = 1 ]; then
    echo "== waiting for fleet.done on $INSTANCE_ID (poll 60 s; Ctrl-C keeps the box)"
    until ssh_box "test -f ~/ff/fleet.done || ! test -f ~/ff/fleet.pid" 2>/dev/null; do sleep 60; done
  fi
  mkdir -p "$OUT"
  echo "== fetch $INSTANCE_ID ($IP) -> $OUT"
  scp_box "$REMOTE_USER@$IP:~/ff/runs/*.jsonl" "$OUT/" 2>/dev/null || echo "   (no results jsonl yet)"
  ssh_box "cd ~/ff && tar -czf - --exclude='*.pt' --exclude='*.bin' --exclude='*/out/*' logs runs fleet.done fleet.pid BOOTSTRAP_WARN 2>/dev/null; true" \
    | tar -xzf - -C "$OUT" 2>/dev/null || echo "   (no logs yet)"
  ssh_box "cat /var/log/ffsim-bootstrap.log 2>/dev/null" > "$OUT/bootstrap.log" || true
  { echo "instance $INSTANCE_ID $TYPE $REGION"; echo "fetched $STAMP"; echo "repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
    ls -l "$OUT"/*.jsonl 2>/dev/null; } > "$OUT/fetch-meta.txt"
  echo "== in $OUT:"; ls "$OUT" | sed 's/^/   /'
  echo "checkpoints (*.pt) are not fetched. Next: verdict locally, then 'down --yes'."
  ;;

down)
  command -v aws >/dev/null 2>&1 || die "aws CLI not found"
  [ "$YES" = 1 ] || refuse "down needs --yes (it terminates the instance and its root volume)"
  load_state
  IDS="$( { [ -n "$INSTANCE_ID" ] && echo "$INSTANCE_ID"; find_instances | cut -f1; } | sort -u | tr '\n' ' ' | sed 's/ $//')"
  if [ -z "$IDS" ]; then echo "== nothing to terminate in $REGION (no $TAG instance, no state)"; rm -f "$STATE"; exit 0; fi
  echo "== terminating $IDS in $REGION (fetch first if you have not)"
  aws_rw ec2 terminate-instances --instance-ids $IDS --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text \
    || echo "   (terminate failed for some ids: already gone?)"
  aws_ro ec2 wait instance-terminated --instance-ids $IDS 2>/dev/null || true
  if [ -n "$SG_ID" ]; then aws_ro ec2 delete-security-group --group-id "$SG_ID" 2>/dev/null && echo "   deleted $SG_ID" || echo "   (security group $SG_ID kept: still referenced or already gone; it costs nothing)"; fi
  rm -f "$STATE"
  echo "== terminated. Leftover 'available' volumes in $REGION (should be none):"
  aws_ro ec2 describe-volumes --filters Name=status,Values=available --query 'Volumes[].[VolumeId,Size,CreateTime]' --output text | sed 's/^/   /' || true
  echo "== still-running G/VT instances in $REGION (should be none):"
  aws_ro ec2 describe-instances --filters Name=instance-state-name,Values=pending,running \
    --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)].[InstanceId,InstanceType]' --output text | sed 's/^/   /' || true
  ;;

ssh)
  resolve_box
  k=(); [ -n "$SSH_KEY" ] && k=(-i "$SSH_KEY")
  exec ssh "${SSH_OPTS[@]}" ${k[@]+"${k[@]}"} "$REMOTE_USER@$IP" ${EXTRA[@]+"${EXTRA[@]}"}
  ;;

-h|--help)
  sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0;;
"")
  sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2;;
*)
  echo "unknown subcommand: $CMD (plan|up|push|run|status|fetch|down|ssh)" >&2; exit 2;;
esac
