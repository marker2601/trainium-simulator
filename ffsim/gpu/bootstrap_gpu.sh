#!/bin/bash
# ffsim/gpu/bootstrap_gpu.sh -- first-boot (EC2 user-data) script for the ffsim GPU proxy box.
#
# launch.sh `up` sends this file as user-data with FFSIM_* assignments prepended; run standalone
# it uses the defaults below. It runs once, as root, on the Deep Learning OSS Nvidia Driver AMI
# (Ubuntu 22.04, PyTorch 2.x, CUDA). It never calls the AWS API: the auto-shutdown guard is
# `poweroff`, which the launcher turns into a terminate with
# --instance-initiated-shutdown-behavior terminate, so no IAM role is needed.
#
# What it sets up:
#   /home/ubuntu/ff/                 work dir (launch.sh push fills it)
#   /home/ubuntu/ff/env.sh           NEURON_COMPETITION_R1_CACHE_DIR, FFSIM_PY (the DLAMI torch python)
#   /home/ubuntu/ff/python_path      the interpreter launch.sh run/push use
#   ffsim-maxhours.timer             poweroff FFSIM_MAX_HOURS after boot, no matter what
#   ffsim-autodown.timer             (only when FFSIM_AUTO_DOWN=1) poweroff FFSIM_GRACE_MIN minutes after
#                                    fleet.py exits (launch.sh run writes ~/ff/fleet.done on exit; a new
#                                    `run` removes it and so cancels the countdown)
#   /home/ubuntu/ff/BOOTSTRAP_DONE   marker launch.sh push waits for (BOOTSTRAP_WARN lists soft failures)
#   /var/log/ffsim-bootstrap.log     everything this script printed
: "${FFSIM_MAX_HOURS:=12}"
: "${FFSIM_AUTO_DOWN:=0}"
: "${FFSIM_GRACE_MIN:=30}"
: "${FFSIM_S3:=}"
: "${FFSIM_USER:=ubuntu}"
set -uo pipefail
exec > >(tee -a /var/log/ffsim-bootstrap.log | logger -t ffsim-bootstrap) 2>&1
echo "ffsim bootstrap start $(date -u +%FT%TZ) max_hours=$FFSIM_MAX_HOURS auto_down=$FFSIM_AUTO_DOWN grace_min=$FFSIM_GRACE_MIN s3=${FFSIM_S3:-none}"

FF="/home/$FFSIM_USER/ff"
mkdir -p "$FF/runs" "$FF/logs" "$FF/artifacts" "$FF/cache" /opt/ffsim/bin
WARN="$FF/BOOTSTRAP_WARN"; : > "$WARN"

# ---- 1. the halt helper: sync results out if asked, then power off (= terminate) -----------------
cat > /opt/ffsim/bin/ffsim-halt.sh <<EOF
#!/bin/bash
# \$1 = reason. Best-effort S3 sync of results (needs an instance profile), then poweroff.
echo "ffsim-halt: \$(date -u +%FT%TZ) reason=\${1:-manual}" | tee -a /var/log/ffsim-bootstrap.log
if [ -n "$FFSIM_S3" ] && command -v aws >/dev/null 2>&1; then
  aws s3 sync "$FF/runs" "$FFSIM_S3/runs" --exclude '*.pt' --exclude '*.bin' --exclude '*/out/*' 2>&1 | tail -3
  aws s3 sync "$FF/logs" "$FFSIM_S3/logs" 2>&1 | tail -3
  aws s3 cp /var/log/ffsim-bootstrap.log "$FFSIM_S3/logs/bootstrap.log" 2>&1 | tail -1
fi
sync
/sbin/poweroff
EOF
chmod 0755 /opt/ffsim/bin/ffsim-halt.sh

# ---- 2. auto-down check: fleet.done older than the grace period -> halt ---------------------------
cat > /opt/ffsim/bin/ffsim-autodown-check.sh <<EOF
#!/bin/bash
[ "$FFSIM_AUTO_DOWN" = "1" ] || exit 0
DONE="$FF/fleet.done"
[ -f "\$DONE" ] || exit 0
age=\$(( \$(date +%s) - \$(stat -c %Y "\$DONE") ))
[ "\$age" -ge $(( FFSIM_GRACE_MIN * 60 )) ] || exit 0
/opt/ffsim/bin/ffsim-halt.sh "auto-down (fleet.done \${age}s old, rc=\$(cat "\$DONE"))"
EOF
chmod 0755 /opt/ffsim/bin/ffsim-autodown-check.sh

# ---- 3. systemd units ---------------------------------------------------------------------------
cat > /etc/systemd/system/ffsim-maxhours.service <<'EOF'
[Unit]
Description=ffsim GPU proxy hard stop (poweroff = terminate)
[Service]
Type=oneshot
ExecStart=/opt/ffsim/bin/ffsim-halt.sh max-hours
EOF
cat > /etc/systemd/system/ffsim-maxhours.timer <<EOF
[Unit]
Description=ffsim GPU proxy hard stop after ${FFSIM_MAX_HOURS} h
[Timer]
OnBootSec=${FFSIM_MAX_HOURS}h
AccuracySec=1min
[Install]
WantedBy=timers.target
EOF
cat > /etc/systemd/system/ffsim-autodown.service <<'EOF'
[Unit]
Description=ffsim GPU proxy auto-down check (fleet.py finished + grace period)
[Service]
Type=oneshot
ExecStart=/opt/ffsim/bin/ffsim-autodown-check.sh
EOF
cat > /etc/systemd/system/ffsim-autodown.timer <<'EOF'
[Unit]
Description=ffsim GPU proxy auto-down poll
[Timer]
OnBootSec=5min
OnUnitActiveSec=1min
AccuracySec=30s
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now ffsim-maxhours.timer ffsim-autodown.timer || echo "systemd timers: enable failed" >> "$WARN"
systemctl list-timers --all 2>/dev/null | grep -E 'ffsim|NEXT' || true

# ---- 4. the box is disposable: no apt reboots mid-run (the daily upgrade killed Trainium runs) ----
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
systemctl disable unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true

# ---- 5. python with torch+CUDA: the DLAMI ships it in /opt/pytorch (venv) or a conda env ----------
PY=""
for cand in /opt/pytorch/bin/python /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python; do
  [ -x "$cand" ] && { PY="$cand"; break; }
done
[ -n "$PY" ] || PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "no python found" >> "$WARN"; PY=python3; }
echo "$PY" > "$FF/python_path"
cat > "$FF/env.sh" <<EOF
# sourced by launch.sh push/run and by the owner: `source ~/ff/env.sh`
export NEURON_COMPETITION_R1_CACHE_DIR=$FF/cache
export FFSIM_PY=$PY
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export FFSIM_GPU_BOX=1
EOF

echo "=== nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "nvidia-smi failed (driver mismatch?)" >> "$WARN"
echo "=== torch"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count(), 'bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())" 2>&1 \
  || echo "torch import / cuda check failed" >> "$WARN"
echo "=== prepare.py deps (pyarrow tiktoken requests)"
"$PY" -m pip install -q pyarrow tiktoken requests 2>&1 | tail -2 || echo "pip install of prepare.py deps failed" >> "$WARN"
echo "=== disk"
df -h / | tail -1

chown -R "$FFSIM_USER:$FFSIM_USER" "$FF"
[ -s "$WARN" ] && { echo "bootstrap finished WITH WARNINGS:"; cat "$WARN"; } || echo "bootstrap finished clean"
date -u +%FT%TZ > "$FF/BOOTSTRAP_DONE"
chown "$FFSIM_USER:$FFSIM_USER" "$FF/BOOTSTRAP_DONE" "$WARN"
echo "ffsim bootstrap end $(date -u +%FT%TZ)"
