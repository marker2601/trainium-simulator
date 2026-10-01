#!/usr/bin/env bash
# ffsim batch: fit the three models (when runs.jsonl exists) and run one search over a space json.
#
#   bash ffsim/cloud/run_batch.sh [SPACE_JSON] [GEN] [N_SIMS] [TOP] [WORKERS]
#
# Defaults: ffsim/examples/space-k59-local.json local 1000 20 <nproc>.
# Environment:
#   AWS_RESULTS_URI=s3://bucket/prefix   upload the search outputs (md/csv/json/summary, models.pkl) there
#   FFSIM_MODELS=path|stub               models.pkl to use (default research/sim-data/models.pkl);
#                                        'stub' = the knob-blind anchor stubs (pipeline check only)
#   FFSIM_SKIP_FIT=1                     use the existing models.pkl without refitting
# Outputs: research/sim-data/search-<space>-<UTC stamp>.{md,csv,json}, -summary.txt, -fit.txt.
# Needs python3 with numpy only; the aws CLI only when AWS_RESULTS_URI is set.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

SPACE="${1:-ffsim/examples/space-k59-local.json}"
GEN="${2:-local}"
N_SIMS="${3:-1000}"
TOP="${4:-20}"
WORKERS="${5:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2) )}"
# interpreter: $PYTHON, else the first of python3 / python that really runs (Windows ships a
# `python3` Store alias that only prints an install hint)
PY=""
for cand in "${PYTHON:-}" python3 python; do
  [ -n "$cand" ] || continue
  if "$cand" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || { echo "no working python3/python found (set PYTHON=...)" >&2; exit 2; }

NAME="$(basename "$SPACE" .json)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p research/sim-data
OUT="research/sim-data/search-${NAME}-${STAMP}"
MODELS="${FFSIM_MODELS:-research/sim-data/models.pkl}"

echo "ffsim run_batch: space=$SPACE gen=$GEN n_sims=$N_SIMS top=$TOP workers=$WORKERS models=$MODELS"
"$PY" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"

if [ "$MODELS" != "stub" ]; then
  if [ -f research/sim-data/runs.jsonl ] && [ "${FFSIM_SKIP_FIT:-0}" != "1" ]; then
    "$PY" -m ffsim fit --out "$MODELS" 2>&1 | tee "$OUT-fit.txt"
  elif [ ! -f "$MODELS" ]; then
    echo "WARNING: no research/sim-data/runs.jsonl and no $MODELS: using the knob-blind anchor stubs" >&2
    MODELS=stub
  fi
fi

"$PY" -m ffsim search --models "$MODELS" --space "$SPACE" --gen "$GEN" --n-sims "$N_SIMS" \
  --top "$TOP" --workers "$WORKERS" --out "$OUT.md" 2>&1 | tee "$OUT-summary.txt"

if [ -n "${AWS_RESULTS_URI:-}" ]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI missing: installing awscli into the current python" >&2
    "$PY" -m pip install --quiet awscli
  fi
  DEST="${AWS_RESULTS_URI%/}"
  for f in "$OUT.md" "$OUT.csv" "$OUT.json" "$OUT-summary.txt" "$OUT-fit.txt"; do
    [ -f "$f" ] && aws s3 cp "$f" "$DEST/$(basename "$f")"
  done
  if [ "$MODELS" != "stub" ] && [ -f "$MODELS" ]; then
    aws s3 cp "$MODELS" "$DEST/models.pkl"
  fi
  echo "results uploaded to $DEST/"
fi
echo "done: $OUT.{md,csv,json}"
