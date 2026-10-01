# ffsim: numpy-only surrogate search box.
# Build from the REPO ROOT (the context must contain ffsim/ and research/sim-data/):
#   docker build -f ffsim/cloud/Dockerfile -t ffsim .
# Run a search (results land in the mounted research/sim-data/):
#   docker run --rm -v "$PWD/research/sim-data:/work/research/sim-data" ffsim \
#       ffsim/examples/space-k59-local.json local 1000 20 4
# Upload results too (needs AWS credentials in the environment or a mounted ~/.aws):
#   docker run --rm -e AWS_RESULTS_URI=s3://BUCKET/ffsim/results -e AWS_PROFILE=... -v ~/.aws:/root/.aws:ro ffsim
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "numpy>=2,<3"

WORKDIR /work
COPY ffsim/ /work/ffsim/
# research/sim-data/ holds runs.jsonl, validation-pairs.json, official-uploads.csv and/or models.pkl.
# It must exist in the build context (create it empty if you only want the knob-blind stubs).
COPY research/sim-data/ /work/research/sim-data/

ENTRYPOINT ["bash", "/work/ffsim/cloud/run_batch.sh"]
CMD ["ffsim/examples/space-k59-local.json", "local", "1000", "20"]
