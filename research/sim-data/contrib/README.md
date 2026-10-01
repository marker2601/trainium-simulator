# Contributed runs

`runs-contrib.jsonl` holds the community's merged run records, one JSON object per line. Each line is a
`RunRecord` (`ffsim/schema.py`), the format the fitter reads, plus a `contrib` key with the original submission
(`contrib/schema.json`), its content hash, the checker's flags and the model's prediction at ingest time
(`RunRecord.from_dict` ignores that key).

Lines are appended only from the pull request the contribution workflow opens after a maintainer approves the
issue (or by `python -m ffsim contrib ingest`). Do not edit lines by hand; to withdraw a record, delete its whole
line. Every reader skips a line it cannot use (unreadable, a repeated content hash, non-finite or implausible
numbers) and says so on stderr. A push to `main` that touches this directory runs the guarded refit
(`.github/workflows/refit.yml`), which opens a pull request with the refreshed model and statistics.

Data in this directory is published under CC-BY-4.0. How to contribute: `CONTRIBUTING.md`.
