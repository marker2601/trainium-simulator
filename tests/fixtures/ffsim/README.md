# ffsim parser fixtures

`chipC/C28_cold_K51/train.log` and `eval.log` are REAL chip-C output (C28 cold rehearsal of K51,
26 Sep 2026): the SSM-capped head of train.log merged with the harvested tail (from step 1290),
with the OperatorEntry / torch.distributed / ShardIndexInjection noise lines removed except the
first two (they carry the W926 / W0926 date stamps). The `t_*`
files hold the real launch/eval epochs from the timing trailer.

`overrides`, `code.sha256` and `chipC/_meta/base.env` are SYNTHETIC (written to exercise the
override / base.env / code-dir paths); do not treat the sha or the base.env values as evidence.
`monitor-runs.csv` and `official-uploads.csv` are small synthetic tables in the monitor schema.
