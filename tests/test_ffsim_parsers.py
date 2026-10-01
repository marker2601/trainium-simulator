"""Tests for ffsim.parse_logs / parse_csv / parse_monitor / dataset (numpy + stdlib only)."""
from __future__ import annotations

import json
import os

import pytest

from ffsim import dataset
from ffsim.parse_csv import load_experiments_csv, run_family, seed_from_name
from ffsim.parse_logs import (
    compute_step_time,
    is_run_dir,
    is_screen_name,
    parse_eval_log,
    parse_overrides,
    parse_run_dir,
    parse_train_log,
)
from ffsim.parse_monitor import load_monitor_csv, load_official_uploads, parse_knob_changes
from ffsim.schema import RunRecord, StepTime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures", "ffsim")
C28 = os.path.join(FIX, "chipC", "C28_cold_K51")
EXPERIMENTS_CSV = os.path.join(REPO, "research", "experiments.csv")
SIM_DATA_C = os.path.join(REPO, "research", "sim-data", "chipC")

MICRO = 65_536


# ----------------------------------------------------------------------------- helpers
def step_line(step, dt_exact, k, loss=5.0, charged=0, lrm=0.7, printed_dt=None):
    tok_s = int(round(k * MICRO / dt_exact))
    dt = dt_exact if printed_dt is None else printed_dt
    return (f"step {step:05d} | loss {loss:.4f} | lrm {lrm:.3f} | dt {dt:.2f}s | tok/s {tok_s:,} "
            f"| charged {charged}s | next_level 20")


def k59_style_log(explicit_k1_phase: bool, with_summary: bool = False, num_steps: int = 1_000_000) -> str:
    """Synthetic train.log for FF_ACCUM_SCHED=1:0.06,2:0.2,4 (k1 from the warm-up end)."""
    lines = [
        "device=neuron dtype=torch.bfloat16 world_size=8",
        f"schedule=time time_target=1793.0s num_steps={num_steps} cooldown_fraction=0.45 bucket_allreduce=False official_budget=1800s",
        "num_params=124,256,278 flops_per_token=7.581204e+08",
        "micro_tokens=65,536 grad_accum_steps=4",
        "FF_ACCUM_SCHED=1:0.06,2:0.2,4: 1 micro-batches per update (65,536 tokens) with LR x 0.5 from step 20 "
        "(end of warm-up) until 0.06 of charged time, 2 (131,072 tokens) with LR x 0.7071 until 0.2, 4 (262,144 tokens) otherwise",
        "FF_LEAKY_RELU2=0.35: leaky relu^2 slope",
        "FF_ACCUM_LR: 0.5,0.7071",
        "[W926 18:37:54.216084465 OperatorEntry.cpp:208] Warning: noise",
        "W0926 18:37:54.726000 334802 torch/distributed/run.py:851] noise",
        "ShardIndexInjection: conflicting shard indices {0, 1}, skipping injection",
        step_line(0, 500.0, 4, loss=15.4, charged=505),
    ]
    for s in range(1, 5):
        lines.append(step_line(s, 0.93, 4, loss=15.0, charged=505 + s))
    lines.append(step_line(10, 0.93, 4, loss=13.0, charged=10))
    if explicit_k1_phase:
        lines.append("FF_ACCUM_SCHED phase 1: 1 micro-batches per update (65,536 tokens) from step 20 (charged 20s)")
    # k1: steps 20..190; the two lines after the switch are slow, step 100 is a recompile spike
    for s in range(20, 200, 10):
        dt = 0.80 if s in (20, 30) else (1.50 if s == 100 else 0.3312)
        lines.append(step_line(s, dt, 1, loss=10.0 - s / 100, charged=20 + s // 3))
    p2 = 2 if explicit_k1_phase else 1
    lines.append(f"FF_ACCUM_SCHED phase {p2}: 2 micro-batches per update (131,072 tokens) from step 200 (charged 90s)")
    for s in range(200, 600, 10):
        dt = 1.10 if s in (200, 210) else 0.5666
        lines.append(step_line(s, dt, 2, loss=8.0 - s / 200, charged=90 + (s - 200) // 2))
    lines.append(f"FF_ACCUM_SCHED phase {p2 + 1}: 4 micro-batches per update (262,144 tokens) from step 600 (charged 300s)")
    for s in range(600, 1210, 10):
        dt = 2.00 if s in (600, 610) else 0.9280
        lines.append(step_line(s, dt, 4, loss=5.0 - s / 1000, charged=300 + (s - 600)))
    if with_summary:
        lines.append('ff_summary: {"steps": 1205, "seed": 73, "accum_sched": {"median_step_seconds_by_k": {"1": 0.3312, "2": 0.5666, "4": 0.928}, "steps_by_k": {"1": 180, "2": 400, "4": 625}}}')
    return "\n".join(lines) + "\n"


def write_run_dir(root, chip, run, train_log, overrides=None, eval_log=None, eval20_log=None, sha=None, base_env=None):
    d = root / f"chip{chip}" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.log").write_text(train_log, encoding="utf-8")
    if overrides is not None:
        (d / "overrides").write_text(overrides, encoding="utf-8")
    if eval_log is not None:
        (d / "eval.log").write_text(eval_log, encoding="utf-8")
    if eval20_log is not None:
        (d / "eval20.log").write_text(eval20_log, encoding="utf-8")
    if sha is not None:
        (d / "code.sha256").write_text(sha, encoding="utf-8")
    if base_env is not None:
        meta = root / f"chip{chip}" / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "base.env").write_text(base_env, encoding="utf-8")
    return str(d)


# ----------------------------------------------------------------------------- parse_logs: real C28 fixture
@pytest.fixture(scope="module")
def c28() -> RunRecord:
    return parse_run_dir(C28)


def test_c28_identity_and_sources(c28):
    assert (c28.run_id, c28.chip, c28.run) == ("C:C28_cold_K51", "C", "C28_cold_K51")
    assert c28.sources == ["chip_log"]
    assert c28.submission == "K51" and "rehearsal" in c28.tags
    assert c28.date_utc == "2026-09-26"          # from the real t_launch epoch
    assert c28.is_screen is False


def test_c28_steps_and_times(c28):
    assert c28.steps == 2036                      # ff_summary steps (last step line is 02035)
    assert c28.charged_seconds == 1782.0
    assert c28.startup_seconds == 447.05
    assert c28.time_target == 1788.0


def test_c28_step_time_prefers_log_summary(c28):
    st = c28.step_time
    assert st.k1 is None and st.n_k1 == 0
    assert (st.k2, st.n_k2) == (0.5667, 590)      # ff_summary median_step_seconds_by_k / steps_by_k
    assert (st.k4, st.n_k4) == (0.9925, 1446)
    assert "step_time_source=summary" in c28.notes and "lines_median_by_k=" in c28.notes
    with open(os.path.join(C28, "train.log"), encoding="utf-8") as fh:
        lines = parse_train_log(fh.read()).step_time_lines
    assert abs(lines.k2 - 0.5667) < 0.003 and abs(lines.k4 - 0.9925) < 0.003   # line-based medians agree
    assert lines.n_k2 == 57 and lines.n_k4 == 161  # 59 k2 lines - 2 post-switch; 168 k4 - 5 startup - 2 post-switch


def test_c28_loss_curve(c28):
    assert c28.loss_curve[0] == [0.0, 15.4792]
    assert c28.loss_curve[-1] == [2035.0, 2.6596]
    assert len(c28.loss_curve) == 227
    steps = [p[0] for p in c28.loss_curve]
    assert steps == sorted(steps)


def test_c28_eval_and_code(c28):
    assert c28.bpb_2m == pytest.approx(0.9678859569240843)
    assert c28.bpb_2m_ema is None and c28.bpb_20m is None
    assert c28.code_sha == "0f0e0d0c0b0a09080706050403020100ffeeddccbbaa99887766554433221100"
    assert c28.code_version == "code_k51"         # FF_CODE_DIR=/root/ff-claude/code_k51


def test_c28_knobs(c28):
    k = c28.knobs
    assert c28.seed == 67 and k["FF_SEED"] == "67"          # last FF_SEED in overrides wins over base.env 58
    assert k["FF_COOLDOWN_FRAC"] == "0.45"                    # from _meta/base.env
    assert k["FF_CC_FLAGS"] == "--model-type=transformer -O3 --auto-cast=matmult"
    assert k["FF_MTP"] == "1" and k["FF_ACCUM_SCHED"] == "2:0.2,4" and k["FF_MASK_BOS_TARGET"] == "1"
    assert c28.knobs_complete is True
    assert "echo_flags=FF_KEY_OFFSET,FF_MUON_SCHED,FF_MUON_VIEWS" in c28.notes


def test_c28_fixture_size_budget():
    assert os.path.getsize(os.path.join(C28, "train.log")) <= 60 * 1024


def test_c28_date_stamp_on_noise_lines():
    with open(os.path.join(C28, "train.log"), encoding="utf-8") as fh:
        log = parse_train_log(fh.read())
    assert log.date_mmdd == (9, 26)                  # from the real "[W926 18:37:54 ... OperatorEntry" line
    assert log.summary["steps"] == 2036 and log.summary["seed"] == 67
    assert log.config["num_steps"] == "1000000" and log.config["cooldown_fraction"] == "0.6"


# ----------------------------------------------------------------------------- parse_logs: synthetic
@pytest.mark.parametrize("explicit", [True, False])
def test_k1_k2_k4_step_time_exclusions(explicit):
    log = parse_train_log(k59_style_log(explicit_k1_phase=explicit))
    assert log.initial_k == 4 and log.micro_tokens == MICRO
    assert log.switches == [(20, 1), (200, 2), (600, 4)]
    st = log.step_time
    assert st.k1 == pytest.approx(0.3312, abs=2e-4)
    assert st.k2 == pytest.approx(0.5666, abs=2e-4)
    assert st.k4 == pytest.approx(0.9280, abs=2e-4)
    # 18 k1 lines - 2 post-switch - 1 recompile; 40 k2 - 2; 61 k4 - 2 + step 10 (steps 1-4 are startup)
    assert (st.n_k1, st.n_k2, st.n_k4) == (15, 38, 60)
    assert log.k_mismatches == 0
    assert log.steps == 1201 and log.startup_seconds == 500.0 and log.time_target == 1793.0
    assert log.knob_hints["FF_ACCUM_SCHED"] == "1:0.06,2:0.2,4"
    assert log.knob_hints["FF_LEAKY_RELU2"] == "0.35"
    assert "FF_ACCUM_LR" in log.echo_flags


def test_summary_steps_and_medians_win_over_lines():
    log = parse_train_log(k59_style_log(True, with_summary=True))
    assert log.steps == 1205 and log.summary["seed"] == 73
    assert log.step_time_source == "summary"
    st = log.step_time
    assert (st.k1, st.n_k1, st.k2, st.n_k2, st.k4, st.n_k4) == (0.3312, 180, 0.5666, 400, 0.928, 625)
    assert (log.step_time_lines.n_k1, log.step_time_lines.n_k2, log.step_time_lines.n_k4) == (15, 38, 60)
    nolog = parse_train_log(k59_style_log(True))
    assert nolog.step_time_source == "lines" and nolog.step_time == nolog.step_time_lines


def test_precise_dt_from_tok_s():
    # printed dt is 0.93 but tok/s encodes 0.9280 exactly
    line = step_line(700, 0.9280, 4, printed_dt=0.93)
    rows = parse_train_log("micro_tokens=65,536 grad_accum_steps=4\n" + line + "\n").rows
    assert rows[0]["dt"] == 0.93 and rows[0]["dt_precise"] == pytest.approx(0.9280, abs=1e-4)
    assert rows[0]["k"] == 4


def test_tok_s_overrides_wrong_phase_lines():
    # phase line claims k2 from step 100, but the tok/s column says the steps ran at k4
    text = "micro_tokens=65,536 grad_accum_steps=4\nFF_ACCUM_SCHED phase 1: 2 micro-batches per update (131,072 tokens) from step 100 (charged 50s)\n"
    text += "\n".join(step_line(s, 0.93, 4) for s in range(100, 400, 10)) + "\n"
    log = parse_train_log(text)
    assert log.k_mismatches == 30 and log.step_time.k2 is None and log.step_time.k4 == pytest.approx(0.93, abs=1e-3)


def test_compute_step_time_recompile_rule():
    rows = [{"step": s, "dt": 1.0, "dt_precise": 1.0, "k": 4} for s in range(10, 110, 10)]
    rows[5]["dt"] = rows[5]["dt_precise"] = 3.5      # > 3x median -> dropped
    rows[6]["dt"] = rows[6]["dt_precise"] = 2.9      # kept
    st = compute_step_time(rows, [])
    assert st.k4 == 1.0 and st.n_k4 == 9


def test_truncated_log_without_summary_or_eval(tmp_path):
    text = k59_style_log(True)
    text = text[: text.index("step 00600")]          # cut mid-run, like the SSM 24,000-char cap
    d = write_run_dir(tmp_path, "D", "D7_lk35_s58", text)
    rec = parse_run_dir(d)
    assert rec.chip == "D" and rec.run_id == "D:D7_lk35_s58"
    assert rec.steps == 591 and rec.bpb_2m is None and rec.code_sha is None
    assert rec.step_time.k2 == pytest.approx(0.5666, abs=2e-4) and rec.step_time.n_k2 == 38
    assert rec.step_time.k4 == pytest.approx(0.93, abs=1e-3) and rec.step_time.n_k4 == 1   # only warm-up step 10 ran at k4
    assert rec.code_version == "code" and rec.knobs_complete is False
    assert "no overrides file" in rec.notes
    assert rec.date_utc == "2026-09-26" or rec.date_utc.endswith("-09-26")   # W0926 stamp, year from mtime


def test_overrides_base_env_seed_and_code_dir(tmp_path):
    ov = "FF_SEED=58\nexport FF_SEED=67\nFF_CODE_DIR=/root/ff-claude/code_m9/\nFF_CC_FLAGS=--model-type=transformer -O3 --auto-cast=matmult\n# comment\n"
    be = "FF_COOLDOWN_FRAC=0.45\nFF_SEED=51\nFF_TIME_TARGET=1793\n"
    d = write_run_dir(tmp_path, "C", "C41_lk35_s67", k59_style_log(True), overrides=ov, base_env=be,
                      sha="abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789  train_ff.py\n")
    rec = parse_run_dir(d)
    assert rec.seed == 67 and rec.code_version == "code_m9"
    assert rec.knobs["FF_CC_FLAGS"] == "--model-type=transformer -O3 --auto-cast=matmult"
    assert rec.knobs["FF_COOLDOWN_FRAC"] == "0.45" and rec.knobs["FF_TIME_TARGET"] == "1793"
    assert rec.knobs_complete is True
    assert rec.code_sha.startswith("abcdef0123456789")
    assert rec.time_target == 1793.0


def test_non_run_dirs_give_none(tmp_path):
    (tmp_path / "chipC" / "_meta").mkdir(parents=True)
    (tmp_path / "chipC" / "_meta" / "base.env").write_text("FF_SEED=1\n", encoding="utf-8")
    (tmp_path / "chipC" / "_transfer").mkdir()
    (tmp_path / "chipC" / "empty_run").mkdir()
    assert parse_run_dir(str(tmp_path / "chipC" / "_meta")) is None
    assert parse_run_dir(str(tmp_path / "chipC" / "_transfer")) is None
    assert parse_run_dir(str(tmp_path / "chipC" / "empty_run")) is None
    assert parse_run_dir(str(FIX) + "/chipC/_meta") is None
    assert is_run_dir(C28)


def test_cli_entry_point_names():
    import ffsim.parse_csv as pc
    import ffsim.parse_monitor as pm
    assert pc.parse_experiments_csv is pc.load_experiments_csv
    assert pm.parse_monitor_csv is pm.load_monitor_csv
    flat = dataset.merge_runs(load_monitor_csv(os.path.join(FIX, "monitor-runs.csv")) + [None])   # cli passes a flat list
    assert len(flat) == 5


def test_parse_overrides_one_line_form():
    ov = parse_overrides("FF_SEED=58 NEURON_COMPETITION_R1_NUM_STEPS=120 FF_PROFILE=1 FF_CC_FLAGS=--model-type=transformer -O3 --auto-cast=matmult")
    assert ov == {"FF_SEED": "58", "NEURON_COMPETITION_R1_NUM_STEPS": "120", "FF_PROFILE": "1",
                  "FF_CC_FLAGS": "--model-type=transformer -O3 --auto-cast=matmult"}
    assert parse_overrides('FF_X="a b"\nFF_Y=1') == {"FF_X": "a b", "FF_Y": "1"}


def test_eval_logs_plain_ema_and_20m(tmp_path):
    ev = "eval: trained weights\n{\n  \"val_bpb\": 0.96092\n}\nEMA weights val_bpb 0.96050\n"
    ev20 = "num_tokens 20971520 val_bpb=0.96792\n"
    d = write_run_dir(tmp_path, "C", "C40_cold_K59", k59_style_log(True), overrides="FF_SEED=73\n", eval_log=ev, eval20_log=ev20)
    rec = parse_run_dir(d)
    assert rec.bpb_2m == 0.96092 and rec.bpb_2m_ema == 0.96050 and rec.bpb_20m == 0.96792
    assert rec.submission == "K59"
    assert parse_eval_log("no numbers here") == (None, None)


def test_screen_detection(tmp_path):
    d = write_run_dir(tmp_path, "C", "m13_qkv_probe", k59_style_log(True, num_steps=60), overrides="FF_SEED=73\n")
    assert parse_run_dir(d).is_screen is True                       # num_steps=60
    d = write_run_dir(tmp_path, "C", "C42_lk35_s58", k59_style_log(True), overrides="FF_SEED=58\nFF_STEP_TOTAL=120\n")
    assert parse_run_dir(d).is_screen is True                       # FF_STEP_TOTAL
    d = write_run_dir(tmp_path, "C", "C43_lk35_s58", k59_style_log(True), overrides="FF_SEED=58\n")
    assert parse_run_dir(d).is_screen is False
    d = write_run_dir(tmp_path, "C", "C44_lk35_s60", k59_style_log(True), overrides="FF_SEED=60\n")
    assert parse_run_dir(d).is_screen is False                      # `_s60` is the seed suffix, not the S60 marker
    short = "\n".join(step_line(s, 0.93, 4) for s in range(0, 60)) + "\n"
    d = write_run_dir(tmp_path, "D", "D9_lk35_s58", short)
    assert parse_run_dir(d).is_screen is True                       # steps <= 200
    assert is_screen_name("S60_m13_qkv") and is_screen_name("ff-driver-k59") and is_screen_name("S1_rope")
    assert is_screen_name("M13_S60_qkv") and is_screen_name("m13_qkv_probe")
    assert not is_screen_name("C28_cold_K51") and not is_screen_name("0315_C_s132_confirm") and not is_screen_name("0816_R_g7lk35_s73")
    # seed 60 (`_s60`) is a full run, not the S60 screen marker (F6)
    assert not is_screen_name("2147_D_s60") and not is_screen_name("C45_lk35_s60") and not is_screen_name("0315_C_s60_confirm")


# ----------------------------------------------------------------------------- parse_csv
@pytest.fixture(scope="module")
def experiments():
    return load_experiments_csv(EXPERIMENTS_CSV)


def test_experiments_csv_counts(experiments):
    assert len(experiments) == 348
    assert len({r.run_id for r in experiments}) == 348
    chips = {c: sum(1 for r in experiments if r.chip == c) for c in "AB"}
    assert chips == {"A": 211, "B": 137}
    assert sum(1 for r in experiments if r.is_screen) == 50
    assert sum(1 for r in experiments if r.in_window) == 184
    assert all(r.sources == ["experiments.csv"] and r.date_utc is None and not r.knobs_complete for r in experiments)
    assert all(r.steps is not None and r.bpb_2m is not None for r in experiments)
    assert sum(1 for r in experiments if r.seed is None) == 7      # rows without FF_SEED and without _s suffix


def test_experiments_csv_row_values(experiments):
    by = {r.run_id: r for r in experiments}
    r = by["A:0347_R_cd060_s132"]
    assert r.seed == 132 and r.steps == 1553 and r.bpb_2m == pytest.approx(0.9845605347817307)
    assert r.knobs == {"FF_COOLDOWN_FRAC": "0.60", "FF_SEED": "132"}
    assert r.in_window is True and r.is_screen is False
    assert "family:R" in r.tags and "era:R" in r.tags
    assert r.notes == "norm_bpb=0.983835"
    r = by["A:1906_P_cc_fp8"]                                       # column cut at the first space; overrides carry the full value
    assert r.knobs["FF_CC_FLAGS"] == "--model-type=transformer -O3 --auto-cast=matmult --auto-cast-type=fp8_e4m3"
    assert r.is_screen is True and r.knobs["NEURON_COMPETITION_R1_NUM_STEPS"] == "120" and r.steps == 119
    r = by["A:0646_G3_d12w640"]
    assert r.seed is None and r.knobs == {"FF_ASPECT_RATIO": "53"} and "family:G3" in r.tags and "era:G" in r.tags
    r = by["A:0315_C_s132_confirm"]
    assert r.seed == 132
    r = by["B:1515_H256ss_s52"]
    assert r.seed == 52 and r.in_window is True and r.knobs["FF_SCALAR_STACK"] == "1"
    assert run_family("0730_S1_stream") == "S1" and seed_from_name("1519_VE1s51_stream") is None


# ----------------------------------------------------------------------------- parse_monitor
def test_monitor_csv_fixture():
    recs = load_monitor_csv(os.path.join(FIX, "monitor-runs.csv"))
    assert len(recs) == 5
    by = {r.run_id: r for r in recs}
    r = by["C:C40_cold_K59"]
    assert r.sources == ["monitor"] and r.date_utc == "2026-09-29" and r.code_version == "M12"
    assert r.seed == 73 and r.steps == 2357 and r.bpb_2m == 0.96092 and r.bpb_20m == 0.96792 and r.bpb_2m_ema is None
    assert r.submission == "K59" and r.control_of == "C:C36_cold_K57"
    assert r.knobs == {"FF_LEAKY_RELU2": "0.35", "FF_SEED": "73"} and not r.knobs_complete
    assert "recipe:lk35" in r.tags and "control_bpb=0.96193" in r.notes and "delta_norm=-0.00098" in r.notes and "anchor rehearsal" in r.notes
    s = by["C:S60_m13_qkv"]
    assert s.is_screen is True and s.bpb_2m is None and s.control_of == "C:S60_m12_base" and "screen" in s.tags
    d = by["D:D3_lk35_s58"]
    assert d.chip == "D" and d.bpb_2m_ema == 0.96470 and d.control_of == "D:D2_rf_s58"
    assert parse_knob_changes("FF_A=1 FF_CC_FLAGS=--x -O3") == {"FF_A": "1", "FF_CC_FLAGS": "--x -O3"}


def test_monitor_knob_changes_are_ff_env_only(tmp_path):
    """DF-4: knobs are the FF_* / NEURON_* env train_ff.py reads; monitor shorthand is renamed to its FF_ key when
    known and otherwise kept out of knobs (notes `unmapped_changes=`)."""
    from ffsim.parse_monitor import KNOB_ALIASES, KNOB_PREFIXES, MONITOR_COLUMNS, UNMAPPED_NOTE, split_knob_changes
    assert all(v.startswith("FF_") for v in KNOB_ALIASES.values())
    knobs, other = split_knob_changes("softcap=18 warmup=40 FF_SEED=58 x0_init=9r0.16 XU_QUEUE_DEFAULT=32 "
                                      "NEURON_CC_FLAGS=--x -O3")
    assert knobs == {"FF_SOFTCAP": "18", "FF_WARMUP": "40", "FF_SEED": "58", "NEURON_CC_FLAGS": "--x -O3"}
    assert other == {"x0_init": "9r0.16", "XU_QUEUE_DEFAULT": "32"}
    assert parse_knob_changes("head_init_std=0.004 wte_init_std=1.6 z_loss=1e-4 POOL=3 CFC=1.0 QKfold=1") == {
        "FF_HEAD_STD": "0.004", "FF_WTE_STD": "1.6", "FF_ZLOSS": "1e-4", "FF_POOL_LAST": "3", "FF_CFC_INIT": "1.0",
        "FF_QK_GAIN_FOLD": "1"}
    assert parse_knob_changes(None) == {} and split_knob_changes("") == ({}, {})
    path = tmp_path / "monitor.csv"
    path.write_text(",".join(MONITOR_COLUMNS) + "\n"
                    "2026-09-25,C,0519_G_S123,M13,S123,73,60,,,,,1,,,,,"
                    "S123=FF_ROPE_FN+QKfold+FF_RMSNORM_FN EWD0=1 softcap=18,three flags\n", encoding="utf-8")
    (rec,) = load_monitor_csv(str(path))
    assert rec.knobs == {"FF_SOFTCAP": "18"} and all(k.startswith(KNOB_PREFIXES) for k in rec.knobs)
    assert f"{UNMAPPED_NOTE}=S123=FF_ROPE_FN+QKfold+FF_RMSNORM_FN EWD0=1" in rec.notes and "three flags" in rec.notes


def test_monitor_csv_missing(tmp_path):
    missing = str(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        load_monitor_csv(missing)
    assert load_monitor_csv(missing, missing_ok=True) == []
    assert load_official_uploads(missing, missing_ok=True) == []


def test_official_uploads_fixture():
    ups = load_official_uploads(os.path.join(FIX, "official-uploads.csv"))
    assert [u["submission"] for u in ups] == ["K57", "K59"]
    k57, k59 = ups
    assert k57["official_bpb"] == 0.9686 and k57["rehearsal_bpb"] == 0.96193
    assert k57["offset"] == pytest.approx(0.00667, abs=1e-6)        # filled from official - rehearsal
    assert k59["offset"] == 0.0062 and k59["seed"] == 73 and k59["note"] == "best" and k57["note"] is None


# ----------------------------------------------------------------------------- dataset
def test_save_load_roundtrip(tmp_path, c28):
    recs = [c28] + load_monitor_csv(os.path.join(FIX, "monitor-runs.csv"))
    path = str(tmp_path / "sub" / "runs.jsonl")
    assert dataset.save_runs(recs, path) == len(recs)
    back = dataset.load_runs(path)
    assert [r.to_json() for r in back] == [r.to_json() for r in recs]
    assert isinstance(back[0].step_time, StepTime) and back[0].step_time.k4 == c28.step_time.k4


def test_merge_precedence_and_unions(c28):
    mon = load_monitor_csv(os.path.join(FIX, "monitor-runs.csv"))
    csv_rec = RunRecord(run_id="C:C28_cold_K51", chip="C", run="C28_cold_K51", sources=["experiments.csv"],
                        bpb_2m=0.9999, steps=1, in_window=True, knobs={"FF_SEED": "1", "FF_WD": "0.02"},
                        tags=["family:C"], notes="csv note", is_screen=False)
    merged = dataset.merge_runs([[csv_rec], mon, [c28]])
    assert set(merged) == {"C:C28_cold_K51", "C:C40_cold_K59", "C:C36_cold_K57", "D:D3_lk35_s58", "C:S60_m13_qkv"}
    m = merged["C:C28_cold_K51"]
    assert m.sources == ["chip_log", "monitor", "experiments.csv"]     # union, in precedence order
    assert m.bpb_2m == c28.bpb_2m and m.steps == 2036                # chip_log wins
    assert m.code_version == "code_k51"                               # the log's FF_CODE_DIR value beats the monitor's K51
    assert m.in_window is True                                        # ...but a field only the csv has is kept
    assert m.knobs["FF_SEED"] == "67" and m.knobs["FF_WD"] == "0.02" and m.knobs["FF_TIME_TARGET"] == "1788"
    assert m.knobs_complete is True
    assert m.step_time.k4 == c28.step_time.k4 and m.loss_curve == c28.loss_curve
    assert "family:C" in m.tags and "rehearsal" in m.tags and "recipe:cold-rehearsal" in m.tags
    assert "csv note" in m.notes and "cold rehearsal of K51" in m.notes and "lines_median_by_k" in m.notes
    assert m.date_utc == "2026-09-26" and m.submission == "K51"
    # a weak 'code' survives only when nothing better exists; a screen flag from any source sticks
    lone = dataset.merge_runs([[RunRecord(run_id="C:x", chip="C", run="x", sources=["chip_log"], code_version="code")]])
    assert lone["C:x"].code_version == "code"
    weak = dataset.merge_runs([[RunRecord(run_id="C:w", chip="C", run="w", sources=["chip_log"], code_version="code")],
                               [RunRecord(run_id="C:w", chip="C", run="w", sources=["monitor"], code_version="K51")]])
    assert weak["C:w"].code_version == "K51"                          # a log without FF_CODE_DIR ('code') loses to the monitor
    both = dataset.merge_runs([[RunRecord(run_id="C:y", chip="C", run="y", sources=["chip_log"], is_screen=False)],
                               [RunRecord(run_id="C:y", chip="C", run="y", sources=["monitor"], is_screen=True)]])
    assert both["C:y"].is_screen is True


def test_merge_csv_exact_bpb_beats_monitor():
    """DF-3: experiments.csv carries raw_bpb at full precision; the monitor's bpb_2m is LLM-extracted and 5-6 dp
    rounded, so for bpb_2m the csv outranks the monitor. Every other field keeps monitor > csv, including steps
    (the csv's steps column is the last logged step index, one below the contract's "@N" count)."""
    rid = "B:1429_Q_adamw1cd050_s58"

    def rec(src, **kw):
        return RunRecord(run_id=rid, chip="B", run="1429_Q_adamw1cd050_s58", sources=[src], **kw)

    assert dataset.CSV_EXACT_FIELDS == ("bpb_2m",)
    csv_rec = rec("experiments.csv", bpb_2m=0.985078274649089, steps=1568, in_window=True, seed=58)
    mon_rec = rec("monitor", bpb_2m=0.98508, steps=1569, date_utc="2026-09-22", bpb_2m_ema=0.9849, seed=58)
    m = dataset.merge_runs([[csv_rec], [mon_rec]])[rid]
    assert m.bpb_2m == 0.985078274649089                              # the csv's exact bpb wins
    assert m.steps == 1569 and m.date_utc == "2026-09-22" and m.bpb_2m_ema == 0.9849   # other fields: monitor first
    assert m.in_window is True and m.sources == ["monitor", "experiments.csv"]
    # the monitor's bpb is kept only when the csv has no value
    m = dataset.merge_runs([[rec("experiments.csv", steps=119)], [rec("monitor", bpb_2m=0.98508, steps=120)]])[rid]
    assert m.bpb_2m == 0.98508 and m.steps == 120
    # a chip_log still outranks both...
    m = dataset.merge_runs([[csv_rec], [mon_rec], [rec("chip_log", bpb_2m=0.985, steps=1570)]])[rid]
    assert m.bpb_2m == 0.985 and m.steps == 1570
    # ...unless it is log_incomplete: then the csv's bpb, the monitor's steps
    partial = rec("chip_log", bpb_2m=None, steps=601, tags=["log_incomplete"])
    m = dataset.merge_runs([[mon_rec], [partial], [csv_rec]])[rid]
    assert m.bpb_2m == 0.985078274649089 and m.steps == 1569
    m = dataset.merge_runs([[mon_rec], [partial]])[rid]
    assert m.bpb_2m == 0.98508 and m.steps == 1569


def test_effective_knobs_full_runs_summarize(c28):
    base = {"FF_COOLDOWN_FRAC": "0.45", "FF_WARMUP": "20", "FF_SEED": "58"}
    env = dataset.effective_knobs(RunRecord(run_id="C:z", chip="C", run="z", knobs={"FF_SEED": "73"}), base)
    assert env == {"FF_COOLDOWN_FRAC": "0.45", "FF_WARMUP": "20", "FF_SEED": "73"}
    assert base["FF_SEED"] == "58"                                    # input not mutated
    recs = [c28] + load_monitor_csv(os.path.join(FIX, "monitor-runs.csv"))
    full = dataset.full_runs(recs)
    assert {r.run_id for r in full} == {"C:C28_cold_K51", "C:C40_cold_K59", "C:C36_cold_K57", "D:D3_lk35_s58"}
    s = dataset.summarize(recs)
    assert s["n"] == 6 and s["by_chip"] == {"C": 5, "D": 1}
    assert s["by_source"] == {"chip_log": 1, "monitor": 5}
    assert s["by_code_version"] == {"K51": 1, "M12": 3, "M13": 1, "code_k51": 1}
    assert s["screens"] == 1 and s["full_runs"] == 5 and s["with_step_time"] == 1 and s["with_bpb_20m"] == 1   # 6 records, S60 is the only screen


def test_experiments_summary(experiments):
    s = dataset.summarize(experiments)
    assert s["n"] == 348 and s["screens"] == 50 and s["full_runs"] == 298 and s["with_seed"] == 341
    assert json.dumps(s)                                              # JSON-serialisable for QA reports


# ----------------------------------------------------------------------------- real harvested dirs (when present)
def _harvested_dirs():
    out = []
    for chip in ("C", "D"):
        root = os.path.join(REPO, "research", "sim-data", f"chip{chip}")
        if os.path.isdir(root):
            out += sorted(os.path.join(root, d) for d in os.listdir(root)
                          if not d.startswith("_") and os.path.isfile(os.path.join(root, d, "train.log")))
    return out


CHIP_D = os.path.join(REPO, "research", "sim-data", "chipD")


@pytest.mark.skipif(not os.path.isfile(os.path.join(CHIP_D, "1202_R_g7lk35_s73", "train.log")),
                    reason="chip D harvest not present")
def test_real_chip_d_full_run_and_screen():
    r = parse_run_dir(os.path.join(CHIP_D, "1202_R_g7lk35_s73"))
    assert (r.chip, r.seed, r.steps, r.date_utc) == ("D", 73, 2153, "2026-09-29")
    assert r.charged_seconds == 1754.1 and r.startup_seconds == 377.21 and r.time_target == 1760.0
    assert r.bpb_2m == pytest.approx(0.9664382633713863) and r.bpb_2m_ema == pytest.approx(0.966475158787183)
    assert r.code_version == "code_k12off" and r.code_sha.startswith("32cbcdd3f6eb84344e254bb2cd8a0c48")
    assert r.knobs_complete and r.knobs["FF_ACCUM_SCHED"] == "1:0.06,2:0.2,4" and r.knobs["FF_LEAKY_RELU2"] == "0.35"
    assert r.knobs["FF_COOLDOWN_FRAC"] == "0.60"                     # job env wins over base.env 0.45
    st = r.step_time
    assert (st.k1, st.k2, st.k4) == (0.2897, 0.5132, 1.0359) and (st.n_k1, st.n_k2, st.n_k4) == (314, 478, 1361)
    assert not r.is_screen and "status=train_exit=0" in r.notes and "timing=startup_s=414.1 charged_s=1756.8" in r.notes
    s = parse_run_dir(os.path.join(CHIP_D, "1142_G_C0"))
    assert s.is_screen and s.steps == 60 and s.startup_seconds == 484.86 and s.bpb_2m is None
    assert (s.step_time.k1, s.step_time.k2, s.step_time.k4) == (0.293, 0.5107, 0.941)   # forced 10-step phases: only the summary has them
    assert s.knobs["NEURON_COMPETITION_R1_NUM_STEPS"] == "60" and s.code_version == "code_k12off"
    warm = parse_run_dir(os.path.join(CHIP_D, "1240_R_g7lk35_s97"))
    assert warm.startup_seconds is None and warm.seed == 97          # warm compile cache: step-0 dt 38 s is not a cold start
    assert parse_run_dir(os.path.join(CHIP_D, "1552_R_m14asa_s73")).code_version == "code_k14off"


@pytest.mark.skipif(not _harvested_dirs(), reason="no harvested run dirs under research/sim-data/chip{C,D} yet")
def test_harvested_chip_dirs_parse():
    recs = [parse_run_dir(d) for d in _harvested_dirs()]
    assert all(r is not None for r in recs)
    assert all(r.chip in ("C", "D") and r.run_id == f"{r.chip}:{r.run}" for r in recs)
    assert all(json.loads(r.to_json())["run_id"] == r.run_id for r in recs)
    assert all(r.sources == ["chip_log"] for r in recs)
    with_steps = [r for r in recs if r.steps]
    assert with_steps, "no run dir yielded step lines"
    for r in with_steps:
        assert r.loss_curve and r.loss_curve[-1][0] < r.steps
        for k in ("k1", "k2", "k4"):
            v = getattr(r.step_time, k)
            assert v is None or 0.2 < v < 3.0, (r.run_id, k, v)
        if r.bpb_2m is not None and not r.is_screen:                 # finished, evaluated full runs only
            assert 300 < r.charged_seconds < 1900, (r.run_id, r.charged_seconds)   # early-EMA failure tests stop at ~840 s
            assert r.steps > 300, (r.run_id, r.steps)              # 0036_F34 (depth 14) finished at 436 steps
            # no phase-median requirement: k2-only runs (1339_v0_early_raw) and 524k-token runs (2112_F28_b524k, k8) exist
