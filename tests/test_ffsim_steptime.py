"""Tests for ffsim/steptime.py (Part 1: step-time model + steps-from-step-time).

Built against synthetic RunRecords with a known generating model; the tests on harvested data
(research/sim-data/runs.jsonl) skip when the dataset is absent.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics

import numpy as np
import pytest

from ffsim.schema import RunRecord, StepTime
from ffsim.steptime import (
    CODE_DIR_KNOBS,
    EXCLUDE_VERSIONS,
    FEATURE_SETS,
    KNOB_DEFAULTS,
    KNOB_FEATURES,
    MEDIAN_TO_MEAN_OVERHEAD,
    MIN_N,
    NOVEL_PRIOR_LOG,
    PHASE_RATIO_FALLBACK,
    StepTimeModel,
    effective_env,
    features_from_code,
    knob_features,
    lineage_label,
    nearest_version,
    parse_accum_sched,
    record_weight,
    simulate_phases,
    steps_from_step_time,
    structural_features,
    version_key,
    version_similarity,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DATA = os.path.join(REPO, "research", "sim-data")
RUNS_JSONL = os.path.join(SIM_DATA, "runs.jsonl")

# ----------------------------------------------------------------------------------------------
# Synthetic truth: lineage baselines (k4, chip C, reference recipe) within 0.3% of their
# neighbours, the campaign's mechanism multipliers, chip D +12% in full runs / +1% in screens,
# chips A/B +2%. Noise: per-step dt sd 1.5% (a 20-step screen median then scatters ~0.4%, like
# paired S60 controls 0.9373/0.9368, 0.9340/0.9304) plus 0.2% between-run chip jitter.
# ----------------------------------------------------------------------------------------------
TRUE_BASE_K4 = {"M9": 0.931, "M10": 0.930, "M11": 0.929, "M12": 0.927, "M13": 0.925, "M14": 0.923}
# The real M11 -> M12 step (RF: one elementwise pass removed, -1.8%) is a lineage JUMP that no
# nearest-neighbour fallback can predict; used by the "LOO detects a jump" test.
JUMP_BASE_K4 = {"M9": 0.931, "M10": 0.930, "M11": 0.929, "M12": 0.912, "M13": 0.913, "M14": 0.915}
TRUE_RATIO = {"k1": 0.32, "k2": 0.55, "k4": 1.0}
STEP_SD = 0.015
JITTER_SD = 0.002
TRUE_LOG_EFFECT = {
    "leaky_fnm": math.log(0.996), "leaky_abs": math.log(1.036), "leaky_max": math.log(1.063),
    "relu2_fn": math.log(0.981), "xsa_layers": math.log(1.10), "acc_in_graph_off": math.log(1.021),
    "muon_zero_off": math.log(1.03), "log2_fused_ce": math.log(1.027), "mtp_off": math.log(0.99),
    "attn_src_layers": math.log(1.02), "lanes_extra": math.log(1.023), "ema_off": math.log(0.995),
}
TRUE_CHIP = {"C_full": 0.0, "C_screen": 0.005, "D_full": math.log(1.12), "D_screen": math.log(1.01),
             "AB_full": math.log(1.02), "AB_screen": math.log(1.02)}
KNOB_VARIANTS = [
    {},
    {"FF_RELU2_FN": "1"},
    {"FF_XSA_LAYERS": "5,6,7,8"},
    {"FF_XSA_LAYERS": "all"},
    {"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "abs"},
    {"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "max"},
    {"FF_LEAKY_RELU2": "0.35", "FF_LEAKY_FORM": "fnm"},
    {"FF_ACC_IN_GRAPH": "0"},
    {"FF_MUON_ZERO": "0"},
    {"FF_FUSED_CE": "16"},
    {"FF_MTP": "0"},
    {"FF_ATTN_SRC": "5:6,7,8"},
    {"FF_LANES": "2"},
    {"FF_EMA": "0"},
]


def true_step_time(version: str, knobs: dict, chip: str, is_screen: bool, phase: str, base=None) -> float:
    base = base or TRUE_BASE_K4
    f = knob_features(knobs, version)
    log_t = math.log(base[version] * TRUE_RATIO[phase])
    log_t += sum(TRUE_LOG_EFFECT.get(k, 0.0) * v for k, v in f.items())
    group = ("AB" if chip in "AB" else chip) + ("_screen" if is_screen else "_full")
    log_t += TRUE_CHIP[group]
    return math.exp(log_t)


def make_record(rng, version: str, chip: str, is_screen: bool, knobs: dict, idx: int, base=None) -> RunRecord:
    jitter = rng.normal(0.0, JITTER_SD)  # between-run chip jitter
    n = {"k1": 20, "k2": 10, "k4": 20} if is_screen else {"k1": 330, "k2": 490, "k4": 1520}
    st = StepTime()
    for phase in ("k1", "k2", "k4"):
        med_noise = rng.normal(0.0, STEP_SD * math.sqrt(math.pi / (2 * n[phase])))
        t = true_step_time(version, knobs, chip, is_screen, phase, base) * math.exp(jitter + med_noise)
        setattr(st, phase, round(t, 4))
        setattr(st, "n_" + phase, n[phase])
    run = f"{idx:04d}_{'S60' if is_screen else 'R'}_{version.lower()}_s{idx % 7 + 50}"
    return RunRecord(run_id=f"{chip}:{run}", chip=chip, run=run, code_version=version, is_screen=is_screen,
                     knobs=dict(knobs), knobs_complete=True, step_time=st, sources=["chip_log"])


def synthetic_records(seed: int = 0, base=None) -> list:
    base = base or TRUE_BASE_K4
    rng = np.random.default_rng(seed)
    recs = []
    idx = 0
    for version in base:
        for _ in range(3):                                   # full runs, chip C, reference knobs
            recs.append(make_record(rng, version, "C", False, {}, idx, base)); idx += 1
        for knobs in KNOB_VARIANTS:                          # S60 screens, chip C
            recs.append(make_record(rng, version, "C", True, knobs, idx, base)); idx += 1
        recs.append(make_record(rng, version, "C", False, {"FF_RELU2_FN": "1"}, idx, base)); idx += 1
        recs.append(make_record(rng, version, "D", False, {}, idx, base)); idx += 1
        recs.append(make_record(rng, version, "D", True, {}, idx, base)); idx += 1
        recs.append(make_record(rng, version, "A", False, {}, idx, base)); idx += 1
    return recs


@pytest.fixture(scope="module")
def fitted():
    recs = synthetic_records()
    return StepTimeModel().fit(recs), recs


def load_dataset() -> list:
    recs = []
    with open(RUNS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(RunRecord.from_dict(json.loads(line)))
    return recs


@pytest.fixture(scope="module")
def real_model():
    if not os.path.exists(RUNS_JSONL):
        pytest.skip("research/sim-data/runs.jsonl absent")
    recs = [r for r in load_dataset() if r.chip in ("C", "D") and r.step_time.k4]
    if len(recs) < 50:
        pytest.skip("fewer than 50 chip C/D records with k4")
    return StepTimeModel().fit(recs), recs


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
def test_knob_features_reference_env_is_all_zero():
    f = knob_features({})
    assert set(f) == set(KNOB_FEATURES)
    assert all(v == 0.0 for v in f.values())
    # A fully spelled-out reference env is the same as an empty one.
    assert knob_features(dict(KNOB_DEFAULTS)) == f
    # The reference is the K57 recipe: relu^2, no custom backward, accumulation in graph, zero-init.
    assert KNOB_DEFAULTS["FF_LEAKY_RELU2"] == "0" and KNOB_DEFAULTS["FF_ACC_IN_GRAPH"] == "1"


def test_knob_features_parse_values():
    f = knob_features({
        "FF_LEAKY_RELU2": "0.35", "FF_LEAKY_FORM": "fnm", "FF_RELU2_FN": "1", "FF_XSA_LAYERS": "5,6,7,8",
        "FF_FUSE_QKV": "1", "FF_EMA_EVERY": "16", "FF_FUSED_CE": "16", "FF_KV_HEADS": "0", "FF_MTP": "2",
        "FF_ACC_IN_GRAPH": "0", "FF_MUON_ZERO": "0", "FF_DEPTH": "12", "FF_MB": "16", "FF_HEAD_DIM": "128",
        "FF_TOTAL_BATCH": "131072", "FF_ATTN_SRC": "4:5,6,7,8", "FF_LANES": "3", "FF_POOL_LAST": "3",
        "FF_ROPE_FN": "1", "FF_CE_ONEHOT": "2", "FF_VALUE_EMBED": "3",
    })
    assert f["leaky_fnm"] == 1.0 and f["leaky_abs"] == 0.0 and f["leaky_max"] == 0.0
    assert f["relu2_fn"] == 1.0 and f["fuse_qkv"] == 1.0
    assert f["xsa_layers"] == pytest.approx(4 / 12)
    assert f["attn_src_layers"] == pytest.approx(4 / 12)
    assert f["lanes_extra"] == 2.0 and f["pool_last"] == 1.0 and f["rope_fn"] == 1.0
    assert f["qk_gain_fold"] == 0.0 and f["rmsnorm_fn"] == 0.0 and f["ce_onehot"] == 1.0
    assert f["ema_rate"] == pytest.approx(1.0) and f["ema_off"] == 0.0
    assert f["log2_fused_ce"] == pytest.approx(1.0) and f["fused_ce_off"] == 0.0
    assert f["log2_kv_heads"] == pytest.approx(math.log2(round(12 * 113 / 128)))
    assert f["mtp_2"] == 1.0 and f["mtp_off"] == 0.0
    assert f["acc_in_graph_off"] == 1.0 and f["muon_zero_off"] == 1.0
    assert f["value_embed_gate"] == 1.0 and f["value_embed_off"] == 0.0
    assert f["depth_rel"] == pytest.approx(12 / 9 - 1)
    assert f["log2_mb"] == pytest.approx(1.0)
    assert f["log2_head_dim"] == pytest.approx(-1.0)
    assert f["log2_total_batch"] == pytest.approx(-1.0)
    g = knob_features({"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "max", "FF_XSA_LAYERS": "all", "FF_FUSED_CE": "0",
                       "FF_EMA": "0", "FF_EMA_EVERY": "4"})
    assert g["leaky_max"] == 1.0 and g["leaky_fnm"] == 0.0 and g["xsa_layers"] == 1.0
    assert g["fused_ce_off"] == 1.0 and g["log2_fused_ce"] == 0.0
    assert g["ema_off"] == 1.0 and g["ema_rate"] == 0.0  # no EMA updates at all: no rate term
    # fn form is treated like fnm; a zero slope is plain relu^2 whatever the form says;
    # garbage values fall back to the reference.
    assert knob_features({"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "fn"})["leaky_fnm"] == 1.0
    assert knob_features({"FF_LEAKY_RELU2": "0", "FF_LEAKY_FORM": "max"}) == knob_features({})
    assert knob_features({"FF_DEPTH": "nine"}) == knob_features({})


def test_effective_env_applies_code_dir_bakes():
    # Pre-G7 lineages ran without FF_ACC_IN_GRAPH unless the env says so; the env always wins.
    assert knob_features({}, "M7")["acc_in_graph_off"] == 1.0
    assert knob_features({"FF_ACC_IN_GRAPH": "1"}, "M7")["acc_in_graph_off"] == 0.0
    assert knob_features({}, "M5")["muon_zero_off"] == 1.0
    # M9..M12 and the K57 bakes take the reference.
    for cv in ("M9", "code_m9", "code_k57", "code_k12off", "M12", "K59"):
        assert knob_features({}, cv) == knob_features({}), cv
    # code_m9off is M9 with the K54c recipe (accumulation off-graph).
    assert knob_features({}, "code_m9off")["acc_in_graph_off"] == 1.0
    # code_m10b / code_m11c bake the leaky form; the screens of code_m10xa set FF_XSA_LAYERS=1 as a flag.
    assert knob_features({"FF_LEAKY_RELU2": "0.5"}, "code_m10b")["leaky_abs"] == 1.0
    assert knob_features({"FF_LEAKY_RELU2": "0.5"}, "code_m11c")["leaky_max"] == 1.0
    assert knob_features({"FF_XSA_LAYERS": "1"}, "code_m10xa")["xsa_layers"] == 1.0
    assert knob_features({"FF_XSA_LAYERS": "1"}, "code_m10xl")["xsa_layers"] == pytest.approx(4 / 9)
    assert effective_env({"FF_XSA_LAYERS": "1"}, "code_m10xl")["FF_XSA_LAYERS"] == "5,6,7,8"
    assert "code_k12off" not in CODE_DIR_KNOBS  # the K57 recipe IS the reference


def test_structural_features_interaction():
    assert structural_features("C", False) == {"chip_D": 0, "chip_AB": 0, "screen": 0, "chip_D_x_screen": 0, "chip_AB_x_screen": 0}
    d = structural_features("D", True)
    assert d["chip_D"] == 1 and d["screen"] == 1 and d["chip_D_x_screen"] == 1 and d["chip_AB"] == 0
    b = structural_features("b", False)
    assert b["chip_AB"] == 1 and b["chip_AB_x_screen"] == 0


def test_record_weight_saturates():
    assert record_weight(0) == record_weight(1)
    assert record_weight(20) < record_weight(1500) < 1.0
    assert record_weight(1500) / record_weight(20) < 4.0  # full runs weigh ~3x a screen, not 75x
    assert record_weight(20, n_half=0) == record_weight(1500, n_half=0) == 1.0


# ----------------------------------------------------------------------------------------------
# Lineage keys / nearest version
# ----------------------------------------------------------------------------------------------
def test_version_key_maps_code_dirs_and_submissions_to_lineages():
    assert version_key("code_m9") == "m9" == version_key(" M9 ")
    assert version_key("code_k12off") == "m12" == version_key("M12") == version_key("K59") == version_key("K58")
    assert version_key("code_k57") == "m9" == version_key("K57") == version_key("code_m9off") == version_key("code_m9acc1")
    assert version_key("code_k13off") == "m13" and version_key("code_k15off") == "m15"
    assert version_key("code_k52m6") == "m6" == version_key("code_k53")
    assert version_key("code_k54bmin") == "m7" and version_key("code_k54cmin") == "m8"
    assert version_key("code_m10b") == "m10" == version_key("code_m10xa") and version_key("code_m11c") == "m11"
    assert version_key("code_m7a") == "m7a" and version_key("off") == "off"
    assert lineage_label("m12") == "M12" and lineage_label("m7a") == "M7a"
    assert version_similarity("M12", "code_k12off") == 3.0
    assert version_similarity("M12", "M13") > version_similarity("M12", "M14") > version_similarity("M12", "M3")


def test_nearest_version_prefers_lineage_neighbour():
    known = ["M12", "M13", "M14", "M9", "code_m9"]
    assert nearest_version("M15", known) == "M14"
    assert nearest_version("m13", known) == "M13"
    assert nearest_version("M10", known) == "M9"
    assert nearest_version("K59", known) == "M12"        # K59 IS lineage M12
    assert nearest_version("code_k57", known) == "M9"    # K57 baked on M9
    assert nearest_version("M16", ["M12", "M14"]) == "M14"
    assert nearest_version("X", []) is None


# ----------------------------------------------------------------------------------------------
# Fit / predict
# ----------------------------------------------------------------------------------------------
def test_fit_recovers_baselines_and_knob_effects(fitted):
    model, _ = fitted
    for version, base in TRUE_BASE_K4.items():
        p = model.predict(version, {}, "C")
        assert p["notes"] == ""
        assert p["k4"] == pytest.approx(base, rel=0.005), version
    p0 = model.predict("M12", {}, "C")
    assert model.predict("M12", {"FF_RELU2_FN": "1"}, "C")["k4"] / p0["k4"] == pytest.approx(0.981, abs=0.006)
    assert model.predict("M12", {"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "max"}, "C")["k4"] / p0["k4"] == pytest.approx(1.063, abs=0.01)
    assert model.predict("M12", {"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "abs"}, "C")["k4"] / p0["k4"] == pytest.approx(1.036, abs=0.01)
    assert model.predict("M12", {"FF_XSA_LAYERS": "all"}, "C")["k4"] / p0["k4"] == pytest.approx(1.10, abs=0.012)
    assert model.predict("M12", {"FF_ACC_IN_GRAPH": "0"}, "C")["k4"] / p0["k4"] == pytest.approx(1.021, abs=0.008)
    assert model.predict("M12", {"FF_LANES": "2"}, "C")["k4"] / p0["k4"] == pytest.approx(1.023, abs=0.01)


def test_fit_recovers_chip_and_screen_interaction(fitted):
    model, _ = fitted
    c_full = model.predict("M13", {}, "C")["k4"]
    assert model.predict("M13", {}, "D")["k4"] / c_full == pytest.approx(1.12, abs=0.012)
    c_screen = model.predict("M13", {}, "C", is_screen=True)["k4"]
    assert model.predict("M13", {}, "D", is_screen=True)["k4"] / c_screen == pytest.approx(1.01, abs=0.012)
    assert model.predict("M13", {}, "A")["k4"] / c_full == pytest.approx(1.02, abs=0.012)


def test_predict_returns_all_phases_and_sd(fitted):
    model, _ = fitted
    p = model.predict("M12", {}, "C")
    for key in ("k1", "k2", "k4", "sd_k4", "version_used", "notes"):
        assert key in p
    assert p["k1"] / p["k4"] == pytest.approx(TRUE_RATIO["k1"], rel=0.02)
    assert p["k2"] / p["k4"] == pytest.approx(TRUE_RATIO["k2"], rel=0.02)
    assert 0.0 < p["sd_k4"] < 0.02 * p["k4"]
    assert p["version_used"] == "M12"
    assert model.predict("m12", {}, "c")["k4"] == pytest.approx(p["k4"])  # case-insensitive
    assert model.predict("code_k12off", {}, "C")["k4"] == pytest.approx(p["k4"])  # alias of the lineage


def test_unseen_version_falls_back_to_nearest_with_note(fitted):
    model, _ = fitted
    p = model.predict("M15", {}, "C")
    assert "fallback" in p["notes"] and "M14" in p["notes"]
    assert p["version_used"] == "M14"
    assert p["k4"] == pytest.approx(model.predict("M14", {}, "C")["k4"], rel=1e-9)
    assert p["sd_k4"] > model.predict("M14", {}, "C")["sd_k4"]  # extra uncertainty for an unseen lineage


def test_unsupported_mechanism_is_flagged_and_charged_the_prior(fitted):
    """A mechanism no fitted run carries is named, charged NOVEL_PRIOR_LOG (+3%) per mechanism in
    every phase, and given a wider sd. Charging 0 was not neutral: the simulator's symmetric jitter
    factor on the widened sd handed such a recipe MORE steps than its base (review F10)."""
    model, _ = fitted
    base = model.predict("M12", {}, "C")
    p = model.predict("M12", {"FF_FUSE_QKV": "1"}, "C")   # no synthetic run carries QKV fusion
    assert "fuse_qkv" in p["notes"] and "cost unknown" in p["notes"] and "NOVEL_PRIOR" in p["notes"]
    for phase in ("k1", "k2", "k4"):
        assert p[phase] == pytest.approx(base[phase] * math.exp(NOVEL_PRIOR_LOG), rel=1e-9), phase
    assert math.exp(NOVEL_PRIOR_LOG) == pytest.approx(1.03)
    assert p["sd_k4"] > 2 * base["sd_k4"]
    assert steps_from_step_time(p, 1790.0) < steps_from_step_time(base, 1790.0)
    two = model.predict("M12", {"FF_FUSE_QKV": "1", "FF_VALUE_EMBED": "3"}, "C")   # two unknown mechanisms
    assert two["k4"] == pytest.approx(base["k4"] * math.exp(2 * NOVEL_PRIOR_LOG), rel=1e-9) and "x2" in two["notes"]
    ex = model.explain("M12", {"FF_FUSE_QKV": "1"}, "C")
    assert ex["predicted"] == pytest.approx(p["k4"])
    assert [c["feature"] for c in ex["contributions"] if c["feature"].startswith("novel_prior:")] == ["novel_prior:fuse_qkv"]
    assert model.unsupported_features("M12", {"FF_FUSE_QKV": "1", "FF_RELU2_FN": "1"}) == ["fuse_qkv"]
    assert model.unsupported_features("M12", {}, chip="D", is_screen=True) == []
    # A structural gap (a chip-D screen when no D screen was fitted) is not a graph change: no prior.
    no_d_screen = StepTimeModel().fit([r for r in synthetic_records() if not (r.chip == "D" and r.is_screen)])
    q = no_d_screen.predict("M12", {}, "D", is_screen=True)
    assert "chip_D_x_screen" in q["notes"] and "NOVEL_PRIOR" not in q["notes"]


def test_known_versions_summary_and_support(fitted):
    model, _ = fitted
    assert model.known_versions() == sorted(TRUE_BASE_K4, key=version_key)
    assert set(model.known_versions("k4")) == set(TRUE_BASE_K4)
    summary = model.phase_fit_summary()
    assert set(summary) == {"k1", "k2", "k4"}
    assert summary["k4"]["n"] == len(synthetic_records())
    assert summary["k4"]["rmse_pct"] < 0.5
    table = {c["feature"]: c for c in model.coefficient_table("k4")}
    assert table["relu2_fn"]["support"] == 2 * len(TRUE_BASE_K4) and table["fuse_qkv"]["support"] == 0
    assert table["relu2_fn"]["pct"] == pytest.approx(-1.9, abs=0.5)
    sup = model.support("k4")
    assert sup["FF_LEAKY_RELU2"]["values"] == [0.0, 0.35, 0.5] and sup["FF_FUSE_QKV"]["values"] == [0.0]
    assert set(sup["FF_LEAKY_FORM"]) == {"fnm", "abs", "max"}


def test_explain_lists_contributions_that_sum_to_prediction(fitted):
    model, _ = fitted
    ex = model.explain("M12", {"FF_RELU2_FN": "1", "FF_XSA_LAYERS": "5,6,7,8"}, "D")
    names = [c["feature"] for c in ex["contributions"]]
    assert names[0] == "intercept"
    assert "ver:m12" in names and "relu2_fn" in names and "xsa_layers" in names and "chip_D" in names
    assert "screen" not in names  # zero-valued features are not listed
    total = sum(c["contribution_log"] for c in ex["contributions"])
    assert total == pytest.approx(ex["log_total"])
    assert math.exp(total) == pytest.approx(ex["predicted"])
    assert ex["predicted"] == pytest.approx(model.predict("M12", {"FF_RELU2_FN": "1", "FF_XSA_LAYERS": "5,6,7,8"}, "D")["k4"])
    rf = next(c for c in ex["contributions"] if c["feature"] == "relu2_fn")
    assert rf["pct"] == pytest.approx(-1.9, abs=0.5) and rf["doc"] and rf["support"] > 0
    assert ex["effective_env"]["FF_RELU2_FN"] == "1"


def test_loo_report_meets_target_on_smooth_lineage(fitted):
    model, _ = fitted
    rep = model.loo_report("k4")
    assert rep["n_versions"] == len(TRUE_BASE_K4) and rep["n_scored"] == len(TRUE_BASE_K4)
    assert rep["target_pct"] == 0.5
    for v, d in rep["per_version"].items():
        assert d["fallback_to"] in TRUE_BASE_K4 and d["fallback_to"] != v
        assert d["n"] > 0 and d["mean_abs_pct"] is not None and d["n_novel"] == 0
    assert rep["per_version"]["M13"]["fallback_to"] in ("M12", "M14")
    assert rep["n_novel"] == 0 and rep["mean_abs_pct_reachable"] == pytest.approx(rep["mean_abs_pct"])
    # Both averaging conventions are reported; the unsuffixed keys are the run-weighted ones and
    # `passes` is judged on the run-weighted reachable error (review F2).
    assert rep["mean_abs_pct"] == rep["mean_abs_pct_by_run"]
    assert rep["mean_abs_pct_reachable"] == rep["mean_abs_pct_reachable_by_run"]
    assert rep["judged_on"] == "mean_abs_pct_reachable_by_run"
    assert rep["n_reachable"] == rep["n_runs"] == len(synthetic_records())
    assert rep["mean_abs_pct_reachable_by_lineage"] == pytest.approx(
        np.mean([d["mean_abs_pct_reachable"] for d in rep["per_version"].values()]))
    assert rep["mean_abs_pct_by_run"] == pytest.approx(
        np.mean([abs(r["err_pct"]) for d in rep["per_version"].values() for r in d["runs"]]))
    assert rep["n_reachable_above_target"] == sum(abs(r["err_pct"]) > 0.5 for d in rep["per_version"].values() for r in d["runs"])
    assert rep["mean_abs_pct_by_run"] <= 0.5 and rep["mean_abs_pct_by_lineage"] <= 0.5 and rep["passes"], rep
    # After loo_report the unseen-lineage sd comes from the measured LOO rmse.
    assert model._fallback_sd_log == pytest.approx(rep["rmse_pct_reachable"] / 100.0)


def test_loo_report_detects_a_lineage_jump():
    """M11 -> M12 is a -1.8% jump (RF). The nearest-lineage fallback cannot know it, and the report
    must say so (error ~ the true gap, passes False) rather than hide it."""
    recs = synthetic_records(seed=1, base=JUMP_BASE_K4)
    model = StepTimeModel().fit(recs)
    rep = model.loo_report("k4")
    assert not rep["passes"] and rep["mean_abs_pct"] > 0.5
    for held in ("M11", "M12"):
        d = rep["per_version"][held]
        gap = 100.0 * abs(JUMP_BASE_K4[d["fallback_to"]] / JUMP_BASE_K4[held] - 1.0)
        assert d["mean_abs_pct"] == pytest.approx(gap, abs=0.5), (held, d, gap)
    # Within-lineage predictions are still fine: the jump is a lineage fact, not a model defect.
    assert model.loro_report("k4")["mean_abs_pct"] < 0.6
    # And an unseen lineage now carries the measured LOO rmse as extra sd.
    p = model.predict("M15", {}, "C")
    assert p["sd_k4"] / p["k4"] >= rep["rmse_pct_reachable"] / 100.0


def test_loo_report_separates_novel_mechanisms():
    """A mechanism only one lineage ran cannot be predicted when that lineage is held out: the
    run is scored as 'novel', named, and kept out of the reachable error."""
    recs = [r for r in synthetic_records() if not (r.code_version != "M13" and "FF_LANES" in r.knobs)]
    model = StepTimeModel().fit(recs)
    rep = model.loo_report("k4")
    d = rep["per_version"]["M13"]
    assert d["n_novel"] == 1 and d["novel_features"] == ["lanes_extra"]
    assert d["mean_abs_pct_reachable"] < d["mean_abs_pct"] or d["mean_abs_pct_reachable"] <= 0.5
    novel_runs = [r for r in d["runs"] if r["novel"]]
    assert len(novel_runs) == 1 and abs(novel_runs[0]["err_pct"]) == pytest.approx(2.3, abs=0.8)
    assert rep["n_novel"] == 1


def test_loro_report_within_lineage_noise_floor(fitted):
    model, _ = fitted
    rep = model.loro_report("k4")
    assert rep["n"] == len(synthetic_records())
    assert rep["mean_abs_pct"] < 0.6 and rep["rmse_pct"] < 0.7


def test_fit_skips_unusable_records_and_counts_them():
    recs = synthetic_records()
    bad1 = RunRecord(run_id="C:x", chip="C", run="x", code_version=None, step_time=StepTime(k4=0.9, n_k4=20))
    bad2 = RunRecord(run_id="Z:y", chip="Z", run="y", code_version="M12", step_time=StepTime(k4=0.9, n_k4=20))
    bad3 = RunRecord(run_id="C:z", chip="C", run="z", code_version="M12", step_time=StepTime())  # no phases measured
    bad4 = RunRecord(run_id="C:c", chip="C", run="c", code_version="code", step_time=StepTime(k4=1.9, n_k4=100))
    bad5 = RunRecord(run_id="C:p", chip="C", run="p", code_version="M12", step_time=StepTime(k4=1.02, n_k4=5))  # 5-step probe
    model = StepTimeModel().fit(recs + [bad1, bad2, bad3, bad4, bad5])
    assert model.skipped == {"no_code_version": 1, "excluded_version": 1, "bad_chip": 1, "low_n": 1}
    assert model.phase_fit_summary()["k4"]["n"] == len(recs)
    assert model.predict("M12", {}, "C")["k4"] == pytest.approx(TRUE_BASE_K4["M12"], rel=0.005)
    assert "code" in EXCLUDE_VERSIONS and MIN_N == 10
    # min_n=1 keeps the probe (its weight is small); exclude_versions=() keeps the placeholder.
    keep = StepTimeModel(min_n=1, exclude_versions=()).fit(recs + [bad4, bad5])
    assert keep.skipped["low_n"] == 0 and keep.skipped["excluded_version"] == 0
    assert keep.phase_fit_summary()["k4"]["n"] == len(recs) + 2


def test_feature_sets_and_options_keep_the_api():
    recs = synthetic_records()
    for name in FEATURE_SETS:
        m = StepTimeModel(features=name).fit(recs)
        p = m.predict("M12", {"FF_RELU2_FN": "1"}, "C")
        assert 0.85 < p["k4"] < 1.0 and set(m.features) == set(FEATURE_SETS[name])
    m = StepTimeModel(knn_k=3, n_half=0).fit(recs)
    p = m.predict("M12", {}, "C")
    assert p["k4"] == pytest.approx(TRUE_BASE_K4["M12"], rel=0.006)
    assert any(c["feature"] == "knn_residual" for c in m.explain("M12", {}, "C")["contributions"])


def test_phase_without_data_uses_ratio_fallback():
    recs = []
    for i in range(6):
        r = make_record(np.random.default_rng(i), "M12", "C", False, {}, i)
        r.step_time.k1 = r.step_time.k2 = None
        r.step_time.n_k1 = r.step_time.n_k2 = 0
        recs.append(r)
    model = StepTimeModel().fit(recs)
    p = model.predict("M12", {}, "C")
    assert p["k1"] == pytest.approx(p["k4"] * PHASE_RATIO_FALLBACK["k1"])
    assert p["k2"] == pytest.approx(p["k4"] * PHASE_RATIO_FALLBACK["k2"])
    assert "k1: no fitted data" in p["notes"] and "k2: no fitted data" in p["notes"]
    assert model.known_versions("k1") == []


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        StepTimeModel().predict("M12", {}, "C")


def test_features_from_code_hook_is_a_stub(tmp_path):
    p = tmp_path / "train.py"
    p.write_text("print('hi')\n")
    assert features_from_code(str(p)) == {}


# ----------------------------------------------------------------------------------------------
# Steps from step time
# ----------------------------------------------------------------------------------------------
def test_parse_accum_sched_forms():
    assert parse_accum_sched("1:0.06,2:0.2,4") == [(1, 0.06), (2, 0.2), (4, None)]
    assert parse_accum_sched("2:0.12,4") == [(2, 0.12), (4, None)]
    assert parse_accum_sched("4") == [(4, None)]
    for bad in ("", "4:0.5", "1:0.2,2:0.1,4", "3:0.1,4", "1,2,4", "1:0.06,2:0.2,0"):
        with pytest.raises(ValueError):
            parse_accum_sched(bad)


STEP_LINE = re.compile(r"^step (\d+) \|.*?\| dt ([\d.]+)s \|.*?charged (\d+)s")
PHASE_LINE = re.compile(r"FF_ACCUM_SCHED phase (\d+): (\d+) micro-batch(?:es)? per update.*?from step (\d+)")


def _medians_from_train_log(path: str):
    """Per-k medians, charged seconds and step count from a harvested train.log (test-only parser)."""
    dts, switches = [], []
    charged = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_LINE.match(line.strip())
            if m:
                dts.append((int(m.group(1)), float(m.group(2))))
                charged = float(m.group(3))
                continue
            m = PHASE_LINE.search(line)
            if m:
                switches.append((int(m.group(3)), int(m.group(2))))
    if not dts or not switches or charged is None:
        return None
    switches.sort()
    k_final = 4
    by_k = {}
    for step, dt in dts:
        if step < 5:
            continue
        k = k_final
        if step >= switches[0][0]:
            k = next(kk for s, kk in reversed(switches) if step >= s)
        by_k.setdefault(k, []).append(dt)
    if 4 not in by_k:
        return None
    st = {f"k{k}": statistics.median(v) for k, v in by_k.items()}
    return st, charged, dts[-1][0] + 1


def test_steps_anchor_k59_c40():
    """Chip C k4 0.928 / k2 ~0.51 / k1 ~0.29, charged 1790 s -> ~2357 steps (K59 C40) within 2%.
    Uses the harvested LK0.35 s73 logs when present, else the CONTRACT.md anchors."""
    logs = sorted(glob.glob(os.path.join(SIM_DATA, "chipC", "*lk35_s73*", "train.log")))
    parsed = [(_medians_from_train_log(p), p) for p in logs]
    parsed = [(x, p) for x, p in parsed if x is not None and x[2] > 1000]
    if parsed:
        for (st, charged, actual_steps), path in parsed:
            steps = steps_from_step_time(st, charged, time_target=1793.0)
            assert abs(steps / actual_steps - 1.0) <= 0.02, (path, st, charged, steps, actual_steps)
    else:
        st = {"k1": 0.2937, "k2": 0.5069, "k4": 0.9312}
        steps = steps_from_step_time(st, 1790.0, time_target=1793.0)
        assert abs(steps / 2357 - 1.0) <= 0.02, steps


def test_steps_reproduces_k54b_phase_switches():
    """C32 K54b: medians k1 0.2866 / k2 0.5103 / k4 0.9356, charged 1788 s (FF_TIME_TARGET 1793):
    2353 steps, k=1 from step 20, k=2 from 343, k=4 from 832 (campaign ledger, 27 Sep 12:05 UTC)."""
    st = {"k1": 0.2866, "k2": 0.5103, "k4": 0.9356}
    r = simulate_phases(st, 1788.0, time_target=1793.0)
    assert r["switches"][0] == (0, 4) and r["switches"][1] == (20, 1)
    assert abs(r["switches"][2][0] - 343) <= 6 and r["switches"][2][1] == 2
    assert abs(r["switches"][3][0] - 832) <= 10 and r["switches"][3][1] == 4
    assert abs(r["steps"] / 2353 - 1.0) <= 0.01
    assert r["charged_used"] <= 1788.0 < r["charged_used"] + st["k4"]
    assert sum(r["steps_by_k"].values()) == r["steps"]
    # The median -> mean overhead brings the full-run count within 0.3%.
    assert abs(steps_from_step_time(st, 1788.0, time_target=1793.0, overhead_frac=MEDIAN_TO_MEAN_OVERHEAD) / 2353 - 1.0) <= 0.003


def test_steps_single_phase_and_warmup_semantics():
    # '4': every step at k4; steps 0-4 are free, then floor(1790 / 0.928) charged steps.
    assert steps_from_step_time({"k4": 0.928}, 1790.0, "4") == 5 + math.floor(1790.0 / 0.928)
    # Warm-up runs at k = K, so k1 is only needed from step 20 on; with charged time that ends
    # inside the warm-up the k1/k2 values are never consulted.
    assert simulate_phases({"k1": 0.3, "k2": 0.5, "k4": 1.0}, 10.0)["steps"] == 15
    # Two-phase schedule (K50 era): k2 until 12%, then k4.
    r = simulate_phases({"k2": 0.5, "k4": 1.0}, 1000.0, "2:0.12,4")
    assert [k for _, k in r["switches"]] == [4, 2, 4]
    assert r["switches"][1][0] == 20


def test_steps_missing_phase_or_bad_budget_raises():
    with pytest.raises(ValueError):
        steps_from_step_time({"k4": 0.928}, 1790.0)  # k1/k2 needed by the default schedule
    with pytest.raises(ValueError):
        steps_from_step_time({"k1": 0.3, "k2": 0.5, "k4": 0.0}, 1790.0)
    with pytest.raises(ValueError):
        steps_from_step_time({"k1": 0.3, "k2": 0.5, "k4": 0.9}, 0.0)


def test_steps_monotone_in_step_time():
    base = {"k1": 0.30, "k2": 0.51, "k4": 0.928}
    n0 = steps_from_step_time(base, 1790.0)
    slower = {k: v * 1.01 for k, v in base.items()}
    n1 = steps_from_step_time(slower, 1790.0)
    assert n1 < n0 and abs((n0 - n1) / n0 - 0.01) < 0.002  # 1% slower -> ~1% fewer steps


# ----------------------------------------------------------------------------------------------
# Harvested data (skip when absent): the numbers in research/sim-data/steptime-validation.md
# ----------------------------------------------------------------------------------------------
def test_real_fit_skips_placeholder_and_probes(real_model):
    model, recs = real_model
    assert model.skipped["excluded_version"] == sum(1 for r in recs if r.code_version == "code")
    assert model.skipped["low_n"] >= 1                      # 5-step probes / partial logs
    fs = model.phase_fit_summary()["k4"]
    assert fs["n"] >= 150 and fs["rmse_pct"] < 0.6
    assert {"M9", "M12"} <= set(fs["versions"]) and "code" not in fs["versions"]


def test_real_hard_lessons_from_the_data(real_model):
    """RF -1.8% (contract) vs its control lineage; M11c max +5.9% vs M10b abs +3.4% (steps-based;
    +6.4% / +3.7% on the k4 median); XSA-all ~+10%; chip D +12% full / +1% screen."""
    model, _ = real_model
    ref = model.predict("M9", {}, "C")["k4"]
    assert ref == pytest.approx(0.929, abs=0.004)          # K57 recipe full-run k4 on chip C
    rf = model.predict("M12", {"FF_RELU2_FN": "1"}, "C")["k4"] / ref - 1.0
    assert -0.025 <= rf <= -0.015, rf
    mx = model.predict("M11", {"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "max"}, "C")["k4"] / ref - 1.0
    ab = model.predict("M10", {"FF_LEAKY_RELU2": "0.5", "FF_LEAKY_FORM": "abs"}, "C")["k4"] / ref - 1.0
    assert 0.055 <= mx <= 0.070 and 0.030 <= ab <= 0.042 and mx > ab, (mx, ab)
    xa = model.predict("M10", {"FF_XSA_LAYERS": "all"}, "C")["k4"] / ref - 1.0
    assert 0.085 <= xa <= 0.105, xa
    d_full = model.predict("M12", {"FF_LEAKY_RELU2": "0.35"}, "D")["k4"] / model.predict("M12", {"FF_LEAKY_RELU2": "0.35"}, "C")["k4"]
    d_scr = model.predict("M12", {}, "D", is_screen=True)["k4"] / model.predict("M12", {}, "C", is_screen=True)["k4"]
    assert 1.11 <= d_full <= 1.14 and 1.0 <= d_scr <= 1.02, (d_full, d_scr)
    # QKV fusion and the value-embedding gate are NOT in the chip C/D data: flagged, not invented.
    assert "fuse_qkv" in model.predict("M12", {"FF_FUSE_QKV": "1"}, "C")["notes"]
    assert "value_embed_gate" in model.predict("M12", {"FF_VALUE_EMBED": "3"}, "C")["notes"]


def test_real_k59_step_count(real_model):
    """predict('M12', K59 knobs, 'C') -> steps within 1% of C40 (2357 @ 1787.1 s charged, target
    1793) and of the LK0.35 seeds (2322-2329 @ ~1754 s charged, target 1760)."""
    model, _ = real_model
    k59 = {"FF_LEAKY_RELU2": "0.35", "FF_LEAKY_FORM": "fnm", "FF_ACC_IN_GRAPH": "1", "FF_MUON_ZERO": "1",
           "FF_EMA": "1", "FF_EMA_EVERY": "32", "FF_FUSED_CE": "8", "FF_DEPTH": "9"}
    p = model.predict("M12", k59, "C")
    assert p["notes"] == "" and p["version_used"] == "M12"
    assert abs(p["k4"] / 0.927 - 1.0) < 0.01 and p["k1"] < p["k2"] < p["k4"]
    steps_c40 = steps_from_step_time(p, 1787.1, time_target=1793.0, overhead_frac=MEDIAN_TO_MEAN_OVERHEAD)
    assert abs(steps_c40 / 2357 - 1.0) < 0.01, steps_c40
    steps_seed = steps_from_step_time(p, 1754.3, time_target=1760.0, overhead_frac=MEDIAN_TO_MEAN_OVERHEAD)
    assert abs(steps_seed / 2325 - 1.0) < 0.01, steps_seed


def test_real_loo_reachable_both_conventions(real_model):
    """Leave-one-lineage-out on k4 over the runs whose mechanisms another lineage measured. The
    lineage-averaged error (the mean of 13 per-lineage means) is 0.43%, under the 0.5% target; the
    run-weighted error over the same 161 runs is 0.545% (60 of them above 0.5%) and MISSES it,
    because the two biggest lineages (M9 0.51%, M12 0.66%) are the hard ones. `passes` follows the
    run-weighted number; both are pinned so neither can be quoted as the other (review F2)."""
    model, _ = real_model
    rep = model.loo_report("k4")
    assert rep["n_versions"] >= 10 and rep["n_novel"] > 0
    assert 150 <= rep["n_reachable"] == rep["n_runs"] - rep["n_novel"] <= 175
    assert rep["mean_abs_pct_reachable_by_lineage"] == pytest.approx(0.43, abs=0.05)
    assert rep["mean_abs_pct_reachable_by_run"] == pytest.approx(0.545, abs=0.05)
    assert rep["mean_abs_pct_reachable"] == rep["mean_abs_pct_reachable_by_run"]
    assert rep["judged_on"] == "mean_abs_pct_reachable_by_run"
    assert rep["mean_abs_pct_reachable_by_run"] > 0.5 and not rep["passes"], rep["mean_abs_pct_reachable_by_run"]
    assert rep["n_reachable_above_target"] >= 0.3 * rep["n_reachable"]
    assert rep["rmse_pct_reachable"] == pytest.approx(0.73, abs=0.05)
    assert rep["mean_abs_pct_by_lineage"] == pytest.approx(1.02, abs=0.1)
    assert rep["mean_abs_pct_by_run"] == pytest.approx(0.95, abs=0.1)
    assert rep["mean_abs_pct"] > rep["mean_abs_pct_reachable"]   # the novel runs are the honest gap
    m12 = rep["per_version"]["M12"]
    assert "relu2_fn" in m12["novel_features"] and m12["fallback_to"] in ("M11", "M13")
    m10 = rep["per_version"]["M10"]
    assert "xsa_layers" in m10["novel_features"] and m10["mean_abs_pct_reachable"] <= 0.5

def test_real_full_run_medians_exclude_partial_logs(real_model):
    """steptime-validation.md sections 4-5 direct numbers (review DF-8): a 'full run' median never
    includes a `log_incomplete` partial log (D:1630_R_lk35sc165_s73: 63 k4 lines, no ff_summary), so
    the chip-D `code_k12off` full-run k4 median is 1.0461 over n=7 (1.04545 over n=8 with it), the
    seven chip-C LK0.35 seeds sit at 0.9270 (ratio 1.1285), and the 96-parameter gate pair is F68
    (FF_VALUE_EMBED=3) against its single true control C7_cold_K26, +7.65%: the other two K26 runs
    are arms (an early-EMA probe and the unet arm), not controls."""
    _, recs = real_model
    d_full = [r for r in recs if r.chip == "D" and not r.is_screen
              and version_key(r.code_version or "") == version_key("code_k12off")]
    partial = [r for r in d_full if "log_incomplete" in r.tags]
    assert [r.run for r in partial] == ["1630_R_lk35sc165_s73"] and partial[0].step_time.n_k4 < 100
    clean = sorted(r.step_time.k4 for r in d_full if "log_incomplete" not in r.tags)
    assert len(clean) == 7 and clean[0] == pytest.approx(1.0359, abs=1e-4) and clean[-1] == pytest.approx(1.0526, abs=1e-4)
    assert statistics.median(clean) == pytest.approx(1.0461, abs=1e-4)
    assert statistics.median([r.step_time.k4 for r in d_full]) == pytest.approx(1.04545, abs=1e-4)  # n=8 would read this
    c_seeds = sorted(r.step_time.k4 for r in recs if r.chip == "C" and not r.is_screen and r.submission is None
                     and version_key(r.code_version or "") == version_key("code_k12off")
                     and r.knobs.get("FF_LEAKY_RELU2") == "0.35" and re.fullmatch(r"\d{4}_R_g7lk35_s\d+", r.run))
    assert len(c_seeds) == 7 and statistics.median(c_seeds) == pytest.approx(0.9270, abs=1e-4)
    assert statistics.median(clean) / statistics.median(c_seeds) == pytest.approx(1.1285, abs=0.001)
    k26 = {r.run: r.step_time.k4 for r in recs if r.chip == "C" and r.code_version == "code" and r.run.endswith("_K26")}
    assert set(k26) == {"0403_v12_early_ema_K26", "0427_F68_ve3_K26", "0507_F69_unet_K26", "C7_cold_K26"}
    assert k26["0427_F68_ve3_K26"] / k26["C7_cold_K26"] - 1.0 == pytest.approx(0.0765, abs=0.0005)
    # the old "controls median 1.8489" was C7 alone: the median of the three non-F68 runs is C7
    assert statistics.median([k26[k] for k in k26 if k != "0427_F68_ve3_K26"]) == k26["C7_cold_K26"]
