"""Tests for ffsim.gpu.fleet (plan / run / report / from-search) and ffsim.gpu.calibrate.

No torch, no GPU, no AWS: train_gpu.py is replaced by a fake runner, results are synthetic."""
from __future__ import annotations

import csv
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from ffsim.gpu import calibrate as cal
from ffsim.gpu import fleet

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "ffsim" / "gpu" / "configs"
PAIRS_JSON = ROOT / "research" / "sim-data" / "validation-pairs.json"
GATE = CONFIGS / "gate-k60.json"
PRESCREEN = CONFIGS / "prescreen.json"
CANDIDATES = CONFIGS / "candidates-from-ffsim.json"

# "true" GPU-side arm effects (bpb, relative to k60) used by the synthetic faithful proxy;
# Trainium equal-step deltas are A_TRUE times the GPU deltas.
A_TRUE = 1.25
ARM_EFFECT = {"k60": 0.0, "repeat": 0.0, "repeat2": 0.0, "relu2": 0.0014, "rf": 0.0013, "lk05": 0.0007,
              "as0": 0.0017, "asb": 0.0009, "mp97": 0.0019, "mp93": 0.0004, "ko01": 0.0004,
              "noramp": -0.0031, "r3p30": 0.0035, "qk16": 0.0001, "wu40": -0.0002, "wds1": 0.0,
              "sc165": -0.0001, "elr36": 0.0, "abs_base": 0.0017, "abs_asa": 0.0}
SEED_EFFECT = {73: 0.0, 58: 0.0002, 67: -0.0003, 53: 0.0004}


def synthetic_results(config, tmp_path: Path, flip: bool = False, noise: float = 0.00015,
                      drop_labels=(), fail_labels=(), name="results.jsonl"):
    rng = np.random.default_rng(1)
    results = tmp_path / name
    manifest = tmp_path / (results.stem + "-manifest.jsonl")
    with results.open("w", encoding="utf-8") as fr, manifest.open("w", encoding="utf-8") as fm:
        for r in config["runs"]:
            if r["label"] in drop_labels:
                continue
            if r["label"] in fail_labels:
                fm.write(json.dumps({"label": r["label"], "arm": r["arm"], "seed": r["seed"], "steps": r["steps"],
                                     "status": "exit 1", "gpu": 0}) + "\n")
                fr.write(json.dumps({"label": r["label"], "arm": r["arm"], "steps": r["steps"], "val_bpb": None,
                                     "config": {"seed": r["seed"]}}) + "\n")
                continue
            eff = ARM_EFFECT[r["arm"]] / A_TRUE * (-1.0 if flip else 1.0)
            bpb = 0.9600 + SEED_EFFECT[r["seed"]] + eff + rng.normal(0.0, noise)
            fm.write(json.dumps({"label": r["label"], "arm": r["arm"], "seed": r["seed"], "steps": r["steps"],
                                 "status": "ok", "gpu": 0, "seconds": 100.0}) + "\n")
            fr.write(json.dumps({"label": r["label"], "arm": r["arm"], "steps": r["steps"], "val_bpb": round(bpb, 6),
                                 "config": {"seed": r["seed"]}, "median_step_seconds": 4.2,
                                 "eval_num_tokens": 2097152, "eval_num_bytes": 8387440, "eval_denominator_ok": True}) + "\n")
    return results, manifest


# ----------------------------------------------------------------------------- configs

@pytest.mark.parametrize("path", [GATE, PRESCREEN, CANDIDATES])
def test_configs_load_and_are_consistent(path):
    cfg = fleet.load_config(path)
    runs = cfg["runs"]
    assert runs and len({r["label"] for r in runs}) == len(runs)
    for r in runs:
        assert r["env"]["FF_SEED"] == str(r["seed"])
        assert r["steps"] > 0 and r["arm"]
        assert "--data-world" in r["args"] and "--eval-tokens" in r["args"]
    arms = {r["arm"] for r in runs}
    keyset = {(r["arm"], r["seed"], r["steps"]) for r in runs}
    for p in cfg["pairs"]:
        assert p["treatment_arm"] in arms and p["control_arm"] in arms, p["pair_id"]
        matched = [(a, s, st) for (a, s, st) in keyset if a == p["treatment_arm"]
                   and (p["control_arm"], s, st) in keyset]
        assert matched, f"{p['pair_id']} has no matched-seed pair of runs"


def test_gate_config_is_built_from_real_trainium_pairs():
    cfg = fleet.load_config(GATE)
    idx = cal.load_validation_pairs(PAIRS_JSON)
    assert len(idx) == 99
    assert 35 <= len(cfg["runs"]) <= 45
    families = set()
    n_refs = 0
    for p in cfg["pairs"]:
        families.add(p["family"])
        refs = cal.trainium_reference(p, idx)
        assert refs and not any(r.get("missing") for r in refs), p["pair_id"]
        n_refs += len(refs)
        for r in refs:
            if r["source"] == "validation-pairs.json":
                src = idx[r["pair_id"]]
                assert src["same_chip"] and src["chip"] == src["control_chip"]
                assert math.isclose(r["delta_raw"], src["delta_raw"])
    assert {"leaky_slope", "relu2_fn", "attn_src", "mom_peak", "key_offset_rad", "accum_sched",
            "qk_gain", "warmup", "wd_sched", "softcap_asym", "lr"} <= families
    # F4: FF_ACC_IN_GRAPH is forced to 0 by train_gpu.py under --compile none, so k60 vs acc0 was a self-null
    assert "acc_in_graph" not in families and "acc0" not in {r["arm"] for r in cfg["runs"]}
    dargs = cfg["defaults"]["args"]
    assert dargs[dargs.index("--compile") + 1] == "none"
    assert not any("acc_in_graph" in r.get("pair", "").split(",") for r in cfg["runs"])
    assert n_refs >= 30
    # attention-source reuse: 4 Trainium AS pairs across asa + asb
    as_refs = sum(len(p["trainium"]) for p in cfg["pairs"] if p["family"] == "attn_src")
    assert as_refs == 4
    # the frozen rule (calibrate.FROZEN_ROWS) resolves completely on this config: 11 decisive + 9 null/tie rows
    rule = cal.freeze_rule(cfg["pairs"], idx, cal.config_keys(cfg))
    assert rule["unresolved"] == [] and rule["n_decisive"] == 11 and rule["n_null_tie"] == 9
    by_pair = {}
    for r in rule["rows"]:
        by_pair.setdefault(r["pair_id"], []).append(r["seed"])
    assert {p: sorted(s) for p, s in by_pair.items()} == {
        "lk35_vs_relu2": [58, 67, 73], "lk35_vs_rf": [58, 67, 73], "lk35_vs_lk05": [58, 73], "asa": [73],
        "mp97": [73], "mp93": [73], "rf_vs_relu2": [58, 67, 73], "wd_sched1": [58, 73], "softcap165": [58, 73],
        "qk16": [58, 73]}
    # ramp3 / r3p30 (they test the correction), asb (no step count) and the ambiguous pairs never gate
    assert all(rule["pair_roles"][p] == "reported" for p in ("ramp3", "r3p30", "asb", "ko03", "warmup40"))
    # asa gates on its chip-C s73 reference alone: the chip-D row is reported, the s67 row has no step counts
    asa = next(r for r in rule["rows"] if r["pair_id"] == "asa")
    assert asa["trainium_refs"] == ["asa_C_s73"] and asa["delta_eq"] < -0.0015
    asa_refs = cal.trainium_reference(next(p for p in cfg["pairs"] if p["pair_id"] == "asa"), idx)
    assert {r["pair_id"]: r["gates"] for r in asa_refs} == {"m14_asa_D_s73": False, "asa_C_s73": True, "asa_C_s67": False}
    # the equal-step correction turns RF (a throughput lever) into a null pair
    assert cal.pair_role("rf_vs_relu2") == "null"
    assert not cal.is_decisive(cal.trainium_reference(next(p for p in cfg["pairs"] if p["pair_id"] == "rf_vs_relu2"), idx))[0]
    # the digest is a function of config + Trainium data + the code's thresholds only
    assert rule["digest"] == cal.freeze_rule(cfg["pairs"], idx, cal.config_keys(cfg))["digest"]
    assert rule["thresholds"]["decisive_abs"] == 0.0003 and rule["thresholds"]["scale_band"] == [0.5, 2.0]
    assert rule["thresholds"]["sign_agreement_min"] == 0.8 and rule["thresholds"]["tie_max_outside"] == 2
    # arm env: the pair's own knobs on top of the K60 base; rf resets LEAKY to the pair's base
    by_arm = {r["arm"]: r["env"] for r in cfg["runs"]}
    assert {k: v for k, v in by_arm["k60"].items() if k != "FF_SEED"} == {}
    assert by_arm["rf"]["FF_RELU2_FN"] == "1" and by_arm["rf"]["FF_LEAKY_RELU2"] == "0"
    assert by_arm["relu2"] == {"FF_LEAKY_RELU2": "0", "FF_RELU2_FN": "0", "FF_SEED": by_arm["relu2"]["FF_SEED"]}
    assert by_arm["as0"]["FF_ATTN_SRC"] == ""
    assert by_arm["noramp"]["FF_ACCUM_SCHED"] == "2:0.2,4"
    # nulls pinned: same GPU as k60_s73, other GPU for repeat2
    pins = {r["label"]: r.get("gpu") for r in cfg["runs"]}
    assert pins["k60_s73"] == 0 and pins["k60_s73_repeat"] == 0 and pins["k60_s73_repeat2"] == 1


def test_gate_is_mode_a_at_2314_plus_two_absolute_level_runs():
    """F2: the fleet runs mode (a) (one shared K60 triple, --steps 2314 for every pair run) and the config says so;
    gate 2 (absolute level) gets its own base s73 @ 2357 and as_a s73 @ 2361 runs outside `pairs`."""
    raw = json.loads(GATE.read_text(encoding="utf-8"))
    cfg = fleet.load_config(GATE)
    assert raw["defaults"]["steps"] == 2314 and "MODE (a)" in raw["_steps_policy"] and "MODE (a)" in raw["_comment"]
    assert "step-seconds-from-log" in raw["_steps_policy"] and "SMOKE TEST" in raw["_steps_policy"]
    by_label = {r["label"]: r for r in cfg["runs"]}
    abs_runs = raw["_absolute_level"]["runs"]
    assert set(abs_runs) == {"abs_base_s73", "abs_asa_s73"}
    idx = cal.load_validation_pairs(PAIRS_JSON)
    c40 = idx["reh_k59_vs_k57t"]
    assert abs_runs["abs_base_s73"]["steps"] == c40["treatment_steps"] == 2357
    assert abs_runs["abs_base_s73"]["target_bpb"] == c40["treatment_bpb"] == 0.96092
    assert abs_runs["abs_asa_s73"]["steps"] == fleet.K60_STEPS == 2361
    assert abs_runs["abs_asa_s73"]["target_bpb"] == 0.95910
    for label, spec in abs_runs.items():
        r = by_label[label]
        assert r["arm"] == spec["arm"] and r["steps"] == spec["steps"] and r["seed"] == 73
        assert {k: v for k, v in r["env"].items() if k != "FF_SEED"} == spec["env"]
        assert spec["pass_within"] == 0.01 and spec["structural_beyond"] == 0.03
    env_of = {r["arm"]: {k: v for k, v in r["env"].items() if k != "FF_SEED"} for r in cfg["runs"]}
    assert env_of["abs_base"] == env_of["as0"] == {"FF_ATTN_SRC": ""} and env_of["abs_asa"] == env_of["k60"] == {}
    # every pair run stays at 2314 (mode a); only the two gate-2 runs differ, and no pair uses their arms
    assert {r["steps"] for r in cfg["runs"] if r["arm"] not in ("abs_base", "abs_asa")} == {2314}
    pair_arms = {p["treatment_arm"] for p in cfg["pairs"]} | {p["control_arm"] for p in cfg["pairs"]}
    assert not ({"abs_base", "abs_asa"} & pair_arms)


def test_absolute_level_runs_stay_out_of_calibration(tmp_path):
    cfg = fleet.load_config(GATE)
    results, manifest = synthetic_results(cfg, tmp_path)
    index = fleet.merge_results(results, manifest, cfg)
    assert {"abs_base_s73", "abs_asa_s73"} <= set(index)
    c = cal.calibrate(index, cfg["pairs"], cal.load_validation_pairs(PAIRS_JSON))
    used = {lab for p in c["pairs"] for g in p.get("gpu_deltas", []) for lab in g.get("labels", [])}
    used |= {lab for g in c["noise"]["gpu_null_deltas"] for lab in g.get("labels", [])}
    assert not ({"abs_base_s73", "abs_asa_s73"} & used)
    assert c["noise"]["gpu_base_n"] == 4      # the 2361-step K60-env run is not in the k60 seed spread


def test_prescreen_is_300_steps_subset_of_gate():
    gate = fleet.load_config(GATE)
    pre = fleet.load_config(PRESCREEN)
    assert all(r["steps"] == 300 for r in pre["runs"])
    assert len(pre["pairs"]) == 6 and len(pre["runs"]) <= 10
    gate_pairs = {p["pair_id"] for p in gate["pairs"]}
    assert {p["pair_id"] for p in pre["pairs"]} <= gate_pairs
    gate_env = {r["arm"]: {k: v for k, v in r["env"].items() if k != "FF_SEED"} for r in gate["runs"]}
    for r in pre["runs"]:
        assert {k: v for k, v in r["env"].items() if k != "FF_SEED"} == gate_env[r["arm"]]


def test_duplicate_label_refused(tmp_path):
    p = tmp_path / "dup.json"
    p.write_text(json.dumps({"runs": [{"label": "a", "arm": "x"}, {"label": "a", "arm": "y"}]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        fleet.load_config(p)


# ----------------------------------------------------------------------------- commands / env

def test_build_command_virtual_clock(tmp_path):
    cfg = fleet.load_config(GATE)["runs"][0]
    cmd = fleet.build_command(cfg, "python", tmp_path / "r.jsonl", tmp_path / "runs", (0.295, 0.506, 0.928))
    s = " ".join(cmd)
    assert "--clock virtual" in s and "--step-seconds k1=0.2950,k2=0.5060,k4=0.9280" in s
    assert f"--steps {cfg['steps']}" in s and "--label k60_s73" in s and "--arm k60" in s
    assert cmd.count("--data-world") == 1 and str(fleet.TRAIN_GPU) in cmd


def test_parse_step_seconds_triple_default_and_models():
    trip, how = fleet.parse_step_seconds("0.3,0.5,0.9")
    assert trip == (0.3, 0.5, 0.9) and "explicit" in how
    assert fleet.parse_step_seconds("default")[0] == fleet.DEFAULT_STEP_SECONDS
    if fleet.DEFAULT_MODELS.exists():
        trip, how = fleet.parse_step_seconds("models.pkl", knobs={"FF_ATTN_SRC": "5:6,7,8"})
        assert 0.85 < trip[2] < 1.0 and trip[0] < trip[1] < trip[2] and "steptime.predict" in how
    n = fleet.virtual_steps(fleet.DEFAULT_STEP_SECONDS)
    assert 2300 < n < 2500


def test_strip_env_uses_gpu_env_when_present_and_fallback_otherwise(monkeypatch):
    env = {"FF_SEED": "73", "FF_SCHEDULE": "time", "NEURON_RT_X": "1", "FF_CC_FLAGS": "--foo"}
    fake = types.ModuleType("ffsim.gpu.gpu_env")
    fake.strip = lambda e: ({k: v for k, v in e.items() if k == "FF_SEED"}, ["ALL_BUT_SEED"])
    monkeypatch.setitem(sys.modules, "ffsim.gpu.gpu_env", fake)
    kept, removed, how = fleet.strip_env(env)
    assert kept == {"FF_SEED": "73"} and removed == ["ALL_BUT_SEED"] and "gpu_env" in how
    monkeypatch.setitem(sys.modules, "ffsim.gpu.gpu_env", None)   # import -> ImportError
    kept, removed, how = fleet.strip_env(env)
    assert "FALLBACK" in how
    assert kept == {"FF_SEED": "73", "FF_SCHEDULE": "time"}       # virtual clock keeps FF_SCHEDULE
    assert sorted(removed) == ["FF_CC_FLAGS=--foo", "NEURON_RT_X=1"]


def test_assign_lanes_honours_pins():
    runs = [{"label": "a", "gpu": 0}, {"label": "b"}, {"label": "c"}, {"label": "d", "gpu": 0}, {"label": "e", "gpu": 1}]
    lanes = fleet.assign_lanes(runs, 2)
    assert [r["label"] for r in lanes[0]] == ["a", "b", "d"]
    assert [r["label"] for r in lanes[1]] == ["c", "e"]
    # F5: a config pinned to a GPU the fleet lacks is never folded onto gpu 0 (repeat2, the
    # cross-device null, would otherwise run on the same GPU and be read as the across-GPU null)
    skipped = []
    lanes1 = fleet.assign_lanes(runs, 1, skipped)
    assert len(lanes1) == 1 and [r["label"] for r in lanes1[0]] == ["a", "b", "c", "d"]
    assert [r["label"] for r in skipped] == ["e"] and "needs gpu 1" in fleet.skip_reason(runs[4], 1)
    assert fleet.skip_reason(runs[0], 1) == "" and fleet.skip_reason(runs[1], 1) == ""
    assert len(fleet.assign_lanes(runs, 1)[0]) == 4                     # dropped even without a skipped list


def test_fallback_strip_keeps_prepare_allowlist(monkeypatch):
    monkeypatch.setitem(sys.modules, "ffsim.gpu.gpu_env", None)
    kept, removed, how = fleet.strip_env({"NEURON_COMPETITION_R1_CACHE_DIR": "/home/u/ff/cache",
                                          "NEURON_COMPETITION_R1_NUM_STEPS": "100", "NEURON_RT_LOG": "x"})
    assert "FALLBACK" in how and kept == {"NEURON_COMPETITION_R1_CACHE_DIR": "/home/u/ff/cache"}
    assert sorted(removed) == ["NEURON_COMPETITION_R1_NUM_STEPS=100", "NEURON_RT_LOG=x"]


@pytest.mark.parametrize("use_gpu_env", [True, False])
def test_inherited_cache_dir_reaches_subprocess(tmp_path, monkeypatch, use_gpu_env):
    """F1: launch.sh's env.sh exports NEURON_COMPETITION_R1_CACHE_DIR (where push put the shards);
    the fleet must pass it through (prepare.py reads it) while still stripping inherited Neuron junk."""
    if not use_gpu_env:
        monkeypatch.setitem(sys.modules, "ffsim.gpu.gpu_env", None)
    monkeypatch.setenv("NEURON_COMPETITION_R1_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NEURON_RT_LOG_LEVEL", "INFO")            # inherited Neuron-only: must go
    monkeypatch.setenv("FF_CC_FLAGS", "--model-type=transformer")  # inherited Neuron-only: must go
    monkeypatch.setenv("NEURON_COMPETITION_R1_NUM_STEPS", "5")   # refused by train_gpu.py: must go
    cfg = {"label": "k60_s73", "arm": "k60", "pair": "", "seed": 73, "steps": 10,
           "env": {"FF_SEED": "73", "NEURON_RT_X": "1"}, "args": []}
    seen = {}

    def runner(cmd, env, fh):
        seen.update(env)
        return 0

    rec = fleet.run_one(cfg, 0, tmp_path / "runs", tmp_path / "res.jsonl", tmp_path / "man.jsonl",
                        "python", (0.3, 0.5, 0.9), [], runner=runner)
    assert rec["status"] == "ok"
    assert rec["strip_how"].startswith("ffsim.gpu.gpu_env") is use_gpu_env
    assert ("FALLBACK" in rec["strip_how"]) is (not use_gpu_env)
    assert seen["NEURON_COMPETITION_R1_CACHE_DIR"] == str(tmp_path / "cache")
    assert seen["FF_SEED"] == "73" and seen["CUDA_VISIBLE_DEVICES"] == "0"
    for junk in ("NEURON_RT_LOG_LEVEL", "FF_CC_FLAGS", "NEURON_COMPETITION_R1_NUM_STEPS", "NEURON_RT_X"):
        assert junk not in seen
    log = (tmp_path / "runs" / "k60_s73" / "train.log").read_text()
    assert "stripped inherited Neuron-only env:" in log and "NEURON_RT_LOG_LEVEL" in log
    assert "NEURON_COMPETITION_R1_CACHE_DIR" not in log.split("stripped inherited")[1].splitlines()[0]


def test_run_skips_pins_to_absent_gpu(tmp_path, monkeypatch):
    """F5 end to end: on a 1-GPU fleet the gpu-1 null is skipped (logged + manifest row), never run on gpu 0."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"defaults": {"steps": 10}, "runs": [
        {"label": "k60_s73", "arm": "k60", "seed": 73, "gpu": 0},
        {"label": "k60_s73_repeat", "arm": "repeat", "pair": "null_repeat", "seed": 73, "gpu": 0},
        {"label": "k60_s73_repeat2", "arm": "repeat2", "pair": "null_repeat2", "seed": 73, "gpu": 1},
    ]}), encoding="utf-8")
    results = tmp_path / "res.jsonl"
    seen = []

    class Proc:
        returncode = 0

    def fake_run(cmd, env=None, stdout=None, stderr=None, **kw):
        seen.append((cmd[cmd.index("--label") + 1], env.get("CUDA_VISIBLE_DEVICES")))
        stdout.write("fake\n")
        return Proc()

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    monkeypatch.setattr(fleet, "TRAIN_GPU", tmp_path / "train_gpu.py")
    (tmp_path / "train_gpu.py").write_text("# stub\n", encoding="utf-8")
    rc = fleet.main(["run", "--configs", str(cfg_path), "--results", str(results),
                     "--out-root", str(tmp_path / "runs"), "--gpus", "1", "--step-seconds", "0.3,0.5,0.9"])
    assert rc == 0                                        # a skip is not a failure
    assert seen == [("k60_s73", "0"), ("k60_s73_repeat", "0")]
    rows, bad = fleet.read_jsonl(tmp_path / "res-manifest.jsonl")
    by = {r["label"]: r for r in rows}
    assert bad == 0 and set(by) == {"k60_s73", "k60_s73_repeat", "k60_s73_repeat2"}
    assert by["k60_s73_repeat2"]["status"].startswith("skipped: needs gpu 1") and by["k60_s73_repeat2"]["gpu"] == 1
    assert not (tmp_path / "runs" / "k60_s73_repeat2").exists()
    # the skipped null never becomes a result row, so report shows it as not-ok and calibrate ignores it
    index = fleet.merge_results(results, tmp_path / "res-manifest.jsonl", fleet.load_config(cfg_path))
    assert index["k60_s73_repeat2"]["status"].startswith("skipped")
    # on a 2-GPU fleet the same config runs it on gpu 1
    seen.clear()
    rc = fleet.main(["run", "--configs", str(cfg_path), "--results", str(tmp_path / "res2.jsonl"),
                     "--out-root", str(tmp_path / "runs2"), "--gpus", "2", "--step-seconds", "0.3,0.5,0.9"])
    assert rc == 0 and sorted(seen) == [("k60_s73", "0"), ("k60_s73_repeat", "0"), ("k60_s73_repeat2", "1")]


def test_run_records_failures_and_continues(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"defaults": {"steps": 10, "args": ["--data-world", "8"]}, "runs": [
        {"label": "ok_s1", "arm": "base", "seed": 1, "env": {"NEURON_RT_LOG": "x"}},
        {"label": "bad_s1", "arm": "t", "seed": 1, "env": {"FF_X": "1"}},
        {"label": "ok2_s2", "arm": "base", "seed": 2},
    ]}), encoding="utf-8")
    results = tmp_path / "res.jsonl"
    seen = []

    class Proc:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, env=None, stdout=None, stderr=None, **kw):
        label = cmd[cmd.index("--label") + 1]
        seen.append((label, env.get("CUDA_VISIBLE_DEVICES"), env.get("FF_SEED"), "NEURON_RT_LOG" in env))
        stdout.write("fake train_gpu\n")
        if label.startswith("bad"):
            return Proc(3)
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"label": label, "arm": cmd[cmd.index("--arm") + 1], "steps": 10,
                                 "val_bpb": 1.0, "config": {"seed": int(env["FF_SEED"])}}) + "\n")
        return Proc(0)

    monkeypatch.setattr(fleet.subprocess, "run", fake_run)
    monkeypatch.setattr(fleet, "TRAIN_GPU", tmp_path / "train_gpu.py")
    (tmp_path / "train_gpu.py").write_text("# stub\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "ffsim.gpu.gpu_env", None)
    rc = fleet.main(["run", "--configs", str(cfg_path), "--results", str(results), "--out-root", str(tmp_path / "runs"),
                     "--gpus", "2", "--step-seconds", "0.3,0.5,0.9"])
    assert rc == 1                                    # a failure is reported, not fatal
    assert sorted(l for l, *_ in seen) == ["bad_s1", "ok2_s2", "ok_s1"]
    assert all(not has_neuron for *_, has_neuron in seen)
    man = tmp_path / "res-manifest.jsonl"
    rows, bad = fleet.read_jsonl(man)
    assert bad == 0 and {r["label"]: r["status"] for r in rows} == {"ok_s1": "ok", "bad_s1": "exit 3", "ok2_s2": "ok"}
    assert (tmp_path / "runs" / "bad_s1" / "train.log").exists()
    assert "stripped Neuron-only env: NEURON_RT_LOG=x" in (tmp_path / "runs" / "ok_s1" / "train.log").read_text()
    # dry run touches nothing
    n_before = len(seen)
    assert fleet.main(["run", "--configs", str(cfg_path), "--results", str(tmp_path / "x.jsonl"), "--dry-run"]) == 0
    assert len(seen) == n_before and not (tmp_path / "x.jsonl").exists()


# ----------------------------------------------------------------------------- plan / cost

def test_plan_prints_runs_hours_and_dollars_without_running(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert fleet.main(["plan", "--configs", str(GATE), "--seconds-per-step", "4.0", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "k60_s73" in text and "GPU-hours" in text and "g6e.48xlarge" in text and "Nothing above was launched" in text
    plan = json.loads(out.read_text(encoding="utf-8"))
    n = len(plan["runs"])
    hours = sum((r["steps"] * 4.0 + fleet.EVAL_SECONDS + fleet.STARTUP_SECONDS) / 3600 for r in plan["runs"])
    by_steps = {int(k): v for k, v in plan["cost"]["by_steps"].items()}
    assert by_steps == {2314: n - 2, 2357: 1, 2361: 1}
    assert math.isclose(plan["cost"]["gpu_hours"], hours, rel_tol=1e-3)     # the plan json rounds gpu_hours
    inst = {r["instance"]: r for r in plan["cost"]["instances"]}
    assert inst["g6e.xlarge"]["usd_per_hour"] == 1.861 and inst["g6e.12xlarge"]["gpus"] == 4
    assert inst["g6e.xlarge"]["waves"] == n and inst["g6e.48xlarge"]["waves"] == math.ceil(n / 8)
    assert math.isclose(inst["g6e.xlarge"]["usd"], inst["g6e.xlarge"]["wall_hours"] * 1.861, rel_tol=1e-3)
    assert plan["cost"]["price_source"].startswith("aws pricing")
    quote = fleet.cost_table(plan["runs"], 4.0, price_source="quote")
    assert quote["price_source"] == fleet.PRICE_SOURCE_FALLBACK
    assert {r["instance"]: r["usd_per_hour"] for r in quote["instances"]}["g6e.12xlarge"] == 10.4912
    assert not list(tmp_path.glob("runs*"))


# ----------------------------------------------------------------------------- from-search

def test_from_search_emits_paired_candidates(tmp_path):
    csv_path = tmp_path / "search-test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "name", "changes", "official_mean", "official_sd", "delta_vs_base",
                                           "p_beat_base", "steps_mean"])
        w.writeheader()
        w.writerow({"rank": 1, "name": "A", "changes": "FF_A=1 FF_B=0.3", "official_mean": "0.9660", "official_sd": "0.0009",
                    "delta_vs_base": "-0.0001", "p_beat_base": "0.6", "steps_mean": "2364"})
        w.writerow({"rank": 2, "name": "A-dup", "changes": "FF_A=1 FF_B=0.3", "official_mean": "0.9661"})
        w.writerow({"rank": 3, "name": "B", "changes": "FF_C=2", "official_mean": "0.9662"})
        w.writerow({"rank": 4, "name": "bad", "changes": "FF_D=1", "official_mean": ""})
        w.writerow({"rank": 5, "name": "C", "changes": "FF_E=x", "official_mean": "0.9659"})
    out = tmp_path / "cand.json"
    assert fleet.main(["from-search", "--search", str(csv_path), "--top-k", "2", "--seeds", "73,58", "--steps", "2361",
                       "--out", str(out)]) == 0
    cfg = fleet.load_config(out)
    arms = [r["arm"] for r in cfg["runs"]]
    assert arms.count("k60") == 2 and arms.count("cand01") == 2 and arms.count("cand02") == 2 and len(arms) == 6
    assert len(cfg["pairs"]) == 2 and cfg["pairs"][0]["ffsim"]["name"] == "C"        # best official_mean first
    assert cfg["pairs"][1]["ffsim"]["changes"] == "FF_A=1 FF_B=0.3"                    # duplicate collapsed
    env = {r["arm"]: r["env"] for r in cfg["runs"]}
    assert env["cand02"]["FF_A"] == "1" and env["cand02"]["FF_B"] == "0.3" and env["cand01"]["FF_E"] == "x"
    assert all(r["steps"] == 2361 for r in cfg["runs"]) and all(not p["gating"] for p in cfg["pairs"])
    assert fleet.parse_changes("FF_A=1+FF_B=2") == {"FF_A": "1", "FF_B": "2"}


def test_candidates_config_matches_search_csv():
    cfg = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    assert cfg["_generator"]["top_k"] == len(cfg["pairs"])
    for p in cfg["pairs"]:
        assert p["control_arm"] == "k60" and p["ffsim"]["changes"]
        assert fleet.parse_changes(p["ffsim"]["changes"]) == {
            k: v for k, v in next(r for r in cfg["runs"] if r["arm"] == p["treatment_arm"])["env"].items()}


# ----------------------------------------------------------------------------- results / report

def test_read_jsonl_and_merge_partial_results(tmp_path):
    cfg = fleet.load_config(PRESCREEN)
    results, manifest = synthetic_results(cfg, tmp_path, drop_labels=("mp97_s73",), fail_labels=("rf_s73",))
    with results.open("a", encoding="utf-8") as fh:
        fh.write('{"label": "truncated", "val_bpb": 0.9')      # a run killed mid-write
    rows, bad = fleet.read_jsonl(results)
    assert bad == 1 and len(rows) == len(cfg["runs"]) - 1
    index = fleet.merge_results(results, manifest, cfg)
    assert index["mp97_s73"]["status"] == "pending" and index["rf_s73"]["status"] == "exit 1"
    assert index["k60_s73"]["status"] == "ok" and index["k60_s73"]["seed"] == 73 and index["k60_s73"]["steps"] == 300
    assert fleet.merge_results(tmp_path / "missing.jsonl", None, None) == {}
    # seed recovered from the label when neither results nor manifest carry it
    (tmp_path / "bare.jsonl").write_text(json.dumps({"label": "x_s58", "arm": "x", "steps": 5, "val_bpb": 1.0}) + "\n")
    assert fleet.merge_results(tmp_path / "bare.jsonl")["x_s58"]["seed"] == 58


def test_report_lists_status(tmp_path, capsys):
    cfg = fleet.load_config(PRESCREEN)
    results, manifest = synthetic_results(cfg, tmp_path, fail_labels=("lk05_s73",))
    assert fleet.main(["report", "--results", str(results), "--configs", str(PRESCREEN)]) == 0
    out = capsys.readouterr().out
    assert "7 ok, 1 failed, 0 pending" in out and "lk05_s73" in out


# ----------------------------------------------------------------------------- calibrate statistics

def test_equal_step_delta_and_decisive():
    d, corrected = cal.equal_step_delta(-0.00037, 2287, 2150)
    assert corrected and 0.0028 < d < 0.0034                      # ramp3: a loss at equal steps
    assert cal.equal_step_delta(-0.001, None, 2150) == (-0.001, False)
    refs = [{"delta_eq": -0.0012, "delta_eq_range": [-0.0014, -0.0010]}]
    assert cal.is_decisive(refs)[0]
    assert cal.row_decisive(0.00042, 0.00045, 0.00048)[0]                        # mp93 at the PROTOCOL 0.0003 threshold
    assert not cal.row_decisive(0.00029, 0.00025, 0.00033)[0]
    assert not cal.is_decisive([{"delta_eq": -0.0003, "delta_eq_range": [-0.0004, -0.0002]}])[0]
    assert not cal.is_decisive([{"delta_eq": 0.0007, "delta_eq_range": [-0.0002, 0.0012]}])[0]
    assert not cal.is_decisive(refs, gating=False)[0]
    assert cal.is_decisive([{"missing": True}])[0] is False


def test_spearman_fit_and_normal_helpers():
    assert cal.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert cal.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert cal.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.866, abs=1e-3)
    assert cal.spearman([1, 2], [1, 2]) is None
    fit = cal.fit_scale([1.0, 2.0, 3.0, 4.0], [1.5, 3.0, 4.5, 6.0])
    assert fit["a"] == pytest.approx(1.5) and fit["rms_residual"] == pytest.approx(0.0) and fit["ci95"][0] == pytest.approx(1.5)
    assert cal.fit_scale([], [])["a"] is None
    assert cal._inv_norm_cdf(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert cal.norm_cdf(0.0) == pytest.approx(0.5)
    assert cal.pooled_sd([[1.0, 3.0], [5.0], [2.0, 4.0]]) == pytest.approx(math.sqrt(2.0))
    assert cal.pooled_sd([[1.0]]) is None


def test_faithful_proxy_is_accepted_and_predicts(tmp_path, capsys):
    cfg = fleet.load_config(GATE)
    results, manifest = synthetic_results(cfg, tmp_path)
    index = fleet.merge_results(results, manifest, cfg)
    c = cal.calibrate(index, cfg["pairs"], cal.load_validation_pairs(PAIRS_JSON))
    assert c["status"] == "accepted", c["reasons"]
    assert c["n_decisive_rows"] == 11 and c["n_decisive_with_data"] == 11 and c["sign_agreement_decisive"] >= 0.8
    assert [g["ok"] for g in c["gates"]] == [True, True, True, True, True, True, None]
    assert c["rule"]["unresolved"] == [] and c["pending"] == [] and c["rule"]["n_null_tie"] == 9
    # gate 2 reads the two dedicated absolute-level runs at their reference schedules, nothing else
    assert {a: (v["label"], v["steps"]) for a, v in c["gates"][2]["levels"].items()} == {
        "abs_asa": ("abs_asa_s73", 2361), "abs_base": ("abs_base_s73", 2357)}
    assert abs(c["scale"]["a"] - A_TRUE) < 0.3 and c["scale"]["ci95"][0] > 0
    assert c["spearman"] > 0.7
    nz = c["noise"]
    assert nz["gpu_pair_sd_pooled"] is not None and nz["gpu_pair_sd_pooled"] < 0.001
    assert nz["gpu_base_n"] == 4 and nz["gpu_null_worst_abs"] is not None and len(nz["gpu_null_deltas"]) == 2
    assert nz["trainium_run_sd"] == 0.0006 and nz["calibration_rms_residual"] is not None
    assert set(c["family_offsets"]) >= {"leaky_slope", "attn_src", "accum_sched"}
    text = cal.render(c)
    assert "ACCEPTED" in text and "lk35_vs_relu2" in text and "gate 3 sign agreement" in text and "0.0003" in text
    # prediction for a new candidate
    pred = cal.predict([-0.0020, -0.0018], c, family="leaky_slope", seconds_per_step=4.0)
    assert pred["p_trainium_delta_negative"] > 0.9 and pred["predicted_trainium_delta"] < -0.001
    assert pred["warning"] is None and pred["cost"] and pred["cost"][0]["seed_pairs"] is not None
    assert "g6e.xlarge" in pred["cost"][0]["usd"] or pred["cost"][0]["extra_runs"] == 0
    null = cal.predict([0.00001, -0.00002], c)
    assert 0.3 < null["p_trainium_delta_negative"] < 0.7
    assert "P(Trainium delta < 0)" in cal.render_prediction(null)
    # CLI: freeze the rule at plan time (no results needed), then verdict against it + json
    frozen = tmp_path / "rule.json"
    assert cal.main(["--freeze", str(frozen), "--configs", str(GATE), "--pairs", str(PAIRS_JSON)]) == 0
    rule = json.loads(frozen.read_text(encoding="utf-8"))
    assert rule["n_decisive"] == 11 and rule["digest"] == c["rule"]["digest"] and rule["written"]
    out = tmp_path / "cal.json"
    rc = cal.main(["--results", str(results), "--configs", str(GATE), "--pairs", str(PAIRS_JSON), "--json", str(out),
                   "--rule", str(frozen), "--predict=-0.002,-0.0015", "--family", "attn_src"])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0 and saved["prediction"]["p_trainium_delta_negative"] > 0.8
    assert saved["calibration"]["rule"]["digest"] == rule["digest"] and len(saved["calibration"]["rule"]["rows"]) == 20
    assert "ACCEPTED" in capsys.readouterr().out


def test_inverted_proxy_is_rejected(tmp_path):
    cfg = fleet.load_config(GATE)
    results, manifest = synthetic_results(cfg, tmp_path, flip=True)
    index = fleet.merge_results(results, manifest, cfg)
    c = cal.calibrate(index, cfg["pairs"], cal.load_validation_pairs(PAIRS_JSON))
    assert c["status"] == "rejected" and c["sign_agreement_decisive"] < 0.5
    assert any("sign agreement" in r for r in c["reasons"])
    pred = cal.predict([-0.002], c)
    assert pred["warning"] and "NOT decision-grade" in pred["warning"]


def test_partial_and_failed_results_are_pending(tmp_path):
    cfg = fleet.load_config(GATE)
    keep = {"k60_s73", "k60_s58", "relu2_s73", "relu2_s58", "as0_s73", "mp97_s73"}
    drop = tuple(r["label"] for r in cfg["runs"] if r["label"] not in keep)
    results, manifest = synthetic_results(cfg, tmp_path, drop_labels=drop, fail_labels=("as0_s73",))
    index = fleet.merge_results(results, manifest, cfg)
    c = cal.calibrate(index, cfg["pairs"], cal.load_validation_pairs(PAIRS_JSON))
    assert c["status"] == "pending" and c["n_decisive_with_data"] == 3      # lk35_vs_relu2 s73/s58 + mp97 s73; asa failed
    assert "asa@s73" in c["pending_decisive"] and "asa@s73 (gate 3)" in c["pending"]
    by_id = {p["pair_id"]: p for p in c["pairs"]}
    assert by_id["asa"]["gpu_summary"]["n"] == 0 and by_id["lk35_vs_relu2"]["gpu_summary"]["n"] == 2
    assert cal.main(["--results", str(results), "--configs", str(GATE), "--pairs", str(PAIRS_JSON)]) == 2
    assert cal.calibrate({}, cfg["pairs"], {})["status"] == "pending"


def test_subset_of_decisive_rows_never_accepts(tmp_path):
    """F1: six agreeing decisive pairs used to accept; now every frozen row, tie row and null must be scored."""
    cfg = fleet.load_config(GATE)
    pairs_index = cal.load_validation_pairs(PAIRS_JSON)
    results, manifest = synthetic_results(cfg, tmp_path, drop_labels=("lk05_s58",))
    c = cal.calibrate(fleet.merge_results(results, manifest, cfg), cfg["pairs"], pairs_index)
    assert c["status"] == "pending" and c["n_decisive_with_data"] == 10 and c["sign_agreement_decisive"] >= 0.8
    assert c["pending_decisive"] == ["lk35_vs_lk05@s58"] and c["gates"][3]["ok"] is None and c["gates"][4]["ok"] is None
    results, manifest = synthetic_results(cfg, tmp_path, drop_labels=("sc165_s58",), name="r2.jsonl")
    c = cal.calibrate(fleet.merge_results(results, manifest, cfg), cfg["pairs"], pairs_index)
    assert c["status"] == "pending" and "softcap165@s58 (gate 5)" in c["pending"] and c["gates"][3]["ok"] is True
    results, manifest = synthetic_results(cfg, tmp_path, drop_labels=("k60_s73_repeat", "k60_s73_repeat2"), name="r3.jsonl")
    c = cal.calibrate(fleet.merge_results(results, manifest, cfg), cfg["pairs"], pairs_index)
    assert c["status"] == "pending" and c["gates"][1]["ok"] is None and "nulls (gate 1)" in c["pending"]


def test_family_clause_and_tie_band_reject(tmp_path):
    cfg = fleet.load_config(GATE)
    pairs_index = cal.load_validation_pairs(PAIRS_JSON)
    results, manifest = synthetic_results(cfg, tmp_path)
    # attn_src is a one-row family: flipping it alone leaves 10/11 = 91% yet fails the per-family clause
    index = fleet.merge_results(results, manifest, cfg)
    index["as0_s73"]["val_bpb"] = index["k60_s73"]["val_bpb"] - 0.0015
    c = cal.calibrate(index, cfg["pairs"], pairs_index)
    assert c["status"] == "rejected" and c["sign_agreement_decisive"] >= 0.9 and c["gates"][3]["ok"] is False
    assert any("family with no agreeing seed: attn_src" in r for r in c["reasons"])
    # three null/tie rows outside the band (2 allowed) -> gate 5 fails; two are tolerated
    index = fleet.merge_results(results, manifest, cfg)
    for lab in ("wds1_s73", "sc165_s73", "qk16_s73"):
        index[lab]["val_bpb"] = index["k60_s73"]["val_bpb"] + 0.0015
    c = cal.calibrate(index, cfg["pairs"], pairs_index)
    assert c["status"] == "rejected" and c["gates"][5]["ok"] is False and any(r.startswith("gate 5") for r in c["reasons"])
    index = fleet.merge_results(results, manifest, cfg)
    for lab in ("wds1_s73", "sc165_s73"):
        index[lab]["val_bpb"] = index["k60_s73"]["val_bpb"] + 0.0015
    assert cal.calibrate(index, cfg["pairs"], pairs_index)["gates"][5]["ok"] is True
    # a null above 0.0006 fails gate 1
    index = fleet.merge_results(results, manifest, cfg)
    index["k60_s73_repeat"]["val_bpb"] = index["k60_s73"]["val_bpb"] + 0.0009
    c = cal.calibrate(index, cfg["pairs"], pairs_index)
    assert c["status"] == "rejected" and c["gates"][1]["ok"] is False and "CUBLAS_WORKSPACE_CONFIG" in c["gates"][1]["detail"]
    # an absolute level 0.02 off fails gate 2
    index = fleet.merge_results(results, manifest, cfg)
    for lab in index:
        if index[lab].get("val_bpb") is not None:
            index[lab]["val_bpb"] += 0.02
    c = cal.calibrate(index, cfg["pairs"], pairs_index)
    assert c["status"] == "rejected" and c["gates"][2]["ok"] is False


def test_gate0_and_frozen_rule_mismatch_reject(tmp_path):
    cfg = fleet.load_config(GATE)
    pairs_index = cal.load_validation_pairs(PAIRS_JSON)
    results, manifest = synthetic_results(cfg, tmp_path)
    index = fleet.merge_results(results, manifest, cfg)
    index["relu2_s58"]["eval_denominator_ok"] = False
    c = cal.calibrate(index, cfg["pairs"], pairs_index)
    assert c["status"] == "rejected" and c["gates"][0]["ok"] is False and c["reasons"][0].startswith("gate 0")
    # a rule frozen against another row set (the prescreen config) refuses the verdict outright
    pre = fleet.load_config(PRESCREEN)
    other = cal.freeze_rule(pre["pairs"], pairs_index, cal.config_keys(pre))
    assert other["unresolved"] and other["digest"] != c["rule"]["digest"]
    index = fleet.merge_results(results, manifest, cfg)
    c = cal.calibrate(index, cfg["pairs"], pairs_index, frozen=other)
    assert c["status"] == "rejected" and c["reasons"][0].startswith("frozen rule")
    same = cal.freeze_rule(cfg["pairs"], pairs_index, cal.config_keys(cfg))
    assert cal.calibrate(index, cfg["pairs"], pairs_index, frozen=same)["status"] == "accepted"
    # the prescreen config alone can never be accepted: its frozen rows are unresolvable
    c = cal.calibrate(fleet.merge_results(*synthetic_results(pre, tmp_path, name="pre.jsonl"), pre), pre["pairs"], pairs_index)
    assert c["status"] == "pending" and c["reasons"][0].startswith("frozen row")
    assert "0.0003" in cal.rule_text() and "[0.5, 2.0]" in cal.rule_text() and "80%" in cal.rule_text()


def test_fleet_verdict_delegates(tmp_path, capsys):
    cfg = fleet.load_config(PRESCREEN)
    results, manifest = synthetic_results(cfg, tmp_path)
    rc = fleet.main(["verdict", "--results", str(results), "--configs", str(PRESCREEN), "--pairs", str(PAIRS_JSON)])
    assert rc == 2 and "PENDING" in capsys.readouterr().out          # 6 pairs, one seed: not enough decisive


# ----------------------------------------------------------------------------- K60 pinned schedule (F3)
# configs/schedule-K60-TT1793.json is GENERATED (fleet.schedule_table = vclock.rescale_to_steps on a triple
# derived from the reference chip-C log's own switch lines); PROTOCOL.md 2.2 and README.md quote it.

SCHEDULE = CONFIGS / "schedule-K60-TT1793.json"
PROTOCOL_MD = ROOT / "ffsim" / "gpu" / "PROTOCOL.md"
README_MD = ROOT / "ffsim" / "gpu" / "README.md"
LK35_LOGS = ("0816_R_g7lk35_s73", "0913_R_g7lk35_s58", "0945_R_g7lk35_s67")

needs_reference = pytest.mark.skipif(fleet.reference_log() is None, reason="no K60 reference log harvested")
needs_lk35 = pytest.mark.skipif(not all((fleet.CHIP_LOG_DIR / r / "train.log").exists() for r in LK35_LOGS),
                                reason="LK35 chip-C logs not harvested")


def _table():
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


@needs_reference
def test_schedule_json_is_the_regenerated_code_walk():
    from ffsim.gpu import vclock
    table = _table()
    ref = fleet.reference_log()
    trip, src = fleet.derive_step_seconds(ref)
    regen = json.loads(json.dumps(fleet.schedule_table(trip, fleet.K60_STEPS, src, ref)))
    assert regen == table, "configs/schedule-K60-TT1793.json is stale: python -m ffsim.gpu.fleet schedule --write"
    assert table["steps"] == fleet.K60_STEPS == 2361 and table["stop_reason"] == "stop_step"
    assert table["reference"]["derived_from"] == ref.parent.name
    assert table["reference"]["provisional"] is (ref.parent.name != "C41_cold_K60")
    # the fleet's default triple IS the JSON's (single source); the 4-decimal form build_command passes walks the same table
    assert fleet.DEFAULT_STEP_SECONDS == tuple(table["step_seconds_input"][k] for k in ("1", "2", "4"))
    assert fleet.parse_step_seconds("default") == (fleet.DEFAULT_STEP_SECONDS, fleet.DEFAULT_STEP_SECONDS_SOURCE)
    four = tuple(float(f"{v:.4f}") for v in fleet.DEFAULT_STEP_SECONDS)
    t4 = fleet.schedule_table(four, fleet.K60_STEPS)
    assert [e[:2] for e in t4["accum"]] == [e[:2] for e in table["accum"]] and t4["steps_by_k"] == table["steps_by_k"]
    assert [e[:2] for e in t4["mtp"]] == [e[:2] for e in table["mtp"]] and t4["levels"][0][1] == table["levels"][0][1]
    # what train_gpu.py --steps does: rescale_to_steps + pinned stop, switches at their charged fractions of TT
    scaled, scale, walk = vclock.rescale_to_steps(vclock.Recipe(), {1: trip[0], 2: trip[1], 4: trip[2]}, 2361)
    assert walk["accum"] == table["accum"] and walk["levels"] == table["levels"] and walk["steps"] == 2361
    assert math.isclose(scale, table["scale"], rel_tol=1e-6)
    assert table["accum"][1][2] == pytest.approx(0.06 * 1793, abs=0.6)
    assert table["accum"][2][2] == pytest.approx(0.20 * 1793, abs=1.0)
    mtp = {m[0]: m[1] for m in table["mtp"]}
    assert table["phase2_step"] == table["accum"][1][1] and table["phase3_step"] == table["accum"][2][1] == mtp[4]
    assert table["cooldown_start_step"] == table["levels"][0][1] and table["levels"][0][0] == 19 and table["final_level"] == 1
    assert table["ema_save_level_step"] < table["level1_step"] < table["steps"]
    # token / FLOP arithmetic follows from steps_by_k, never hand-computed
    assert sum(table["steps_by_k"].values()) == 2361
    assert table["tokens_by_k"] == {k: int(k) * n * 65536 for k, n in table["steps_by_k"].items()}
    assert table["tokens"] == sum(table["tokens_by_k"].values()) and table["all_k4_tokens"] == 2361 * 262144
    assert math.isclose(table["flops"], 7.581204e8 * table["tokens"], rel_tol=1e-12)


@needs_reference
def test_protocol_and_readme_quote_the_schedule_json():
    import re
    t = _table()
    p2, p3, cd, stop = t["phase2_step"], t["phase3_step"], t["cooldown_start_step"], t["steps"]
    mtp = {m[0]: m[1] for m in t["mtp"]}
    proto = PROTOCOL_MD.read_text(encoding="utf-8")

    def one(pattern, text=proto):
        m = re.search(pattern, text)
        assert m, pattern
        return m

    assert int(one(r"\| phase 1 -> 2 \(k=2, LR x 0\.7071\) \| \*\*(\d+)\*\*").group(1)) == p2
    assert int(one(r"\| phase 2 -> 3 \(k=4\) = MTP stage 4 \| \*\*(\d+)\*\*").group(1)) == p3
    assert int(one(r"\| cooldown level 19 \| \*\*(\d+)\*\*").group(1)) == cd
    assert one(r"\| MTP stages 1, 2, 3 \| ([\d, ]+) \|").group(1) == ", ".join(str(mtp[i]) for i in (1, 2, 3))
    assert one(r"\| MTP stages 5, 6, 7, 8 \| ([\d, ]+) \|").group(1) == ", ".join(str(mtp[i]) for i in (5, 6, 7, 8))
    assert int(one(r"\| EMA save level <= 3 \| (\d+) \|").group(1)) == t["ema_save_level_step"]
    assert int(one(r"\| level 1 \(floor\) \| (\d+) \|").group(1)) == t["level1_step"]
    assert int(one(r"\| stop \(pinned, `--steps 2361`\) \| \*\*(\d+)\*\*").group(1)) == stop
    by, tok = t["steps_by_k"], t["tokens_by_k"]
    assert one(r"\nk1: +([\d,]+) x +65,536 = +([\d,]+)").groups() == (f"{by['1']:,}", f"{tok['1']:,}")
    assert one(r"\nk2: +([\d,]+) x +131,072 = +([\d,]+)").groups() == (f"{by['2']:,}", f"{tok['2']:,}")
    assert one(r"\nk4: +([\d,]+) x +262,144 = +([\d,]+)").groups() == (f"{by['4']:,}", f"{tok['4']:,}")
    assert one(r"total 2,361 steps = ([\d,]+) tokens").group(1) == f"{t['tokens']:,}"
    flops = "%.3fe17" % (t["flops"] / 1e17)
    assert one(r"7\.581204e8 x ([\d,]+) = \*\*([\d.]+e17)\*\*").groups() == (f"{t['tokens']:,}", flops)
    assert f"FLOPs = 7.581204e8 x {t['tokens']:,} = {flops}" in proto
    assert f"x{t['scale']:.6f}" in proto
    assert "491,323,392" not in proto and "3.725e17" not in proto          # the hand-computed draft numbers
    readme = README_MD.read_text(encoding="utf-8")
    m = one(r"phase 2 at step (\d+), phase 3 at (\d+),\s*#?\s*cooldown at (\d+), stop (\d+); ([\d,]+) tokens, ([\d.]+e17) FLOPs", readme)
    assert tuple(int(x) for x in m.groups()[:4]) == (p2, p3, cd, stop)
    assert m.group(5) == f"{t['tokens']:,}" and m.group(6) == flops
    assert f"K60: 2,361 steps, {t['tokens']:,} tokens" in readme and "491,323,392" not in readme
    assert "schedule-K60-TT1793.json" in readme and SCHEDULE.exists()


@needs_lk35
def test_walk_reproduces_each_lk35_log_from_its_own_means_and_c40_triple_is_not_theirs():
    from ffsim.gpu import vclock
    # out of sample for the loop semantics: only the 3 accum switch lines + final charged seconds enter the
    # means; the 8 MTP stages, the 19 cooldown levels and the stop are predicted
    for run in LK35_LOGS:
        path = str(fleet.CHIP_LOG_DIR / run / "train.log")
        log = vclock.parse_chip_log(path)
        assert log["time_target"] == 1760.0 and len(log["mtp"]) == 8 and len(log["levels"]) == 19
        walk = vclock.walk_schedule(vclock.Recipe(time_target=log["time_target"]), vclock.phase_seconds_from_log(path))
        cmp = vclock.compare_schedules(walk, log)
        assert not cmp["missing"] and cmp["max_abs_switch_diff"] <= 2 and cmp["steps_diff"] == 0, \
            (run, vclock.format_comparison(cmp))
    # the C40 (master M14) triple is NOT the LK35 (older master) triple: 11-13 steps late at their TT 1760
    # (PROTOCOL.md 2.2); a K60 triple has to come from an M14 log
    c40 = fleet.CHIP_LOG_DIR / "C40_cold_K59" / "train.log"
    if c40.exists():
        trip, src = fleet.derive_step_seconds(c40)
        assert "C40_cold_K59" in src and trip == pytest.approx((0.2968, 0.5102, 0.9341), abs=1e-4)
        log = vclock.parse_chip_log(str(fleet.CHIP_LOG_DIR / LK35_LOGS[0] / "train.log"))
        walk = vclock.walk_schedule(vclock.Recipe(time_target=1760.0), {1: trip[0], 2: trip[1], 4: trip[2]})
        cmp = vclock.compare_schedules(walk, log)
        assert 8 <= cmp["max_abs_switch_diff"] <= 16 and -16 <= cmp["steps_diff"] <= -8


@needs_reference
def test_schedule_command_prints_writes_and_plan_shows_the_walk(tmp_path, capsys):
    import re
    t = _table()
    assert fleet.main(["schedule", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == t
    out = tmp_path / "sched.json"
    assert fleet.main(["schedule", "--write", str(out)]) == 0
    text = capsys.readouterr().out
    assert "K60 pinned schedule" in text and f"{t['tokens']:,} tokens" in text and str(out) in text
    assert json.loads(out.read_text(encoding="utf-8")) == t
    assert fleet.main(["schedule", "--from-log", str(tmp_path / "missing.log")]) == 2
    capsys.readouterr()
    assert fleet.main(["schedule", "--step-seconds", "0.3,0.5,0.9", "--steps", "2000", "--json"]) == 0
    small = json.loads(capsys.readouterr().out)
    assert small["steps"] == 2000 and sum(small["steps_by_k"].values()) == 2000
    assert small["step_seconds_input"] == {"1": 0.3, "2": 0.5, "4": 0.9} and "explicit" in small["step_seconds_source"]
    # plan prints the same walk (and one line per other step count in the config) before the run list
    assert fleet.main(["plan", "--configs", str(GATE), "--quote-prices"]) == 0
    text = capsys.readouterr().out
    assert "K60 pinned schedule" in text and text.index("K60 pinned schedule") < text.index("k60_s73")
    assert re.search(r"phase 2 \(k=2, LR x 0\.7071\) +%d " % t["phase2_step"], text)
    assert re.search(r"phase 3 \(k=4\) = MTP stage 4 +%d " % t["phase3_step"], text)
    assert "--steps 2314:" in text and "--steps 2357:" in text and f"{t['tokens']:,} tokens" in text
    assert not list(tmp_path.glob("runs*"))
