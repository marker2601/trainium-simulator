"""Tests for ffsim.search: train.py validation, standardised common random numbers and analytic ranking keys,
support grading (weak / interaction / special dims), quality notes and the chip-queue launch lines.

Self-contained duck-typed stubs (no other ffsim module needed); tests/test_ffsim_simulate.py covers the
simulate / report / cli side with its own stubs."""
from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ffsim import report as rep
from ffsim import search as srch
from ffsim import simulate as sim
from ffsim.schema import Recipe, RunRecord

ROOT = Path(__file__).resolve().parents[1]
RUNNER_LOG = ROOT / "research" / "sim-data" / "chipC" / "_meta" / "queue" / "runner.log"


# --- stubs -------------------------------------------------------------------------------------------------
class StubStepTime:
    jitter_sd = 0.004
    support = {"FF_LEAKY_RELU2": {"values": [0.35, 0.5], "min": 0.35, "max": 0.5}, "FF_RELU2_FN": ["0", "1"]}

    def predict(self, code_version, knobs, chip):
        k4 = 0.928 * (1.116 if chip == "D" else 1.0)
        if knobs.get("FF_RELU2_FN") == "1":
            k4 *= 0.982
        return {"k1": k4 * 0.316, "k2": k4 * 0.545, "k4": k4}

    def known_versions(self):
        return ["M12", "M14"]

    def nearest_version(self, code_version):
        return "M14"


class StubQuality:
    support = {"FF_LEAKY_RELU2": ["0.25", "0.35", "0.5"], "FF_SOFTCAP": {"min": 12, "max": 18},
               "FF_RELU2_FN": ["0", "1"], "FF_WD_SCHED": ["0", "1"]}
    steps_slope = -0.057
    sigma = 0.0002

    def predict(self, knobs, steps, code_version, chip, seed=None):
        lk = float(knobs.get("FF_LEAKY_RELU2", 0.35))
        mean = 0.96092 - 0.057 * (math.log(steps) - math.log(2357)) + 0.02 * (lk - 0.35) ** 2
        if knobs.get("FF_WD_SCHED") == "1":
            mean -= 0.0004
        sd = math.hypot(0.0006, self.sigma) if seed is None else self.sigma
        return mean, sd


class CovQuality:
    """A ridge-style model like ffsim.quality.QualityModel: design_row / beta / cov / sigma. The LK column has a
    wide prior, so an out-of-range LK is a wide-sd (EXTRAP) candidate."""
    support = {"FF_LEAKY_RELU2": ["0.25", "0.35", "0.5"], "FF_SOFTCAP": {"min": 12, "max": 18}}
    SEED_SD = 0.0006

    def __init__(self):
        self.beta = np.array([0.96092, -0.057, -0.0002, 0.0001])
        self.cov = np.diag([1e-8, 4e-6, 4e-5, 1.6e-7])
        self.sigma = 0.0003

    def design_row(self, knobs, steps, code_version, chip, seed=None):
        x = np.array([1.0, math.log(steps / 2357.0), float(knobs.get("FF_LEAKY_RELU2", 0.35)) - 0.35,
                      (float(knobs.get("FF_SOFTCAP", 15)) - 15.0) / 3.0])
        return x, (0.0 if seed is not None else self.SEED_SD ** 2), []

    def predict(self, knobs, steps, code_version, chip, seed=None):
        x, extra, _ = self.design_row(knobs, steps, code_version, chip, seed)
        return float(x @ self.beta), math.sqrt(float(x @ self.cov @ x) + self.sigma ** 2 + extra)

    def steps_slope(self, steps=2300.0):
        return float(self.beta[1])


def _rec(i, code_version, chip, **knobs):
    return RunRecord(run_id=f"{chip}:r{i}", chip=chip, run=f"r{i}", code_version=code_version,
                     knobs={k: str(v) for k, v in knobs.items()}, bpb_2m=0.96, steps=2300)


class EraQuality(StubQuality):
    """StubQuality plus what ffsim.quality.QualityModel exposes for support grading and notes: the fitted
    records, the era lineage (resolve_era / version_to_era / version_aliases), the fitted chips, the config
    threshold and design_row notes. code_version M99 is 3 generations off and 20x the sd."""
    support = dict(StubQuality.support, FF_COOLDOWN_FRAC=["0.6", "0.7"], FF_ATTN_SRC=["", "5:6,7,8"])
    version_to_era = {"M12": "M9+", "M14": "M9+", "M3": "M3"}
    version_aliases = {"code_k12off": "M12"}
    chips = {"D": 0}
    config = SimpleNamespace(reference_chip="C", min_value_runs=2)

    def __init__(self):
        self.fit_records = [
            _rec(0, "M12", "C", FF_LEAKY_RELU2=0.35, FF_WD_SCHED=1),
            _rec(1, "M12", "C", FF_LEAKY_RELU2=0.35, FF_WD_SCHED=1),
            _rec(2, "code_k12off", "C", FF_LEAKY_RELU2=0.35, FF_WD_SCHED=1),
            _rec(3, "M3", "C", FF_COOLDOWN_FRAC=0.7),                       # the one M3-era cooldown-0.7 run
            _rec(4, "M12", "C", FF_LEAKY_RELU2=0.25),
            _rec(5, "M12", "C", FF_LEAKY_RELU2=0.25),
            _rec(6, "M12", "D", FF_LEAKY_RELU2=0.35),
            _rec(7, "M14", "C", FF_LEAKY_RELU2=0.35),
            _rec(8, "M12", "C", FF_LEAKY_RELU2=0.25, FF_WD_SCHED=1),        # LK 0.25 and WD 1 co-observed
        ]

    def resolve_era(self, code_version, tags=None):
        v = self.version_aliases.get(code_version, code_version)
        if v in self.version_to_era:
            return self.version_to_era[v], 0, False
        if str(v).startswith("M"):
            return "M9+", 3, False
        return "M9+", 1, True

    def design_row(self, knobs, steps, code_version, chip, seed=None):
        notes = []
        era, dist, cross = self.resolve_era(code_version)
        if dist > 0:
            notes.append(f"code_version {code_version!r} unseen: using {era} ({dist} generation(s) away)")
        return np.array([1.0]), 0.0, notes

    def predict(self, knobs, steps, code_version, chip, seed=None):
        mean, sd = super().predict(knobs, steps, code_version, chip, seed)
        if code_version == "M99":
            sd *= 20.0
        return mean, sd


class StubOffset:
    mean = 0.0066
    sd = 0.0004

    def sample(self, rng, n):
        return rng.normal(self.mean, self.sd, n)


def models(quality=None, steptime=None) -> sim.Models:
    return sim.Models(steptime or StubStepTime(), quality or StubQuality(), StubOffset(), meta={"kind": "test-stub"})


def k59_knobs(**extra):
    knobs = {"FF_TIME_TARGET": "1793", "FF_ACCUM_SCHED": "1:0.06,2:0.2,4", "FF_WARMUP": "20", "FF_LEAKY_RELU2": "0.35"}
    knobs.update({k: str(v) for k, v in extra.items()})
    return knobs


def space(dims=None, name="K59", max_changes=2, queue=None, **base_extra) -> srch.Space:
    d = {"name": "small",
         "base": {"name": name, "code_version": "M12", "chip": "C", "time_target": 1793, "knobs": k59_knobs(**base_extra)},
         "dims": dims if dims is not None else [
             {"knob": "FF_LEAKY_RELU2", "values": [0.25, 0.35, 0.5, 0.9]},
             {"knob": "FF_SOFTCAP", "values": [10, 14, 18]},
             {"knob": "FF_RELU2_FN", "values": [0, 1]},
             {"knob": "FF_WD_SCHED", "values": [1]}],
         "max_changes": max_changes}
    if queue is not None:
        d["queue"] = queue
    return srch.space_from_dict(d)


def rows_by_name(res):
    return {r["name"]: r for r in res["rows"]}


# --- train.py startup rules (F7) -----------------------------------------------------------------------------
def test_validate_knobs_encodes_train_py_asserts():
    v = srch.validate_knobs
    assert v({}) == []                                                    # every rule needs a knob: all skipped
    assert v({"FF_RELU2_FN": "0", "FF_LEAKY_RELU2": "0.35"}) == []
    out = v({"FF_RELU2_FN": "1", "FF_LEAKY_RELU2": "0.35"})
    assert len(out) == 1 and "train.py:565" in out[0] and "FF_RELU2_FN" in out[0]
    assert v({"FF_RELU2_FN": "1", "FF_LEAKY_RELU2": "0"}) == []            # RF without LK: the K58R recipe
    assert any("train.py:1109" in x for x in v({"FF_NULL_MLP": "1", "FF_LEAKY_RELU2": "0.35"}))
    assert any("train.py:1111" in x for x in v({"FF_NULL_MLP": "2", "FF_RELU2_FN": "1", "FF_LEAKY_RELU2": "0"}))
    assert any("train.py:562" in x for x in v({"FF_LEAKY_FORM": "fnm", "FF_LEAKY_RELU2": "0"}))
    assert v({"FF_LEAKY_FORM": "fnm", "FF_LEAKY_RELU2": "0.35"}) == []
    assert any("train.py:561" in x for x in v({"FF_LEAKY_FORM": "weird"}))
    assert any("train.py:567" in x for x in v({"FF_LEAKY_INPLACE": "0", "FF_LEAKY_RELU2": "0"}))
    assert any("train.py:566" in x for x in v({"FF_LEAKY_RELU2": "1.5"}))
    assert any("train.py:566" in x for x in v({"FF_LEAKY_RELU2": "abc"}))  # unparsable = train.py would crash
    assert any("train.py:37" in x for x in v({"FF_XU_QUEUE": "64"}))
    assert any("train.py:629" in x for x in v({"FF_ASYNC_OPT": "1", "FF_SCALAR_CONSTS": "1", "FF_TENSOR_LR": "1"}))
    assert v({"FF_ASYNC_OPT": "1", "FF_SCALAR_CONSTS": "2", "FF_TENSOR_LR": "1"}) == []
    assert any("train.py:111" in x for x in v({"FF_EMA_PREWARM": "1", "FF_EMA": "1", "FF_EMA_EVERY": "4"}))
    assert v({"FF_EMA_PREWARM": "1", "FF_EMA": "1", "FF_EMA_EVERY": "32"}) == []
    assert v({"FF_EMA_PREWARM": "1", "FF_EMA": "1"}) == []                # FF_EMA_EVERY unknown: skipped
    assert any("train.py:594" in x for x in v({"FF_CE_BF16": "1", "FF_FUSED_CE": "0"}))
    assert any("train.py:1116" in x for x in v({"FF_NO_CAUSAL": "1", "FF_NKI_ATTN": "1"}))
    assert srch.TRAIN_PY_SOURCE.startswith("K59 upload train.py")


def test_evaluate_rejects_candidates_train_py_would_refuse():
    sp = space()
    res = srch.evaluate(sp, models(), n_sims=100, gen="oat", rng_seed=0)
    names = {r["name"] for r in res["rows"]}
    assert "FF_RELU2_FN=1" not in names and "FF_RELU2_FN=0" in names
    assert res["meta"]["n_invalid"] == 1 and res["meta"]["n_candidates"] == 9
    inv = res["invalid"][0]
    assert inv["name"] == "FF_RELU2_FN=1" and "train.py:565" in inv["invalid"] and inv["changes"] == {"FF_RELU2_FN": "1"}
    assert res["meta"]["train_py_rules"] == srch.TRAIN_PY_SOURCE and res["meta"]["base_violations"] == []
    # the local generator: every double change carrying FF_RELU2_FN=1 goes too (LK 0 is not in this space)
    res2 = srch.evaluate(sp, models(), n_sims=100, gen="local", rng_seed=0)
    assert res2["meta"]["n_invalid"] == 8 and all("FF_RELU2_FN=1" in i["name"] for i in res2["invalid"])
    assert res2["meta"]["n_candidates"] == 1 + 9 + 29 - 8
    # validate=False keeps them (the stubs treat RF as pure speed, which is what the campaign proved)
    res3 = srch.evaluate(sp, models(), n_sims=100, gen="oat", rng_seed=0, validate=False)
    assert res3["meta"]["n_invalid"] == 0 and "FF_RELU2_FN=1" in rows_by_name(res3) and res3["meta"]["train_py_rules"] is None
    # LK 0 + RF 1 (the K58R form) is a valid double change
    sp0 = space(dims=[{"knob": "FF_LEAKY_RELU2", "values": [0]}, {"knob": "FF_RELU2_FN", "values": [1]}])
    res4 = srch.evaluate(sp0, models(), n_sims=100, gen="local", rng_seed=0)
    assert "FF_LEAKY_RELU2=0+FF_RELU2_FN=1" in rows_by_name(res4) and [i["name"] for i in res4["invalid"]] == ["FF_RELU2_FN=1"]


def test_base_violations_are_not_charged_to_candidates():
    """A base that itself trips a rule (RF on top of LK) does not make every candidate invalid: only a NEW
    violation rejects a candidate, and the base's own violations are reported in meta."""
    sp = space(dims=[{"knob": "FF_SOFTCAP", "values": [14]}, {"knob": "FF_NULL_MLP", "values": [1]}], FF_RELU2_FN=1)
    res = srch.evaluate(sp, models(), n_sims=100, gen="oat", rng_seed=0)
    assert any("train.py:565" in v for v in res["meta"]["base_violations"])
    assert "FF_SOFTCAP=14" in rows_by_name(res) and "base" in rows_by_name(res)
    assert [i["name"] for i in res["invalid"]] == ["FF_NULL_MLP=1"] and "train.py:1109" in res["invalid"][0]["invalid"]


# --- standardised common random numbers and analytic ranking keys (F4) ----------------------------------------
def test_standardised_noise_has_no_sample_mean():
    raw = sim.draw_noise(3, 500, StubOffset())
    nz = srch.standardise_noise(raw, StubOffset())
    assert isinstance(nz, srch.StandardisedNoise) and nz.n == 500
    for z in (nz.z_time, nz.z_quality):
        assert abs(float(z.mean())) < 1e-12 and float(z.std()) == pytest.approx(1.0, abs=1e-12)
    assert float(nz.offset.mean()) == pytest.approx(0.0066, abs=1e-12) and float(nz.offset.std()) == pytest.approx(0.0004, abs=1e-12)
    assert nz.param_seed == raw.param_seed and nz.resid_seed == raw.resid_seed       # the draws still follow the rng
    pd = nz.param_draw(4)
    assert pd.shape == (500, 4) and np.allclose(pd.mean(0), 0.0, atol=1e-12) and np.allclose(pd.std(0), 1.0, atol=1e-12)
    raw_pd = raw.param_draw(4)
    assert not np.allclose(raw_pd.mean(0), 0.0, atol=1e-4)                            # what the plain Noise gives
    assert np.corrcoef(pd[:, 0], raw_pd[:, 0])[0, 1] == pytest.approx(1.0)            # same draws, only centred/scaled
    # a different rng seed gives different (standardised) draws; the offset model's mean wins over the sample's
    other = srch.standardise_noise(sim.draw_noise(4, 500, StubOffset()), StubOffset())
    assert not np.array_equal(other.z_time, nz.z_time) and other.param_seed != nz.param_seed
    plain = srch.standardise_noise(sim.draw_noise(3, 500, StubOffset()), None)
    assert abs(float(plain.offset.mean() - raw.offset.mean())) < 1e-12               # no model mean: draws kept
    assert srch.offset_mean_of(StubOffset(), nz) == 0.0066 and srch.offset_mean_of(object(), nz) == pytest.approx(0.0066, abs=1e-12)


def test_ranking_keys_are_analytic_and_rng_invariant():
    """official_mean / bpb_2m_mean / delta_vs_base are the model's mean at its own step count (+ offset mean):
    identical for every --rng and draw count; only official_mc_mean, the sd, the percentiles and P(...) are
    Monte Carlo. Before, wide-sd (EXTRAP) candidates moved by sd_i x mean(z) with the rng seed and draw count."""
    sp = space()
    a = srch.evaluate(sp, models(), n_sims=200, gen="local", rng_seed=0)
    b = srch.evaluate(sp, models(), n_sims=1000, gen="local", rng_seed=9)
    ra, rb = rows_by_name(a), rows_by_name(b)
    assert set(ra) == set(rb) and len(ra) == 31
    assert [r["name"] for r in a["rows"]] == [r["name"] for r in b["rows"]]           # identical ranking
    for nm in ra:
        assert ra[nm]["official_mean"] == rb[nm]["official_mean"] and ra[nm]["delta_vs_base"] == rb[nm]["delta_vs_base"]
        assert ra[nm]["bpb_2m_mean"] == rb[nm]["bpb_2m_mean"]
        assert ra[nm]["official_mean"] == pytest.approx(ra[nm]["bpb_2m_mean"] + 0.0066, abs=1e-12)
        assert abs(ra[nm]["official_mc_mean"] - ra[nm]["official_mean"]) < 8e-5       # residual 0.0002 / sqrt(200)
        assert abs(rb[nm]["official_mc_mean"] - rb[nm]["official_mean"]) < 4e-5
        assert ra[nm]["official_mc_mean"] != rb[nm]["official_mc_mean"] or nm == "base"
    assert ra["base"]["delta_vs_base"] == 0.0 and ra["base"]["p_beat_base"] == 0.5
    assert ra["FF_WD_SCHED=1"]["delta_vs_base"] == pytest.approx(-0.0004, abs=1e-12)  # exact, not MC
    assert ra["FF_LEAKY_RELU2=0.9"]["delta_vs_base"] == pytest.approx(0.02 * 0.55 ** 2, abs=1e-12)
    # WD=1 ties at exactly -0.0004 with the four doubles that add a no-effect knob (RF=0, softcap): rank by name
    assert a["rows"][0]["delta_vs_base"] == ra["FF_WD_SCHED=1"]["delta_vs_base"] and ra["FF_WD_SCHED=1"]["rank"] == 5
    assert 0.8 < ra["FF_WD_SCHED=1"]["p_beat_base"] < 0.99
    assert a["meta"]["ranking"].startswith("analytic") and "standardised" in a["meta"]["crn"]
    # with a cov-bearing model the wide-sd candidate (LK 0.9: prior sd 0.0035) is as invariant as the others
    q = CovQuality()
    m = sim.Models(StubStepTime(), q, StubOffset())
    sp2 = space(dims=[{"knob": "FF_LEAKY_RELU2", "values": [0.3, 0.9]}, {"knob": "FF_SOFTCAP", "values": [14]}])
    c = rows_by_name(srch.evaluate(sp2, m, n_sims=200, gen="oat", rng_seed=0))
    d = rows_by_name(srch.evaluate(sp2, m, n_sims=1500, gen="oat", rng_seed=11))
    assert c["FF_LEAKY_RELU2=0.9"]["bpb_2m_sd"] > 0.003
    for nm in c:
        assert c[nm]["official_mean"] == d[nm]["official_mean"] and c[nm]["delta_vs_base"] == d[nm]["delta_vs_base"]
    wide = c["FF_LEAKY_RELU2=0.9"]
    assert wide["delta_vs_base"] == pytest.approx(-0.0002 * 0.55, abs=1e-12)          # beta x (0.9 - 0.35), exactly
    assert abs(wide["official_mc_mean"] - wide["official_mean"]) < 2e-4               # residual 0.0003 / sqrt(200)
    assert abs(d["FF_LEAKY_RELU2=0.9"]["official_mc_mean"] - wide["official_mean"]) < 6e-5


def test_workers_give_identical_rows():
    sp = space()
    one = srch.evaluate(sp, models(), n_sims=300, gen="local", rng_seed=5, workers=1)
    two = srch.evaluate(sp, models(), n_sims=300, gen="local", rng_seed=5, workers=2)
    for k in ("name", "official_mean", "official_mc_mean", "p_beat_base", "support_note", "job_env"):
        assert [r[k] for r in one["rows"]] == [r[k] for r in two["rows"]]
    assert one["invalid"] == two["invalid"]


# --- support grading (F5, F7 minor, F11) ------------------------------------------------------------------------
def test_weak_support_and_interaction_extrap():
    m = models(EraQuality())
    sp = space(dims=[{"knob": "FF_COOLDOWN_FRAC", "values": [0.7]}, {"knob": "FF_WD_SCHED", "values": [1]},
                     {"knob": "FF_LEAKY_RELU2", "values": [0.25, 0.5]}], FF_COOLDOWN_FRAC=0.6)
    fitted = srch.FittedRows(m.quality)
    assert not fitted.empty and len(fitted.rows) == 9
    assert fitted.count("FF_WD_SCHED", 1, "M9+") == (4, 4) and fitted.count("FF_COOLDOWN_FRAC", "0.7", "M9+") == (1, 0)
    assert fitted.count("code_version", "M12") == (7, 0) and fitted.count("chip", "D") == (1, 0)   # code_k12off == M12
    assert fitted.co_observed("FF_LEAKY_RELU2", 0.25, "FF_WD_SCHED", 1)
    assert not fitted.co_observed("FF_COOLDOWN_FRAC", 0.7, "FF_WD_SCHED", 1)
    assert srch.base_era_of(m, sp) == "M9+" and srch.min_value_runs(m.quality) == 2
    res = srch.evaluate(sp, m, n_sims=100, gen="local", rng_seed=0)
    by = rows_by_name(res)
    cd = by["FF_COOLDOWN_FRAC=0.7"]                                  # one M3-era run: not `ok` any more
    assert cd["in_support"] is False and cd["support_level"] == "weak" and cd["support_note"] == "FF_COOLDOWN_FRAC:weak(n=1, 0 in era M9+)"
    assert by["FF_WD_SCHED=1"]["in_support"] is True and by["FF_WD_SCHED=1"]["support_note"] == "FF_WD_SCHED:ok(n=4)"
    assert by["FF_LEAKY_RELU2=0.25"]["support_note"].startswith("FF_LEAKY_RELU2:ok(n=3)")
    lk5 = by["FF_LEAKY_RELU2=0.5"]                                    # in the support table, but no fitted run
    assert lk5["in_support"] is False and lk5["support_note"].startswith("FF_LEAKY_RELU2:weak(n=0, 0 in era M9+)")
    pair = by["FF_COOLDOWN_FRAC=0.7+FF_WD_SCHED=1"]
    assert pair["support_level"] == "interaction" and "FF_COOLDOWN_FRAC+FF_WD_SCHED:never co-observed (interaction EXTRAP)" in pair["support_note"]
    good = by["FF_LEAKY_RELU2=0.25+FF_WD_SCHED=1"]                     # combined in a fitted run
    assert good["in_support"] is True and good["support_level"] == "ok" and "co-observed" not in good["support_note"]
    assert res["meta"]["min_value_runs"] == 2 and res["meta"]["support_levels"]["weak"] >= 2
    # a stricter threshold makes LK 0.25 (3 runs) weak too; the table-only path (no fitted rows) still says ok
    ok, note, level = srch.support_check(m, srch.make_candidate({"FF_LEAKY_RELU2": 0.25}), sp, fitted, "M9+", min_runs=4)
    assert not ok and level == "weak" and note == "FF_LEAKY_RELU2:weak(n=3, 3 in era M9+)/steptime-extrap"
    ok, note, level = srch.support_check(models(), srch.make_candidate({"FF_LEAKY_RELU2": 0.25}))
    assert ok and level == "ok" and note == "FF_LEAKY_RELU2:ok/steptime-extrap"
    # the confirmation list (in-support only) therefore skips the weak and interaction rows
    conf = srch.top_by(res, "delta_vs_base", 5, in_support=True)
    assert all(r["support_level"] in ("ok", "interp") for r in conf) and conf[0]["name"] in ("FF_WD_SCHED=1", "FF_LEAKY_RELU2=0.25+FF_WD_SCHED=1")


def test_special_dims_code_version_and_chip():
    m = models(EraQuality())
    sp = space(dims=[{"knob": "code_version", "values": ["M14", "M99", "K57x"]}, {"knob": "chip", "values": ["D", "E"]},
                     {"knob": "time_target", "values": [1800]}], max_changes=1)
    res = srch.evaluate(sp, m, n_sims=100, gen="oat", rng_seed=0)
    by = rows_by_name(res)
    m14 = by["code_version=M14"]                                      # fitted (era lineage distance 0)
    assert m14["in_support"] is True and m14["support_note"] == "code_version:fitted(1 runs, era M9+)" and m14["notes"] == ""
    m99 = by["code_version=M99"]                                      # unknown version: quality + step-time fallbacks
    assert m99["in_support"] is False and m99["support_level"] == "extrap"
    assert m99["support_note"] == "code_version:fallback->M9+ (distance 3)/steptime-fallback->M14"
    assert "quality: code_version 'M99' unseen: using M9+ (3 generation(s) away)" in m99["notes"]
    assert re.search(r"quality sd 0\.0126 is 20x the base's 0\.0006: the surrogate is extrapolating", m99["notes"])
    assert by["code_version=K57x"]["support_note"].startswith("code_version:fallback->M9+ (different lineage)")
    assert by["chip=D"]["in_support"] is True and by["chip=D"]["support_note"] == "chip:fitted(1 runs)"
    assert by["chip=E"]["in_support"] is False and by["chip=E"]["support_note"] == "chip:unseen chip (treated as C + prior sd)"
    assert by["time_target=1800"]["in_support"] is True and by["time_target=1800"]["support_note"] == "time_target:steps-only"
    assert by["time_target=1800"]["steps_mean"] > by["base"]["steps_mean"]
    # a model without eras or fitted rows: the old NEVER-VARIED verdict is not repeated for a fitted special dim
    plain = rows_by_name(srch.evaluate(sp, models(), n_sims=50, gen="oat", rng_seed=0))
    assert plain["code_version=M14"]["support_note"] == "code_version:UNKNOWN(model has no eras)"
    assert plain["chip=D"]["support_note"] == "chip:UNKNOWN(model has no chip table)"
    assert "NEVER-VARIED" not in plain["code_version=M14"]["support_note"] + plain["chip=D"]["support_note"]


def test_steptime_extrap_tag_only_for_knobs_the_steptime_model_models():
    res = srch.evaluate(space(), models(), n_sims=50, gen="oat", rng_seed=0)
    by = rows_by_name(res)
    assert by["FF_LEAKY_RELU2=0.25"]["support_note"] == "FF_LEAKY_RELU2:ok/steptime-extrap"   # LK is a step-time knob
    assert by["FF_LEAKY_RELU2=0.9"]["support_note"] == "FF_LEAKY_RELU2:EXTRAP/steptime-extrap"
    assert by["FF_SOFTCAP=10"]["support_note"] == "FF_SOFTCAP:EXTRAP"                          # quality-only knob
    assert by["FF_SOFTCAP=14"]["support_note"] == "FF_SOFTCAP:interp"
    assert by["FF_WD_SCHED=1"]["support_note"] == "FF_WD_SCHED:ok"
    assert "steptime-extrap" not in by["FF_SOFTCAP=10"]["support_note"] + by["FF_WD_SCHED=1"]["support_note"]
    never = rows_by_name(srch.evaluate(space(dims=[{"knob": "FF_QK_GAIN", "values": [1.2]}]), models(), n_sims=50, gen="oat"))
    assert never["FF_QK_GAIN=1.2"]["support_note"] == "FF_QK_GAIN:NEVER-VARIED" and never["FF_QK_GAIN=1.2"]["support_level"] == "never"


# --- chip queue launch lines (F6) ------------------------------------------------------------------------------
def test_k59_base_gets_the_full_queue_job_env():
    sp = space()                                                       # base named K59 -> the built-in queue arm
    q = srch.resolve_queue(sp)
    assert q["env"] and q["source"] == "built-in K59 queue arm" and q["reference_run"] == "C:0816_R_g7lk35_s73"
    assert "FF_SEED" not in q["env"] and q["env"]["FF_CODE_DIR"] == "/root/ff-claude/code_k12off"
    assert q["env"]["FF_LEAKY_RELU2"] == "0.35" and q["env"]["FF_LEAKY_FORM"] == "fnm" and q["env"]["FF_COOLDOWN_FRAC"] == "0.60"
    assert q["env"]["FF_ACCUM_SCHED"] == "1:0.06,2:0.2,4" and q["env"]["FF_ACCUM_LR"] == "0.5,0.7071" and q["env"]["FF_EMA_EVERY"] == "32"
    c = srch.make_candidate({"FF_WD_SCHED": 1, "FF_LEAKY_RELU2": 0.3})
    job = srch.job_env_for(q, c)
    env = srch.parse_env_line(job)
    assert env["FF_LEAKY_RELU2"] == "0.3" and env["FF_WD_SCHED"] == "1" and env["FF_CODE_DIR"] == "/root/ff-claude/code_k12off"
    assert len(env) == len(q["env"]) + 1
    line = srch.env_overrides({"job_env": job, "changes": c.changes}, 73)
    assert line.startswith("FF_SEED=73 FF_COOLDOWN_FRAC=0.60 FF_ASYNC_OPT=1 ") and line.endswith(" FF_LEAKY_FORM=fnm FF_LEAKY_RELU2=0.3 FF_WD_SCHED=1")
    assert srch.parse_env_line(srch.job_env_for(q, srch.make_candidate({"code_version": "M14"})))["FF_CODE_DIR"] == "/root/ff-claude/code_k14off"
    assert srch.parse_env_line(srch.job_env_for(q, srch.make_candidate({"time_target": 1793})))["FF_TIME_TARGET"] == "1793"
    assert srch.job_env_for(q, srch.make_candidate({"chip": "D", "FF_WD_SCHED": 1})) is None   # D's base.env unknown
    assert srch.env_overrides({"job_env": None, "changes": {"chip": "D", "FF_WD_SCHED": 1}}, 58) == "FF_SEED=58 FF_WD_SCHED=1"
    res = srch.evaluate(sp, models(), n_sims=50, gen="oat", rng_seed=0)
    meta = res["meta"]["queue"]
    assert meta["reference_run"] == "C:0816_R_g7lk35_s73" and meta["code_dir"] == "/root/ff-claude/code_k12off"
    assert "base.env" in meta["note"] and "FF_TIME_TARGET stays base.env's" in meta["note"]
    assert res["base"]["job_env"] == srch.format_env(q["env"])
    assert all(r["job_env"] for r in res["rows"])
    text = "\n".join(rep.markdown_lines(res, top=5))
    assert re.search(r"`FF_SEED=73 FF_COOLDOWN_FRAC=0\.60 .*FF_CODE_DIR=/root/ff-claude/code_k12off .*FF_LEAKY_RELU2=0\.35 FF_WD_SCHED=1`", text)


def test_other_bases_need_a_queue_block(tmp_path: Path):
    sp = space(name="K57")
    q = srch.resolve_queue(sp)
    assert q["env"] is None and "no queue job env known for base 'K57'" in q["note"]
    res = srch.evaluate(sp, models(), n_sims=50, gen="oat", rng_seed=0)
    assert all(r["job_env"] is None for r in res["rows"]) and "no queue job env known" in res["meta"]["queue"]["note"]
    assert srch.env_overrides(rows_by_name(res)["FF_WD_SCHED=1"], 73) == "FF_SEED=73 FF_WD_SCHED=1"
    # a queue block pointing at a runner.log: the START line's env (last value of a repeated key wins, FF_SEED off)
    log = tmp_path / "runner.log"
    log.write_text("START other 08:00:00 :: FF_SEED=1 FF_X=9\n"
                   "START r1 08:16:50 :: FF_SEED=73 FF_A=1 FF_ACCUM_SCHED=2:0.2,4 FF_ACCUM_SCHED=1:0.06,2:0.2,4 "
                   "FF_CODE_DIR=/root/ff-claude/code_k12off\nEND r1 08:50:29 bpb=0.96\n", encoding="utf-8")
    assert srch.job_env_from_runner_log(log, "C:r1") == {"FF_SEED": "73", "FF_A": "1", "FF_ACCUM_SCHED": "1:0.06,2:0.2,4",
                                                         "FF_CODE_DIR": "/root/ff-claude/code_k12off"}
    assert srch.job_env_from_runner_log(log, "nope") is None and srch.job_env_from_runner_log(tmp_path / "none", "r1") is None
    sp2 = space(name="K57", queue={"runner_log": str(log), "reference_run": "C:r1", "code_dirs": {"M14": "/root/ff-claude/code_custom"}})
    q2 = srch.resolve_queue(sp2)
    assert q2["source"] == "space.queue" and q2["env"] == {"FF_A": "1", "FF_ACCUM_SCHED": "1:0.06,2:0.2,4", "FF_CODE_DIR": "/root/ff-claude/code_k12off"}
    assert srch.job_env_for(q2, srch.make_candidate({"code_version": "M14", "FF_WD_SCHED": 1})) == \
        "FF_A=1 FF_ACCUM_SCHED=1:0.06,2:0.2,4 FF_CODE_DIR=/root/ff-claude/code_custom FF_WD_SCHED=1"
    sp3 = space(name="K57", queue={"runner_log": str(log), "reference_run": "C:missing"})
    assert srch.resolve_queue(sp3)["env"] is None and "no START line" in srch.resolve_queue(sp3)["note"]
    # a job env that disagrees with the base recipe is refused (the line would not be the base + change)
    sp4 = space(queue={"job_env": "FF_LEAKY_RELU2=0.5 FF_CODE_DIR=/root/ff-claude/code_k12off"})
    q4 = srch.resolve_queue(sp4)
    assert q4["env"] is None and "disagrees" in q4["note"] and "FF_LEAKY_RELU2: base '0.35' vs queue '0.5'" in q4["note"]
    sp5 = space(queue={"job_env": "FF_LEAKY_RELU2=0.35 FF_CODE_DIR=/root/ff-claude/code_k14off"})
    assert "FF_CODE_DIR /root/ff-claude/code_k14off vs code_version 'M12'" in srch.resolve_queue(sp5)["note"]
    assert srch.code_dir_for("M12") == "/root/ff-claude/code_k12off" and srch.code_dir_for("code_k12off") == "/root/ff-claude/code_k12off"
    assert srch.code_dir_for("M77") == "/root/ff-claude/code_m77" and srch.code_dir_for("M14", {"M14": "/x"}) == "/x"


@pytest.mark.skipif(not RUNNER_LOG.exists(), reason="needs the harvested chip C queue log")
def test_built_in_k59_job_env_matches_the_harvested_runner_log():
    env = srch.job_env_from_runner_log(RUNNER_LOG, "C:0816_R_g7lk35_s73")
    assert env is not None and env.pop("FF_SEED") == "73"
    assert env == srch.parse_env_line(srch.K59_QUEUE["job_env"])


# --- generators and formatting (unchanged behaviour, kept as a guard) ------------------------------------------
def test_generators_and_recipe_building():
    sp = space()
    assert [len(d.values) for d in sp.dims] == [3, 3, 2, 1]
    assert len(srch.gen_oat(sp)) == 10 and len(srch.gen_local(sp)) == 1 + 9 + 29
    assert len(srch.gen_grid(space(max_changes=None))) == 4 * 4 * 3 * 2
    r = srch.to_recipe(sp, srch.make_candidate({"FF_WD_SCHED": 1, "code_version": "M14", "time_target": 1800}))
    assert r.code_version == "M14" and r.knobs["FF_WD_SCHED"] == "1" and r.knobs["FF_TIME_TARGET"] == "1800" and r.time_target == 1800.0
    assert srch.same_version("code_k12off", "M12") and not srch.same_version("M12", "M14")
    assert srch.format_env({"A": "1", "B": "x y", "C": ""}) == "A=1 B='x y' C=''"
    assert srch.parse_env_line("A=1 B='x y' junk C=2 C=3") == {"A": "1", "B": "x y", "C": "3"}
