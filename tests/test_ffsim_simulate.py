"""Tests for ffsim.simulate / search / report / cli with small duck-typed stub models.

They do not need ffsim.steptime / quality / offset / dataset (written by other agents)."""
from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pytest

from ffsim import report as rep
from ffsim import search as srch
from ffsim import simulate as sim
from ffsim.cli import main as cli_main
from ffsim.schema import Recipe, SimResult

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "ffsim" / "examples"


# --- stubs (module level so multiprocessing can pickle them) -------------------------------------
class StubStepTime:
    jitter_sd = 0.004
    support = {"FF_LEAKY_RELU2": ["0.35", "0.5"], "FF_RELU2_FN": ["0", "1"]}

    def __init__(self, k4: float = 0.928):
        self.k4 = k4

    def predict(self, code_version, knobs, chip):
        k4 = self.k4 * (1.116 if chip == "D" else 1.0)
        if knobs.get("FF_RELU2_FN") == "1":      # RF: one elementwise pass less, 1.8% faster
            k4 *= 0.982
        return {"k1": k4 * 0.325, "k2": k4 * 0.56, "k4": k4}


class StubQuality:
    # FF_RELU2_FN is listed because the campaign proved RF is bit-identical in function (speed only)
    support = {"FF_LEAKY_RELU2": ["0.25", "0.35", "0.5"], "FF_SOFTCAP": {"min": 12, "max": 18}, "FF_RELU2_FN": ["0", "1"]}
    steps_slope = -0.057
    SEED_EFFECT = {73: 0.0, 58: 0.0003, 67: -0.0002}
    sigma = 0.0002          # per-run residual: drawn independently per recipe by the simulator

    def predict(self, knobs, steps, code_version, chip, seed=None):
        lk = float(knobs.get("FF_LEAKY_RELU2", 0.35))
        mean = 0.96092 - 0.057 * (math.log(steps) - math.log(2357)) + 0.02 * (lk - 0.35) ** 2
        if seed is None:
            return mean, math.hypot(0.0006, self.sigma)
        return mean + self.SEED_EFFECT.get(int(seed), 0.0), self.sigma


class SharedOnlyQuality(StubQuality):
    """A model exposing neither sigma nor cov: the simulator can only share its sd (and says so)."""
    sigma = None

    def predict(self, knobs, steps, code_version, chip, seed=None):
        mean = 0.96092 - 0.057 * (math.log(steps) - math.log(2357))
        return (mean, 0.0006) if seed is None else (mean + self.SEED_EFFECT.get(int(seed), 0.0), 0.0002)


class CovQuality:
    """A ridge-style model like ffsim.quality.QualityModel: design_row / beta / cov / sigma, so the
    simulator can split parameter uncertainty (shared beta draw) from the per-run residual."""
    support = {"FF_LEAKY_RELU2": ["0.25", "0.35", "0.5"], "FF_SOFTCAP": {"min": 12, "max": 18}}
    SEED_SD = 0.0006

    def __init__(self):
        # columns: intercept, ln(steps/2357), (lk - 0.35), (softcap - 15) / 3
        self.beta = np.array([0.96092, -0.057, -0.0002, 0.0001])
        self.cov = np.diag([1e-8, 4e-6, 4e-7, 1.6e-7])
        self.sigma = 0.0003

    def design_row(self, knobs, steps, code_version, chip, seed=None):
        x = np.array([1.0, math.log(steps / 2357.0), float(knobs.get("FF_LEAKY_RELU2", 0.35)) - 0.35,
                      (float(knobs.get("FF_SOFTCAP", 15)) - 15.0) / 3.0])
        extra = 0.0 if seed is not None else self.SEED_SD ** 2
        return x, extra, []

    def predict(self, knobs, steps, code_version, chip, seed=None):
        x, extra, _ = self.design_row(knobs, steps, code_version, chip, seed)
        return float(x @ self.beta), math.sqrt(float(x @ self.cov @ x) + self.sigma ** 2 + extra)

    def steps_slope(self, steps=2300.0):
        return float(self.beta[1])

    def delta_sd(self, a, b, steps, seed=None):
        d = self.design_row(a.knobs, steps, a.code_version, a.chip, seed)[0] - \
            self.design_row(b.knobs, steps, b.code_version, b.chip, seed)[0]
        return math.sqrt(float(d @ self.cov @ d) + 2.0 * self.sigma ** 2)


class RaisingQuality(StubQuality):
    def predict(self, knobs, steps, code_version, chip, seed=None):
        if knobs.get("FF_SOFTCAP") == "10":
            raise ValueError("softcap 10 unsupported")
        return super().predict(knobs, steps, code_version, chip, seed)


class StubOffset:
    mean = 0.0066
    sd = 0.0004

    def sample(self, rng, n):
        return rng.normal(self.mean, self.sd, n)


def models(quality=None) -> sim.Models:
    return sim.Models(StubStepTime(), quality or StubQuality(), StubOffset(), meta={"kind": "test-stub"})


def k59(**extra) -> Recipe:
    knobs = {"FF_TIME_TARGET": "1793", "FF_ACCUM_SCHED": "1:0.06,2:0.2,4", "FF_WARMUP": "20", "FF_LEAKY_RELU2": "0.35"}
    knobs.update({k: str(v) for k, v in extra.items()})
    return Recipe("K59", "M12", knobs, "C", 1793)


def small_space(max_changes=2) -> srch.Space:
    return srch.space_from_dict({
        "name": "small",
        "base": {"name": "K59", "code_version": "M12", "chip": "C", "time_target": 1793, "knobs": k59().knobs},
        "dims": [{"knob": "FF_LEAKY_RELU2", "values": [0.25, 0.35, 0.5, 0.9]},
                 {"knob": "FF_SOFTCAP", "values": [10, 14, 18]},
                 {"knob": "FF_RELU2_FN", "values": [0, 1]}],
        "max_changes": max_changes})


# --- accumulation schedule and step counting -----------------------------------------------------
def test_parse_accum_sched_forms():
    assert sim.parse_accum_sched("1:0.06,2:0.2,4") == [("k1", 0.06), ("k2", 0.2), ("k4", 1.0)]
    assert sim.parse_accum_sched("2:0.12,4") == [("k2", 0.12), ("k4", 1.0)]
    assert sim.parse_accum_sched("4") == [("k4", 1.0)]
    assert sim.parse_accum_sched("") == [("k4", 1.0)]
    with pytest.raises(ValueError):
        sim.parse_accum_sched("1:0.5,2:0.2,4")
    with pytest.raises(ValueError):
        sim.parse_accum_sched("1:0.1:2,4")


def test_steps_scalar_exact_arithmetic():
    # startup 5 (uncharged) + warm-up 15 at K + final phase floor((100 - 15) / 1.0)
    assert sim.steps_scalar({"k4": 1.0}, 100.0, "4", 20) == 105
    # k1 phase until 50% of time_target: 35 s left after the warm-up at 0.5 s -> 70 steps, then 50 at k4
    assert sim.steps_scalar({"k1": 0.5, "k4": 1.0}, 100.0, "1:0.5,4", 20, time_target=100.0) == 140
    # boundary already passed by the warm-up -> zero k1 steps
    assert sim.steps_scalar({"k1": 0.5, "k4": 1.0}, 100.0, "1:0.1,4", 20, time_target=100.0) == 105


def test_steps_reproduce_k59_anchor():
    # the raw walk on the old calibrated ratios and the harness clock (1790 s): an arithmetic identity kept as a guard
    steps = sim.steps_scalar({"k1": 0.928 * 0.325, "k2": 0.928 * 0.56, "k4": 0.928}, 1790.0,
                             "1:0.06,2:0.2,4", 20, time_target=1793.0)
    assert 2345 <= steps <= 2370
    # C40 from its own logged medians (k1 0.2937, k2 0.5069, k4 0.9312), the median -> mean overhead and the
    # step-clock reserve (1793 -> 1787.1 s at the last step): 2355 vs 2357 measured
    own = sim.walk_step_times({"k1": 0.2937, "k2": 0.5069, "k4": 0.9312}, sim.MEDIAN_TO_MEAN_OVERHEAD)
    assert sim.steps_scalar(own, 1793.0 - sim.SAVE_RESERVE_S, "1:0.06,2:0.2,4", 20, time_target=1793.0) == 2355


def test_step_walk_overhead_and_reserve_fix_the_k59_bias():
    """F3: the fitted K59 medians (k1 0.2921, k2 0.5038, k4 0.9259) walked with the 0.57% overhead and the
    5.5 s step-clock reserve give 2369 steps at FF_TIME_TARGET 1793 (C40 2357, a 0.45% slow run) and 2324 at
    the queue budget 1760 (the seven LK0.35 seeds: 2322-2329). Raw medians over 1793 - 3 s gave 2385."""
    assert sim.SAVE_RESERVE_S == 5.5 and sim.MEDIAN_TO_MEAN_OVERHEAD == pytest.approx(0.0057)
    try:
        from ffsim import steptime
        assert sim.MEDIAN_TO_MEAN_OVERHEAD == steptime.MEDIAN_TO_MEAN_OVERHEAD
    except ImportError:
        pass

    class FittedK59StepTime(StubStepTime):
        def predict(self, code_version, knobs, chip):
            return {"k1": 0.2921, "k2": 0.5038, "k4": 0.9259, "sd": 0.0036}

    m = sim.Models(FittedK59StepTime(), StubQuality(), StubOffset())
    nz = sim.draw_noise(0, 400, m.offset)
    bd = sim.simulate_batch([k59(), k59(FF_TIME_TARGET=1760)], m.steptime, m.quality, nz, seed=73)
    assert bd.steps_point[0] == 2369 and bd.steps_point[1] == 2324
    assert bd.step_time[0]["k4"] == pytest.approx(0.9259)           # reported step times stay the model's medians
    old = sim.simulate_batch([k59()], m.steptime, m.quality, nz, seed=73, save_reserve=3.0, overhead_frac=0.0)
    assert old.steps_point[0] == 2385                                 # the biased walk, on request only
    r = sim.simulate(k59(), *m.as_tuple(), n=400, rng=0, seed=73)
    assert abs(r.steps_mean - 2369) < 3 and "cross-check" not in r.notes


def test_steps_vectorised_matches_scalar():
    rng = np.random.default_rng(3)
    n = 200
    f = 1.0 + 0.01 * rng.standard_normal(n)
    st = {"k1": 0.30 * f, "k2": 0.52 * f, "k4": 0.928 * f}
    vec = sim.steps_from_step_times(st, 1790.0, "1:0.06,2:0.2,4", 20, time_target=1793.0)
    ref = np.array([sim.steps_scalar({k: float(v[i]) for k, v in st.items()}, 1790.0, "1:0.06,2:0.2,4", 20,
                                     time_target=1793.0) for i in range(n)])
    assert vec.shape == (n,)
    assert np.array_equal(vec, ref)
    # 2-D broadcasting (recipes x draws) is what simulate_batch relies on
    st2 = {k: v[None, :] * np.array([[1.0], [1.05]]) for k, v in st.items()}
    vec2 = sim.steps_from_step_times(st2, 1790.0, "1:0.06,2:0.2,4", 20, time_target=1793.0)
    assert vec2.shape == (2, n) and np.array_equal(vec2[0], ref) and np.all(vec2[1] < vec2[0])


def test_fill_phases():
    st, notes = sim.fill_phases({"k4": 1.0}, ["k1", "k2", "k4"])
    # measured full-run ratios on chip C (C40: 0.3154 / 0.5444), not the old calibrated 0.325 / 0.56
    assert st["k1"] == pytest.approx(0.316) and st["k2"] == pytest.approx(0.545) and notes
    with pytest.raises(ValueError):
        sim.fill_phases({"k8": 2.0}, ["k4"])


# --- simulate -------------------------------------------------------------------------------------
def test_simulate_basic_and_reproducible():
    m = models()
    r = sim.simulate(k59(), *m.as_tuple(), n=1000, rng=0)
    assert isinstance(r, SimResult) and r.n == 1000
    assert 2330 <= r.steps_mean <= 2380 and 3 <= r.steps_sd <= 20
    assert r.official_mean == pytest.approx(r.bpb_2m_mean + 0.0066, abs=1e-4)
    assert r.official_p10 < r.official_p50 < r.official_p90
    assert 0.0 <= r.p_beat_best <= 1.0
    assert r.step_time["k4"] == pytest.approx(0.928)
    again = sim.simulate(k59(), *m.as_tuple(), n=1000, rng=0)
    assert again == r
    other = sim.simulate(k59(), *m.as_tuple(), n=1000, rng=1)
    assert other.official_mean != r.official_mean


def test_seed_given_is_narrower_than_marginal():
    m = models()
    marginal = sim.simulate(k59(), *m.as_tuple(), n=2000, rng=0)
    s73 = sim.simulate(k59(), *m.as_tuple(), n=2000, rng=0, seed=73)
    s58 = sim.simulate(k59(), *m.as_tuple(), n=2000, rng=0, seed=58)
    assert s73.bpb_2m_sd < marginal.bpb_2m_sd
    assert s58.bpb_2m_mean - s73.bpb_2m_mean == pytest.approx(0.0003, abs=1e-6)


def test_time_target_knob_wins_over_field():
    m = models()
    long = Recipe("tt", "M12", {"FF_TIME_TARGET": "1900"}, "C", 1793)
    short = Recipe("tt", "M12", {}, "C", 1793)
    assert sim.recipe_time_target(long) == 1900.0 and sim.recipe_time_target(short) == 1793.0
    assert sim.simulate(long, *m.as_tuple(), n=200, rng=0).steps_mean > sim.simulate(short, *m.as_tuple(), n=200, rng=0).steps_mean


def test_pair_common_random_numbers():
    m = models()
    same = sim.simulate_pair(k59(), k59(), *m.as_tuple(), n=1000, rng=0)
    assert same.p_a_better == 0.5 and same.diff_sd == 0.0 and same.diff_mean == 0.0
    rf = sim.simulate_pair(k59(FF_RELU2_FN=1), k59(), *m.as_tuple(), n=2000, rng=0)
    expected = -0.057 * math.log(1 / 0.982)          # +1.8% steps -> about -0.00104 bpb
    assert rf.p_a_better > 0.95
    assert rf.diff_mean == pytest.approx(expected, abs=2.5e-4)
    # jitter, seed luck and offset cancel; each arm keeps its own per-run residual (sigma 0.0002), so the
    # paired sd is the pair noise floor sqrt(2) sigma, never ~0 (the old sign-indicator behaviour)
    assert rf.diff_sd == pytest.approx(math.sqrt(2) * 0.0002, rel=0.15)
    assert rf.a.official_sd > 5e-4                    # while each arm alone carries the full noise
    assert rf.a.bpb_2m_sd == pytest.approx(math.hypot(0.0006, 0.0002), rel=0.1)   # marginal sd = the model's sd


def test_pair_probability_reproduces_predict_delta_sd():
    """F1: with a cov-bearing model the paired sd is sqrt(d cov d + 2 sigma^2) (what QualityModel.predict_delta
    reports) and P(a < b) is the matching normal probability, not 1.0 for every negative mean delta."""
    q = CovQuality()
    m = sim.Models(StubStepTime(), q, StubOffset())
    seen = []
    for seed in (None, 73):
        for cand in (k59(FF_LEAKY_RELU2=0.3), k59(FF_SOFTCAP=12), k59(FF_LEAKY_RELU2=0.5, FF_SOFTCAP=18)):
            pr = sim.simulate_pair(cand, k59(), *m.as_tuple(), n=6000, rng=1, seed=seed)
            sd = q.delta_sd(cand, k59(), pr.b.steps_mean, seed)
            assert pr.diff_sd == pytest.approx(sd, rel=0.08), (cand.knobs, seed)
            p = 0.5 * (1.0 + math.erf(-pr.diff_mean / sd / math.sqrt(2)))
            assert pr.p_a_better == pytest.approx(p, abs=0.03), (cand.knobs, seed)
            # each arm's marginal sd is still the model's own sd (without chip jitter moving the steps)
            alone = sim.simulate(cand, *m.as_tuple(), n=6000, rng=1, seed=seed, jitter_sd=0.0)
            assert alone.bpb_2m_sd == pytest.approx(q.predict(cand.knobs, alone.steps_mean, "M12", "C", seed)[1], rel=0.08)
            seen.append(pr.p_a_better)
    assert max(seen) < 0.9 and min(seen) > 0.1          # honest: these deltas are inside the pair noise floor
    same = sim.simulate_pair(k59(), k59(), *m.as_tuple(), n=1000, rng=1)
    assert same.p_a_better == 0.5 and same.diff_sd == 0.0
    # the parameter draw is shared: recipes with the same design row differ only by their residual
    nz = sim.draw_noise(1, 3000, m.offset)
    bd = sim.simulate_batch([k59(), k59(FF_NEVER_VARIED="1")], m.steptime, q, nz, seed=73)
    assert bd.quality_param_sd[0] == pytest.approx(bd.quality_param_sd[1]) and bd.quality_param_sd[0] > 0
    assert (bd.bpb_2m[0] - bd.bpb_2m[1]).std() == pytest.approx(math.sqrt(2) * q.sigma, rel=0.08)
    rows = sim.summarise_batch(bd, 0.9671, bd.official[0])
    assert rows[1]["delta_sd_vs_base"] == pytest.approx(math.sqrt(2) * q.sigma, rel=0.08) and rows[0]["delta_sd_vs_base"] == 0.0


def test_quality_noise_split_fallbacks():
    r = k59()
    q = CovQuality()
    sp = sim.quality_noise_split(q, r, 2360.0, None, q.predict(r.knobs, 2360.0, "M12", "C", None)[1])
    assert sp.x is not None and sp.resid_sd == q.sigma and sp.shared_sd == pytest.approx(q.SEED_SD, rel=1e-6)
    sp = sim.quality_noise_split(StubQuality(), r, 2360.0, None, math.hypot(0.0006, 0.0002))
    assert sp.x is None and sp.resid_sd == 0.0002 and sp.shared_sd == pytest.approx(0.0006) and not sp.notes
    sp = sim.quality_noise_split(SharedOnlyQuality(), r, 2360.0, None, 0.0006)
    assert sp.resid_sd == 0.0 and sp.shared_sd == 0.0006 and "sign indicator" in sp.notes[0]
    m = sim.Models(StubStepTime(), SharedOnlyQuality(), StubOffset())
    res = sim.simulate(k59(), *m.as_tuple(), n=100, rng=0)
    assert "sign indicator" in res.notes
    # the residual draw is keyed on the recipe, never its name, and differs between different recipes
    a = sim.residual_draw(7, sim.recipe_key(k59(), None), 50)
    assert np.array_equal(a, sim.residual_draw(7, sim.recipe_key(Recipe("other name", "M12", k59().knobs, "C", 1793), None), 50))
    assert not np.array_equal(a, sim.residual_draw(7, sim.recipe_key(k59(FF_SOFTCAP=14), None), 50))
    assert not np.array_equal(a, sim.residual_draw(8, sim.recipe_key(k59(), None), 50))


def test_steps_slope_is_taken_at_the_predicted_steps():
    """F4: a QualityModel-style steps_slope(steps) is called with the recipe's step count, not at STEPS_REF."""
    class SlopeQuality(StubQuality):
        calls = []

        def steps_slope(self, steps=2300.0):
            self.calls.append(steps)
            return -0.057 + 0.00001 * (steps - 2300.0)

    q = SlopeQuality()
    m = sim.Models(StubStepTime(), q, StubOffset())
    nz = sim.draw_noise(0, 50, m.offset)
    bd = sim.simulate_batch([k59()], m.steptime, q, nz)
    assert q.calls == [bd.steps_point[0]] and bd.steps_point[0] != 2300.0
    assert bd.steps_slope[0] == pytest.approx(-0.057 + 0.00001 * (bd.steps_point[0] - 2300.0))
    # a knob-aware 4-argument form and a constant callable still work
    class FourArg(StubQuality):
        def steps_slope(self, knobs, steps, code_version, chip):
            return -0.05
    assert sim.simulate_batch([k59()], m.steptime, FourArg(), nz).steps_slope[0] == -0.05
    class NoArg(StubQuality):
        def steps_slope(self):
            return -0.04
    assert sim.simulate_batch([k59()], m.steptime, NoArg(), nz).steps_slope[0] == -0.04


def test_batch_error_collection():
    m = models(RaisingQuality())
    nz = sim.draw_noise(0, 100, m.offset)
    bd = sim.simulate_batch([k59(), k59(FF_SOFTCAP=10), k59(FF_SOFTCAP=14)], m.steptime, m.quality, nz, on_error="collect")
    assert list(bd.errors) == [1] and "softcap" in bd.errors[1]
    rows = sim.summarise_batch(bd, 0.9671, bd.official[0])
    assert "error" in rows[1] and "official_mean" in rows[0] and rows[0]["p_beat_base"] == 0.5
    # a knob the model ignores: same mean, but a different recipe carries its own residual -> 0.5 in expectation only
    assert rows[2]["p_beat_base"] == pytest.approx(0.5, abs=0.15) and rows[2]["delta_vs_base"] == pytest.approx(0.0, abs=1e-4)
    assert rows[2]["delta_sd_vs_base"] == pytest.approx(math.sqrt(2) * 0.0002, rel=0.25)
    with pytest.raises(ValueError):
        sim.simulate_batch([k59(FF_SOFTCAP=10)], m.steptime, m.quality, nz)


def test_batch_speed_hundreds_of_candidates():
    m = models()
    nz = sim.draw_noise(0, 1000, m.offset)
    recipes = [k59(FF_SOFTCAP=12 + (i % 9), FF_LEAKY_RELU2=0.25 + 0.01 * (i % 20)) for i in range(300)]
    sim.summarise_batch(sim.simulate_batch(recipes[:8], m.steptime, m.quality, nz), 0.9671, nz.offset)  # warm-up
    t0 = time.perf_counter()
    bd = sim.simulate_batch(recipes, m.steptime, m.quality, nz)
    rows = sim.summarise_batch(bd, 0.9671, bd.official[0])
    per = (time.perf_counter() - t0) / len(recipes)
    assert len(rows) == 300 and bd.steps.shape == (300, 1000)
    assert per < 0.005, f"{per * 1e3:.2f} ms per candidate (n=1000) - must stay well under 10 ms"


# --- search ----------------------------------------------------------------------------------------
def test_dim_values_and_formatting():
    assert srch.dim_values({"knob": "x", "range": [12, 20], "step": 1}) == list(range(12, 21))
    assert srch.dim_values({"knob": "x", "range": [0.2, 0.4], "step": 0.1}) == [0.2, 0.3, 0.4]
    assert srch.dim_values({"knob": "x", "range": [0, 1], "n": 3}) == [0.0, 0.5, 1.0]
    assert srch.fmt_value(15.0) == "15" and srch.fmt_value(0.35) == "0.35" and srch.fmt_value(True) == "1"
    assert srch.same_value("0.35", 0.35) and not srch.same_value("0.35", 0.5)


def test_generators_counts_and_dedupe():
    sp = small_space()
    # base values are removed from the dims: LK 0.35 and RELU2_FN 0 (base has no FF_RELU2_FN -> both kept)
    assert [len(d.values) for d in sp.dims] == [3, 3, 2]
    oat = srch.gen_oat(sp)
    assert oat[0].name == "base" and len(oat) == 1 + 8
    local = srch.gen_local(sp)
    assert len(local) == 1 + 8 + (3 * 3 + 3 * 2 + 3 * 2)
    assert len({c.key() for c in local}) == len(local)
    grid = srch.gen_grid(sp)
    assert len(grid) == len(local)                       # max_changes 2 cuts the triples
    assert len(srch.gen_grid(small_space(max_changes=None))) == 4 * 4 * 3
    rnd = srch.gen_random(sp, 20, rng=0)
    assert len(rnd) == 21 and len({c.key() for c in rnd}) == 21 and all(len(c.changes) <= 2 for c in rnd)
    with pytest.raises(ValueError):
        srch.generate(sp, "nope")


def test_to_recipe_and_special_dims():
    sp = srch.space_from_dict({"base": {"name": "b", "code_version": "M12", "chip": "C", "time_target": 1793, "knobs": {"A": "1"}},
                               "dims": [{"knob": "code_version", "values": ["M12", "M14"]}, {"knob": "chip", "values": ["D"]}]})
    cands = srch.gen_local(sp)
    names = {c.name for c in cands}
    assert "code_version=M14" in names and "chip=D+code_version=M14" in names and len(cands) == 4
    r = srch.to_recipe(sp, srch.make_candidate({"code_version": "M14", "chip": "D", "A": 2}))
    assert r.code_version == "M14" and r.chip == "D" and r.knobs == {"A": "2"}


def test_support_status():
    sup = StubQuality.support
    assert srch.support_status(sup, "FF_LEAKY_RELU2", 0.35) == "seen"
    assert srch.support_status(sup, "FF_LEAKY_RELU2", 0.3) == "interp"
    assert srch.support_status(sup, "FF_LEAKY_RELU2", 0.9) == "unseen"
    assert srch.support_status(sup, "FF_SOFTCAP", 14) == "interp"
    assert srch.support_status(sup, "FF_SOFTCAP", 10) == "unseen"
    assert srch.support_status(sup, "FF_QK_GAIN", 1.2) == "never"
    assert srch.support_status({}, "x", 1) == "never"


def test_evaluate_ranks_flags_and_crn():
    # validate=False: the stubs treat FF_RELU2_FN as a pure speed change next to FF_LEAKY_RELU2=0.35 (the campaign
    # proved RF bit-identical in function), which search.py's train.py rule 565 would otherwise refuse to simulate
    sp = small_space()
    res = srch.evaluate(sp, models(), n_sims=500, gen="oat", rng_seed=0, validate=False)
    rows = res["rows"]
    assert res["meta"]["n_candidates"] == 9 and res["meta"]["n_errors"] == 0
    assert [r["official_mean"] for r in rows] == sorted(r["official_mean"] for r in rows)
    assert [r["rank"] for r in rows] == list(range(1, 10))
    by = {r["name"]: r for r in rows}
    base = by["base"]
    assert res["base"] is base and base["delta_vs_base"] == 0.0 and base["p_beat_base"] == 0.5
    assert by["FF_RELU2_FN=1"]["rank"] == 1 and by["FF_RELU2_FN=1"]["p_beat_base"] > 0.95
    assert by["FF_RELU2_FN=1"]["step_k4"] == pytest.approx(0.928 * 0.982)
    assert by["FF_LEAKY_RELU2=0.9"]["in_support"] is False and "EXTRAP" in by["FF_LEAKY_RELU2=0.9"]["support_note"]
    assert by["FF_LEAKY_RELU2=0.25"]["in_support"] is True
    assert by["FF_SOFTCAP=14"]["in_support"] is True and "interp" in by["FF_SOFTCAP=14"]["support_note"]
    assert by["FF_SOFTCAP=10"]["in_support"] is False
    assert by["FF_LEAKY_RELU2=0.9"]["delta_vs_base"] == pytest.approx(0.02 * 0.55 ** 2, abs=1e-4)
    top = srch.top_by(res, "delta_vs_base", 3, in_support=True)
    assert top[0]["name"] == "FF_RELU2_FN=1" and all(t["n_changes"] > 0 for t in top)
    env = srch.env_overrides(top[0], 73)      # search.py may prepend the base's effective env; the change is last
    assert env.startswith("FF_SEED=73 ") and env.endswith("FF_RELU2_FN=1")


def test_evaluate_collects_candidate_errors():
    res = srch.evaluate(small_space(), models(RaisingQuality()), n_sims=200, gen="oat", rng_seed=0, validate=False)
    assert res["meta"]["n_errors"] == 1 and res["errors"][0]["name"] == "FF_SOFTCAP=10"
    assert res["meta"]["n_candidates"] == 8


def test_evaluate_workers_are_deterministic():
    sp = small_space()
    one = srch.evaluate(sp, models(), n_sims=300, gen="local", rng_seed=5, workers=1)
    two = srch.evaluate(sp, models(), n_sims=300, gen="local", rng_seed=5, workers=2)
    assert [r["name"] for r in one["rows"]] == [r["name"] for r in two["rows"]]
    assert [r["official_mean"] for r in one["rows"]] == [r["official_mean"] for r in two["rows"]]
    assert [r["p_beat_base"] for r in one["rows"]] == [r["p_beat_base"] for r in two["rows"]]


# --- report --------------------------------------------------------------------------------------------
def test_report_writers_roundtrip(tmp_path: Path):
    res = srch.evaluate(small_space(), models(), n_sims=300, gen="local", rng_seed=0, validate=False)
    md = rep.write_markdown(res, tmp_path / "s.md", top=5)
    text = md.read_text(encoding="utf-8")
    assert "## Next Trainium confirmations" in text and re.search(r"FF_SEED=73 .*FF_RELU2_FN=1`", text)
    assert "FF_SEED=58" in text and "FF_SEED=67" in text and "EXTRAP" in text and "P(beat best)" in text
    csv_path = rep.write_csv(res, tmp_path / "s.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == res["meta"]["n_candidates"] and "official_mean" in rows[0] and rows[0]["rank"] == "1"
    js = rep.write_json(res, tmp_path / "s.json")
    back = rep.load_json(js)
    assert rep.markdown_lines(back, top=5) == rep.markdown_lines(res, top=5)
    summary = rep.format_summary(res, top=3)
    assert "confirm next" in summary and summary.isascii()
    single = sim.simulate(k59(), *models().as_tuple(), n=100, rng=0)
    assert rep.format_simresult(single).isascii() and f"P(official < {sim.BEST_OFFICIAL:.4f})" in rep.format_simresult(single)


# --- examples and CLI -----------------------------------------------------------------------------------
def test_examples_are_the_k59_recipe():
    recipe = json.loads((EXAMPLES / "recipe-K59.json").read_text(encoding="utf-8"))
    space = json.loads((EXAMPLES / "space-k59-local.json").read_text(encoding="utf-8"))
    k = recipe["knobs"]
    assert recipe["code_version"] == "M12" and recipe["chip"] == "C" and recipe["time_target"] == 1793
    assert k["FF_LEAKY_RELU2"] == "0.35" and k["FF_LEAKY_FORM"] == "fnm" and k["FF_TIME_TARGET"] == "1793"
    assert k["FF_ACCUM_SCHED"] == "1:0.06,2:0.2,4" and k["FF_ACCUM_LR"] == "0.5,0.7071" and k["FF_SOFTCAP"] == "15"
    assert k["FF_DEPTH"] == "9" and k["FF_SEED"] == "73" and k["FF_COOLDOWN_FRAC"] == "0.6"
    assert space["base"]["knobs"] == k and space["max_changes"] == 2
    sp = srch.load_space(EXAMPLES / "space-k59-local.json")
    assert 200 <= len(srch.gen_local(sp)) <= 400          # "hundreds of candidates"
    r = sim.load_recipe(EXAMPLES / "recipe-K59.json")
    res = sim.simulate(r, *sim.anchor_models().as_tuple(), n=500, rng=0, seed=73)
    assert abs(res.steps_mean - 2357) < 15 and abs(res.bpb_2m_mean - 0.96092) < 0.0005


def test_cli_simulate_search_report(tmp_path: Path, capsys):
    rc = cli_main(["simulate", "--models", "stub", "--recipe", str(EXAMPLES / "recipe-K59.json"), "--n", "300",
                   "--seed", "73", "--vs", str(EXAMPLES / "recipe-K59.json"), "--json", str(tmp_path / "sim.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "recipe K59" in out and "P(K59 < K59) = 0.500" in out and (tmp_path / "sim.json").exists()
    rc = cli_main(["search", "--models", "stub", "--space", str(EXAMPLES / "space-k59-local.json"), "--n-sims", "200",
                   "--gen", "oat", "--top", "5", "--out", str(tmp_path / "s.md")])
    out = capsys.readouterr().out
    assert rc == 0 and "ffsim search k59-local" in out
    assert (tmp_path / "s.md").exists() and (tmp_path / "s.csv").exists() and (tmp_path / "s.json").exists()
    rc = cli_main(["report", "--results", str(tmp_path / "s.json"), "--top", "3", "--out", str(tmp_path / "r.md")])
    out = capsys.readouterr().out
    assert rc == 0 and (tmp_path / "r.md").exists() and "wrote" in out
    rc = cli_main(["validate", "--models", "stub", "--runs", str(tmp_path / "missing.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0 and "offset model" in out and "anchor check" in out


def test_cli_missing_models_and_modules(tmp_path: Path, capsys):
    # no pickle AND no runs.jsonl to fit one from: exit 2 with both remedies named
    rc = cli_main(["simulate", "--models", str(tmp_path / "none.pkl"), "--runs", str(tmp_path / "none.jsonl"),
                   "--recipe", str(EXAMPLES / "recipe-K59.json")])
    out = capsys.readouterr().out
    assert rc == 2 and "--models stub" in out and "build-dataset" in out
    assert not (tmp_path / "none.pkl").exists()
    rc = cli_main(["fit", "--runs", str(tmp_path / "none.jsonl")])
    assert rc == 2 and "build-dataset" in capsys.readouterr().out


def test_cli_fits_missing_models_on_the_fly(tmp_path: Path, capsys, monkeypatch):
    """A missing models.pkl with a runs.jsonl at hand: validate/search/simulate fit it (via cli.fit_models),
    save it to the requested path and continue; the next call loads the pickle without refitting."""
    from ffsim import cli
    runs = tmp_path / "runs.jsonl"
    runs.write_text("", encoding="utf-8")
    calls = []

    def fake_fit(args, out):
        calls.append(str(out))
        m = models()
        sim.save_models(m, out)
        return m, {"records": [], "pairs": [], "uploads": [], "uploads_path": None,
                   "runs_path": str(runs), "pairs_path": ""}

    monkeypatch.setattr(cli, "fit_models", fake_fit)
    pkl = tmp_path / "auto" / "models.pkl"
    rc = cli_main(["validate", "--models", str(pkl), "--runs", str(runs)])
    out = capsys.readouterr().out
    assert rc == 0 and "fitting it on the fly" in out and "anchor check" in out
    assert pkl.exists() and calls == [str(pkl)]
    rc = cli_main(["search", "--models", str(pkl), "--runs", str(runs), "--space", str(EXAMPLES / "space-k59-local.json"),
                   "--n-sims", "50", "--gen", "oat", "--top", "3", "--out", str(tmp_path / "s.md")])
    out = capsys.readouterr().out
    assert rc == 0 and "fitting it on the fly" not in out and calls == [str(pkl)] and (tmp_path / "s.json").exists()
    rc = cli_main(["simulate", "--models", str(tmp_path / "auto2.pkl"), "--runs", str(runs),
                   "--recipe", str(EXAMPLES / "recipe-K59.json"), "--n", "50"])
    out = capsys.readouterr().out
    assert rc == 0 and "fitting it on the fly" in out and calls == [str(pkl), str(tmp_path / "auto2.pkl")]


@pytest.mark.skipif(not (ROOT / "research" / "sim-data" / "runs.jsonl").exists(), reason="needs the built dataset")
def test_cli_autofit_real_dataset(tmp_path: Path, capsys):
    """The README quickstart out of the box: `validate` with no models.pkl fits from research/sim-data."""
    pkl = tmp_path / "models.pkl"
    rc = cli_main(["validate", "--models", str(pkl)])
    out = capsys.readouterr().out
    assert rc == 0 and "fitting it on the fly" in out and "saved" in out and "anchor check" in out and pkl.exists()
    rc = cli_main(["simulate", "--models", str(pkl), "--recipe", str(EXAMPLES / "recipe-K59.json"), "--n", "200"])
    out = capsys.readouterr().out
    assert rc == 0 and "fitting it on the fly" not in out and "models: models.pkl" in out


def test_models_pickle_roundtrip(tmp_path: Path):
    m = models()
    sim.save_models(m, tmp_path / "m.pkl")
    back = sim.load_models(tmp_path / "m.pkl")
    assert back.meta == {"kind": "test-stub"}
    a = sim.simulate(k59(), *m.as_tuple(), n=200, rng=0)
    b = sim.simulate(k59(), *back.as_tuple(), n=200, rng=0)
    assert a == b
