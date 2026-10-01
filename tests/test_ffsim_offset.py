"""Review O1 regression tests for ffsim.offset at the simulate boundary.

The unit tests of OffsetModel itself live in tests/test_ffsim_quality.py (offset section). These pin what the O1
fix promised the *simulator*: an official-score draw carries the uncertainty of a mean estimated from six points
(predictive sd, not the bare upload-noise sd), the sd is quoted with its chi-square interval, and the model says
what it is calibrated for.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from ffsim.offset import CONTRACT_OFFSETS, OffsetModel, sd_confidence_interval
from ffsim.simulate import draw_noise, sample_offset

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS_CSV = os.path.join(os.path.dirname(HERE), "research", "sim-data", "official-uploads.csv")


def test_simulate_draws_offsets_with_the_predictive_sd():
    m = OffsetModel()
    n = 20000
    assert m.predictive_sd == pytest.approx(m.sd * math.sqrt(1.0 + 1.0 / len(CONTRACT_OFFSETS)))
    # the sd gap the test must resolve: sqrt(7/6) - 1 = 8% of 0.00044, i.e. 0.000035 on 20000 draws (se 0.000002)
    draws = sample_offset(m, np.random.default_rng(11), n)
    assert draws.shape == (n,)
    assert abs(float(draws.mean()) - m.mean) < 2e-5
    assert abs(float(draws.std()) - m.predictive_sd) < 1e-5
    assert abs(float(draws.std()) - m.sd) > 2e-5          # NOT the upload-noise-only sd
    # draw_noise (the common-random-numbers bundle every simulate/simulate_pair call uses) goes the same way
    noise = draw_noise(11, n, m)
    assert abs(float(noise.offset.std()) - m.predictive_sd) < 1e-5
    assert abs(float(noise.offset.mean()) - m.mean) < 2e-5
    # the pre-O1 draw is still reachable explicitly, and is narrower
    bare = m.sample(np.random.default_rng(11), n, with_mean_uncertainty=False)
    assert abs(float(bare.std()) - m.sd) < 1e-5 and float(bare.std()) < float(draws.std())


def test_offset_quotes_its_sd_interval_and_scope():
    m = OffsetModel()
    s = m.summary()
    lo, hi = s["sd_ci95"]
    assert (lo, hi) == pytest.approx(sd_confidence_interval(m.sd, m.sd_n))
    assert lo < m.sd < hi and hi / lo > 3.5           # a six-point sd is known to a factor of ~2 either way
    assert s["predictive_sd"] == pytest.approx(math.sqrt(s["sd"] ** 2 + s["mean_se"] ** 2))
    note = s["note"].lower()
    assert "chip c" in note and "k5x" in note and "runtime" in note and "chip d" in note
    # a model whose sd fell back to the contract's six keeps the contract's interval, not one for its 2 points
    thin = OffsetModel().fit([{"submission": "a", "offset": 0.0060}, {"submission": "b", "offset": 0.0070}])
    assert thin.n == 2 and thin.sd_n == 6 and thin.sd_ci95 == pytest.approx(OffsetModel().sd_ci95)


@pytest.mark.skipif(not os.path.exists(UPLOADS_CSV), reason="research/sim-data/official-uploads.csv not present")
def test_fitted_offset_predictive_sd_and_interval_on_the_real_table():
    m = OffsetModel().fit(UPLOADS_CSV)
    # seven chip-C points since K60 (29 Sep 19:50 UTC: C41 0.95910 -> 0.9655, +0.0064) joined K51 -> K59
    assert m.n == 7 and m.sd_n == 7
    assert m.mean == pytest.approx(0.00653, abs=5e-5)
    assert m.sd == pytest.approx(0.00042, abs=1e-5)
    assert m.mean_se == pytest.approx(0.00016, abs=1e-5)
    assert m.predictive_sd == pytest.approx(0.00045, abs=1e-5)
    lo, hi = m.sd_ci95
    assert lo == pytest.approx(0.00027, abs=1e-5) and hi == pytest.approx(0.00092, abs=1e-5)
    draws = sample_offset(m, np.random.default_rng(0), 20000)
    assert abs(float(draws.std()) - m.predictive_sd) < 1e-5
