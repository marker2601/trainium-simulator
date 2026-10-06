"""Tests for ffsim.quality (Part 2) and ffsim.offset (Part 3).

Two layers:

1. A synthetic campaign generated from a KNOWN truth of the same form as the surrogate, anchored so that the K59
   rehearsal record (M12, chip C, seed 73, 2357 steps) is exactly 0.96092:

       bpb = era + chipD - 0.057 L + 0.026 L^2 + sum_j beta_j x_j + seed + N(0, 0.00012)   (L = ln(steps/2300))

   Its eras M9 / M11 / M12 / M14 have DISTINCT true levels, so the synthetic fixture fits with ``era_groups={}``
   (every version its own era) and no bowl prior; the default grouping / bowl prior / leakage guard / record
   conventions have their own targeted tests below.
2. The real dataset (research/sim-data/runs.jsonl + validation-pairs.json + official-uploads.csv) when present:
   regression guards on the headline numbers of research/sim-data/quality-validation.md.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from ffsim.offset import (CONTRACT_OFFSETS, CONTRACT_SUBMISSIONS, OffsetModel, _chi2_ppf, load_official_uploads,
                          offsets_from_uploads, sd_confidence_interval)
from ffsim.quality import (BOWL_BASES, CODE_DEFAULTS, FEATURE_NAMES, FEATURES, K59_DEFAULTS, LINEAGE_IMPLIED_KNOBS,
                           QualityConfig, QualityModel, canonical_version, code_defaults_for, default_era_group,
                           knob_feature_dict, knob_features, pairs_from_records, parse_accum_sched, parse_version,
                           record_knobs)
from ffsim.schema import RunRecord

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_JSONL = os.path.join(REPO, "research", "sim-data", "runs.jsonl")
PAIRS_JSON = os.path.join(REPO, "research", "sim-data", "validation-pairs.json")
UPLOADS_CSV = os.path.join(REPO, "research", "sim-data", "official-uploads.csv")

ANCHOR_BPB, ANCHOR_STEPS = 0.96092, 2357

# ----------------------------------------------------------------------------- synthetic truth

TRUE_SLOPE, TRUE_CURV, TRUE_CHIP_D = -0.057, 0.026, 0.0003   # more steps -> lower bpb; the slope flattens
TRUE_SEED = {51: 0.0003, 53: 0.0002, 58: 0.0004, 67: -0.0003, 73: -0.0005, 79: 0.0006, 83: 0.0, 97: 0.0002,
             107: 0.0005, 137: 0.0007}
# per SCALED feature unit (see ffsim.quality.FEATURES)
TRUE_KNOB = {
    "leaky": 0.0054 * 0.15, "leaky_sq": 0.026 * 0.15 ** 2, "leaky_form_abs": 0.0003, "leaky_form_max": 0.0010,
    "relu2_fn": -0.0009, "softcap_a": -0.0002, "wd_sched": -0.0001, "mom_peak": 0.0008, "muon_beta2": -0.00025,
    "rope_log": 0.0002, "scalar_beta2": 0.0001, "qk_gain": 0.0001, "warmup_log": 0.0007, "emb_lr_log": 0.0001,
    "unemb_lr_log": 0.0, "key_offset_log": -0.00037, "acc_in_graph": -0.0006, "accum_until": -0.002,
    "accum_k1_until": -0.0002, "cooldown_frac": 0.0001, "mtp_off": 0.004, "mtp_p2": 0.0004, "xsa_on": 0.0005,
    "xsa_frac": 0.003, "depth": 0.001,
}
TRUE_BETA = np.array([TRUE_KNOB.get(n, 0.0) for n in FEATURE_NAMES])
ERA_DATE = {"M9": "2026-09-28", "M11": "2026-09-28", "M12": "2026-09-29", "M14": "2026-09-29"}
SYNTH_CONFIG = QualityConfig(era_groups={}, bowl_prior_mean=0.0)


def _era_base():
    L = math.log(ANCHOR_STEPS / 2300.0)
    m12 = ANCHOR_BPB - TRUE_SLOPE * L - TRUE_CURV * L * L - TRUE_SEED[73]
    return {"M12": m12, "M9": m12 + 0.0017, "M11": m12 + 0.0012, "M14": m12 - 0.0002}


ERA_BASE = _era_base()


def truth(knobs, steps, era, chip, seed):
    L = math.log(steps / 2300.0)
    return (ERA_BASE[era] + (TRUE_CHIP_D if chip == "D" else 0.0) + TRUE_SLOPE * L + TRUE_CURV * L * L
            + float(knob_features(knobs) @ TRUE_BETA) + TRUE_SEED.get(seed, 0.0))


def k59(**over):
    k = dict(K59_DEFAULTS)
    k.update({a: str(b) for a, b in over.items()})
    return k


K57 = k59(FF_LEAKY_RELU2="0")  # the pre-leaky recipe


def make_campaign(noise_sd=0.00012, rng_seed=7):
    """A synthetic two-week campaign: (records, pairs). Treatment arms carry control_of."""
    rng = np.random.default_rng(rng_seed)
    records, counter = [], [0]

    def add(era, chip, seed, knobs, base_steps, step_factor=1.0, name=None, control=None, noise=True,
            is_screen=False, bpb=None):
        counter[0] += 1
        steps = int(round(base_steps * step_factor * math.exp(rng.normal(0.0, 0.008)))) if not is_screen else 60
        y = truth(knobs, steps, era, chip, seed) + (rng.normal(0.0, noise_sd) if noise else 0.0)
        if bpb is not None:
            y = bpb
        run = name or f"{counter[0]:04d}_R_s{seed}"
        r = RunRecord(run_id=f"{chip}:{run}", chip=chip, run=run, code_version=era, seed=seed, steps=steps,
                      bpb_2m=y if not is_screen else None, is_screen=is_screen, knobs=dict(knobs),
                      knobs_complete=True, date_utc=ERA_DATE[era], control_of=control)
        records.append(r)
        return r

    # --- era M12 on chip C: LK0.35 controls and the last two days of arms
    ctl = {}
    for s in (58, 67, 73, 79, 83, 97):
        ctl[("M12", "C", s)] = add("M12", "C", s, k59(), 2325)
    anchor = add("M12", "C", 73, k59(), ANCHOR_STEPS, name="C40_K59", noise=False, bpb=ANCHOR_BPB)
    anchor.steps = ANCHOR_STEPS
    anchor.submission = "K59"
    arms = [
        (k59(FF_LEAKY_RELU2="0"), 0.99, (58, 67, 73)),
        (k59(FF_LEAKY_RELU2="0.25"), 0.995, (58, 73)),
        (k59(FF_LEAKY_RELU2="0.5"), 0.966, (58, 67, 73)),
        (k59(FF_LEAKY_RELU2="0", FF_RELU2_FN="1"), 1.01, (58, 67, 73)),
        (k59(FF_SOFTCAP_A="16.5"), 1.0, (58, 73)),
        (k59(FF_WD_SCHED="1"), 1.0, (58, 67, 73)),
        (k59(FF_WD_SCHED="2"), 1.0, (73,)),
        (k59(FF_MOM_PEAK="0.93"), 1.0, (73,)),
        (k59(FF_MOM_PEAK="0.97"), 1.0, (73,)),
        (k59(FF_MUON_BETA2="0.8"), 1.0, (73,)),
        (k59(FF_ROPE_BASE="300000"), 1.0, (73,)),
        (k59(FF_SCALAR_BETA2="0.99"), 1.0, (73,)),
        (k59(FF_QK_GAIN="1.6"), 1.0, (58, 67, 73)),
        (k59(FF_WARMUP="40"), 1.0, (73,)),
        (k59(FF_EMB_LR="0.36"), 1.0, (73,)),
        (k59(FF_UNEMB_LR="0.0072"), 1.0, (73,)),
        (k59(FF_XSA_LAYERS="0,1,2,3"), 0.966, (73,)),
        (k59(FF_XSA_LAYERS="all"), 0.91, (73,)),
        (k59(FF_LEAKY_RELU2="0.5", FF_LEAKY_FORM="abs"), 0.966, (73,)),
        (k59(FF_LEAKY_RELU2="0.5", FF_LEAKY_FORM="max"), 0.94, (73,)),
        (k59(FF_DEPTH="10"), 0.90, (73,)),
    ]
    for knobs, sf, seeds in arms:
        for s in seeds:
            add("M12", "C", s, knobs, 2325, sf, control=ctl[("M12", "C", s)].run_id)
    # --- era M12 on chip D (slower runtime; judged only against D's own control)
    for s in (73, 137):
        ctl[("M12", "D", s)] = add("M12", "D", s, k59(), 2325, 0.93)
    add("M12", "D", 73, k59(FF_SOFTCAP_A="16.5"), 2325, 0.93, control=ctl[("M12", "D", 73)].run_id)
    # --- era M9 (K56/K57 recipe: no leaky) on chips C and A
    for chip, seeds in (("C", (51, 58, 67, 73, 79, 83, 97, 107)), ("A", (53, 67))):
        for s in seeds:
            ctl[("M9", chip, s)] = add("M9", chip, s, K57, 2310)
    m9_arms = [
        ("C", k59(FF_LEAKY_RELU2="0", FF_ACC_IN_GRAPH="0"), 0.99, (58, 67, 73)),
        ("C", k59(FF_LEAKY_RELU2="0", FF_KEY_OFFSET_RAD="0.1"), 1.0, (58, 67, 73)),
        ("C", k59(FF_LEAKY_RELU2="0", FF_MTP_PHASES="0.2,0.5"), 1.0, (58,)),
        ("C", k59(FF_LEAKY_RELU2="0", FF_COOLDOWN_FRAC="0.55"), 1.0, (58,)),
        ("A", k59(FF_LEAKY_RELU2="0", FF_ACCUM_SCHED="4", FF_ACCUM_LR="1"), 1.02, (53, 67)),
        ("C", k59(FF_LEAKY_RELU2="0", FF_ACCUM_SCHED="2:0.2,4", FF_ACCUM_LR="0.5"), 1.005, (58, 67)),
        ("A", k59(FF_LEAKY_RELU2="0", FF_MTP="0"), 1.03, (67,)),
    ]
    for chip, knobs, sf, seeds in m9_arms:
        for s in seeds:
            add("M9", chip, s, knobs, 2310, sf, control=ctl[("M9", chip, s)].run_id)
    # --- era M11 (M9 + flags): one control and three arms at s73
    ctl[("M11", "C", 73)] = add("M11", "C", 73, K57, 2310)
    for knobs, sf in ((k59(FF_LEAKY_RELU2="0", FF_WD_SCHED="1"), 1.0), (k59(FF_LEAKY_RELU2="0", FF_MOM_PEAK="0.97"), 1.0),
                      (k59(FF_LEAKY_RELU2="0.5", FF_LEAKY_FORM="max"), 0.94)):
        add("M11", "C", 73, knobs, 2310, sf, control=ctl[("M11", "C", 73)].run_id)
    # --- era M14: two controls and one arm
    for s in (58, 73):
        ctl[("M14", "C", s)] = add("M14", "C", s, k59(), 2330)
    add("M14", "C", 73, k59(FF_SOFTCAP_A="16.5"), 2330, control=ctl[("M14", "C", 73)].run_id)
    # --- things the fit must ignore: screens and a run without a bpb
    add("M12", "C", 73, k59(FF_QK_GAIN="1.6"), 2325, is_screen=True, name="S60_qk16")
    nobpb = add("M12", "C", 58, k59(), 2325, name="crashed_s58")
    nobpb.bpb_2m = None
    return records, pairs_from_records(records)


@pytest.fixture(scope="module")
def campaign():
    return make_campaign()


@pytest.fixture(scope="module")
def model(campaign):
    records, pairs = campaign
    return QualityModel(SYNTH_CONFIG).fit(records, pairs)


# ----------------------------------------------------------------------------- feature table


def test_k59_reference_is_all_zero():
    x = knob_features(K59_DEFAULTS)
    assert x.shape == (len(FEATURES),)
    assert np.all(x == 0.0)
    assert np.all(knob_features({}) == 0.0)      # a CANDIDATE's missing knobs read as the K59 defaults
    assert np.all(knob_features(None) == 0.0)


def test_feature_table_is_consistent():
    names = [f.name for f in FEATURES]
    assert len(set(names)) == len(names)
    d = knob_feature_dict(K59_DEFAULTS)
    assert set(d) == set(names)
    assert all(f.scale > 0 for f in FEATURES)
    assert all(f.transform and f.knobs for f in FEATURES)
    bowls = [f for f in FEATURES if f.bowl]
    assert {f.base for f in bowls} == set(BOWL_BASES) | {"leaky"}
    assert all(f.name == f"{f.base}_sq" for f in bowls)
    assert all(f.base in names for f in bowls)
    # every feature knob has a K59 default and a code default
    for f in FEATURES:
        for k in f.knobs:
            assert k in K59_DEFAULTS and k in CODE_DEFAULTS, k


def test_feature_encodings():
    d = knob_feature_dict(k59(FF_LEAKY_RELU2="0", FF_RELU2_FN="1"))
    assert d["leaky"] == pytest.approx(-0.35) and d["leaky_sq"] == pytest.approx(0.1225)
    assert d["relu2_fn"] == 1.0 and d["leaky_form_abs"] == 0.0
    d = knob_feature_dict(k59(FF_LEAKY_RELU2="0.5", FF_LEAKY_FORM="max"))
    assert d["leaky_form_max"] == 1.0 and d["leaky_form_abs"] == 0.0
    d = knob_feature_dict(k59(FF_SOFTCAP_A="16.5"))
    assert d["softcap_a"] == pytest.approx(0.1)
    d = knob_feature_dict(k59(FF_SOFTCAP="0"))
    assert d["softcap_off"] == 1.0 and d["softcap_log"] == 0.0 and d["softcap_a"] == 0.0
    d = knob_feature_dict(k59(FF_ROPE_BASE="300000"))
    assert d["rope_log"] == pytest.approx(math.log(3.0))
    assert knob_features(k59(FF_ROPE_BASE="300000"))[FEATURE_NAMES.index("rope_log")] == pytest.approx(1.0)
    d = knob_feature_dict(k59(FF_KEY_OFFSET="0"))
    assert d["key_offset_off"] == 1.0 and d["key_offset_log"] == 0.0
    d = knob_feature_dict(k59(FF_KEY_OFFSET_RAD="0.1"))
    assert d["key_offset_off"] == 0.0 and d["key_offset_log"] == pytest.approx(math.log(1.0 / 3.0))
    assert knob_features(k59(FF_KEY_OFFSET_RAD="0.1"))[FEATURE_NAMES.index("key_offset_log")] == pytest.approx(-1.0)
    d = knob_feature_dict(k59(FF_XSA_LAYERS="0,1,2,3"))
    assert d["xsa_on"] == 1.0 and d["xsa_frac"] == pytest.approx(4 / 9)
    assert knob_feature_dict(k59(FF_XSA_LAYERS="all"))["xsa_frac"] == 1.0
    d = knob_feature_dict(k59(FF_MTP="0"))
    assert d["mtp_off"] == 1.0 and d["mtp_p1"] == 0.0 and d["mtp_p2"] == 0.0 and d["mtp_w_tail"] == 0.0
    assert knob_feature_dict(k59(FF_MTP_PHASES="0.2,0.5"))["mtp_p2"] == pytest.approx(0.05)
    assert knob_feature_dict(k59(FF_MTP_W="1,0.6,0.3"))["mtp_w_tail"] == pytest.approx(0.15)
    d = knob_feature_dict(k59(FF_ACCUM_SCHED="4", FF_ACCUM_LR="1"))
    assert d["accum_until"] == pytest.approx(-0.2) and d["accum_k1_until"] == pytest.approx(-0.06)
    assert d["accum_lr0"] == 0.0 and d["accum_lr1"] == 0.0
    d = knob_feature_dict(k59(FF_ACCUM_SCHED="2:0.2,4", FF_ACCUM_LR="0.6"))
    assert d["accum_until"] == pytest.approx(0.0) and d["accum_k1_until"] == pytest.approx(-0.06)
    assert d["accum_lr0"] == pytest.approx(0.1) and d["accum_lr1"] == 0.0
    assert knob_feature_dict(k59(FF_TOTAL_BATCH="131072"))["total_batch_log2"] == pytest.approx(-1.0)
    assert knob_feature_dict(k59(FF_WD_SCHED="p"))["wd_sched"] == 1.0
    # the features added for the 29 Sep pairs
    d = knob_feature_dict(k59(FF_ATTN_SRC="5:6,7,8", FF_ROPE_FN="1", FF_RMSNORM_FN="1", FF_QK_GAIN_FOLD="0"))
    assert d["attn_src"] == 1.0 and d["fn_flags"] == pytest.approx(2 / 3)
    d = knob_feature_dict(k59(FF_MATRIX_LR_SCALE="2.3", FF_WD="0.03", FF_ADAMW_LR_SCALE="1.7", FF_SCALAR_LR="0.1664"))
    assert d["matrix_lr_log"] == pytest.approx(math.log(1.15)) and d["wd_log"] == pytest.approx(math.log(1.5))
    assert d["adamw_lr_log"] == pytest.approx(math.log(1.7 / 1.4142))
    assert d["scalar_lr_log"] == pytest.approx(math.log(0.1664 / 0.2))
    d = knob_feature_dict(k59(FF_CE_BF16="1", FF_PTP_W="0.25", FF_MOM_RAMP="450"))
    assert d["ce_bf16"] == 1.0 and d["ptp_w"] == 0.25 and d["mom_ramp_log"] == pytest.approx(math.log(1.5))
    # bowls are the square of their base, in raw units
    d = knob_feature_dict(k59(FF_MOM_PEAK="0.97"))
    assert d["mom_peak"] == pytest.approx(0.02) and d["mom_peak_sq"] == pytest.approx(0.02 ** 2)
    x = knob_features(k59(FF_MOM_PEAK="0.97"))
    assert x[FEATURE_NAMES.index("mom_peak")] == pytest.approx(1.0)
    assert x[FEATURE_NAMES.index("mom_peak_sq")] == pytest.approx(1.0)


def test_record_convention_code_defaults():
    """A record without FF_LEAKY_RELU2 in its env ran the K57 recipe (LK 0), not K59's 0.35."""
    d = knob_feature_dict({}, CODE_DEFAULTS)
    assert d["leaky"] == pytest.approx(-0.35) and d["key_offset_off"] == 1.0 and d["acc_in_graph"] == -1.0
    assert d["mtp_off"] == 1.0
    assert knob_feature_dict({})["leaky"] == 0.0
    assert CODE_DEFAULTS["FF_LEAKY_RELU2"] == "0" and K59_DEFAULTS["FF_LEAKY_RELU2"] == "0.35"
    # the rotary key-offset default moved 0.1 -> 0.3 at K55 / M8
    assert code_defaults_for("M7")["FF_KEY_OFFSET_RAD"] == "0.1"
    assert code_defaults_for("M4")["FF_KEY_OFFSET_RAD"] == "0.1"
    assert code_defaults_for("M8")["FF_KEY_OFFSET_RAD"] == "0.3"
    assert code_defaults_for("M12") == CODE_DEFAULTS and code_defaults_for(None) == CODE_DEFAULTS
    m7 = knob_feature_dict({"FF_KEY_OFFSET": "1"}, code_defaults_for("M7"))
    assert m7["key_offset_log"] == pytest.approx(math.log(0.1 / 0.3))
    m9 = knob_feature_dict({"FF_KEY_OFFSET": "1"}, code_defaults_for("M9"))
    assert m9["key_offset_log"] == 0.0


def test_lineage_implied_knobs():
    r = RunRecord(run_id="C:x", chip="C", run="x", code_version="code_m10b", tags=["lineage:M10b"],
                  knobs={"FF_LEAKY_RELU2": "0.5"})
    k = record_knobs(r)
    assert k["FF_LEAKY_FORM"] == "abs" and k["FF_LEAKY_RELU2"] == "0.5"
    assert knob_feature_dict(k, CODE_DEFAULTS)["leaky_form_abs"] == 1.0
    r.knobs["FF_LEAKY_FORM"] = "fnm"                      # an explicit knob wins over the lineage
    assert record_knobs(r)["FF_LEAKY_FORM"] == "fnm"
    assert set(LINEAGE_IMPLIED_KNOBS) == {"M10b", "M11c"}


def test_numeric_and_string_knob_values_agree():
    a = knob_features({"FF_LEAKY_RELU2": "0.5", "FF_QK_GAIN": "1.6", "FF_WARMUP": "40"})
    b = knob_features({"FF_LEAKY_RELU2": 0.5, "FF_QK_GAIN": 1.6, "FF_WARMUP": 40})
    assert np.allclose(a, b)


def test_garbage_knob_values_fall_back_to_defaults():
    x = knob_features({"FF_LEAKY_RELU2": "banana", "FF_ACCUM_SCHED": "??", "FF_MTP_PHASES": "x,y", "FF_ROPE_BASE": ""})
    assert np.all(np.isfinite(x))
    assert x[FEATURE_NAMES.index("leaky")] == 0.0
    assert x[FEATURE_NAMES.index("accum_until")] == pytest.approx(-1.0)   # unparseable -> plain K, no warm-up


def test_parse_accum_sched():
    assert parse_accum_sched("1:0.06,2:0.2,4") == (1, 0.06, 2, 0.2, 4)
    assert parse_accum_sched("2:0.2,4") == (2, 0.2, None, 0.0, 4)
    assert parse_accum_sched("4") == (None, 0.0, None, 0.0, 4)
    assert parse_accum_sched(None) == (1, 0.06, 2, 0.2, 4)


def test_parse_version_and_lineage():
    assert parse_version("M12") == ("M", 12, "")
    assert parse_version("M11c") == ("M", 11, "c")
    assert parse_version("code_m9") == ("M", 9, "")
    assert parse_version("K57") == ("K", 57, "")
    assert parse_version("code_k12off") == ("K", 12, "off")
    assert parse_version("") is None and parse_version(None) is None and parse_version("master") is None
    assert canonical_version("code_k12off", ["recipe:LK0.35", "lineage:M12"]) == "M12"
    assert canonical_version("code_m9") == "M9" and canonical_version("code_m11c") == "M11c"
    assert canonical_version("M11c") == "M11c" and canonical_version("ad7f75a5") == "ad7f75a5"
    assert canonical_version("code_k12off") == "code_k12off"     # no tag: the raw dir name (aliases learn it)
    assert canonical_version(None) is None and canonical_version("", ["lineage:"]) is None
    assert default_era_group("M9") == "M9+" and default_era_group("M14") == "M9+" and default_era_group("M15") == "M9+"
    assert default_era_group("M7") == "M7" and default_era_group("M3") == "M3" and default_era_group("code") == "code"


# ----------------------------------------------------------------------------- fitting


def test_fit_uses_full_runs_only(campaign, model):
    records, _ = campaign
    assert model.dropped["is_screen"] == 1
    assert model.dropped["no_bpb"] == 1
    assert model.n == len(records) - 2
    assert "C:S60_qk16" not in model.records_by_id and "C:crashed_s58" not in model.records_by_id
    assert set(model.eras) == {"M9", "M11", "M12", "M14"}
    assert model.era_merges == {}
    assert set(model.chips) == {"A", "D"}
    assert 73 in model.seeds and 137 in model.seeds


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        QualityModel().fit([])
    with pytest.raises(ValueError):
        QualityModel().fit([RunRecord(run_id="C:x", chip="C", run="x", is_screen=True, bpb_2m=0.96, steps=2300,
                                      code_version="M12")])


def test_fit_is_deterministic(campaign):
    records, pairs = campaign
    a = QualityModel(SYNTH_CONFIG).fit(records, pairs)
    b = QualityModel(SYNTH_CONFIG).fit(records, pairs)
    assert np.array_equal(a.beta, b.beta)
    assert a.sigma == b.sigma


def test_anchor_k59_prediction(model):
    mean, sd = model.predict(K59_DEFAULTS, ANCHOR_STEPS, "M12", chip="C", seed=73)
    assert abs(mean - ANCHOR_BPB) < 0.0004, (mean, sd)
    assert 0.0 < sd < 0.001
    # the seed-averaged prediction is the anchor minus seed 73's (negative) effect, with a wider sd
    mean_avg, sd_avg = model.predict(K59_DEFAULTS, ANCHOR_STEPS, "M12")
    assert abs((mean_avg - mean) - (-TRUE_SEED[73])) < 0.0004
    assert sd_avg > sd


def test_steps_slope_and_curvature(model):
    s = model.summary()
    b, b_sd = s["slope"]
    assert abs(b - TRUE_SLOPE) < 0.012, s["slope"]
    assert b_sd < 0.006
    c, _ = s["curvature"]
    assert abs(c - TRUE_CURV) < 0.02
    # 1% more steps ~ -0.00057 bpb around 2300
    m0, _ = model.predict(K59_DEFAULTS, 2300, "M12", seed=73)
    m1, _ = model.predict(K59_DEFAULTS, 2323, "M12", seed=73)
    assert -0.0008 < (m1 - m0) < -0.0003
    assert model.steps_slope() == pytest.approx(b)
    assert model.steps_slope(900) == pytest.approx(b + 2 * c * math.log(900 / 2300))
    assert QualityModel().steps_slope() == pytest.approx(-0.057)


def test_slope_prior_holds_when_steps_never_vary():
    """With every run at the same step count the slope is not identified: it must stay at the prior."""
    recs = []
    for s in (58, 67, 73):
        recs.append(RunRecord(run_id=f"C:a{s}", chip="C", run=f"a{s}", code_version="M12", seed=s, steps=2300,
                              bpb_2m=0.9615 + 0.0002 * (s - 67) / 10, knobs=k59()))
    m = QualityModel().fit(recs)
    b, b_sd = m.summary()["slope"]
    assert b == pytest.approx(-0.057, abs=1e-9)
    assert b_sd == pytest.approx(0.005, abs=1e-6)


def test_knob_effects_recovered(model):
    knobs = model.summary()["knobs"]
    for name in ("mtp_off", "xsa_frac", "accum_until", "leaky_sq", "acc_in_graph", "relu2_fn", "warmup_log"):
        est, sd = knobs[name]
        assert abs(est - TRUE_KNOB[name]) < 0.0007, (name, est, TRUE_KNOB[name])
    # the LK slope bowl: 0 and 0.5 are worse than 0.35
    base, _ = model.predict(k59(), 2325, "M12", seed=73)
    lk0, _ = model.predict(k59(FF_LEAKY_RELU2="0"), 2325, "M12", seed=73)
    lk5, _ = model.predict(k59(FF_LEAKY_RELU2="0.5"), 2325, "M12", seed=73)
    assert lk0 > base + 0.0005 and lk5 > base + 0.0005


def test_holdout_predictions(campaign, model):
    """Fresh runs from the same truth (new seeds -> seed-averaged, and seen seeds) are predicted within noise."""
    rng = np.random.default_rng(11)
    errs = []
    for knobs, steps, era, chip, seed in [
        (k59(FF_QK_GAIN="1.6", FF_WARMUP="40"), 2300, "M12", "C", 73),
        (k59(FF_LEAKY_RELU2="0.25"), 2340, "M12", "C", 67),
        (k59(FF_SOFTCAP_A="16.5", FF_WD_SCHED="1"), 2280, "M12", "C", 58),
        (k59(), 2200, "M12", "D", 137),
        (K57, 2310, "M9", "C", 79),
        (k59(FF_LEAKY_RELU2="0", FF_KEY_OFFSET_RAD="0.1"), 2300, "M9", "A", 67),
        (k59(), 2330, "M14", "C", 58),
    ]:
        y = truth(knobs, steps, era, chip, seed) + rng.normal(0.0, 0.00012)
        mean, sd = model.predict(knobs, steps, era, chip=chip, seed=seed)
        errs.append(mean - y)
        assert abs(mean - y) < 3.5 * sd + 0.0002, (knobs, mean, y, sd)
    assert math.sqrt(float(np.mean(np.square(errs)))) < 0.0008
    # a brand-new seed: the seed-averaged truth, wider sd
    y_new = truth(k59(), 2325, "M12", "C", 999)
    mean, sd = model.predict(k59(), 2325, "M12", seed=999)
    assert abs(mean - y_new) < 0.0006
    assert sd > model.config.seed_prior_sd


def test_seed_effects_are_shrunk_random_effects(model):
    seeds = model.summary()["seeds"]
    for s, true in TRUE_SEED.items():
        est, sd = seeds[s]
        assert abs(est - true) < 0.0005, (s, est, true)
        assert sd < model.config.seed_prior_sd  # the data narrowed the prior
    # a seed seen once (137, chip D) is shrunk toward 0 more than a seed seen many times (73)
    _, sd137 = seeds[137]
    _, sd73 = seeds[73]
    assert sd137 > sd73
    assert 0.0 < seeds[137][0] < TRUE_SEED[137] + 0.0004
    assert model.summary()["seed_counts"][73] > model.summary()["seed_counts"][137] == 1
    # unseen seed -> wider sd than a seen seed
    _, sd_seen = model.predict(k59(), 2325, "M12", seed=73)
    _, sd_unseen = model.predict(k59(), 2325, "M12", seed=4242)
    _, sd_none = model.predict(k59(), 2325, "M12", seed=None)
    assert sd_unseen > sd_seen and sd_none == pytest.approx(sd_unseen)


def test_era_lineage_fallback(model):
    assert model.resolve_era("M12") == ("M12", 0, False)
    assert model.resolve_era("M15") == ("M14", 1, False)
    assert model.resolve_era("M16") == ("M14", 2, False)
    assert model.resolve_era("M13") == ("M12", 1, False)      # prefers the older neighbour
    assert model.resolve_era("M11c") == ("M11", 1, False)
    assert model.resolve_era("code_m9") == ("M9", 0, False)
    assert model.resolve_era("code_k12off", ["lineage:M12"]) == ("M12", 0, False)
    era, dist, cross = model.resolve_era("K57")
    assert cross and dist == 1 and era in ("M12", "M14")       # latest-dated era of another lineage
    era, dist, cross = model.resolve_era(None)
    assert cross and era in model.eras
    m14, sd14 = model.predict(k59(), 2330, "M14", seed=73)
    m15, sd15 = model.predict(k59(), 2330, "M15", seed=73)
    m16, sd16 = model.predict(k59(), 2330, "M16", seed=73)
    assert m15 == pytest.approx(m14) and m16 == pytest.approx(m14)
    assert sd15 > sd14 and sd16 > sd15
    _, sd_k = model.predict(k59(), 2330, "K99", seed=73)
    assert sd_k > sd16
    _, _, notes = model.design_row(k59(), 2330, "M15", "C", 73)
    assert any("unseen" in n for n in notes)


def test_era_effects_ordering(model):
    eras = model.era_effects()
    for e, base in ERA_BASE.items():
        assert abs(eras[e][0] - base) < 0.0006, (e, eras[e], base)


def test_default_grouping_merges_the_flag_gated_family(campaign):
    """Default config: M9 / M11 / M12 / M14 are one era ``M9+``; version aliases and unseen versions resolve."""
    records, pairs = campaign
    m = QualityModel().fit(records, pairs)
    assert set(m.eras) == {"M9+"}
    assert set(m.era_members["M9+"]) == {"M9", "M11", "M12", "M14"}
    assert m.resolve_era("M12") == ("M9+", 0, False)
    assert m.resolve_era("code_k12off", ["lineage:M12"]) == ("M9+", 0, False)
    assert m.resolve_era("M13") == ("M9+", 1, False)        # never run: one generation from M12
    assert m.resolve_era("M15") == ("M9+", 1, False)
    assert m.summary()["era_grouping"].startswith("default")
    # the anchor still holds within the synthetic era spread
    mean, _ = m.predict(K59_DEFAULTS, ANCHOR_STEPS, "M12", chip="C", seed=73)
    assert abs(mean - ANCHOR_BPB) < 0.0015


def _controls(era, seeds, knobs, steps=2320, chip="C", base=0.9615, prefix="ctl"):
    return [RunRecord(run_id=f"{chip}:{prefix}_{era}_{s}", chip=chip, run=f"{prefix}_{era}_{s}", code_version=era,
                      seed=s, steps=steps, bpb_2m=base + 0.0001 * (i - 1), knobs=dict(knobs), knobs_complete=True,
                      sources=["chip_log"]) for i, s in enumerate(seeds)]


def test_leakage_guard_single_run_era_shares_its_parent():
    """A version seen once (M13, the only run with the treatment knob) must not get its own era: its era is its
    parent lineage's and the treatment lives in the knob feature."""
    recs = _controls("M12", (58, 67, 73), k59())
    recs.append(RunRecord(run_id="C:asa", chip="C", run="asa", code_version="code_k13off", tags=["lineage:M13"], seed=73,
                          steps=2320, bpb_2m=0.9615 - 0.0018, knobs=k59(FF_ATTN_SRC="5:6,7,8"), knobs_complete=True,
                          sources=["chip_log"]))
    m = QualityModel(QualityConfig(era_groups={})).fit(recs, [])
    assert "M13" not in m.eras and set(m.eras) == {"M12"}
    assert m.era_merges == {"M13": "M12"}
    assert m.era_members["M12"] == ["M12", "M13"]
    assert m.version_to_era["M13"] == "M12" and m.version_aliases["code_k13off"] == "M13"
    est, _ = m.summary()["knobs"]["attn_src"]
    assert est < -0.0005                                   # the treatment is carried by the knob, not an era
    assert m.resolve_era("M13") == ("M12", 0, False)
    assert m.resolve_era("code_k13off") == ("M12", 0, False)
    # with two runs the era is kept (min_era_runs = 2), with min_era_runs=1 nothing is merged
    recs2 = recs + [RunRecord(run_id="C:asa2", chip="C", run="asa2", code_version="M13", seed=58, steps=2320,
                              bpb_2m=0.9615 - 0.0016, knobs=k59(FF_ATTN_SRC="5:6,7,8"), knobs_complete=True,
                              sources=["chip_log"])]
    m2 = QualityModel(QualityConfig(era_groups={})).fit(recs2, [])
    assert set(m2.eras) == {"M12", "M13"} and m2.era_merges == {}
    m3 = QualityModel(QualityConfig(era_groups={}, min_era_runs=1)).fit(recs, [])
    assert set(m3.eras) == {"M12", "M13"}


def test_unknown_era_and_monitor_only_records_are_not_fitted():
    recs = _controls("M12", (58, 67, 73), k59())
    recs.append(RunRecord(run_id="C:old", chip="C", run="old", code_version="code", seed=73, steps=900, bpb_2m=1.05,
                          knobs=k59(), knobs_complete=True, sources=["chip_log"]))
    recs.append(RunRecord(run_id="A:desc:G4_s67", chip="A", run="desc:G4_s67", code_version="M12", seed=67, steps=2140,
                          bpb_2m=0.9651, knobs={"FF_MUON_ZERO2": "1"}, knobs_complete=False, sources=["monitor"]))
    m = QualityModel().fit(recs, [])
    assert m.n == 3
    assert m.dropped["unknown_era"] == 1 and m.dropped["no_effective_knobs"] == 1
    assert set(m.holdout) == {"A:desc:G4_s67"}
    # a record with no sources at all (synthetic / hand-made) is kept even without knobs_complete
    bare = RunRecord(run_id="C:bare", chip="C", run="bare", code_version="M12", seed=79, steps=2320, bpb_2m=0.9617, knobs=k59())
    assert QualityModel().fit(recs + [bare], []).n == 4


def test_predict_record_uses_the_record_convention():
    recs = _controls("M9", (58, 67, 73), {"FF_KEY_OFFSET": "1", "FF_ACC_IN_GRAPH": "1", "FF_MTP": "1",
                                          "FF_TIME_TARGET": "1760"}, base=0.9630)
    recs += _controls("M12", (58, 67, 73), {"FF_KEY_OFFSET": "1", "FF_ACC_IN_GRAPH": "1", "FF_MTP": "1",
                                            "FF_TIME_TARGET": "1760", "FF_LEAKY_RELU2": "0.35",
                                            "FF_LEAKY_FORM": "fnm"}, base=0.9617, prefix="lk")
    m = QualityModel().fit(recs, [])
    k57 = recs[0]
    rec_mean, _ = m.predict_record(k57)
    assert abs(rec_mean - k57.bpb_2m) < 0.0004               # read as LK 0 (code default)
    cand_mean, _ = m.predict(k57.knobs, k57.steps, "M9", "C", k57.seed)
    assert cand_mean < rec_mean - 0.0008                      # the same dict as a CANDIDATE reads LK 0.35
    lk = recs[3]
    assert abs(m.predict_record(lk)[0] - lk.bpb_2m) < 0.0004
    assert abs(m.predict(lk.knobs, lk.steps, "M12", "C", lk.seed)[0] - lk.bpb_2m) < 0.0004
    assert m.support["FF_LEAKY_RELU2"] == ["0", "0.35"]       # effective values, defaults included
    assert "FF_TIME_TARGET" in m.support


def test_bowl_prior_expresses_a_tuned_optimum():
    """MOM_PEAK 0.93 and 0.97 both worse than 0.95: a bowl. The default prior predicts the unseen side as worse
    too; with bowl_prior_mean=0 a single observed side is extrapolated linearly."""
    recs = _controls("M12", (58, 67, 73, 79), k59())
    recs.append(RunRecord(run_id="C:mp97", chip="C", run="mp97", code_version="M12", seed=73, steps=2320,
                          bpb_2m=0.9615 + 0.0020, knobs=k59(FF_MOM_PEAK="0.97"), knobs_complete=True, sources=["chip_log"]))
    both = recs + [RunRecord(run_id="C:mp93", chip="C", run="mp93", code_version="M12", seed=73, steps=2320,
                             bpb_2m=0.9615 + 0.0005, knobs=k59(FF_MOM_PEAK="0.93"), knobs_complete=True, sources=["chip_log"])]
    m = QualityModel().fit(both, [])
    base = m.predict(k59(), 2320, "M12", "C", 73)[0]
    assert m.predict(k59(FF_MOM_PEAK="0.97"), 2320, "M12", "C", 73)[0] > base + 0.001
    assert m.predict(k59(FF_MOM_PEAK="0.93"), 2320, "M12", "C", 73)[0] > base + 0.0002
    one = QualityModel().fit(recs, [])
    flat = QualityModel(QualityConfig(bowl_prior_mean=0.0)).fit(recs, [])
    d_one = one.predict(k59(FF_MOM_PEAK="0.93"), 2320, "M12", "C", 73)[0] - one.predict(k59(), 2320, "M12", "C", 73)[0]
    d_flat = flat.predict(k59(FF_MOM_PEAK="0.93"), 2320, "M12", "C", 73)[0] - flat.predict(k59(), 2320, "M12", "C", 73)[0]
    assert d_one > d_flat
    coefs = {c["name"]: c for c in one.coefficients()}
    assert coefs["knob:mom_peak_sq"]["prior_mean"] == pytest.approx(0.0005)
    assert coefs["knob:mom_peak"]["prior_mean"] == 0.0


def test_chip_d_fixed_effect(model):
    est, sd = model.summary()["chips"]["D"]
    assert abs(est - TRUE_CHIP_D) < 0.0005
    mc, _ = model.predict(k59(), 2160, "M12", chip="C", seed=73)
    md, _ = model.predict(k59(), 2160, "M12", chip="D", seed=73)
    assert md - mc == pytest.approx(est)
    _, sd_unseen_chip = model.predict(k59(), 2160, "M12", chip="B", seed=73)
    _, sd_c = model.predict(k59(), 2160, "M12", chip="C", seed=73)
    assert sd_unseen_chip > sd_c


def test_predict_validates_inputs(model):
    with pytest.raises(ValueError):
        model.predict(k59(), 0, "M12")
    with pytest.raises(RuntimeError):
        QualityModel().predict(k59(), 2300, "M12")


# ----------------------------------------------------------------------------- paired validation


def test_pair_validation_leave_one_pair_out(campaign, model):
    _, pairs = campaign
    assert 20 <= len(pairs) < 120
    rep = model.pair_validation()
    assert rep["mode"] == "leave-one-pair-out"
    assert rep["n"] == len(pairs) and rep["n_skipped"] == 0
    assert set(rep) >= {"n", "sign_agreement", "mae", "rmse", "coverage_2sd", "rows", "targets", "pass", "n_decisive",
                        "sign_agreement_decisive", "mae_decisive", "by_family"}
    assert rep["targets"] == {"sign_agreement": 0.8, "mae": 0.0003}
    assert rep["mae"] <= 0.0003, rep["mae"]
    assert rep["sign_agreement"] >= 0.8, rep["sign_agreement"]
    assert rep["pass"] == {"sign_agreement": True, "mae": True}
    assert rep["coverage_2sd"] >= 0.9
    row = rep["rows"][0]
    for key in ("name", "treatment", "control", "observed", "predicted", "pred_sd", "error", "z", "sign_ok", "decisive",
                "steps_t", "steps_c", "seed_t", "seed_c", "chip_t", "chip_c", "code_t", "code_c", "era_t", "era_c",
                "lopo_refit", "holdout", "unsupported_features", "extrapolated_features", "notes"):
        assert key in row
    assert all(r["pred_sd"] > 0 and r["lopo_refit"] and not r["holdout"] for r in rep["rows"])
    assert all(r["seed_t"] == r["seed_c"] for r in rep["rows"])
    # the D-chip pair was judged against D's own control
    d_rows = [r for r in rep["rows"] if r["chip_t"] == "D"]
    assert d_rows and all(r["chip_c"] == "D" for r in d_rows)
    # knobs with a single pair are unsupported once that pair is left out (the refit answers with the prior)
    unsup = [r for r in rep["rows"] if r["unsupported_features"]]
    assert rep["n_unsupported"] == len(unsup) and 0 < len(unsup) < rep["n"]
    flagged = {name for r in unsup for name in r["unsupported_features"]}
    assert {"muon_beta2", "warmup_log", "depth"} <= flagged          # one pair each in the campaign
    assert "mom_peak" not in flagged and "qk_gain" not in flagged   # two / three arms each
    assert all(r["notes"] for r in unsup)
    assert rep["mae_supported"] <= rep["mae"] and rep["sign_agreement_supported"] >= 0.9
    # QK gain 1.6 has three pairs: leaving one out keeps it supported
    qk = [r for r in rep["rows"] if r["treatment"] in
          {x.run_id for x in campaign[0] if x.knobs.get("FF_QK_GAIN") == "1.6" and not x.is_screen}]
    assert qk and all(not r["unsupported_features"] for r in qk)


def test_pair_validation_in_sample_and_explicit_pairs(campaign, model):
    records, pairs = campaign
    rep = model.pair_validation(pairs, leave_pair_out=False)
    assert rep["mode"] == "in-sample" and rep["n"] == len(pairs)
    assert rep["mae"] <= 0.0003
    assert rep["n_unsupported"] == 0 and rep["mae_supported"] == rep["mae"]
    assert all(not r["lopo_refit"] for r in rep["rows"])
    # explicit delta_raw and alias keys are honoured; unknown runs are skipped, not fatal
    t = next(r for r in records if r.control_of)
    custom = [{"name": "custom", "arm": t.run_id, "ctl": t.control_of, "delta_raw": -0.001},
              {"name": "ghost", "treatment": "C:nope", "control": t.control_of},
              {"name": "screen", "treatment": "C:S60_qk16", "control": t.control_of}]
    rep = model.pair_validation(custom, leave_pair_out=False)
    assert rep["n"] == 1 and rep["n_skipped"] == 2
    assert rep["rows"][0]["observed"] == -0.001
    assert rep["skipped"][0]["treatment"] == "C:nope"
    empty = model.pair_validation([], leave_pair_out=False)
    assert empty["n"] == 0 and empty["sign_agreement"] is None and empty["mae"] is None
    assert empty["pass"] == {"sign_agreement": False, "mae": False}


def test_pair_validation_accepts_validation_pairs_json_rows(campaign, model):
    """The research/sim-data/validation-pairs.json row shape: treatment_run + chip, control_run + control_chip."""
    records, _ = campaign
    t = next(r for r in records if r.control_of and r.chip == "C")
    c = model.records_by_id[t.control_of]
    row = {"pair_id": "json_row", "family": "demo", "in_task_list": True, "chip": "C", "treatment_run": t.run,
           "control_run": c.run, "control_chip": "C", "treatment_bpb": t.bpb_2m, "control_bpb": c.bpb_2m,
           "seed": t.seed, "control_seed": c.seed, "delta_raw": t.bpb_2m - c.bpb_2m,
           "knob_changes": {k: v for k, v in t.knobs.items() if c.knobs.get(k) != v}, "control_knobs": {}}
    rep = model.pair_validation([row], leave_pair_out=False)
    assert rep["n"] == 1 and rep["rows"][0]["name"] == "json_row"
    assert rep["rows"][0]["family"] == "demo" and rep["rows"][0]["in_task_list"] is True
    assert rep["by_family"]["demo"]["n"] == 1
    assert rep["rows"][0]["observed"] == pytest.approx(t.bpb_2m - c.bpb_2m)


def test_pair_validation_wildcard_and_holdout_arms():
    """A chip-A monitor-only pair: wildcard run names resolve by chip + bpb + seed, the arms are out of the fit,
    and the pair file's knob_changes / control_knobs supply the knobs the monitor rows lack."""
    recs = _controls("M7", (53, 58, 67), {"FF_KEY_OFFSET": "1", "FF_MTP": "1", "FF_ACCUM_SCHED": "2:0.2,4",
                                          "FF_ACCUM_LR": "0.7071"}, base=0.9650)
    for i, s in enumerate((53, 58, 67)):   # RAMP3 arms on chip C: -0.0003
        recs.append(RunRecord(run_id=f"C:r3_{s}", chip="C", run=f"r3_{s}", code_version="M7", seed=s, steps=2400,
                              bpb_2m=0.9650 + 0.0001 * (i - 1) - 0.0003 - 0.0019,
                              knobs={"FF_KEY_OFFSET": "1", "FF_MTP": "1", "FF_ACCUM_SCHED": "1:0.06,2:0.2,4",
                                     "FF_ACCUM_LR": "0.5,0.7071"}, knobs_complete=True, sources=["chip_log"]))
    a_ctl = RunRecord(run_id="A:desc:G4_s67", chip="A", run="desc:G4_s67", code_version="M7", seed=67, steps=2142,
                      bpb_2m=0.96509, knobs={"FF_MUON_ZERO2": "1"}, knobs_complete=False, sources=["monitor"])
    a_arm = RunRecord(run_id="A:desc:G4_RAMP3_s67", chip="A", run="desc:G4_RAMP3_s67", code_version="M7", seed=67,
                      steps=2291, bpb_2m=0.96480, knobs={"FF_ACCUM_SCHED": "1:0.06,2:0.2,4", "FF_ACCUM_LR": "0.5,0.7071"},
                      knobs_complete=False, sources=["monitor"])
    m = QualityModel().fit(recs + [a_ctl, a_arm], [])
    assert set(m.holdout) == {a_ctl.run_id, a_arm.run_id}
    pair = {"pair_id": "ramp3_g4_A_s67", "family": "accum_sched", "in_task_list": True, "chip": "A",
            "treatment_run": "A:*_R_g4r3_s67", "treatment_bpb": 0.96480, "control_run": "A:*_R_g4_s67",
            "control_bpb": 0.96509, "control_chip": "A", "seed": 67, "control_seed": 67, "delta_raw": -0.00029,
            "knob_changes": {"FF_ACCUM_SCHED": "1:0.06,2:0.2,4", "FF_ACCUM_LR": "0.5,0.7071"},
            "control_knobs": {"FF_ACCUM_SCHED": "2:0.2,4", "FF_ACCUM_LR": "0.7071"}}
    rep = m.pair_validation([pair], leave_pair_out=True)
    assert rep["n"] == 1 and rep["n_skipped"] == 0
    row = rep["rows"][0]
    assert row["treatment"] == a_arm.run_id and row["control"] == a_ctl.run_id
    assert row["holdout"] and not row["lopo_refit"]
    assert any("out-of-sample" in n for n in row["notes"]) and any("laid over" in n for n in row["notes"])
    assert row["predicted"] < 0 and row["sign_ok"]                 # the RAMP3 effect learned on chip C carries over
    assert row["pred_sd"] > 0.003                                  # chip A never fitted: wide
    # an unresolvable wildcard is skipped, not fatal
    bad = dict(pair, treatment_bpb=0.5)
    assert m.pair_validation([bad], leave_pair_out=False)["n_skipped"] == 1


def test_pair_validation_leave_one_knob_value_out(campaign, model):
    """LOKVO: the refit also drops every fitted run carrying the pair's untried knob value, so the pair is predicted
    for a value the model never saw (what the surrogate is used for). Harder than leave-one-pair-out by
    construction; the rows say what was untried, how many runs went, and whether the untried value stayed inside
    the remaining range (interpolation) or not."""
    records, pairs = campaign
    lopo = model.pair_validation()
    rep = model.pair_validation(leave_value_out=True)
    assert rep["mode"] == "leave-one-knob-value-out" and rep["n"] == lopo["n"] == len(pairs)
    for key in ("untried", "n_value_runs_dropped", "value_refit"):
        assert all(key in r for r in rep["rows"])
    assert set(rep) >= {"n_extrapolated", "sign_agreement_interior", "mae_interior", "sign_agreement_extrapolated",
                        "mae_extrapolated", "n_value_refit"}
    assert 0 < rep["n_value_refit"] <= rep["n"]
    assert all(r["lopo_refit"] for r in rep["rows"])
    assert rep["mae"] > lopo["mae"]                              # replicate prediction is the easier question
    assert rep["sign_agreement"] >= 0.75 and rep["mae"] <= 0.0008, (rep["sign_agreement"], rep["mae"])
    assert rep["mae_interior"] < rep["mae"] and rep["mae_interior"] <= 0.0005
    assert rep["n_extrapolated"] > rep["n"] - rep["n_extrapolated"]   # most untried values sit at an edge
    by_id = {r.run_id: r for r in records}
    lk5 = [r for r in rep["rows"] if by_id[r["treatment"]].knobs.get("FF_LEAKY_RELU2") == "0.5"
           and by_id[r["treatment"]].knobs.get("FF_LEAKY_FORM") == "fnm"]
    assert len(lk5) == 3
    for r in lk5:
        assert r["untried"] == {"FF_LEAKY_RELU2": "0.5"} and r["value_refit"]
        assert r["n_value_runs_dropped"] == 5                    # the other two seeds' arms + the abs / max arms
        assert "leaky" in r["extrapolated_features"] and r["sign_ok"]
        assert any("untried value(s): FF_LEAKY_RELU2=0.5" in n for n in r["notes"])
    acc = [r for r in rep["rows"] if r["untried"] == {"FF_ACC_IN_GRAPH": "0"}]
    assert len(acc) == 3 and all(r["n_value_runs_dropped"] == 2 for r in acc)
    # a pair that moves no feature knob (two seeds of the same recipe) drops nothing extra: LOKVO == LOPO
    ctl = [r for r in records if r.code_version == "M12" and r.chip == "C" and r.knobs == k59() and r.bpb_2m
           and r.seed in (58, 67)]
    pair = [{"name": "seed", "treatment": ctl[0].run_id, "control": ctl[1].run_id}]
    a = model.pair_validation(pair, leave_value_out=True)["rows"][0]
    b = model.pair_validation(pair, leave_pair_out=True)["rows"][0]
    assert a["untried"] == {} and a["n_value_runs_dropped"] == 0 and not a["value_refit"] and a["lopo_refit"]
    assert a["predicted"] == pytest.approx(b["predicted"])


def test_support_detail_and_value_support(model):
    d = model.support_detail()
    assert set(d) == set(model.support)
    assert all(sorted(v) == model.support[k] for k, v in d.items())
    assert d["FF_QK_GAIN"]["1.6"] == {"n": 3, "eras": {"M12": 3}}
    assert model.value_support("FF_QK_GAIN", "1.6", "M12")["status"] == "seen"
    assert model.value_support("FF_QK_GAIN", 1.6)["n"] == 3                        # numeric spelling, no era
    assert model.value_support("FF_QK_GAIN", "1.5", "M12")["status"] == "interp"
    assert model.value_support("FF_QK_GAIN", "2.5", "M12")["status"] == "unseen"
    assert model.value_support("FF_MLP_MULT", "6")["status"] == "never"            # never varied
    assert model.value_support("FF_NOT_A_KNOB", "1")["status"] == "never"
    w = model.value_support("FF_WARMUP", "40", "M12")
    assert w["status"] == "weak" and w["n"] == 1                                   # a single run
    ko = model.value_support("FF_KEY_OFFSET_RAD", "0.1", "M12")
    assert ko["status"] == "weak" and ko["n"] == 3 and ko["n_in_era"] == 0 and "none in era M12" in ko["note"]
    assert model.value_support("FF_KEY_OFFSET_RAD", "0.1", "M9")["status"] == "seen"
    assert model.value_support("FF_LEAKY_RELU2", "0.50", "M12")["status"] == "seen"   # '0.50' == '0.5'
    assert QualityModel().support_detail() == {}
    assert QualityModel().value_support("FF_QK_GAIN", "1.6")["status"] == "never"


def test_bowl_constraint_pins_a_concave_bowl():
    """FF_COOLDOWN_FRAC 0.5 and 0.7, one run each, both ~-0.0001 vs the control (noise): unconstrained, the bowl
    turns concave and BOTH directions 'improve'; the default pins the curvature at 0 and the knob is carried by its
    (~0) linear term. A real bowl (both sides worse) is untouched: test_bowl_prior_expresses_a_tuned_optimum."""
    recs = _controls("M12", (58, 67, 73, 79), k59())
    for v, d in (("0.5", -0.00012), ("0.7", -0.00008)):
        recs.append(RunRecord(run_id=f"C:cd{v}", chip="C", run=f"cd{v}", code_version="M12", seed=73, steps=2320,
                              bpb_2m=0.9615 + d, knobs=k59(FF_COOLDOWN_FRAC=v), knobs_complete=True, sources=["chip_log"]))
    free = QualityModel(QualityConfig(bowl_nonneg=False)).fit(recs, [])
    base = free.predict(k59(), 2320, "M12", "C", 73)[0]
    d05 = free.predict(k59(FF_COOLDOWN_FRAC="0.5"), 2320, "M12", "C", 73)[0] - base
    d07 = free.predict(k59(FF_COOLDOWN_FRAC="0.7"), 2320, "M12", "C", 73)[0] - base
    assert d05 < -0.00005 and d07 < -0.00005 and free.pinned == []
    assert free.summary()["knobs"]["cooldown_frac_sq"][0] < 0
    m = QualityModel().fit(recs, [])
    assert m.pinned == ["knob:cooldown_frac_sq"] and m.summary()["bowls_pinned"] == m.pinned
    base = m.predict(k59(), 2320, "M12", "C", 73)[0]
    for v in ("0.5", "0.7"):
        assert abs(m.predict(k59(FF_COOLDOWN_FRAC=v), 2320, "M12", "C", 73)[0] - base) < 0.0001
    coefs = {c["name"]: c for c in m.coefficients()}
    assert coefs["knob:cooldown_frac_sq"]["pinned"] and abs(coefs["knob:cooldown_frac_sq"]["estimate"]) < 1e-9
    assert coefs["knob:cooldown_frac_sq"]["prior_mean"] == pytest.approx(0.0005)    # the prior as specified
    assert not coefs["knob:cooldown_frac"]["pinned"] and not coefs["knob:leaky_sq"]["pinned"]
    fs = m.feature_support()["cooldown_frac_sq"]
    assert fs["pinned"] and not fs["identified"] and fs["supported"]
    assert m.value_support("FF_COOLDOWN_FRAC", "0.7", "M12")["status"] == "weak"     # one run behind the value


def test_era_confounded_features_and_fit_eras():
    """ACC_IN_GRAPH off in every M7 run and on in every M12 run: the feature never varies inside an era, so its
    effect is the M7 -> M12 era difference and the coefficient is the prior. It is reported, noted on candidates
    and pair deltas, and fit_eras restricts the fit (the other eras' runs become holdout arms)."""
    recs = _controls("M12", (58, 67, 73, 79), k59()) + _controls("M7", (58, 67, 73), k59(FF_ACC_IN_GRAPH="0"),
                                                                 base=0.9622, prefix="old")
    m = QualityModel(QualityConfig(era_groups={})).fit(recs, [])
    assert m.era_confounded_features() == ["acc_in_graph"] == m.summary()["era_confounded"]
    fs = m.feature_support()["acc_in_graph"]
    assert fs["era_confounded"] and fs["supported"] and fs["eras_nonzero"] == {"M7": 3}
    assert not m.feature_support()["leaky"]["era_confounded"]
    assert any("era-confounded" in n for n in m.design_row(k59(FF_ACC_IN_GRAPH="0"), 2320, "M12", "C", 73)[2])
    assert not any("era-confounded" in n for n in m.design_row(k59(), 2320, "M12", "C", 73)[2])
    assert not any("era-confounded" in n for n in m._record_row(recs[4])[2])     # a record is not a move
    assert any("era-confounded" in n for n in m.predict_delta(recs[0], recs[4])[2])
    assert m.value_support("FF_ACC_IN_GRAPH", "0", "M12")["status"] == "weak"
    assert m.value_support("FF_ACC_IN_GRAPH", "0", "M7")["status"] == "seen"
    assert m.summary()["fit_eras"] is None and m.dropped["era_excluded"] == 0
    r = QualityModel(QualityConfig(era_groups={}, fit_eras=("M12",))).fit(recs, [])
    assert r.n == 4 and set(r.eras) == {"M12"} and r.dropped["era_excluded"] == 3
    assert set(r.holdout) == {x.run_id for x in recs[4:]} and r.summary()["fit_eras"] == ["M12"]
    assert r.era_confounded_features() == []
    rep = r.pair_validation([{"name": "acc", "treatment": recs[4].run_id, "control": recs[0].run_id}],
                            leave_pair_out=False)
    row = rep["rows"][0]
    assert row["holdout"] and "acc_in_graph" in row["unsupported_features"]


def test_pairs_from_records(campaign):
    records, pairs = campaign
    by_id = {r.run_id: r for r in records}
    for p in pairs:
        assert p["treatment"] in by_id and p["control"] in by_id
        t, c = by_id[p["treatment"]], by_id[p["control"]]
        assert t.control_of == c.run_id
        assert p["delta_raw"] == pytest.approx(t.bpb_2m - c.bpb_2m)


def test_fit_without_pairs_derives_them_from_control_of(campaign):
    records, pairs = campaign
    m = QualityModel(SYNTH_CONFIG).fit(records)
    assert len(m.pairs) == len(pairs)
    m2 = QualityModel(SYNTH_CONFIG).fit(records, pairs=[])
    assert m2.pairs == []


# ----------------------------------------------------------------------------- reporting helpers


def test_summary_and_coefficients(model):
    s = model.summary()
    assert s["fitted"] and s["n"] == model.n and 0.0003 <= s["sigma"] <= 0.01
    assert set(s["knobs"]) == set(FEATURE_NAMES)
    assert len(s["features"]) == len(FEATURES)
    assert set(s["feature_support"]) == set(FEATURE_NAMES)
    for name, sup in s["feature_support"].items():
        assert set(sup) >= {"n_nonzero", "n_rows", "scaled_range", "knob_values", "supported", "identified"}
        assert sup["supported"] == (sup["n_nonzero"] > 0)
    assert s["feature_support"]["mtp_off"]["supported"] and not s["feature_support"]["mlp_mult"]["supported"]
    assert s["feature_support"]["leaky"]["identified"]
    assert set(s["feature_support"]["leaky"]["knob_values"]["FF_LEAKY_RELU2"]) == {"0", "0.25", "0.35", "0.5"}
    assert s["record_defaults"] == "code" and s["era_merges"] == {} and s["n_holdout"] == 0
    assert isinstance(s["unmodelled_knobs"], dict)
    coefs = model.coefficients()
    assert [c["name"] for c in coefs] == model.columns
    assert all(c["sd"] >= 0 for c in coefs)
    res = model.residuals()
    assert len(res) == model.n and {"run_id", "observed", "fitted", "residual", "era", "chip", "seed", "steps"} <= set(res[0])
    assert math.sqrt(np.mean([r["residual"] ** 2 for r in res])) < 0.0005
    assert QualityModel().summary() == {"fitted": False}
    # support: knob -> values seen (what ffsim.search.support_status reads)
    sup = model.support
    assert isinstance(sup, dict) and all(isinstance(v, list) for v in sup.values())
    assert set(sup["FF_QK_GAIN"]) == {"1.44", "1.6"}
    # extrapolation: a value outside the fitted range is flagged, seen values are not
    assert model.extrapolated_features(k59(FF_QK_GAIN="1.6")) == []
    assert "qk_gain" in model.extrapolated_features(k59(FF_QK_GAIN="2.5"))
    assert "mlp_mult" in model.extrapolated_features(k59(FF_MLP_MULT="6"))


def test_config_is_respected():
    cfg = QualityConfig(slope_prior_mean=-0.08, slope_prior_sd=1e-6)
    recs = [RunRecord(run_id=f"C:r{i}", chip="C", run=f"r{i}", code_version="M12", seed=73, steps=2300 + 10 * i,
                      bpb_2m=0.9615 - 0.0001 * i, knobs=k59()) for i in range(4)]
    m = QualityModel(cfg).fit(recs)
    assert m.summary()["slope"][0] == pytest.approx(-0.08, abs=1e-5)


# ----------------------------------------------------------------------------- offset


def test_offset_defaults_match_contract():
    m = OffsetModel()
    assert m.mean == pytest.approx(0.006567, abs=1e-5)
    assert m.sd == pytest.approx(0.00044, abs=3e-5)
    assert m.n == 6 and m.source == "contract"
    assert list(m.offsets) == list(CONTRACT_OFFSETS)
    assert "text difficulty" in m.note and "0.0004" in m.note and "chip C" in m.note
    assert m.predictive_sd > m.sd
    assert m.mean_se == pytest.approx(m.sd / math.sqrt(6))
    # the sd itself rests on six points: chi-square 95% CI for sd 0.00044 with df 5 is 0.00027 .. 0.00107
    assert m.sd_n == 6
    lo, hi = m.sd_ci95
    assert lo == pytest.approx(0.00027, abs=1.5e-5) and hi == pytest.approx(0.00107, abs=1.5e-5)


def test_chi2_quantiles_and_sd_interval():
    # tabulated chi-square quantiles (df 5: 0.831212 / 12.8325; df 21: 10.283 / 35.479; df 1: 0.000982 / 5.02389)
    assert _chi2_ppf(0.025, 5) == pytest.approx(0.831212, rel=1e-5)
    assert _chi2_ppf(0.975, 5) == pytest.approx(12.8325, rel=1e-5)
    assert _chi2_ppf(0.025, 21) == pytest.approx(10.283, rel=1e-4)
    assert _chi2_ppf(0.975, 21) == pytest.approx(35.479, rel=1e-4)
    assert _chi2_ppf(0.975, 1) == pytest.approx(5.02389, rel=1e-5)
    lo, hi = sd_confidence_interval(0.00045, 6)
    assert lo == pytest.approx(0.00045 * math.sqrt(5 / 12.8325), rel=1e-4)
    assert hi == pytest.approx(0.00045 * math.sqrt(5 / 0.831212), rel=1e-4)
    assert lo == pytest.approx(0.000281, abs=2e-6) and hi == pytest.approx(0.001104, abs=2e-6)
    lo22, hi22 = sd_confidence_interval(0.00065, 22)
    assert lo22 == pytest.approx(0.00050, abs=1e-5) and hi22 == pytest.approx(0.00093, abs=1e-5)
    assert all(math.isnan(v) for v in sd_confidence_interval(0.0004, 1))
    with pytest.raises(ValueError):
        _chi2_ppf(1.0, 5)


def test_offset_fit_from_uploads_table():
    table = [{"submission": "K57", "official": 0.9686, "rehearsal": 0.96193},
             {"submission": "K59", "official_bpb": 0.9671, "bpb_2m": 0.96092},
             {"submission": "K55", "offset": 0.0067},
             {"submission": "pending", "official": None, "rehearsal": 0.9600},
             0.0062, "not a row"]
    offs = offsets_from_uploads(table)
    assert len(offs) == 4
    assert offs[0] == pytest.approx(0.9686 - 0.96193)
    m = OffsetModel().fit(table)
    assert m.n == 4 and m.sd_n == 4 and m.source == "uploads"
    assert m.mean == pytest.approx(float(np.mean(offs)))
    assert m.sd == pytest.approx(float(np.std(offs, ddof=1)))
    # RunRecords work too
    recs = [RunRecord(run_id="C:a", chip="C", run="a", bpb_2m=0.96092, official_bpb=0.9671, submission="K59"),
            RunRecord(run_id="C:b", chip="C", run="b", bpb_2m=0.96193, official_bpb=0.9686, submission="K57"),
            RunRecord(run_id="C:c", chip="C", run="c", bpb_2m=0.965)]
    m = OffsetModel().fit(recs)
    assert m.n == 2
    assert m.mean == pytest.approx(((0.9671 - 0.96092) + (0.9686 - 0.96193)) / 2)
    assert m.sd == pytest.approx(OffsetModel().contract_sd)   # 2 rows: sd stays the contract's
    assert m.sd_n == 6 and m.sd_ci95 == pytest.approx(OffsetModel().sd_ci95)   # ... and so does its CI
    assert "contract" in m.source
    m = OffsetModel().fit([])
    assert m.mean == pytest.approx(OffsetModel().mean) and "no usable" in m.source
    assert OffsetModel().fit(None).n == 6


def test_offsets_skip_non_calibration_rows_and_prefer_measured_offsets():
    rows = [{"submission": "K59", "official_bpb": "0.9671", "rehearsal_bpb": "0.96092", "offset": "+0.0062", "recipe_note": "best"},
            {"submission": "K44_rerun", "official_bpb": "0.99021", "rehearsal_bpb": "0.98235", "offset": "+0.00786",
             "recipe_note": "rescored by the organisers; NOT a calibration point for OffsetModel"},
            {"submission": "K54b", "official_bpb": "", "rehearsal_bpb": "0.96326", "offset": "", "recipe_note": "NOT SCORED"},
            {"submission": "X", "official_bpb": 0.97, "rehearsal_bpb": 0.96, "offset": 0.0001, "calibration": False},
            {"submission": "Y", "official": 0.9700, "rehearsal": 0.9640, "offset": 0.0099}]
    offs = offsets_from_uploads(rows)
    assert offs == pytest.approx([0.9671 - 0.96092, 0.9700 - 0.9640])   # measured, not the quoted 0.0099


@pytest.mark.skipif(not os.path.exists(UPLOADS_CSV), reason="research/sim-data/official-uploads.csv not present")
def test_load_official_uploads_real_table():
    contract = load_official_uploads(UPLOADS_CSV)
    assert {r["submission"] for r in contract} == set(CONTRACT_SUBMISSIONS)   # the table is newest-first
    assert all(r["calibration"] and r["offset"] is not None for r in contract)
    m = OffsetModel().fit(contract)
    assert m.n == 7 and m.source.startswith("uploads")        # K51 -> K59 plus K60 (29 Sep, +0.0064)
    assert m.mean == pytest.approx(0.00653, abs=5e-5)
    assert m.sd == pytest.approx(0.00042, abs=6e-5)
    lo, hi = m.sd_ci95                                    # a 7-point sd is known to a factor of ~2
    assert lo == pytest.approx(0.00027, abs=2e-5) and hi == pytest.approx(0.00092, abs=4e-5)
    assert m.predictive_sd == pytest.approx(0.00045, abs=2e-5)
    assert OffsetModel().fit(UPLOADS_CSV).mean == pytest.approx(m.mean)
    everything = load_official_uploads(UPLOADS_CSV, submissions=None)
    names = {r["submission"] for r in everything}
    assert len(everything) >= 15 and set(CONTRACT_SUBMISSIONS) <= names
    assert not ({"K54b", "K44_rerun", "F6_dup"} & names)
    # a full row list (what the CLI passes) is restricted to the contract subset unless told otherwise
    via_rows = OffsetModel().fit(everything)
    assert via_rows.n == 7 and via_rows.mean == pytest.approx(m.mean) and "K51" in via_rows.source
    wide = OffsetModel().fit(everything, contract_subset=False)
    assert wide.n == len(everything) and 0.0055 < wide.mean < 0.0075 and wide.sd > m.sd
    assert wide.sd_ci95[1] - wide.sd_ci95[0] < hi - lo   # 23 points: a narrower interval on a wider sd


def test_offset_sample():
    m = OffsetModel()
    rng = np.random.default_rng(3)
    s = m.sample(rng, 20000)                               # default: predictive sd (upload noise + the mean's se)
    assert s.shape == (20000,)
    assert abs(float(s.mean()) - m.mean) < 2e-5
    assert abs(float(s.std()) - m.predictive_sd) < 2e-5
    assert m.predictive_sd == pytest.approx(m.sd * math.sqrt(1 + 1 / 6))
    np.testing.assert_array_equal(m.sample(np.random.default_rng(3), 50),
                                  m.sample(np.random.default_rng(3), 50, with_mean_uncertainty=True))
    s_noise = m.sample(np.random.default_rng(3), 20000, with_mean_uncertainty=False)   # upload noise only
    assert abs(float(s_noise.std()) - m.sd) < 2e-5 and float(s_noise.std()) < float(s.std())
    assert OffsetModel(mean=0.007, sd=0.0002, n=3).sample(rng, 0).shape == (0,)
    assert OffsetModel(mean=0.007, sd=0.0002, n=3).sd_n == 3
    summ = m.summary()
    assert set(summ) >= {"mean", "sd", "sd_n", "sd_ci95", "n", "mean_se", "predictive_sd", "offsets", "source", "note"}
    assert summ["sd_ci95"] == pytest.approx(list(m.sd_ci95))


# ----------------------------------------------------------------------------- real data


def _load_real():
    records = []
    with open(RUNS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(RunRecord.from_dict(json.loads(line)))
    pairs = None
    if os.path.exists(PAIRS_JSON):
        with open(PAIRS_JSON, encoding="utf-8") as fh:
            pairs = json.load(fh)
            if isinstance(pairs, dict):
                pairs = pairs.get("pairs", [])
    return records, pairs


@pytest.mark.skipif(not os.path.exists(RUNS_JSONL), reason="research/sim-data/runs.jsonl not built yet")
def test_real_dataset_fit_and_anchors():
    """Regression guards on research/sim-data/quality-validation.md (29 Sep 2026 dataset)."""
    records, pairs = _load_real()
    model = QualityModel().fit(records, pairs)
    assert model.n >= 100
    assert model.dropped["unknown_era"] > 0 and model.dropped["no_effective_knobs"] > 0
    assert "M9+" in model.eras and model.resolve_era("M12")[1] == 0 and model.resolve_era("code_k12off")[1] == 0
    assert set(model.era_members["M9+"]) >= {"M9", "M12"}
    assert all(model.era_counts[e] >= model.config.min_era_runs for e in model.eras)
    b, b_sd = model.summary()["slope"]
    assert -0.065 < b < -0.04 and b_sd < 0.006
    mean, sd = model.predict(K59_DEFAULTS, ANCHOR_STEPS, "M12", chip="C", seed=73)
    assert math.isfinite(mean) and sd > 0
    assert abs(mean - ANCHOR_BPB) < 0.0005, (mean, sd)
    k57 = dict(K59_DEFAULTS, FF_LEAKY_RELU2="0", FF_TIME_TARGET="1790.5")
    mean57, _ = model.predict(k57, 2359, "M9", chip="C", seed=73)
    assert abs(mean57 - 0.96193) < 0.0006, mean57
    for s in (73, 58, 67):
        est, _ = model.summary()["seeds"][s]
        assert -0.0015 < est < 0.0
    assert model.support["FF_LEAKY_RELU2"] == ["0", "0.25", "0.35", "0.5"]
    # F2: the two single M3-era runs at FF_COOLDOWN_FRAC 0.5 / 0.7 (-0.0001 each, noise) no longer fit a concave
    # bowl in which both directions improve: the curvature is pinned at 0 and the value is graded weak for M12
    assert "knob:cooldown_frac_sq" in model.pinned
    base = model.predict(K59_DEFAULTS, ANCHOR_STEPS, "M12", chip="C", seed=73)[0]
    for v in ("0.5", "0.7"):
        assert abs(model.predict(dict(K59_DEFAULTS, FF_COOLDOWN_FRAC=v), ANCHOR_STEPS, "M12", "C", 73)[0] - base) < 0.0001
        assert model.value_support("FF_COOLDOWN_FRAC", v, "M12")["status"] == "weak"
    # Q5: FF_ACC_IN_GRAPH is off in every M3-M7 run and on in every M9+ run: era-confounded, weak for M12
    assert model.era_confounded_features() == ["acc_in_graph"]
    assert model.value_support("FF_ACC_IN_GRAPH", "0", "M12")["status"] == "weak"
    assert model.value_support("FF_LEAKY_RELU2", "0.25", "M12")["status"] == "seen"


@pytest.mark.skipif(not (os.path.exists(RUNS_JSONL) and os.path.exists(PAIRS_JSON)),
                    reason="research/sim-data runs.jsonl / validation-pairs.json not built yet")
def test_real_dataset_pair_validation():
    records, pairs = _load_real()
    model = QualityModel().fit(records, pairs)
    task = [p for p in pairs if p.get("in_task_list")]
    rep = model.pair_validation(task, leave_pair_out=True)
    assert rep["mode"] == "leave-one-pair-out"
    assert rep["n"] == len(task) and rep["n_skipped"] == 0            # the chip-A wildcard pairs resolve
    assert rep["sign_agreement"] >= 0.8, rep["sign_agreement"]        # spec target (whole task list 62/73 = 0.849 on 29 Sep)
    assert rep["mae"] <= 0.0005, rep["mae"]                           # 0.00038 on 29 Sep: at the pair noise floor
    assert rep["sign_agreement_decisive"] >= 0.9
    assert rep["coverage_2sd"] >= 0.9
    recipe = [r for r in rep["rows"] if r["family"] not in ("seed", "chip")]
    assert sum(r["sign_ok"] for r in recipe) / len(recipe) >= 0.8
    assert {r["family"] for r in rep["rows"]} >= {"leaky_slope", "relu2_fn", "softcap_asym", "wd_sched", "mom_peak",
                                                  "m13_flags", "m14_attn_src", "seed"}
    # Review Q3 / Q6 / DF-6 (29 Sep): the report's headline is the SUPPORTED RECIPE subset (54 pairs: 44/54 = 81.5%,
    # MAE 0.00031), because the seed pairs' sign is fixed by construction (a new seed is predicted as 'worse than seed
    # 73' whatever the recipe) and the prior-only pairs are answered by the bowl prior alone ('any move is worse'),
    # which the list's rejected-move composition rewards. The 80% target is met within noise there (cluster-bootstrap
    # 90% interval 72-90%), so these guards hold the headline claim, not a margin above it.
    supported = [r for r in recipe if not r["unsupported_features"]]
    assert 50 <= len(supported) <= len(recipe) - 4, len(supported)
    assert sum(r["sign_ok"] for r in supported) / len(supported) >= 0.8
    assert sum(r["abs_error"] for r in supported) / len(supported) <= 0.00035
    seed_rows = [r for r in rep["rows"] if r["family"] == "seed"]
    assert len(seed_rows) == 10 and all(r["predicted"] > 0 for r in seed_rows)   # minus the seed-73 effect, every time
    prior_only = [r for r in recipe if r["unsupported_features"]]
    assert len(prior_only) == 8 and all(r["observed"] > 0 for r in prior_only)   # all observed worse than control
    # DF-6: the spec names ~20 pairs in specific families; 33 of the 73 fall in them (27/33 = 81.8% / 0.00035 on 29 Sep)
    spec_named = {"leaky_slope", "leaky_form", "relu2_fn", "m13_flags", "m14_attn_src", "softcap", "softcap_asym",
                  "wd_sched", "mom_peak"}
    spec_rows = [r for r in rep["rows"] if r["family"] in spec_named]
    assert len(spec_rows) == 33
    assert sum(r["sign_ok"] for r in spec_rows) / len(spec_rows) >= 0.8
    # Q6: the 73 pairs cluster on 28 control runs (the largest controls 12 pairs), which is why the report quotes a
    # cluster-bootstrap interval rather than treating 62/73 as a margin over the target
    controls = {}
    for r in rep["rows"]:
        controls[r["control"]] = controls.get(r["control"], 0) + 1
    assert len(controls) == 28 and max(controls.values()) == 12
    ins = model.pair_validation(task, leave_pair_out=False)
    assert ins["mae"] <= rep["mae"] and ins["mae"] <= 0.0003
    # leave-one-knob-value-out: the untried-value question the surrogate is used for (29 Sep: 84% / 0.00060,
    # interior 0.00042 vs extrapolated 0.00068); MAE roughly doubles relative to leave-one-pair-out
    lokvo = model.pair_validation(task, leave_value_out=True)
    assert lokvo["mode"] == "leave-one-knob-value-out" and lokvo["n"] == len(task) and lokvo["n_skipped"] == 0
    assert lokvo["sign_agreement"] >= 0.78 and lokvo["mae"] <= 0.0008, (lokvo["sign_agreement"], lokvo["mae"])
    assert lokvo["mae"] > rep["mae"] and lokvo["mae_interior"] < lokvo["mae_extrapolated"]
    assert lokvo["n_value_refit"] >= 40


@pytest.mark.skipif(not (os.path.exists(RUNS_JSONL) and os.path.exists(PAIRS_JSON)),
                    reason="research/sim-data runs.jsonl / validation-pairs.json not built yet")
def test_real_dataset_pair_steps_match_records():
    """validation-pairs.json carries the runs.jsonl step convention (optimizer steps completed = ff_summary steps
    = last logged step index + 1), not the monitor's last-index numbers (DF-2). Wildcard arms (A:*_R_...) resolve by
    bpb + seed, not by run_id, and are not checked here."""
    records, pairs = _load_real()
    by_id = {r.run_id: r for r in records}
    checked, wrong = 0, []
    for p in pairs:
        for chip_key, run_key, steps_key in (("chip", "treatment_run", "treatment_steps"),
                                             ("control_chip", "control_run", "control_steps")):
            rec = by_id.get(f"{p[chip_key]}:{p[run_key]}")
            if rec is None or rec.steps is None or p.get(steps_key) is None:
                continue
            checked += 1
            if int(p[steps_key]) != int(rec.steps):
                wrong.append((p["pair_id"], rec.run_id, p[steps_key], rec.steps))
    assert checked >= 150, checked                                    # 192 arms resolve on 29 Sep
    assert wrong == [], wrong[:10]
