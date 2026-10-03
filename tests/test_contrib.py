"""Tests for the community contribution pipeline (ffsim/contrib.py, ffsim/params.py, the app's "Add my runs" tab,
the issue form and the workflows). Pure numpy + stdlib except the app tests (skip without gradio) and the workflow
tests (skip without PyYAML)."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
import urllib.parse
from pathlib import Path

import pytest

from ffsim import contrib as C
from ffsim.dataset import load_runs
from ffsim.params import export_params, load_params, quality_from_params, write_params
from ffsim.quality import QualityModel

_AKID = "AKIA" + "IOSFODNN7EXAMPLE"  # AWS documentation example key, split so scanners do not flag it

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "research" / "sim-data" / "runs.jsonl"
PAIRS = ROOT / "research" / "sim-data" / "validation-pairs.json"
UPLOADS = ROOT / "research" / "sim-data" / "official-uploads.csv"
needs_data = pytest.mark.skipif(not RUNS.exists(), reason="research/sim-data/runs.jsonl not present")

GOOD = {
    "schema_version": "1",
    "hardware": "trn2.3xlarge",
    "time_budget_s": 1800,
    "base_recipe": "K82s4",
    "recipe": {"FF_COOLDOWN_FRAC": 0.65},
    "seed": 58,
    "chip_val_bpb": 0.9551,
    "steps": 2390,
    "contributor": "octocat",
    "consent": True,
}


def good(**over):
    d = copy.deepcopy(GOOD)
    for k, v in over.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    return d


@pytest.fixture(scope="module")
def ctx():
    if not RUNS.exists():
        pytest.skip("needs research/sim-data/runs.jsonl")
    return C.load_context(RUNS, ROOT / "no-such-contrib.jsonl", PAIRS, UPLOADS)


# ------------------------------------------------------------------------------------------------ schema
def test_shipped_schema_file_is_the_code_schema():
    shipped = json.loads((ROOT / "contrib" / "schema.json").read_text(encoding="utf-8"))
    assert shipped == C.SCHEMA
    assert set(C.SCHEMA["required"]) >= {"schema_version", "hardware", "time_budget_s", "recipe", "steps", "consent"}
    assert "trn2.3xlarge" in shipped["properties"]["hardware"]["enum"]
    assert "other" in shipped["properties"]["hardware"]["enum"]


def test_good_record_is_valid_and_converted():
    res = C.validate_record(good())
    assert res.ok, res.errors
    assert res.usable_for_fit
    assert res.record["recipe"] == {"FF_COOLDOWN_FRAC": "0.65"}
    assert res.record["chip_eval_tokens"] == C.EVAL_TOKENS_2M
    rr = res.run_record
    assert rr.chip == "contrib-trn2.3xlarge-K82s4" and rr.run_id.startswith("contrib-trn2.3xlarge-K82s4:contrib-")
    assert rr.code_version == "M14"                    # the K82s4 base recipe's lineage
    assert rr.knobs["FF_COOLDOWN_FRAC"] == "0.65" and rr.knobs["FF_SEED"] == "58"
    assert rr.knobs["FF_ROW_SHUFFLE"]                   # base recipe knobs carried over
    assert rr.knobs_complete and rr.sources == ["contrib"] and not rr.is_screen
    assert rr.bpb_2m == pytest.approx(0.9551) and rr.steps == 2390 and rr.seed == 58
    assert "contributor:octocat" in rr.tags and f"contrib_hash:{res.content_hash}" in rr.tags


@pytest.mark.parametrize("over, needle", [
    ({"hardware": "h100"}, "hardware must be one of"),
    ({"schema_version": "2"}, "schema_version"),
    ({"consent": False}, "consent"),
    ({"consent": "yes"}, "consent"),
    ({"steps": None}, "steps is required"),
    ({"recipe": None}, "recipe is required"),
    ({"chip_val_bpb": None}, "at least one of chip_val_bpb or official_val_bpb"),
    ({"chip_val_bpb": 3.0}, "above the maximum"),
    ({"chip_val_bpb": "0.95"}, "type number"),
    ({"steps": 0}, "below the minimum"),
    ({"steps": True}, "type integer"),
    ({"steps": 2390.5}, "type integer"),
    ({"time_budget_s": 10}, "below the minimum"),
    ({"seed": -1}, "below the minimum"),
    ({"extra_field": 1}, "not an allowed field"),
    ({"recipe": {"ff_lower": "1"}}, "key"),
    ({"recipe": {"PATH": "/usr/bin"}}, "key"),
    ({"recipe": {"FF_X": [1, 2]}}, "FF_X"),
    ({"recipe": {"FF_X": "x" * 201}}, "FF_X"),
    ({"date": "01/10/2026"}, "date"),
    ({"contributor": "not a handle!"}, "contributor"),
    ({"hardware": "other"}, "hardware_other is required"),
    ({"recipe": {"FF_SEED": "73"}}, "disagree"),
    ({"step_seconds": 1.5}, "time budget"),
    ({"chip_val_bpb": None, "official_val_bpb": 0.9614, "chip_eval_tokens": 2097152}, "without $.chip_val_bpb"),
    ({"recipe": {"FF_SEED": "inf"}}, "FF_SEED"),
    ({"recipe": {"FF_SEED": "1e400"}, "seed": None}, "FF_SEED"),
    ({"recipe": {"FF_SEED": "3.5"}, "seed": None}, "FF_SEED"),
    ({"code_version": "anything-goes"}, "code_version"),
    ({"code_version": "M99"}, "code_version"),
])
def test_bad_records_are_rejected(over, needle):
    res = C.validate_record(good(**over))
    assert not res.ok
    assert any(needle in e for e in res.errors), res.errors


@pytest.mark.parametrize("knob", ["FF_DEPTH", "FF_QK_GAIN", "FF_SOFTCAP_B"])
@pytest.mark.parametrize("value", ["inf", "-inf", "Infinity", "nan", "NaN", "1e400", "1e150"])
def test_non_finite_and_absurd_knob_values_are_rejected(knob, value):
    """The High review finding: these used to validate as ok, and once merged made every fit raise."""
    res = C.validate_record(good(recipe={knob: value}))
    assert not res.ok and any(knob in e for e in res.errors), res.errors
    assert value not in " ".join(res.errors)


@pytest.mark.parametrize("over", [{"FF_DEPTH": "12.5"}, {"FF_DEPTH": "0"}, {"FF_COOLDOWN_FRAC": "1.5"},
                                  {"FF_MATRIX_LR_SCALE": "0"}, {"FF_EMB_LR": "-0.3"}, {"FF_MTP_PHASES": "0.2,7"},
                                  {"FF_ACCUM_LR": "0.5,inf"}, {"FF_TOTAL_BATCH": "abc"}, {"FF_UNKNOWN_KNOB": "nan"}])
def test_knob_bounds(over):
    assert not C.validate_record(good(recipe=over)).ok


def test_plausible_knob_changes_still_pass():
    res = C.validate_record(good(recipe={"FF_DEPTH": "10", "FF_QK_GAIN": "1.6", "FF_SOFTCAP_B": "5",
                                         "FF_MTP_PHASES": "0.2,0.5", "FF_ACCUM_SCHED": "2:0.2,4"}))
    assert res.ok, res.errors


@needs_data
@pytest.mark.parametrize("knob, value", [("FF_DEPTH", "inf"), ("FF_DEPTH", "nan"), ("FF_DEPTH", "1e400"),
                                         ("FF_QK_GAIN", "inf"), ("FF_SOFTCAP_B", "1e150"), ("FF_SOFTCAP_B", "1e300")])
def test_a_bad_line_merged_by_hand_cannot_take_the_readers_down(tmp_path, knob, value):
    """Even a line that bypassed validation (hand-merged, or an older validator) is skipped by every reader:
    load_contrib, Context.quality (the check workflow), refit and the CLI fit inputs."""
    contrib = tmp_path / "c.jsonl"
    ctx = C.load_context(RUNS, contrib, PAIRS, UPLOADS)
    C.ingest_records(_records(2), ctx, contrib)
    d = json.loads(contrib.read_text(encoding="utf-8").splitlines()[0])
    d["knobs"][knob] = value
    d["run_id"] = d["run_id"] + "x"
    d["contrib"]["content_hash"] = "f" * 64
    with open(contrib, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(d) + "\n")
        fh.write("{not json\n")
        fh.write("[" * 5000 + "\n")
    skipped = []
    rows = C.load_contrib(contrib, skipped=skipped)
    assert len(rows) == 2 and len(skipped) == 3
    q = C.load_context(RUNS, contrib, PAIRS, UPLOADS).quality()
    assert q is not None and q.n >= 2
    out = C.refit(RUNS, contrib, PAIRS, UPLOADS, None, None, loo=False)
    assert out["stats"]["n_records"] == 2
    from ffsim import cli
    ns = argparse.Namespace(runs=str(RUNS), pairs=str(PAIRS), uploads=str(UPLOADS), contrib=str(contrib))
    assert cli._load_inputs(ns)["records"]


@needs_data
def test_duplicates_from_two_merged_prs_count_once(tmp_path):
    contrib = tmp_path / "c.jsonl"
    C.ingest_records(_records(1), C.load_context(RUNS, contrib, PAIRS, UPLOADS), contrib)
    line = contrib.read_text(encoding="utf-8")
    contrib.write_text(line + line, encoding="utf-8")
    assert len(C.load_contrib(contrib)) == 1


def test_contributions_per_account_are_capped_in_the_fit():
    rows = [(C.validate_record(good(seed=i, chip_val_bpb=0.955 + i * 1e-6)).run_record,
             {"submission": {"contributor": "a"}}) for i in range(C.MAX_FIT_PER_CONTRIBUTOR + 5)]
    rows.append((rows[0][0], {"submission": {"contributor": "b"}}))
    assert len(C.contrib_fit_records(rows)) == C.MAX_FIT_PER_CONTRIBUTOR + 1
    assert C.contributor_counts(rows) == {"a": C.MAX_FIT_PER_CONTRIBUTOR + 5, "b": 1}
    res = C.validate_record(good(contributor="a"), contributor_counts={"a": C.MAX_FIT_PER_CONTRIBUTOR})
    assert res.ok and "contributor_cap" in res.flags


def test_contributor_must_be_the_issue_author():
    assert not C.validate_record(good(contributor="someone-else"), issue_author="octocat").ok
    assert C.validate_record(good(contributor="OctoCat"), issue_author="octocat").ok
    assert C.validate_record(good(contributor=None), issue_author="octocat").ok


def test_custom_recipe_claiming_a_team_lineage_is_flagged():
    res = C.validate_record(good(base_recipe="custom", code_version="M14", recipe={"FF_COOLDOWN_FRAC": "0.6"}))
    assert res.ok and "claims_team_lineage" in res.flags


def test_non_object_and_oversized_records_are_rejected():
    assert not C.validate_record([1, 2]).ok
    assert not C.validate_record("{}").ok
    big = good(recipe={f"FF_K{i}": "v" * 150 for i in range(200)})
    res = C.validate_record(big)
    assert not res.ok and "limit" in res.errors[0]
    nest = cur = {}
    for _ in range(40):
        cur["a"] = {}
        cur = cur["a"]
    res = C.validate_record(good(framework_notes=nest))
    assert not res.ok and "nests deeper" in res.errors[0]


@pytest.mark.parametrize("field, value", [
    ("framework_notes", "creds " + _AKID + " in env"),
    ("framework_notes", "ran on account 123456789012"),
    ("framework_notes", "ping me at someone@example.com"),
    ("framework_notes", "token " + "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("framework_notes", "role arn:aws:iam::role/x"),
    ("framework_notes", "instance i-0123456789abcdef0"),
    ("framework_notes", "host 10.0.12.7"),
    ("framework_notes", "password=hunter2"),
    ("framework_notes", "hf" + "_abcdefghijklmnopqrstuvwxyzABCD"),
    ("framework_notes", "-----BEGIN RSA " + "PRIVATE KEY-----"),
    ("framework_notes", "blob QWxhZGRpbjpvcGVuIHNlc2FtZQQWxhZGRpbjpvcGVuIHNlc2FtZQ"),
    ("hardware_other", "i-0123456789abcdef0"),
    ("recipe", {"FF_CC_FLAGS": "https://user:pw@host/x"}),
    ("recipe", {"FF_NOTE": "sk-abcdefghijklmnop1234"}),
    ("framework_notes", "account 1234-5678-9012"),
    ("framework_notes", "bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"),
    ("framework_notes", "ran on 203.0.113.9"),
])
def test_secret_looking_strings_are_rejected_and_not_echoed(field, value):
    rec = good(**{field: value})
    if field == "hardware_other":
        rec["hardware"] = "other"
    res = C.validate_record(rec)
    assert not res.ok
    assert any("looks like it contains" in e for e in res.errors), res.errors
    secret = value if isinstance(value, str) else next(iter(value.values()))
    assert all(secret not in e for e in res.errors)


@pytest.mark.parametrize("where", ["key", "recipe key"])
def test_secret_used_as_a_field_name_is_rejected_and_never_echoed(where):
    secret = _AKID
    rec = good()
    if where == "key":
        rec[secret] = 1
    else:
        rec["recipe"] = {secret: "1"}
    res = C.validate_record(rec)
    assert not res.ok and res.secret_found
    assert all(secret not in e for e in res.errors), res.errors
    text = C.format_comment(res)
    assert secret not in text and "replaced by this check" in text


def test_attacker_chosen_field_names_are_not_echoed():
    for key in ("[click here](https://evil.example/x)", "user.name@corp.example", "![t](https://x.example/p.png)"):
        res = C.validate_record({**good(), key: 1})
        assert not res.ok
        text = C.format_comment(res)
        assert key not in text and "evil.example" not in text and "corp.example" not in text, text
        assert "](" not in text


def test_ordinary_free_text_is_not_mistaken_for_a_secret():
    res = C.validate_record(good(framework_notes="Neuron SDK 2.26, torch-neuronx 2.8, runtime 2.33.10; seed sweep",
                                 recipe={"FF_CC_FLAGS": "--model-type=transformer -O3", "FF_MTP_W": "1,0.5,0.25",
                                         "FF_ACCUM_SCHED": "1:0.06,2:0.2,4"}))
    assert res.ok, res.errors
    for notes in ("neuronx-cc 2.15.128.0", "Neuron SDK 2.20.0.0", "torch-neuronx 2.1.2.2.3.0",
                  "aws-neuronx-runtime-lib 2.22.14.0", "steps 2390, seeds 58 73 67"):
        res = C.validate_record(good(framework_notes=notes))
        assert res.ok, (notes, res.errors)
    for notes in ("10.0.12.7", "host 10.0.12.7"):
        assert not C.validate_record(good(framework_notes=notes)).ok


def test_warnings_for_short_runs_unknown_knobs_and_imputed_scores():
    res = C.validate_record(good(steps=120, recipe={"FF_BRAND_NEW_KNOB": "1"}))
    assert res.ok and not res.usable_for_fit and res.run_record.is_screen
    assert any("short screen" in w for w in res.warnings)
    assert any("FF_BRAND_NEW_KNOB" in w for w in res.warnings)
    off = C.validate_record(good(chip_val_bpb=None, official_val_bpb=0.9614), offset_mean=0.0067)
    assert off.ok and off.usable_for_fit and "bpb_imputed_from_official" in off.flags
    assert off.run_record.bpb_2m == pytest.approx(0.9614 - 0.0067)
    other = C.validate_record(good(chip_eval_tokens=20 * 2 ** 20))
    assert other.ok and not other.usable_for_fit and other.run_record.bpb_2m is None
    tb = C.validate_record(good(time_budget_s=3600, steps=4700))
    assert tb.ok and any("1800" in w for w in tb.warnings)


# ------------------------------------------------------------------------------------------------ duplicates
def test_content_hash_ignores_spelling_handle_and_notes_but_not_the_measurement():
    a = C.validate_record(good()).content_hash
    assert C.validate_record(good(recipe={"FF_COOLDOWN_FRAC": "0.650"}, contributor="someone-else",
                                  framework_notes="rerun", date="2026-09-30")).content_hash == a
    assert C.validate_record(good(seed=73)).content_hash != a
    assert C.validate_record(good(chip_val_bpb=0.9552)).content_hash != a
    assert C.validate_record(good(recipe={"FF_COOLDOWN_FRAC": "0.7"})).content_hash != a
    # the app sends floats, the CLI and README ints: the same run must be the same hash
    assert C.validate_record(good(time_budget_s=1800.0)).content_hash == a


def test_duplicate_is_rejected():
    first = C.validate_record(good())
    dup = C.validate_record(good(contributor="other"), existing_hashes={first.content_hash})
    assert not dup.ok and "duplicate" in dup.errors[0]


@needs_data
def test_record_matching_a_shipped_run_is_flagged(ctx):
    b = next(r for r in ctx.builtin if r.bpb_2m is not None and r.steps and r.steps > 1000)
    res = C.validate_record(good(chip_val_bpb=round(float(b.bpb_2m), 9), steps=int(b.steps)), builtin=ctx.builtin)
    assert res.ok and "matches_builtin" in res.flags


@needs_data
def test_outlier_is_flagged_not_rejected(ctx):
    q = ctx.quality()
    fine = C.validate_record(good(), quality=q)
    assert fine.ok and "outlier" not in fine.flags and fine.prediction is not None
    assert abs(fine.prediction["z"]) < C.OUTLIER_Z
    odd = C.validate_record(good(chip_val_bpb=1.08), quality=q)
    assert odd.ok and "outlier" in odd.flags and abs(odd.prediction["z"]) > C.OUTLIER_Z


@needs_data
def test_k82s4_records_are_anchored_like_the_app(ctx):
    """K82s4's newest levers are outside the fitted data; the prediction for a K82s4-based record carries the app's
    anchor until its (hardware, base) effect has data, so its own rehearsal is not judged ~0.004 bpb off."""
    q = ctx.quality()
    rec = C.validate_record(good(recipe={}, seed=73, chip_val_bpb=0.954472, steps=2388), quality=q)
    assert rec.ok and abs(rec.prediction["residual"]) < 1e-4, rec.prediction
    raw = q.predict_record(rec.run_record)[0]
    assert abs(rec.prediction["predicted_bpb_2m"] - raw) > 0.002            # the anchor is applied
    assert C.anchor_shift(q, C.validate_record(good(base_recipe="K60")).run_record) == 0.0


# ------------------------------------------------------------------------------------------------ ingest / refit
def _records(n):
    out = []
    for i in range(n):
        out.append(good(recipe={"FF_COOLDOWN_FRAC": [0.6, 0.65, 0.7, 0.55, 0.7, 0.75, 0.6][i % 7]},
                        seed=[73, 58, 67][i % 3], chip_val_bpb=round(0.9546 + 0.0001 * (i % 5), 6),
                        steps=2380 + i, contributor=f"user{i % 3}"))
    return out


def _readme(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"# x\r\n\r\n{C.README_START}old{C.README_END}\r\n\r\nend\r\n", encoding="utf-8", newline="")
    return p


@needs_data
def test_ingest_refit_stats_end_to_end(tmp_path):
    contrib = tmp_path / "contrib" / "runs-contrib.jsonl"
    ctx = C.load_context(RUNS, contrib, PAIRS, UPLOADS)
    recs = _records(6)
    res = C.ingest_records(recs + [recs[0]], ctx, contrib)          # the repeat is caught inside the batch
    assert [r.ok for r in res] == [True] * 6 + [False]
    lines = contrib.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    d = json.loads(lines[0])
    assert d["contrib"]["submission"]["hardware"] == "trn2.3xlarge" and d["chip"] == "contrib-trn2.3xlarge-K82s4"
    assert len(C.load_contrib(contrib)) == 6 and len(load_runs(str(contrib))) == 6   # the fitter's own loader reads it

    params, stats, readme = tmp_path / "params.json", tmp_path / "stats.json", _readme(tmp_path)
    space = tmp_path / "space" / "research" / "sim-data" / "contrib" / "runs-contrib.jsonl"
    space.parent.parent.mkdir(parents=True)
    out = C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, space)
    assert out["params_changed"] and out["stats_changed"] and out["readme_changed"] and out["space_synced"]
    s = json.loads(stats.read_text(encoding="utf-8"))
    assert s["n_records"] == 6 and s["n_contributors"] == 3 and s["hardware"] == {"trn2.3xlarge": 6}
    assert s["n_used_in_fit"] == 6 and s["base_recipes"] == {"K82s4": 6}
    assert s["mae_before"] is not None and s["mae_after_loo"] is not None
    assert s["mae_before"] < 0.001 and s["mae_after_loo"] < 0.001   # anchored: no ~0.004 K82s4 bias in "before"
    assert s["guard"]["ok"] and s["guard"]["worst_change"] < C.GUARD_MAX_WORSE
    assert s["quality_fit_n"] == s["builtin"]["n_quality_fit"] + 6
    text = readme.read_bytes().decode("utf-8")
    assert "**6** from **3** named contributors" in text and "old" not in text and "\r\n" in text
    assert space.read_bytes().replace(b"\r\n", b"\n") == contrib.read_bytes().replace(b"\r\n", b"\n")

    p = load_params(params)
    assert p["data"]["n_contrib_records"] == 6 and "chip:contrib-trn2.3xlarge-K82s4" in p["quality"]["columns"]
    q2 = quality_from_params(p["quality"])
    q = QualityModel().fit(load_runs(str(RUNS)) + [r for r, _ in C.load_contrib(contrib)], C._load_pairs(PAIRS))
    rr = C.load_contrib(contrib)[0][0]
    assert q2.predict_record(rr)[0] == pytest.approx(q.predict_record(rr)[0], abs=1e-8)

    again = C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, space)
    assert not any(again[k] for k in ("params_changed", "stats_changed", "readme_changed", "space_synced"))
    assert again["noise_only"] == []


def test_json_close_tolerates_float_noise_only():
    # a coefficient and a zero-up-to-noise step-time coefficient as the first automatic refit pull request moved them
    a = {"beta": [0.9643896967, -4.067522268e-05, 6.718584022e-15], "n": 131, "sha": "ab", "ok": True, "x": None}
    noisy = copy.deepcopy(a)
    noisy["beta"][1:] = [-4.067522171e-05, -2.029626467e-15]
    assert C.json_close(a, noisy, *C.PARAMS_NOISE)
    for over in ({"n": 132}, {"sha": "ac"}, {"ok": False}, {"x": 0.0}, {"extra": 1}, {"beta": [0.9643896967]},
                 {"beta": [0.9644896967, -4.067522268e-05, 6.718584022e-15]}, {"ok": 1}):
        assert not C.json_close(a, {**a, **over}, *C.PARAMS_NOISE), over


def _nudge(x, rel, abs_):
    """Every float in a JSON value moved by ``rel`` (relative) plus ``abs_``, as a refit on another BLAS build does."""
    if isinstance(x, dict):
        return {k: _nudge(v, rel, abs_) for k, v in x.items()}
    if isinstance(x, list):
        return [_nudge(v, rel, abs_) for v in x]
    return x * (1 + rel) + abs_ if isinstance(x, float) else x


@needs_data
def test_refit_leaves_published_files_alone_when_only_float_noise_differs(tmp_path, capsys):
    """The same data refitted on another machine moves params.json around its 9th significant digit (the first
    automatic refit pull request was only that) and can flip the last digit of a stats.json number. Such a refit
    rewrites nothing, so the refit workflow finds no diff and opens no pull request."""
    contrib = tmp_path / "c.jsonl"
    C.ingest_records(_records(6), C.load_context(RUNS, contrib, PAIRS, UPLOADS), contrib)
    params, stats, readme = tmp_path / "params.json", tmp_path / "stats.json", _readme(tmp_path)
    C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, loo=False)
    write_params(_nudge(load_params(params), 3e-8, 1e-14), params)
    C._write_json(stats, _nudge(json.loads(stats.read_text(encoding="utf-8")), 1e-5, 1e-12))
    published = {f: f.read_bytes() for f in (params, stats, readme)}

    out = C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, loo=False)
    assert out["noise_only"] == ["params.json", "stats.json"]
    assert not any(out[k] for k in ("params_changed", "stats_changed", "readme_changed"))
    assert {f: f.read_bytes() for f in published} == published
    rc = C.main(["refit", "--runs", str(RUNS), "--contrib", str(contrib), "--pairs", str(PAIRS), "--uploads",
                 str(UPLOADS), "--params-out", str(params), "--stats-out", str(stats), "--readme", str(readme),
                 "--no-loo", "--guard"])
    printed = capsys.readouterr().out
    assert rc == 0 and "no material change: left params.json, stats.json untouched" in printed
    assert "changed: nothing" in printed and {f: f.read_bytes() for f in published} == published


@needs_data
def test_refit_still_writes_a_material_change(tmp_path):
    """A new contributed record always reaches params.json, stats.json and the README (the data hashes change; here
    the fit moves too), and a published number that is off by more than noise is put back."""
    contrib = tmp_path / "c.jsonl"
    C.ingest_records(_records(6), C.load_context(RUNS, contrib, PAIRS, UPLOADS), contrib)
    params, stats, readme = tmp_path / "params.json", tmp_path / "stats.json", _readme(tmp_path)
    C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, loo=False)
    before = load_params(params)
    rec = good(recipe={"FF_COOLDOWN_FRAC": 0.7}, seed=67, chip_val_bpb=0.9562, steps=2401, contributor="newcomer")
    assert all(r.ok for r in C.ingest_records([rec], C.load_context(RUNS, contrib, PAIRS, UPLOADS), contrib))

    out = C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, loo=False)
    assert out["params_changed"] and out["stats_changed"] and out["readme_changed"] and out["noise_only"] == []
    after = load_params(params)
    assert after["data"]["n_contrib_records"] == 7
    assert not C.json_close(before["quality"]["beta"], after["quality"]["beta"], *C.PARAMS_NOISE)
    assert json.loads(stats.read_text(encoding="utf-8"))["n_records"] == 7
    assert "**7** from **4** named contributors" in readme.read_text(encoding="utf-8")

    stale = copy.deepcopy(after)
    stale["quality"]["beta"][0] *= 1 + 1e-5
    write_params(stale, params)
    out = C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, readme, loo=False)
    assert out["params_changed"] and out["noise_only"] == [] and load_params(params) == after


@needs_data
def test_refit_guard_refuses_a_coordinated_slope_attack(tmp_path):
    """Plausible-looking records that pull a shared knob slope (cooldown 0.3 'good', 0.8 'bad') move the published
    base predictions far more than real runs do: refit --guard writes nothing and the CLI exits 3."""
    contrib = tmp_path / "c.jsonl"
    attack = [good(base_recipe="K60", recipe={"FF_COOLDOWN_FRAC": cd}, seed=[73, 58, 67][i % 3],
                   chip_val_bpb=round(bpb + 0.0001 * i, 6), steps=2380 + i, contributor=f"sock{i}")
              for i, (cd, bpb) in enumerate([(0.3, 0.950)] * 4 + [(0.8, 0.975)] * 4)]
    res = C.ingest_records(attack, C.load_context(RUNS, contrib, PAIRS, UPLOADS), contrib)
    assert all(r.ok for r in res)
    params, stats = tmp_path / "p.json", tmp_path / "s.json"
    out = C.refit(RUNS, contrib, PAIRS, UPLOADS, params, stats, guard=True, loo=False)
    assert out["guard_failed"] and not params.exists() and not stats.exists()
    rc = C.main(["refit", "--runs", str(RUNS), "--contrib", str(contrib), "--pairs", str(PAIRS), "--uploads",
                 str(UPLOADS), "--params-out", str(params), "--stats-out", str(stats), "--no-loo", "--guard"])
    assert rc == 3 and not params.exists()


@needs_data
def test_stats_without_enough_records_for_leave_one_out(tmp_path):
    contrib = tmp_path / "c.jsonl"
    ctx = C.load_context(RUNS, contrib, PAIRS, UPLOADS)
    C.ingest_records(_records(3), ctx, contrib)
    s = C.compute_stats(C.load_context(RUNS, contrib, PAIRS, UPLOADS))
    assert s["n_records"] == 3 and s["mae_before"] is not None and s["mae_after_loo"] is None
    assert "leave-one-out needs" in C.stats_line(s)
    empty = C.compute_stats(C.load_context(RUNS, tmp_path / "none.jsonl", PAIRS, UPLOADS))
    assert empty["n_records"] == 0 and empty["mae_before"] is None
    assert "**0**" in C.stats_line(empty)


@needs_data
def test_cli_fit_inputs_include_contributions(tmp_path):
    from ffsim import cli
    contrib = tmp_path / "c.jsonl"
    C.ingest_records(_records(2), C.load_context(RUNS, contrib, PAIRS, UPLOADS), contrib)
    base = argparse.Namespace(runs=str(RUNS), pairs=str(PAIRS), uploads=str(UPLOADS), contrib="none")
    with_c = argparse.Namespace(runs=str(RUNS), pairs=str(PAIRS), uploads=str(UPLOADS), contrib=str(contrib))
    assert len(cli._load_inputs(with_c)["records"]) == len(cli._load_inputs(base)["records"]) + 2


@needs_data
def test_shipped_params_and_stats_match_the_shipped_data():
    """params.json is a snapshot: it must match the BUILT-IN data exactly, and the contributed data it was fitted
    on. Between a contribution's merge and the refit pull request's merge, main holds newer contributions than the
    snapshot; that is expected and must not turn CI red, so the comparison then checks the snapshot only."""
    p = load_params(ROOT / "ffsim" / "model" / "params.json")
    assert p["data"]["runs_sha256"] == C.file_sha256(RUNS), "run `python -m ffsim contrib refit` after data changes"
    q = quality_from_params(p["quality"])
    if p["data"]["contrib_sha256"] != C.file_sha256(C.DEFAULT_CONTRIB):
        for args in (({}, 2357.0, "M12", "C", 73), ({"FF_COOLDOWN_FRAC": "0.7"}, 2388.0, "M14", "C", None)):
            assert 0.95 < q.predict(*args)[0] < 0.97
        return
    stats = json.loads((ROOT / "contrib" / "stats.json").read_text(encoding="utf-8"))
    assert stats["n_records"] == len(C.load_contrib(C.DEFAULT_CONTRIB))
    fitted = QualityModel().fit(load_runs(str(RUNS)) + C.contrib_fit_records(C.load_contrib(C.DEFAULT_CONTRIB)),
                                C._load_pairs(PAIRS))
    for args in (({}, 2357.0, "M12", "C", 73), ({"FF_COOLDOWN_FRAC": "0.7"}, 2388.0, "M14", "C", None)):
        a, b = q.predict(*args), fitted.predict(*args)
        assert 0.95 < a[0] < 0.97
        assert a[0] == pytest.approx(b[0], abs=1e-7) and a[1] == pytest.approx(b[1], rel=1e-4)


def test_params_round_trip_small(tmp_path):
    fit = QualityModel().fit([_run(i) for i in range(12)])
    params = export_params(None, fit, None, {"n": 12})
    path = tmp_path / "p.json"
    assert write_params(params, path) and not write_params(params, path)
    q2 = quality_from_params(load_params(path)["quality"])
    for steps in (2000.0, 2400.0):
        a, b = fit.predict({}, steps, "M12", "C", 73), q2.predict({}, steps, "M12", "C", 73)
        assert a[0] == pytest.approx(b[0], abs=1e-9) and a[1] == pytest.approx(b[1], rel=1e-6)


def _run(i):
    from ffsim.schema import RunRecord
    return RunRecord(run_id=f"C:r{i}", chip="C", run=f"r{i}", sources=["chip_log"], code_version="M12", seed=[73, 58][i % 2],
                     steps=2300 + 10 * i, bpb_2m=0.961 - 0.00001 * i, knobs={"FF_COOLDOWN_FRAC": ["0.6", "0.7"][i % 2]},
                     knobs_complete=True)


# ------------------------------------------------------------------------------------------------ CLI
@needs_data
def test_cli_validate_writes_comment_record_and_github_output(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps(good()), encoding="utf-8")
    gh = tmp_path / "gh_output"
    rc = C.main(["validate", str(f), "--out-dir", str(tmp_path / "out"), "--github-output", str(gh),
                 "--contrib", str(tmp_path / "none.jsonl")])
    assert rc == 0
    assert "valid=true" in gh.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out" / "record.json").read_text(encoding="utf-8"))["steps"] == 2390
    assert "this run record is valid" in (tmp_path / "out" / "comment.md").read_text(encoding="utf-8")
    assert "Reviewer checklist" in (tmp_path / "out" / "pr-body.md").read_text(encoding="utf-8")
    line = (tmp_path / "out" / "contrib-line.jsonl").read_text(encoding="utf-8")
    assert line.count("\n") == 1 and json.loads(line)["contrib"]["submission"]["steps"] == 2390
    assert re.search(r"^hash=[0-9a-f]{12}$", gh.read_text(encoding="utf-8"), re.M)
    assert "secret=false" in gh.read_text(encoding="utf-8")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(good(consent=False)), encoding="utf-8")
    assert C.main(["validate", str(bad), "--no-model", "--contrib", str(tmp_path / "none.jsonl")]) == 1
    gh2 = tmp_path / "gh2"
    assert C.main(["validate", str(bad), "--no-model", "--always-zero", "--github-output", str(gh2),
                   "--out-dir", str(tmp_path / "o2"), "--contrib", str(tmp_path / "none.jsonl")]) == 0
    assert "valid=false" in gh2.read_text(encoding="utf-8")
    assert not (tmp_path / "o2" / "record.json").exists()
    assert "cannot be accepted" in (tmp_path / "o2" / "comment.md").read_text(encoding="utf-8")


@needs_data
def test_cli_validate_reads_a_github_issue_event(tmp_path):
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"issue": {"number": 17, "body": C.issue_body(good()), "title": "[run] x"}}),
                  encoding="utf-8")
    gh = tmp_path / "gh"
    rc = C.main(["validate", "--issue-event", str(ev), "--out-dir", str(tmp_path / "o"), "--github-output", str(gh),
                 "--always-zero", "--no-model", "--contrib", str(tmp_path / "none.jsonl")])
    assert rc == 0 and "valid=true" in gh.read_text() and "issue_number=17" in gh.read_text()
    assert "Closes #17." in (tmp_path / "o" / "pr-body.md").read_text(encoding="utf-8")
    unticked = C.issue_body(good()).replace("- [X]", "- [ ]")
    ev.write_text(json.dumps({"issue": {"number": 18, "body": unticked}}), encoding="utf-8")
    gh.write_text("")
    C.main(["validate", "--issue-event", str(ev), "--out-dir", str(tmp_path / "o3"), "--github-output", str(gh),
            "--always-zero", "--no-model", "--contrib", str(tmp_path / "none.jsonl")])
    assert "valid=false" in gh.read_text()
    assert "consent box is not ticked" in (tmp_path / "o3" / "comment.md").read_text(encoding="utf-8")
    ev.write_text(json.dumps({"issue": {"number": 19, "body": "no json here"}}), encoding="utf-8")
    gh.write_text("")
    C.main(["validate", "--issue-event", str(ev), "--github-output", str(gh), "--always-zero", "--no-model",
            "--out-dir", str(tmp_path / "o4"), "--contrib", str(tmp_path / "none.jsonl")])
    assert "valid=false" in gh.read_text()


def _hostile_bodies():
    rec = C.normalize_record(good())
    box = C.issue_body(rec).replace("### Consent", "### Anything else (optional)\n\nmy box is host 10.0.0.7\n\n"
                                                   "### Consent")
    return [
        ("```json\n" + "[" * 30000 + "\n```", "nested"),
        ("x```\n" * 13000, "not valid JSON"),
        (box, "outside the record"),
        (C.issue_body(C.normalize_record(good(contributor="mallory"))), "own GitHub login"),
        (C.issue_body(good(recipe={"FF_DEPTH": "inf"})), "not finite"),
    ]


@needs_data
@pytest.mark.parametrize("body, needle", _hostile_bodies(), ids=["deep", "fences", "box-ip", "not-author", "inf"])
def test_hostile_issue_bodies_always_get_a_comment(tmp_path, body, needle):
    """Every hostile body ends in a written comment and valid=false, quickly: no traceback, no regex blow-up."""
    import time
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"issue": {"number": 7, "body": body, "user": {"login": "octocat"}}}), encoding="utf-8")
    gh = tmp_path / "gh"
    t0 = time.perf_counter()
    rc = C.main(["validate", "--issue-event", str(ev), "--out-dir", str(tmp_path / "o"), "--github-output", str(gh),
                 "--always-zero", "--contrib", str(tmp_path / "none.jsonl")])
    assert rc == 0 and time.perf_counter() - t0 < 10
    assert "valid=false" in gh.read_text(encoding="utf-8")
    comment = (tmp_path / "o" / "comment.md").read_text(encoding="utf-8")
    assert "cannot be accepted" in comment and needle in comment.replace("\\", ""), comment
    assert not (tmp_path / "o" / "contrib-line.jsonl").exists()
    if needle == "outside the record":
        assert "secret=true" in gh.read_text(encoding="utf-8") and "10.0.0.7" not in comment


def test_top_level_cli_delegates_to_contrib(capsys):
    from ffsim import cli
    assert cli.main(["contrib", "schema"]) == 0
    assert json.loads(capsys.readouterr().out) == C.SCHEMA


# ------------------------------------------------------------------------------------------------ issue round trip
def test_issue_body_round_trip_and_form_layout():
    rec = C.normalize_record(good())
    data, consent = C.extract_issue_record(C.issue_body(rec))
    assert data == rec and consent is True
    form = ("### Run record (JSON)\n\n```json\n" + json.dumps(rec, indent=1) + "\n```\n\n### Anything else (optional)"
            "\n\n_No response_\n\n### Consent\n\n- [X] I agree that this record becomes public\n")
    assert C.extract_issue_record(form) == (rec, True)
    crlf = form.replace("\n", "\r\n").replace("- [X]", "- [ ]")
    assert C.extract_issue_record(crlf) == (rec, False)
    assert C.extract_issue_record(json.dumps(rec)) == (rec, None)
    # JSON pasted with its own fence: the form wraps it in a second one
    twice = form.replace("```json\n", "```json\n```json\n").replace("\n```\n\n###", "\n```\n```\n\n###", 1)
    assert C.extract_issue_record(twice) == (rec, True)
    assert C.issue_free_text(form) == "" and "box" in C.issue_free_text(form.replace("_No response_", "my box"))
    for bad in ("", "### Run record (JSON)\n\n_No response_\n", "```json\n{not json}\n```"):
        with pytest.raises(ValueError):
            C.extract_issue_record(bad)


def test_issue_url_is_encoded_bounded_and_decodes_to_the_record():
    rec = C.normalize_record(good())
    url = C.build_issue_url(rec)
    assert url.startswith("https://github.com/marker2601/trainium-simulator/issues/new?template=run-submission.yml&")
    assert len(url) <= C.ISSUE_URL_MAX and " " not in url and "\n" not in url
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert q["template"] == ["run-submission.yml"] and q["title"][0].startswith("[run] trn2.3xlarge K82s4")
    assert json.loads(q["record"][0]) == rec
    plain = C.build_plain_issue_url(rec)
    qp = urllib.parse.parse_qs(urllib.parse.urlsplit(plain).query)
    assert len(plain) <= C.ISSUE_URL_MAX and qp["labels"] == ["run-submission"]
    assert C.extract_issue_record(qp["body"][0]) == (rec, True)
    # a big custom recipe falls back to compact JSON, then refuses rather than truncating
    mid = C.normalize_record(good(base_recipe="custom", recipe={f"FF_KNOB_{i:03d}": "0.5" for i in range(120)}))
    murl = C.build_issue_url(mid)
    assert len(murl) <= C.ISSUE_URL_MAX
    assert json.loads(urllib.parse.parse_qs(urllib.parse.urlsplit(murl).query)["record"][0]) == mid
    huge = C.normalize_record(good(base_recipe="custom", recipe={f"FF_KNOB_{i:03d}": "0.12345" for i in range(390)}))
    with pytest.raises(ValueError):
        C.build_issue_url(huge)


def test_comment_text_is_made_inert():
    res = C.ValidationResult(False, ["bad value `x` from @someone <script> [click](https://evil.example) ![i](x)"])
    text = C.format_comment(res)
    assert "@someone" not in text and "<script>" not in text and "`x`" not in text
    assert "](" not in text and "https://evil" not in text and "![" not in text
    assert text.startswith("<!-- ffsim-contrib-check -->")


# ------------------------------------------------------------------------------------------------ the space copy
def test_space_copy_of_ffsim_is_identical():
    space = ROOT / "space" / "ffsim"
    if not space.is_dir():
        pytest.skip("no space/ffsim")
    for f in sorted(space.rglob("*")):
        if f.is_file() and f.suffix in (".py", ".json") and "__pycache__" not in f.parts:
            src = ROOT / "ffsim" / f.relative_to(space)
            assert src.exists(), f"{f} has no ffsim/ original"
            assert f.read_bytes().replace(b"\r\n", b"\n") == src.read_bytes().replace(b"\r\n", b"\n"), (
                f"space/ffsim/{f.relative_to(space)} differs from ffsim/: copy it over")
    for name in ("contrib.py", "params.py"):
        assert (space / name).exists()


# ------------------------------------------------------------------------------------------------ the app
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("gradio")
    if not (ROOT / "space" / "research" / "sim-data" / "runs.jsonl").exists():
        pytest.skip("space data not present")
    sys.path.insert(0, str(ROOT / "space"))
    try:
        spec = importlib.util.spec_from_file_location("ffsim_space_app", ROOT / "space" / "app.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ffsim_space_app"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(ROOT / "space"))
    return mod


def test_app_contrib_record_endpoint(app):
    names = {d.get("api_name") for d in app.demo.config["dependencies"]}
    assert {"predict", "predict_overrides", "speed_to_score", "score_to_speed", "contrib_record"} <= names
    r = app.contrib_record("trn2.3xlarge", "", 1800, "K82s4", "", "FF_COOLDOWN_FRAC=0.65", 58, 0.9551, 2097152, None,
                           2390, None, "", "Neuron SDK 2.26", "octocat", True)
    assert r["ok"], r["errors"]
    assert r["record"]["recipe"] == {"FF_COOLDOWN_FRAC": "0.65"} and r["record"]["seed"] == 58
    assert r["issue_url"] and len(r["issue_url"]) <= 7000 and r["issue_url_plain"]
    assert r["prediction"] and abs(r["prediction"]["z"]) < 4
    md, js = app.contrib_ui("trn2.3xlarge", "", 1800, "K82s4", "", "FF_COOLDOWN_FRAC=0.65", 58, 0.9551, 2097152, None,
                            2390, None, "", "", "", True)
    assert "Open the pre-filled GitHub issue" in md and json.loads(js)["steps"] == 2390
    bad = app.contrib_record("trn2.3xlarge", "", 1800, "K82s4", "", "FF_X=" + _AKID, 58, 0.9551, 2097152,
                             None, 2390, None, "", "", "", False)
    assert not bad["ok"] and any("access key" in e for e in bad["errors"])     # a secret: only the scan errors
    assert all(_AKID not in e for e in bad["errors"])
    unticked = app.contrib_record("trn2.3xlarge", "", 1800, "K82s4", "", "FF_X=1", 58, 0.9551, 2097152, None, 2390,
                                  None, "", "", "", False)
    assert not unticked["ok"] and any("consent" in e for e in unticked["errors"])
    huge = app.contrib_record("trn2.3xlarge", "", 1800, "K82s4", "", "FF_X=1 " * 5000, 58, 0.9551)
    assert not huge["ok"] and "longer than" in huge["errors"][0]
    assert app.demo._queue.max_size == 64                                  # a bounded public queue
    assert "Not ready" in app.format_contrib(bad)
    junk = app.contrib_record("trn2.3xlarge", "", 1800, "K82s4", "", "not-a-pair", 58, 0.9551)
    assert not junk["ok"] and junk["errors"]


def test_app_existing_endpoints_still_work(app):
    r = app.predict_overrides("K82s4", "FF_COOLDOWN_FRAC=0.65", 73, "C", 500)
    assert "error" not in r and r["candidate"]["local_bpb_mean"] > 0.9
    md, res = app.speed_to_score(0.9614, 10)
    assert res["predicted_score"] < 0.9614
    md, res = app.score_to_speed(0.9614, 0.9554)
    assert res["throughput_gain_pct"] > 0


# ------------------------------------------------------------------------------------------------ GitHub files
def _yaml(path):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_issue_form_matches_the_parser():
    form = _yaml(ROOT / ".github" / "ISSUE_TEMPLATE" / "run-submission.yml")
    ids = {b.get("id"): b for b in form["body"] if b.get("id")}
    assert ids[C.ISSUE_FIELD]["attributes"]["label"] == "Run record (JSON)"
    assert ids[C.ISSUE_FIELD]["attributes"]["render"] == "json"
    assert ids["consent"]["attributes"]["label"] == "Consent"
    assert ids["consent"]["attributes"]["options"][0]["required"] is True
    assert "run-submission" in form["labels"] and form["title"].startswith("[run]")
    _yaml(ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml")


def _run_lines(wf):
    for job in wf["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                yield step["run"]


_SHA_PIN = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{40} # v\d+(\.\d+){0,2}$")


@pytest.mark.parametrize("name", ["contrib.yml", "refit.yml", "ci.yml"])
def test_workflows_never_interpolate_event_text_into_shell(name):
    wf = _yaml(ROOT / ".github" / "workflows" / name)
    for run in _run_lines(wf):
        assert "${{" not in run, f"{name}: expression inside a run: line ({run[:60]})"
    text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    assert "issue.body" not in text and "issue.title }}" not in text
    top = wf.get("permissions")
    assert top in ({}, {"contents": "read"}), f"{name}: top-level permissions must be minimal"


@pytest.mark.parametrize("name", ["contrib.yml", "refit.yml", "ci.yml"])
def test_every_action_is_pinned_to_a_commit_sha_and_no_credentials_persist(name):
    text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    uses = re.findall(r"uses:\s*(\S+(?: # \S+)?)", text)
    assert uses
    for u in uses:
        assert _SHA_PIN.match(u), f"{name}: {u} is not pinned to a full commit SHA with its tag in a comment"
    wf = _yaml(ROOT / ".github" / "workflows" / name)
    for job in wf["jobs"].values():
        for step in job.get("steps", []):
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False, f"{name}: checkout persists the token"
    assert "git push" not in text, f"{name}: workflows open pull requests, they never push"
    dependabot = _yaml(ROOT / ".github" / "dependabot.yml")
    assert {u["package-ecosystem"] for u in dependabot["updates"]} >= {"github-actions", "pip"}


def test_contrib_workflow_permissions_and_approval_gate():
    wf = _yaml(ROOT / ".github" / "workflows" / "contrib.yml")
    on = wf.get("on", wf.get(True))
    assert set(on["issues"]["types"]) == {"opened", "edited", "reopened", "labeled"}
    check, pr = wf["jobs"]["check"], wf["jobs"]["pull-request"]
    assert check["permissions"] == {"contents": "read", "issues": "write"}
    assert pr["permissions"] == {"contents": "write", "pull-requests": "write"}
    # the pull request is opened only on a maintainer's approval label, never on an edit
    assert "github.event.action == 'labeled'" in pr["if"] and "contrib-approved" in pr["if"]
    assert "getCollaboratorPermissionLevel" in pr["steps"][0]["with"]["script"]
    # an edit after approval withdraws it; the comment is matched by the bot's login
    script = next(s["with"]["script"] for s in check["steps"] if "github-script" in str(s.get("uses")))
    assert "removeLabel" in script and "p.changes.body" in script and "'github-actions[bot]'" in script
    assert "c.user.type === 'Bot'" not in script
    # the write job runs no Python and installs nothing
    for step in pr["steps"]:
        run = step.get("run", "")
        assert "pip" not in run and "python" not in run, run
        assert "setup-python" not in str(step.get("uses", ""))
    cpr = next(s for s in pr["steps"] if "create-pull-request" in str(s.get("uses")))
    assert cpr["with"]["branch"].endswith("-${{ needs.check.outputs.hash }}")
    assert cpr["with"]["labels"] == "contrib"
    text = (ROOT / ".github" / "workflows" / "contrib.yml").read_text(encoding="utf-8")
    assert "--issue-event \"$GITHUB_EVENT_PATH\"" in text and "--require-hashes" in text
    assert "cache: pip" not in text
    refit = _yaml(ROOT / ".github" / "workflows" / "refit.yml")
    on = refit.get("on", refit.get(True))
    assert on["push"]["paths"] == ["research/sim-data/contrib/**"]
    job = refit["jobs"]["refit"]
    assert job["permissions"] == {"contents": "write", "pull-requests": "write"}
    runs = " ".join(_run_lines(refit))
    assert "--guard" in runs and "--require-hashes" in runs
    assert "cache: pip" not in (ROOT / ".github" / "workflows" / "refit.yml").read_text(encoding="utf-8")
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    assert re.search(r"^numpy==\d+\.\d+\.\d+ \\$", lock, re.M) and len(re.findall(r"--hash=sha256:[0-9a-f]{64}", lock)) >= 1
