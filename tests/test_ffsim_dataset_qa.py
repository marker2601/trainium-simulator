"""Pins research/sim-data/qa-metrics.json (the lists behind dataset-qa.md sections 2 and 4) to a recomputation
from runs.jsonl and the monitor csv files, so the shipped numbers cannot drift from the data (stdlib only).

Section 4 (DF-7): a `desc:` monitor pseudo-run is a semantic duplicate of an experiments.csv record when the two
bpb_2m agree to 5 dp (|a - b| <= 5e-6) on the same chip, or on any chip when the desc: chip is unknown ('?').
Section 2 (DF-1): the raw extractions monitor-runs.a.csv / .b.csv vs the chip logs, matched by run name + chip.
"""
from __future__ import annotations

import csv
import json
import os

import pytest

from ffsim.dataset import load_runs
from ffsim.parse_logs import parse_eval_log

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(REPO, "research", "sim-data")
RUNS = os.path.join(SIM, "runs.jsonl")
METRICS = os.path.join(SIM, "qa-metrics.json")
DUP_TOL = 5e-6
BPB_TOL = 1e-5

needs_data = pytest.mark.skipif(not (os.path.isfile(RUNS) and os.path.isfile(METRICS)),
                                reason="research/sim-data/runs.jsonl or qa-metrics.json not present")


@pytest.fixture(scope="module")
def data():
    with open(METRICS, encoding="utf-8") as fh:
        metrics = json.load(fh)
    return load_runs(RUNS), metrics


def _semantic_duplicates(recs):
    csv_by_chip = {}
    for r in recs:
        if "experiments.csv" in r.sources and r.bpb_2m is not None:
            csv_by_chip.setdefault(r.chip, []).append(r)
    all_csv = [x for v in csv_by_chip.values() for x in v]
    pairs = []
    for d in recs:
        if not d.run.startswith("desc:") or d.bpb_2m is None:
            continue
        pool = all_csv if d.chip == "?" else csv_by_chip.get(d.chip, [])
        pairs.extend((d.run_id, c.run_id) for c in pool if abs(d.bpb_2m - c.bpb_2m) <= DUP_TOL)
    return pairs


@needs_data
def test_semantic_duplicates_match_shipped_list(data):
    recs, metrics = data
    sd = metrics["semantic_duplicates"]
    pairs = _semantic_duplicates(recs)
    assert set(pairs) == {(p["desc_run_id"], p["csv_run_id"]) for p in sd["pairs"]}
    assert len(pairs) == len(sd["pairs"]) == sd["counts"][sd["primary_rule"]]["pairs"]
    assert sorted({d for d, _ in pairs}) == sd["desc_run_ids"]
    assert sd["counts"][sd["primary_rule"]]["distinct_desc_rows"] == len(sd["desc_run_ids"])
    by_id = {r.run_id: r for r in recs}
    for rid in sd["desc_run_ids"]:                 # monitor-only pseudo-runs, never a chip_log / csv record
        assert by_id[rid].sources == ["monitor"] and by_id[rid].run.startswith("desc:")


@needs_data
def test_semantic_duplicate_seed_flags(data):
    recs, metrics = data
    by_id = {r.run_id: r for r in recs}
    twins = {}
    for p in metrics["semantic_duplicates"]["pairs"]:
        d, c = by_id[p["desc_run_id"]], by_id[p["csv_run_id"]]
        expect = None if d.seed is None or c.seed is None else d.seed == c.seed
        assert p["seed_agrees"] == expect
        assert p["chip_desc"] == d.chip and p["chip_csv"] == c.chip
        assert d.chip == "?" or d.chip == c.chip
        twins.setdefault(p["desc_run_id"], []).append(expect)
    # every listed desc: row has at least one twin whose seed agrees or is unknown (the False ones are 5-dp coincidences)
    assert all(any(v in (True, None) for v in vs) for vs in twins.values())


def _log_bpb(chip, run):
    """The dir's own eval.log val_bpb (a merged runs.jsonl record fills bpb_2m from the monitor when the log has none)."""
    path = os.path.join(SIM, f"chip{chip}", run, "eval.log")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_eval_log(fh.read())[0]


def _monitor_vs_log(rows, logs):
    out = {"matched_by_name": 0, "bpb_both": 0, "bpb_disagree_gt_1e-5": 0, "steps_log_is_monitor_plus_1": 0,
           "steps_other_complete": 0}
    for r in rows:
        run = (r.get("run") or "").strip()
        chip = (r.get("chip") or "").strip().upper()
        lg = logs.get((chip, run))
        if not run or run.startswith("desc:") or lg is None:
            continue
        out["matched_by_name"] += 1
        mb = r.get("bpb_2m", "").strip()
        lb = _log_bpb(chip, run) if mb else None
        if mb and lb is not None:
            out["bpb_both"] += 1
            out["bpb_disagree_gt_1e-5"] += abs(float(mb) - lb) > BPB_TOL
        ms = r.get("steps", "").strip()
        if ms and lg.steps is not None and "log_incomplete" not in lg.tags:   # an incomplete log's steps come from the monitor
            ms = int(float(ms))
            if lg.steps == ms + 1:
                out["steps_log_is_monitor_plus_1"] += 1
            elif lg.steps != ms:
                out["steps_other_complete"] += 1
    return out


@needs_data
@pytest.mark.parametrize("key,name", [("raw_extraction_a", "monitor-runs.a.csv"),
                                      ("raw_extraction_b", "monitor-runs.b.csv"),
                                      ("reconciled_monitor_runs_csv", "monitor-runs.csv")])
def test_monitor_vs_chip_log_counts(data, key, name):
    recs, metrics = data
    if not os.path.isfile(os.path.join(SIM, "chipC", "_meta", "HARVEST.json")):
        pytest.skip("needs the full chip C/D harvest (the public release ships three reference run dirs only)")
    path = os.path.join(SIM, name)
    if not os.path.isfile(path):
        pytest.skip(f"{name} not present")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # a complete chip_log record's steps are the log's (merge precedence chip_log > monitor); bpb is read from eval.log
    logs = {(r.chip, r.run): r for r in recs if "chip_log" in r.sources}
    got = _monitor_vs_log(rows, logs)
    want = metrics["monitor_vs_chip_log"][key]
    assert got["matched_by_name"] == want["matched_by_name"]
    assert got["bpb_both"] == want["bpb_both"]
    assert got["bpb_disagree_gt_1e-5"] == want["bpb_disagree_gt_1e-5"] == 0
    assert got["steps_log_is_monitor_plus_1"] == want["steps_log_is_monitor_plus_1"]
    assert got["steps_other_complete"] == sum(1 for x in want["steps_other_runs"] if not x["log_incomplete"])
    assert len(rows) == want["rows"]


@needs_data
def test_headline_numbers_as_documented(data):
    """The figures quoted in dataset-qa.md sections 2 and 4 (update both when runs.jsonl is rebuilt)."""
    _, metrics = data
    mv = metrics["monitor_vs_chip_log"]
    assert (mv["raw_extraction_a"]["matched_by_name"], mv["raw_extraction_a"]["steps_log_is_monitor_plus_1"]) == (23, 10)
    assert (mv["raw_extraction_b"]["matched_by_name"], mv["raw_extraction_b"]["steps_log_is_monitor_plus_1"]) == (18, 6)
    assert mv["reconciled_monitor_runs_csv"]["rows_tagged_log_confirmed"] == 254
    assert len(metrics["semantic_duplicates"]["desc_run_ids"]) == 128
