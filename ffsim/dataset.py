"""runs.jsonl load/save, merging the three sources by run_id, effective knobs, QA summary.

Merge precedence per field: chip_log > monitor > experiments.csv (first non-None wins), with:
  * bpb_2m (CSV_EXACT_FIELDS): chip_log > experiments.csv > monitor. experiments.csv is machine-written
    (raw_bpb at full precision) while the monitor table is LLM-extracted and rounds bpb to 5-6 dp, so the
    monitor's copy is used only when the csv has no value. `steps` keeps monitor > csv: the csv's steps column
    is the last logged step index (NUM_STEPS - 1 on every screen, one below the monitor's "@N" on 29/31 matched
    full runs) while the contract's steps is the number after "@";
  * sources / tags: union (order preserved);
  * knobs: dict union, higher-precedence source wins per key; knobs_complete: any source complete;
  * step_time / loss_curve: the chip_log's when it has data, else the first source with data;
  * is_screen: True if any source says so (a screen is never a quality data point);
  * code_version: the log's placeholder 'code' (no FF_CODE_DIR override) loses to any real value;
  * notes: distinct non-empty notes joined with ' | '.
  * a chip_log tagged `log_incomplete` (step lines but no ff_summary and no eval: the run was still going
    or died) loses the scalar fields (steps, bpb, charged...) to the monitor / csv, which report the finished
    run; its step_time / loss_curve are still used.
merge_runs() also adds `code_dir:<dir>` and `lineage:<M>` tags to runs whose code_version is a chip code
dir (`code_k12off`): the lineage is learned from runs the monitor also names (code_k12off -> M12).
"""
from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict
from dataclasses import fields
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import RunRecord, StepTime

__all__ = ["save_runs", "load_runs", "merge_runs", "effective_knobs", "full_runs", "summarize",
           "SOURCE_RANK", "CSV_EXACT_FIELDS", "WEAK_CODE_VERSION"]

SOURCE_RANK = {"chip_log": 0, "monitor": 1, "experiments.csv": 2}
# Fields experiments.csv carries as exact machine values: for these the csv outranks the LLM-extracted, 5-6 dp
# rounded monitor table (finding DF-3). Not `steps`: the csv's steps column is the last logged step index
# (119 for a 120-step screen, one below the monitor's "@N" on matched full runs), not the contract's count.
CSV_EXACT_FIELDS = ("bpb_2m",)
_CSV_EXACT_RANK = {"chip_log": 0, "experiments.csv": 1, "monitor": 2}
WEAK_CODE_VERSION = "code"
_KEY_FIELDS = ("run_id", "chip", "run")
_SPECIAL = set(_KEY_FIELDS) | {"sources", "tags", "knobs", "knobs_complete", "step_time", "loss_curve",
                               "is_screen", "code_version", "notes"}
_SCALAR_FIELDS = [f.name for f in fields(RunRecord) if f.name not in _SPECIAL]


def save_runs(records: Iterable[RunRecord], path: str) -> int:
    """Write JSON lines (one RunRecord per line, sorted keys). Returns the number written."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
            n += 1
    return n


def load_runs(path: str) -> List[RunRecord]:
    out: List[RunRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(RunRecord.from_dict(json.loads(line)))
    return out


def _rank(rec: RunRecord, table: Dict[str, int] = SOURCE_RANK) -> int:
    return min((table.get(s, 3) for s in rec.sources), default=3)


def _has_step_time(st: StepTime) -> bool:
    return any(v is not None for v in (st.k1, st.k2, st.k4))


def _union(seqs: Iterable[Sequence[str]]) -> List[str]:
    seen: "OrderedDict[str, None]" = OrderedDict()
    for seq in seqs:
        for x in seq:
            seen.setdefault(x, None)
    return list(seen)


INCOMPLETE_TAG = "log_incomplete"


def merge_group(group: List[RunRecord]) -> RunRecord:
    group = sorted(group, key=_rank)   # stable: keeps input order within a source
    head = group[0]
    out = RunRecord(run_id=head.run_id, chip=head.chip, run=head.run)
    out.sources = _union(r.sources for r in group)
    out.tags = _union(r.tags for r in group)
    scalars = sorted(group, key=lambda r: (1 if INCOMPLETE_TAG in r.tags else 0, _rank(r)))
    exact = sorted(group, key=lambda r: (1 if INCOMPLETE_TAG in r.tags else 0, _rank(r, _CSV_EXACT_RANK)))
    for name in _SCALAR_FIELDS:
        for r in (exact if name in CSV_EXACT_FIELDS else scalars):
            v = getattr(r, name)
            if v is not None:
                setattr(out, name, v)
                break
    cv = [r.code_version for r in group if r.code_version not in (None, WEAK_CODE_VERSION)]
    weak = [r.code_version for r in group if r.code_version == WEAK_CODE_VERSION]
    out.code_version = cv[0] if cv else (weak[0] if weak else None)
    knobs: Dict[str, str] = {}
    for r in reversed(group):          # lowest precedence first, so higher precedence overwrites
        knobs.update(r.knobs)
    out.knobs = knobs
    out.knobs_complete = any(r.knobs_complete for r in group)
    out.is_screen = any(r.is_screen for r in group)
    logs = [r for r in group if "chip_log" in r.sources]
    for pool in (logs, group):
        st = next((r.step_time for r in pool if _has_step_time(r.step_time)), None)
        if st is not None:
            out.step_time = StepTime(**vars(st))
            break
    for pool in (logs, group):
        lc = next((r.loss_curve for r in pool if r.loss_curve), None)
        if lc is not None:
            out.loss_curve = [list(p) for p in lc]
            break
    out.notes = " | ".join(_union([[n for n in (r.notes.strip(),) if n] for r in group]))
    return out


def merge_runs(list_of_lists: Iterable) -> Dict[str, RunRecord]:
    """Merge records from several sources into one RunRecord per run_id (insertion order kept).

    Accepts a list of per-source lists or a flat list of RunRecords (ffsim/cli.py passes a flat list).
    """
    groups: "OrderedDict[str, List[RunRecord]]" = OrderedDict()
    for item in list_of_lists:
        if item is None:
            continue
        recs = [item] if isinstance(item, RunRecord) else item
        for rec in recs:
            if rec is None:
                continue
            groups.setdefault(rec.run_id, []).append(rec)
    merged = OrderedDict((rid, merge_group(g)) for rid, g in groups.items())
    _add_lineage_tags(merged, groups)
    return merged


def _add_lineage_tags(merged: Dict[str, RunRecord], groups: Dict[str, List[RunRecord]]) -> None:
    """`code_dir:<dir>` + `lineage:<M>` tags for chip code-dir versions, the dir -> lineage map learned from
    runs that both a chip_log (code_k12off) and the monitor (M12) name."""
    votes: Dict[str, Counter] = {}
    for g in groups.values():
        dirs = [r.code_version for r in g if "chip_log" in r.sources and str(r.code_version or "").startswith("code_")]
        mons = [r.code_version for r in g if "monitor" in r.sources and r.code_version]
        if dirs and mons:
            votes.setdefault(dirs[0], Counter())[mons[0]] += 1
    lineage = {d: c.most_common(1)[0][0] for d, c in votes.items()}
    for rec in merged.values():
        cv = str(rec.code_version or "")
        if cv.startswith("code_"):
            rec.tags = _union([rec.tags, [f"code_dir:{cv}"] + ([f"lineage:{lineage[cv]}"] if cv in lineage else [])])


def effective_knobs(record: RunRecord, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """base.env with the record's knobs applied on top (the record's overrides win)."""
    env: Dict[str, str] = dict(base_env or {})
    env.update(record.knobs)
    return env


def full_runs(records: Iterable[RunRecord]) -> List[RunRecord]:
    """Quality data points only: no screens, and bpb_2m present."""
    return [r for r in records if not r.is_screen and r.bpb_2m is not None]


def summarize(records: Iterable[RunRecord]) -> Dict[str, object]:
    recs = list(records)
    by_chip = Counter(r.chip for r in recs)
    by_source = Counter(s for r in recs for s in r.sources)
    by_code = Counter(r.code_version or "?" for r in recs)
    return {
        "n": len(recs),
        "by_chip": dict(sorted(by_chip.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_code_version": dict(sorted(by_code.items())),
        "screens": sum(1 for r in recs if r.is_screen),
        "full_runs": len(full_runs(recs)),
        "with_bpb_2m": sum(1 for r in recs if r.bpb_2m is not None),
        "with_bpb_20m": sum(1 for r in recs if r.bpb_20m is not None),
        "with_step_time": sum(1 for r in recs if _has_step_time(r.step_time)),
        "with_loss_curve": sum(1 for r in recs if r.loss_curve),
        "knobs_complete": sum(1 for r in recs if r.knobs_complete),
        "with_seed": sum(1 for r in recs if r.seed is not None),
    }
