"""research/experiments.csv (chips A/B, 348 rows) -> RunRecords.

Columns: chip,run,steps,raw_bpb,norm_bpb,in_window,is_screen,FF_* (63 knobs),NEURON_*,overrides.
knobs = the non-empty FF_*/NEURON_* columns, completed from the `overrides` string (the string
carries full values where a column was cut at the first space, e.g. FF_CC_FLAGS=--model-type=... -O3).
seed = FF_SEED, else the `_s<seed>` run-name suffix. Date is unknown (None); the run-name family
token (`HHMM_<family>_<tag>_s<seed>`) is tagged as family:<token> and era:<leading letters>.
norm_bpb (step-normalised, same-recipe use only) is kept in notes, not as a quality field.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional

from .parse_logs import parse_overrides
from .schema import RunRecord

__all__ = ["load_experiments_csv", "parse_experiments_csv", "row_to_record", "run_family", "seed_from_name"]

SEED_SUFFIX_RE = re.compile(r"_s(\d+)(?:_|$)")
NAME_RE = re.compile(r"^(\d{4})_([A-Za-z0-9]+)(?:_|$)")
SCREEN_MAX_STEPS = 200


def _f(s: Optional[str]) -> Optional[float]:
    if s is None or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _i(s: Optional[str]) -> Optional[int]:
    v = _f(s)
    return int(v) if v is not None else None


def _b(s: Optional[str]) -> Optional[bool]:
    if s is None or s.strip() == "":
        return None
    return s.strip().lower() in ("1", "true", "yes", "y", "t")


def run_family(run: str) -> Optional[str]:
    """'0816_R_g7lk35_s73' -> 'R'; '1515_H256ss_s52' -> 'H256ss'; no HHMM_ prefix -> None."""
    m = NAME_RE.match(run)
    return m.group(2) if m else None


def seed_from_name(run: str) -> Optional[int]:
    m = SEED_SUFFIX_RE.search(run)
    return int(m.group(1)) if m else None


def row_to_record(row: Dict[str, str]) -> RunRecord:
    chip = (row.get("chip") or "?").strip().upper()
    run = (row.get("run") or "").strip()
    rec = RunRecord(run_id=f"{chip}:{run}", chip=chip, run=run, sources=["experiments.csv"])
    rec.steps = _i(row.get("steps"))
    rec.bpb_2m = _f(row.get("raw_bpb"))
    rec.in_window = _b(row.get("in_window"))
    screen_col = _b(row.get("is_screen"))
    knobs: Dict[str, str] = {}
    for key, val in row.items():
        if key and (key.startswith("FF_") or key.startswith("NEURON_")) and val is not None and val.strip() != "":
            knobs[key] = val.strip()
    ov = parse_overrides(row.get("overrides") or "")
    for key, val in ov.items():
        cur = knobs.get(key)
        if cur is None or cur != val:
            knobs[key] = val          # the overrides string is the command that ran; it wins
    rec.knobs = knobs
    rec.knobs_complete = False
    if "FF_SEED" in knobs:
        try:
            rec.seed = int(float(knobs["FF_SEED"]))
        except ValueError:
            rec.seed = seed_from_name(run)
    else:
        rec.seed = seed_from_name(run)
    if knobs.get("FF_TIME_TARGET"):
        rec.time_target = _f(knobs["FF_TIME_TARGET"])
    small = False
    for key in ("NEURON_COMPETITION_R1_NUM_STEPS", "FF_STEP_TOTAL"):
        v = _i(knobs.get(key)) if knobs.get(key) else None
        if v is not None and v <= SCREEN_MAX_STEPS:
            small = True
    rec.is_screen = bool(screen_col) or small or (rec.steps is not None and rec.steps <= SCREEN_MAX_STEPS)
    fam = run_family(run)
    if fam:
        rec.tags.append(f"family:{fam}")
        m = re.match(r"[A-Za-z]+", fam)
        era = m.group(0) if m else fam
        rec.tags.append(f"era:{era}")
    if rec.is_screen:
        rec.tags.append("screen")
    notes = []
    nb = _f(row.get("norm_bpb"))
    if nb is not None:
        notes.append(f"norm_bpb={row['norm_bpb'].strip()}")
    rec.notes = " | ".join(notes)
    return rec


def load_experiments_csv(path: str) -> List[RunRecord]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if (r.get("run") or "").strip()]
    return [row_to_record(r) for r in rows]


parse_experiments_csv = load_experiments_csv   # name ffsim/cli.py looks for
