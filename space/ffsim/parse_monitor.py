"""research/sim-data/monitor-runs.csv (LLM-extracted from the campaign ledger) -> RunRecords,
plus the official-uploads table.

monitor-runs.csv header:
  date_utc,chip,run,code_version,recipe_tag,seed,steps,bpb_2m,bpb_2m_ema,bpb_20m,submission,
  is_screen,control_run,control_bpb,delta_raw,delta_norm,knob_changes,note
knob_changes is 'KEY=VALUE KEY=VALUE' -> knobs (never complete): only FF_* / NEURON_* keys are knobs
(schema.py: the env exactly as train_ff.py reads it), monitor shorthand in KNOB_ALIASES is renamed to
its FF_ key, and every other token (recipe labels, runtime env) goes to notes as `unmapped_changes=`;
control_run -> control_of as a run_id (chip:run, same chip unless the cell already has a chip prefix);
recipe_tag -> tags; control_bpb / delta_raw / delta_norm / note -> notes.

The official uploads table has no frozen schema in CONTRACT.md; load_official_uploads() is
schema-tolerant: every column is kept as a string, *_bpb / offset / steps / seed / rank columns are
coerced to numbers, empty cells become None, and `offset` is filled from official_bpb - rehearsal_bpb
when both are present.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from .parse_logs import parse_overrides
from .schema import RunRecord

__all__ = ["load_monitor_csv", "parse_monitor_csv", "load_official_uploads", "parse_knob_changes",
           "split_knob_changes", "KNOB_ALIASES", "KNOB_PREFIXES", "UNMAPPED_NOTE", "MONITOR_COLUMNS"]

MONITOR_COLUMNS = [
    "date_utc", "chip", "run", "code_version", "recipe_tag", "seed", "steps", "bpb_2m", "bpb_2m_ema",
    "bpb_20m", "submission", "is_screen", "control_run", "control_bpb", "delta_raw", "delta_norm",
    "knob_changes", "note",
]
SCREEN_MAX_STEPS = 200
NUMERIC_UPLOAD_KEYS = ("offset", "steps", "seed", "rank")

# Knobs are the FF_* / NEURON_* env exactly as train_ff.py reads them (schema.py). The monitor's knob_changes
# column also carries recipe shorthand: the shorthand whose FF_ name is known is renamed (value kept verbatim),
# everything else (`x0_init=9r0.16`, `S123=FF_ROPE_FN+QKfold+FF_RMSNORM_FN`, `EWD0=1`, runtime env such as
# XU_QUEUE_DEFAULT / TORCH_*) is kept out of knobs and recorded in notes as `unmapped_changes=...`.
KNOB_PREFIXES = ("FF_", "NEURON_")
KNOB_ALIASES: Dict[str, str] = {
    # shorthand -> train_ff.py env name. Verified against the same runs' chip logs, which carry the FF_ key at
    # the same value (C:0921_R_g7sc18_s58 FF_SOFTCAP=18, C:1100_R_g7wu40_s58 FF_WARMUP=40, C:1529_G_POOL
    # FF_POOL_LAST=3, C:1527_G_CFC FF_CFC_INIT=1.0, C:0504_G_S2 FF_QK_GAIN_FOLD=1), and against the
    # `_env_float("FF_HEAD_STD" | "FF_WTE_STD" | "FF_ZLOSS", ...)` reads in train.py.
    "softcap": "FF_SOFTCAP",
    "warmup": "FF_WARMUP",
    "head_init_std": "FF_HEAD_STD",
    "wte_init_std": "FF_WTE_STD",
    "z_loss": "FF_ZLOSS",
    "POOL": "FF_POOL_LAST",
    "CFC": "FF_CFC_INIT",
    "QKfold": "FF_QK_GAIN_FOLD",
}
UNMAPPED_NOTE = "unmapped_changes"


def _s(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    return v or None


def _f(v: Optional[str]) -> Optional[float]:
    v = _s(v)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(v: Optional[str]) -> Optional[int]:
    x = _f(v)
    return int(x) if x is not None else None


def _b(v: Optional[str]) -> bool:
    v = _s(v)
    return bool(v) and v.lower() in ("1", "true", "yes", "y", "t")


def split_knob_changes(s: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """'softcap=18 FF_SEED=58 x0_init=9flat' -> ({'FF_SOFTCAP': '18', 'FF_SEED': '58'}, {'x0_init': '9flat'}).

    Tokens parse as overrides do (values with spaces glue onto the previous key); KNOB_ALIASES are renamed;
    only FF_* / NEURON_* keys are knobs, every other token comes back in the second dict (original key).
    """
    knobs: Dict[str, str] = {}
    other: Dict[str, str] = {}
    for key, val in parse_overrides(s or "").items():
        name = KNOB_ALIASES.get(key, key)
        if name.startswith(KNOB_PREFIXES):
            knobs[name] = val
        else:
            other[key] = val
    return knobs, other


def parse_knob_changes(s: Optional[str]) -> Dict[str, str]:
    """The knobs half of split_knob_changes(): 'FF_SEED=73 softcap=18' -> {'FF_SEED': '73', 'FF_SOFTCAP': '18'}."""
    return split_knob_changes(s)[0]


def _row_to_record(row: Dict[str, str]) -> RunRecord:
    chip = (_s(row.get("chip")) or "?").upper()
    run = _s(row.get("run")) or ""
    rec = RunRecord(run_id=f"{chip}:{run}", chip=chip, run=run, sources=["monitor"])
    rec.date_utc = _s(row.get("date_utc"))
    rec.code_version = _s(row.get("code_version"))
    rec.seed = _i(row.get("seed"))
    rec.steps = _i(row.get("steps"))
    rec.bpb_2m = _f(row.get("bpb_2m"))
    rec.bpb_2m_ema = _f(row.get("bpb_2m_ema"))
    rec.bpb_20m = _f(row.get("bpb_20m"))
    rec.submission = _s(row.get("submission"))
    rec.knobs, unmapped = split_knob_changes(row.get("knob_changes"))
    rec.knobs_complete = False
    if rec.seed is None and rec.knobs.get("FF_SEED"):
        rec.seed = _i(rec.knobs["FF_SEED"])
    if rec.knobs.get("FF_TIME_TARGET"):
        rec.time_target = _f(rec.knobs["FF_TIME_TARGET"])
    ctrl = _s(row.get("control_run"))
    if ctrl:
        rec.control_of = ctrl if ":" in ctrl else f"{chip}:{ctrl}"
    rec.is_screen = _b(row.get("is_screen")) or (rec.steps is not None and rec.steps <= SCREEN_MAX_STEPS)
    tag = _s(row.get("recipe_tag"))
    if tag:
        rec.tags.append(f"recipe:{tag}")
    if rec.is_screen:
        rec.tags.append("screen")
    notes: List[str] = []
    for key in ("control_bpb", "delta_raw", "delta_norm"):
        v = _s(row.get(key))
        if v is not None:
            notes.append(f"{key}={v}")
    if unmapped:
        notes.append(f"{UNMAPPED_NOTE}=" + " ".join(f"{k}={v}" for k, v in unmapped.items()))
    note = _s(row.get("note"))
    if note:
        notes.append(note)
    rec.notes = " | ".join(notes)
    return rec


def load_monitor_csv(path: str, missing_ok: bool = False) -> List[RunRecord]:
    """Rows without a run name are skipped. With missing_ok=True a missing file gives []."""
    if not os.path.isfile(path):
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if _s(r.get("run"))]
    return [_row_to_record(r) for r in rows]


parse_monitor_csv = load_monitor_csv   # name ffsim/cli.py looks for


def load_official_uploads(path: str, missing_ok: bool = False) -> List[Dict[str, Any]]:
    """The official leaderboard uploads table -> list of dicts (see module doc for coercion)."""
    if not os.path.isfile(path):
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    out: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            d: Dict[str, Any] = {}
            for key, val in row.items():
                if key is None:
                    continue
                k = key.strip()
                if k.endswith("_bpb") or k in NUMERIC_UPLOAD_KEYS:
                    d[k] = _f(val)
                else:
                    d[k] = _s(val)
            if not any(v is not None for v in d.values()):
                continue
            if d.get("offset") is None and d.get("official_bpb") is not None and d.get("rehearsal_bpb") is not None:
                d["offset"] = round(d["official_bpb"] - d["rehearsal_bpb"], 6)
            out.append(d)
    return out
