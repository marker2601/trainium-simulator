"""Shared data contract for every ffsim module. Do not add fields without updating CONTRACT.md.

A RunRecord is one training run (full run or short screen) on one chip. Records are stored as
JSON lines in research/sim-data/runs.jsonl. Missing values are None. Knobs are the FF_* / NEURON_*
environment as strings exactly as train_ff.py reads them (e.g. {"FF_LEAKY_RELU2": "0.35"}).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

CHIPS = ("A", "B", "C", "D")

# Accumulation phases: k = micro-batches per optimizer step (k1 = 65,536 tokens, k2 = 131,072, k4 = 262,144).
K_PHASES = ("k1", "k2", "k4")


@dataclass
class StepTime:
    """Median seconds per optimizer step in each accumulation phase, from train.log `dt` lines.

    n_* is the number of logged step lines the median was taken from. None when the phase never ran.
    """
    k1: Optional[float] = None
    k2: Optional[float] = None
    k4: Optional[float] = None
    n_k1: int = 0
    n_k2: int = 0
    n_k4: int = 0


@dataclass
class RunRecord:
    run_id: str                       # f"{chip}:{run}" - unique
    chip: str                         # "A" | "B" | "C" | "D"
    run: str                          # run directory name, e.g. "0816_R_g7lk35_s73"
    sources: List[str] = field(default_factory=list)   # subset of ["experiments.csv", "monitor", "chip_log"]
    date_utc: Optional[str] = None    # "YYYY-MM-DD" when known
    code_version: Optional[str] = None  # "M12", "K57", "code_m9", ... (the train_ff.py lineage the run used)
    code_sha: Optional[str] = None    # sha256 of the train_ff.py / train.py that ran, when known
    seed: Optional[int] = None
    steps: Optional[int] = None       # optimizer steps completed (the number after "@" in monitor lines)
    bpb_2m: Optional[float] = None    # public shard, first 2,097,152 tokens, trained weights (raw, NOT step-normalised)
    bpb_2m_ema: Optional[float] = None
    bpb_20m: Optional[float] = None   # 20M-token re-eval when done
    official_bpb: Optional[float] = None  # leaderboard score, only for uploaded submissions
    submission: Optional[str] = None  # e.g. "K59" for uploads / rehearsals
    is_screen: bool = False           # short screen (S60 / 120-step) - never a quality data point
    in_window: Optional[bool] = None  # experiments.csv column (era window for the surrogate)
    charged_seconds: Optional[float] = None
    startup_seconds: Optional[float] = None
    time_target: Optional[float] = None
    knobs: Dict[str, str] = field(default_factory=dict)
    knobs_complete: bool = False      # True when knobs is the full effective env (base.env + overrides)
    step_time: StepTime = field(default_factory=StepTime)
    loss_curve: List[List[float]] = field(default_factory=list)  # [[step, train_loss], ...] every 10 steps
    control_of: Optional[str] = None  # run_id of the paired control, when the run was a treatment arm
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        d = dict(d)
        st = d.get("step_time") or {}
        d["step_time"] = StepTime(**st) if isinstance(st, dict) else st
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Recipe:
    """A candidate to simulate: a code version plus the effective FF_* environment."""
    name: str
    code_version: str
    knobs: Dict[str, str]
    chip: str = "C"                   # hardware/runtime the step-time model predicts for
    time_target: float = 1790.0       # charged seconds actually trained (K59: FF_TIME_TARGET 1793 -> ~1790 charged)


@dataclass
class SimResult:
    recipe: str
    n: int
    steps_mean: float
    steps_sd: float
    bpb_2m_mean: float                # predicted public-shard rehearsal bpb (trained weights, 2M tokens)
    bpb_2m_sd: float
    official_mean: float              # rehearsal + offset
    official_sd: float
    official_p10: float
    official_p50: float
    official_p90: float
    p_beat_best: float                # P(official < current best official score)
    step_time: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
