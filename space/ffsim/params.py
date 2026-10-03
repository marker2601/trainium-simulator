"""ffsim.params: the fitted models as plain JSON (no pickle), and a predict-only quality model rebuilt from it.

``export_params(models, data_meta)`` turns a fitted (steptime, quality, offset) triple into a JSON-safe dict:

* ``quality``  everything ``QualityModel.predict`` needs: columns, posterior mean ``beta`` and covariance ``cov``,
  residual ``sigma``, the era / chip / seed maps and the lineage tables, the config, and the column offsets.
  ``quality_from_params`` rebuilds a QualityModel from it whose ``predict`` / ``predict_record`` match the fitted
  model to the rounding used here (10 significant digits). It is predict-only: support tables, pair validation
  and refits need the run records and are not exported.
* ``offset``   mean, sd, n and the offsets behind them.
* ``steptime`` per batch phase: columns, coefficients (log step time), n, rmse and lineages. Informational (the
  step-time model also uses a residual nearest-neighbour term over its training runs, which is not exported).

No timestamps are written, so refitting unchanged data reproduces the file byte for byte on the same machine (the
refit workflow commits only when something changed). Floats are rounded to 10 significant digits for the same
reason; another machine's BLAS can still move the last digits, so ``contrib refit`` keeps the published file when
the new numbers only differ by that noise (``contrib.PARAMS_NOISE``).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from . import __version__
from .quality import QualityConfig, QualityModel

PARAMS_FORMAT = "ffsim-params/1"
SIG_DIGITS = 10

__all__ = ["PARAMS_FORMAT", "export_params", "quality_to_params", "quality_from_params", "write_params",
           "load_params"]


def _r(x: Any) -> Any:
    """Round floats (recursively) to SIG_DIGITS significant digits; non-finite floats become None."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if not math.isfinite(f):
            return None
        return float(f"{f:.{SIG_DIGITS}g}")
    if isinstance(x, np.ndarray):
        return [_r(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    return x


def quality_to_params(q: QualityModel) -> Dict[str, Any]:
    if not q.fitted:
        raise ValueError("quality_to_params: the model is not fitted")
    cfg = dataclasses.asdict(q.config)
    return _r({
        "config": cfg,
        "columns": list(q.columns),
        "beta": q.beta,
        "cov": q.cov,
        "sigma": q.sigma,
        "n": q.n,
        "edf": q.edf,
        "eras": dict(q.eras),
        "chips": dict(q.chips),
        "seeds": {str(k): v for k, v in sorted(q.seeds.items())},
        "era_dates": dict(q.era_dates),
        "era_counts": dict(q.era_counts),
        "era_members": {k: list(v) for k, v in q.era_members.items()},
        "era_merges": dict(q.era_merges),
        "version_to_era": dict(q.version_to_era),
        "version_aliases": dict(q.version_aliases),
        "pinned": list(q.pinned),
        "era_confounded": list(q.era_confounded_features()),
        "prior_mean": q._prior_mean,
        "prior_sd": q._prior_sd,
        "index": {"intercept": q._i_intercept, "era0": q._i_era0, "chip0": q._i_chip0, "L": q._i_L, "L2": q._i_L2,
                  "knob0": q._i_knob0, "seed0": q._i_seed0},
        "dropped": dict(q.dropped),
    })


def quality_from_params(d: Dict[str, Any]) -> QualityModel:
    """A predict-only QualityModel from ``quality_to_params`` output."""
    raw = dict(d.get("config") or {})
    known = {f.name: f for f in dataclasses.fields(QualityConfig)}
    kw: Dict[str, Any] = {}
    for k, v in raw.items():
        if k not in known:
            continue
        if k in ("unknown_eras", "fit_eras") and v is not None:
            v = tuple(v)
        kw[k] = v
    q = QualityModel(QualityConfig(**kw))
    q.columns = list(d["columns"])
    q.beta = np.asarray(d["beta"], dtype=float)
    q.cov = np.asarray(d["cov"], dtype=float)
    q.sigma = float(d["sigma"])
    q.n = int(d.get("n", 0))
    q.edf = float(d.get("edf", 0.0))
    q.eras = {str(k): int(v) for k, v in d["eras"].items()}
    q.chips = {str(k): int(v) for k, v in d["chips"].items()}
    q.seeds = {int(k): int(v) for k, v in d["seeds"].items()}
    q.era_dates = dict(d.get("era_dates") or {})
    q.era_counts = {k: int(v) for k, v in (d.get("era_counts") or {}).items()}
    q.era_members = {k: list(v) for k, v in (d.get("era_members") or {}).items()}
    q.era_merges = dict(d.get("era_merges") or {})
    q.version_to_era = dict(d.get("version_to_era") or {})
    q.version_aliases = dict(d.get("version_aliases") or {})
    q.pinned = list(d.get("pinned") or [])
    q._era_confounded = list(d.get("era_confounded") or [])
    q._prior_mean = np.asarray(d.get("prior_mean") or np.zeros(len(q.columns)), dtype=float)
    q._prior_sd = np.asarray(d.get("prior_sd") or np.zeros(len(q.columns)), dtype=float)
    idx = d["index"]
    q._i_intercept, q._i_era0, q._i_chip0 = int(idx["intercept"]), int(idx["era0"]), int(idx["chip0"])
    q._i_L, q._i_L2, q._i_knob0, q._i_seed0 = int(idx["L"]), int(idx["L2"]), int(idx["knob0"]), int(idx["seed0"])
    q.dropped = dict(d.get("dropped") or {})
    if q.beta.shape != (len(q.columns),) or q.cov.shape != (len(q.columns), len(q.columns)):
        raise ValueError("params: beta / cov do not match the columns")
    q.fitted = True
    return q


def _steptime_params(st: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for phase, f in sorted((getattr(st, "_fits", None) or {}).items()):
        out[phase] = {"columns": list(f.columns), "coef": f.coef, "n": int(f.n), "rmse_log": f.rmse_log,
                      "versions": list(f.versions)}
    return _r(out)


def _offset_params(off: Any) -> Dict[str, Any]:
    return _r({"mean": getattr(off, "mean", None), "sd": getattr(off, "sd", None), "n": getattr(off, "n", None),
               "sd_n": getattr(off, "sd_n", None), "predictive_sd": getattr(off, "predictive_sd", None),
               "offsets": list(getattr(off, "offsets", []) or []), "source": getattr(off, "source", "")})


def export_params(steptime: Any, quality: QualityModel, offset: Any,
                  data_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "format": PARAMS_FORMAT,
        "ffsim_version": __version__,
        "note": ("Fitted route-1 surrogate as JSON. Rebuild the quality model with "
                 "ffsim.params.quality_from_params(load_params(path)['quality']). Regenerate with "
                 "`python -m ffsim contrib refit`. Never load a pickle from someone else; this file is plain data."),
        "data": _r(dict(data_meta or {})),
        "quality": quality_to_params(quality),
        "offset": _offset_params(offset),
        "steptime": _steptime_params(steptime),
    }


def write_params(params: Dict[str, Any], path: Union[str, Path]) -> bool:
    """Write ``params`` as JSON (sorted keys, one space indent). Returns True when the file content changed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(params, indent=1, sort_keys=True, allow_nan=False) + "\n"
    old = p.read_text(encoding="utf-8") if p.exists() else None
    if old == text:
        return False
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return True


def load_params(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("format") != PARAMS_FORMAT:
        raise ValueError(f"{path}: not an {PARAMS_FORMAT} file (format {d.get('format')!r})")
    return d
