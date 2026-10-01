"""ffsim.offset: Part 3 of the simulator, the public-shard rehearsal -> official leaderboard offset.

``official_bpb = bpb_2m(rehearsal on chip C) + offset``

What the offset is
------------------
Mostly TEXT DIFFICULTY, i.e. a constant: the first 2M public tokens are easier than typical text (the same model
scores about +0.0070 worse on 20M public tokens), and the official eval shard behaves like typical text.
On top of that there is upload noise (a different eval shard, the official machine's numerics, the exact
weights uploaded) with sd ~0.0004. Measured K51 -> K59 (CONTRACT.md): +0.0061, +0.0063, +0.0072, +0.0069,
+0.0067, +0.0062; mean +0.0066, sample sd 0.00044. K60 (29 Sep, after the contract was written) added +0.0064
(C41 0.95910 -> 0.9655): seven chip-C points, mean +0.00653, sd 0.00042.

Model
-----
``offset ~ N(mean, sd)`` with ``mean`` = the average of the observed offsets and ``sd`` their sample sd (ddof=1).
The offset is assumed independent of the recipe (it moved by only 0.0011 across nine quite different submissions)
and of the chip that rehearsed it (all six rehearsals ran on chip C; a chip-D rehearsal would need its own table).
Both moments rest on six points, so:

- ``mean_se`` = sd / sqrt(n) is the uncertainty of the constant (0.00018) and ``predictive_sd`` =
  sqrt(sd^2 + mean_se^2) = sd * sqrt(1 + 1/n) is the sd of ONE FUTURE offset (0.00047 from the contract's rounded
  six, 0.00049 from the table's sd 0.00045). ``sample`` draws with ``predictive_sd`` by default, so a simulated
  official score carries the mean's uncertainty too; ``with_mean_uncertainty=False`` draws upload noise only (the
  spec's bare "constant plus upload noise" model). In a paired comparison under common random numbers the offset
  cancels either way.
- ``sd_ci95`` is the chi-square 95% interval for the sd itself: for a 6-point sd of 0.00045 it is
  0.00028 .. 0.00111, i.e. the sd is known to a factor of ~2. A ``p_beat_best`` near 0.5 should not be read to
  the second decimal.

Scope (read before trusting an official prediction)
----------------------------------------------------
The offset is calibrated on K5x-shaped recipes (M4 -> M12 code, TT ~1790, 2036 -> 2360 steps) rehearsed on chip C.
It conflates shard difficulty with whatever the official host does to the STEP COUNT: the organiser's K44 rerun on
a slower host cost +0.0014 for ~2.3% fewer steps, and that host effect is inside every calibration point. Host
sensitivity is mechanism-dependent (chip D, runtime 2.33.10, is ~7-12% slower at k4 in full runs but only ~1% in
60-step screens), so the offset is recipe-independent only for mechanisms whose step time reacts to the organiser
runtime the way K5x does. A mechanism with a different runtime sensitivity needs its own upload before its
simulated official score is trusted; its rehearsal bpb_2m is unaffected.

Defaults: with no table (or fewer than 2 usable rows) the CONTRACT.md numbers are used; with 2 rows the mean is
the data's and the sd stays the contract's (a 2-point sd is not a measurement).

The uploads table (``research/sim-data/official-uploads.csv``, ``load_official_uploads``) carries every upload
since F2. Rows are CALIBRATION points only when they were scored and are not flagged "NOT a calibration point"
(K54b: rejected for size; K44_rerun and F6_dup: organiser-side reruns). The contract's offset is the chip-C era
(K51 -> K59 plus K60, ``CONTRACT_SUBMISSIONS``); the loader returns that subset by default and the whole calibration set
with ``submissions=None`` (older uploads were rehearsed on chips A/B with earlier code).
"""
from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

__all__ = ["CONTRACT_OFFSETS", "CONTRACT_SUBMISSIONS", "OffsetModel", "load_official_uploads", "offsets_from_uploads",
           "sd_confidence_interval"]

# Rehearsal -> official offsets, K51 -> K59 (CONTRACT.md); the fallback when no uploads table is usable.
CONTRACT_OFFSETS = (0.0061, 0.0063, 0.0072, 0.0069, 0.0067, 0.0062)
# The scored chip-C uploads the fitted subset is built from: the six behind CONTRACT_OFFSETS (K54b itself was
# rejected for size; K54b_min is its scored twin) plus K60 (29 Sep 19:50 UTC: C41 0.95910 -> 0.9655, +0.0064,
# the seventh chip-C point; CONTRACT.md line 35 still lists six).
CONTRACT_SUBMISSIONS = ("K51", "K53", "K54b_min", "K56", "K57", "K59", "K60")
NON_CALIBRATION_MARK = "not a calibration point"
_NOTE_KEYS = ("recipe_note", "note", "notes", "comment")
SCOPE_NOTE = ("calibrated on K5x-shaped recipes rehearsed on chip C (M4 -> M12, ~2000-2360 steps); the offset "
              "absorbs whatever the official host does to the step count, so it may not transfer to a mechanism "
              "whose step time reacts differently to the organiser runtime (chip D: ~7-12% slower at k4 in full "
              "runs, ~1% in screens)")


def _is_calibration_row(item: Any) -> bool:
    """False for a dict row that says so (``calibration`` False) or whose note carries NON_CALIBRATION_MARK."""
    if not isinstance(item, dict):
        return True
    cal = item.get("calibration")
    if cal is not None:
        return bool(cal)
    for key in _NOTE_KEYS:
        v = item.get(key)
        if isinstance(v, str) and NON_CALIBRATION_MARK in v.lower():
            return False
    return True


def load_official_uploads(path: Union[str, "os.PathLike[str]"],
                          submissions: Optional[Iterable[str]] = CONTRACT_SUBMISSIONS) -> List[Dict[str, Any]]:
    """Rows of official-uploads.csv as dicts with parsed numbers and a ``calibration`` flag.

    Each row: submission, upload_date_utc, rehearsal_run, rehearsal_bpb, rehearsal_steps, official_bpb, offset
    (recomputed as official - rehearsal when both are present, else the quoted value), offset_quoted, code_version,
    recipe_note, calibration. ``submissions``: keep these submission names (default: the contract's K51 -> K59
    set); None keeps every calibration row. Rows that are not calibration points are dropped either way.
    """
    wanted = None if submissions is None else {s.strip() for s in submissions}
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: Dict[str, Any] = {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            for key in ("rehearsal_bpb", "official_bpb", "rehearsal_steps"):
                row[key] = _get(row, key)
            row["offset_quoted"] = _get(row, "offset")
            if row["official_bpb"] is not None and row["rehearsal_bpb"] is not None:
                row["offset"] = row["official_bpb"] - row["rehearsal_bpb"]
            else:
                row["offset"] = row["offset_quoted"] if row["official_bpb"] is not None else None
            row["calibration"] = row["official_bpb"] is not None and _is_calibration_row(raw)
            if not row["calibration"]:
                continue
            if wanted is not None and str(row.get("submission", "")) not in wanted:
                continue
            out.append(row)
    return out


def _get(d: Any, *names: str) -> Optional[float]:
    for name in names:
        v = d.get(name) if isinstance(d, dict) else getattr(d, name, None)
        if v is not None:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                return f
    return None


def offsets_from_uploads(official_uploads: Optional[Iterable[Any]]) -> List[float]:
    """Extract ``official - rehearsal`` from an uploads table.

    Each item may be: a dict with ``official``/``official_bpb`` and ``rehearsal``/``bpb_2m``/``rehearsal_bpb``
    (or a ready ``offset``), a RunRecord with ``official_bpb`` and ``bpb_2m``, or a bare number (an offset).
    Items without both numbers are skipped, as are dict rows flagged as non-calibration points (``calibration``
    False, or a note containing "NOT a calibration point"). official - rehearsal is preferred over a quoted offset.
    """
    out: List[float] = []
    for item in official_uploads or ():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if math.isfinite(float(item)):
                out.append(float(item))
            continue
        if not _is_calibration_row(item):
            continue
        official = _get(item, "official", "official_bpb")
        rehearsal = _get(item, "rehearsal", "rehearsal_bpb", "bpb_2m")
        if official is not None and rehearsal is not None:
            off: Optional[float] = official - rehearsal
        else:
            off = _get(item, "offset")
        if off is None:
            continue
        out.append(off)
    return out


# --- chi-square interval for a sample sd (stdlib only: no scipy) --------------------------------------------
def _gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x): series for x < a + 1, continued fraction otherwise."""
    if x <= 0.0:
        return 0.0
    lg = math.lgamma(a)
    if x < a + 1.0:
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - lg)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - lg) * h


def _chi2_cdf(x: float, df: float) -> float:
    return _gammainc(df / 2.0, x / 2.0)


def _chi2_ppf(p: float, df: float) -> float:
    """Chi-square quantile by bisection on the CDF (0 < p < 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = 0.0, max(10.0, 4.0 * df)
    while _chi2_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sd_confidence_interval(sd: float, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Chi-square confidence interval for a sample sd (ddof=1) estimated from ``n`` normal draws.

    ``sd * sqrt((n-1) / chi2_{1-a/2, n-1}) .. sd * sqrt((n-1) / chi2_{a/2, n-1})``. For n = 6, sd 0.00045:
    0.00028 .. 0.00111. Returns (nan, nan) when n < 2 (no sd to bound).
    """
    if n < 2:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    df = n - 1
    lo = sd * math.sqrt(df / _chi2_ppf(1.0 - alpha / 2.0, df))
    hi = sd * math.sqrt(df / _chi2_ppf(alpha / 2.0, df))
    return (float(lo), float(hi))


class OffsetModel:
    """offset = official - rehearsal ~ N(mean, sd). ``fit(uploads)`` or use the CONTRACT.md defaults.

    ``n`` is the number of points behind the mean, ``sd_n`` the number behind the sd (they differ only when the
    sd fell back to the contract's six because fewer than three rows were usable).
    """

    def __init__(self, mean: Optional[float] = None, sd: Optional[float] = None, n: Optional[int] = None):
        contract = np.array(CONTRACT_OFFSETS)
        self.contract_mean = float(contract.mean())
        self.contract_sd = float(contract.std(ddof=1))
        self.offsets: List[float] = list(CONTRACT_OFFSETS)
        self.mean = self.contract_mean if mean is None else float(mean)
        self.sd = self.contract_sd if sd is None else float(sd)
        self.n = len(CONTRACT_OFFSETS) if n is None else int(n)
        self.sd_n = len(CONTRACT_OFFSETS) if sd is None else self.n
        self.source = "contract"
        self.note = ("offset = official - chip-C rehearsal: mostly text difficulty (a constant, the first 2M public "
                     "tokens are easier than typical text) plus upload noise sd ~0.0004 (a 6-point sd: 95% CI about "
                     "0.0003..0.0011; sample() draws with predictive_sd, which adds the mean's se); " + SCOPE_NOTE)

    def fit(self, official_uploads: Optional[Union[Iterable[Any], str, "os.PathLike[str]"]],
            contract_subset: bool = True) -> "OffsetModel":
        """Mean/sd from the uploads table (see ``offsets_from_uploads``); falls back to the contract when thin.

        A path is read with ``load_official_uploads``. With ``contract_subset`` (the default) rows that name
        their ``submission`` are restricted to the contract's K51 -> K59 set whenever at least three of those
        are present (the chip-C era the simulator predicts for); older uploads were rehearsed on chips A/B with
        earlier code and widen the sd to ~0.00065. ``contract_subset=False`` uses every calibration row given.
        """
        if isinstance(official_uploads, (str, os.PathLike)):
            official_uploads = load_official_uploads(official_uploads, submissions=None)
        rows = list(official_uploads or ())
        subset_note = ""
        if contract_subset:
            named = [r for r in rows if isinstance(r, dict) and str(r.get("submission", "")).strip() in CONTRACT_SUBMISSIONS]
            if len(named) >= 3:
                rows, subset_note = named, " (K51 -> K60 subset)"
        offs = offsets_from_uploads(rows)
        if len(offs) == 0:
            self.offsets = list(CONTRACT_OFFSETS)
            self.mean, self.sd, self.n = self.contract_mean, self.contract_sd, len(CONTRACT_OFFSETS)
            self.sd_n = len(CONTRACT_OFFSETS)
            self.source = "contract (no usable uploads)"
            return self
        arr = np.array(offs, dtype=float)
        self.offsets = [float(v) for v in arr]
        self.n = int(arr.size)
        self.mean = float(arr.mean())
        if arr.size >= 3:
            self.sd = max(float(arr.std(ddof=1)), 1e-4)
            self.sd_n = self.n
            self.source = "uploads" + subset_note
        else:
            self.sd = self.contract_sd
            self.sd_n = len(CONTRACT_OFFSETS)
            self.source = "uploads (mean) + contract (sd: fewer than 3 rows)"
        return self

    @property
    def mean_se(self) -> float:
        """Standard error of the constant part (sd / sqrt(n))."""
        return self.sd / math.sqrt(max(self.n, 1))

    @property
    def predictive_sd(self) -> float:
        """sd of one future offset including the uncertainty of the mean: sqrt(sd^2 + se^2)."""
        return math.sqrt(self.sd ** 2 + self.mean_se ** 2)

    @property
    def sd_ci95(self) -> Tuple[float, float]:
        """Chi-square 95% interval for the sd (``sd_n`` points): (0.00028, 0.00111) for the K51 -> K59 six."""
        return sd_confidence_interval(self.sd, self.sd_n)

    def sample(self, rng: np.random.Generator, n: int, with_mean_uncertainty: bool = True) -> np.ndarray:
        """``n`` offsets ~ N(mean, predictive_sd); ``with_mean_uncertainty=False`` uses the upload-noise ``sd`` only.

        The mean rests on ``self.n`` points, so a future official score is off by the mean's error as well as by
        upload noise; the default carries both (0.00049 instead of 0.00045 for the K51 -> K59 six).
        """
        sd = self.predictive_sd if with_mean_uncertainty else self.sd
        return self.mean + sd * rng.standard_normal(int(n))

    def summary(self) -> Dict[str, Any]:
        lo, hi = self.sd_ci95
        return {"mean": self.mean, "sd": self.sd, "sd_n": self.sd_n, "sd_ci95": [lo, hi], "n": self.n,
                "mean_se": self.mean_se, "predictive_sd": self.predictive_sd, "offsets": list(self.offsets),
                "source": self.source, "note": self.note}
