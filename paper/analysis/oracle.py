"""Exact-oracle statistics (paper Section 5 and Appendix A) from results/official-scores.csv.

offset = official - rehearsal, computed from the CSV's unrounded rehearsal column and its official column (the
offset column of the CSV is rounded and is not read; printed 4-decimal offsets are rounded half-up from the exact
decimal difference, the CSV's rule). Writes tables/offset_periods.tex, offset_chips.tex, offset_holdout.tex,
oracle_sensitivity.tex, oracle_coverage.tex, uploads.tex, steps_decomp.tex and figures/offset_prospective.pdf, and
registers the numbers it prints.

Units: absolute scores in bpb; offsets, errors and differences in tables in 10^-3 bpb.

Revision 1 (review response): offsets per chip and a "text + chip" reading of the offset; a sign test on the frozen
calibration's final-day errors; every oracle statistic with the chip-E rehearsal of K70a as used (normalised to
chip-C speed), raw, and excluded; uploads labelled by how their salt was chosen (directly by a lottery, or inherited
from one); interval coverage at 50/80/95% for two burn-in lengths; the oracle's resolution for a paired difference;
a within-chip decomposition of the step-count increase.
"""
from __future__ import annotations

import collections
import math
import re
from typing import Dict, List, Optional

from common import (FIGURES, TABLES, Numbers, binom_tail, chi2_quantile, intc, load_scores, mean, milli, num,
                    offset_str, official_str, sd, t_quantile, table, tex_escape)

CHIP_C = ["K51", "K53", "K54b_min", "K56", "K57", "K59", "K60"]          # the simulator's offset model (docs/EXACT-ORACLE.md)
FINAL_DAY = ["K63", "K70a", "K73s4", "K73s6", "K77a", "K82s4", "K82s7"]  # rehearsed on chips C/E/G/H on 30 Sep - 1 Oct
# How each final-day upload's shuffle salt was chosen (docs/FINDINGS.md s3, docs/EXACT-ORACLE.md "Lotteries"):
SALT_DIRECT = {"K73s4", "K73s6", "K82s7"}      # the salt was picked by its rehearsal score in a salt lottery
SALT_INHERITED = {"K77a", "K82s4"}             # carry salt 4, the winner of the K73 lottery
NORMALISED = {"K70a"}                          # chip-E rehearsal normalised to chip-C speed (CSV note)
LEVELS = (0.50, 0.80, 0.95)


def _ols(x: List[float], y: List[float]) -> Dict[str, float]:
    n = len(x)
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    icpt = my - slope * mx
    resid = [b - (icpt + slope * a) for a, b in zip(x, y)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx)
    t = t_quantile(0.975, n - 2)
    return {"slope": slope, "se": se, "lo": slope - t * se, "hi": slope + t * se, "r": sxy / math.sqrt(sxx * syy),
            "n": n}


def _halfup(r: dict, nd: int = 4) -> float:
    from decimal import ROUND_HALF_UP, Decimal
    d = Decimal(repr(r["official"])) - Decimal(repr(r["rehearsal"]))
    return float(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def _mark(name: str) -> str:
    m = ""
    if name in SALT_DIRECT:
        m += "\\textsuperscript{s}"
    if name in SALT_INHERITED:
        m += "\\textsuperscript{i}"
    if name in NORMALISED:
        m += "\\textsuperscript{n}"
    return m


def _expanding(since: List[dict], start: int) -> List[dict]:
    """Predict each upload's offset from the mean of all earlier uploads (since K50), from the ``start``-th on."""
    out = []
    for i in range(start, len(since)):
        prev = [r["offset"] for r in since[:i]]
        m, s = mean(prev), sd(prev)
        psd = s * math.sqrt(1.0 + 1.0 / len(prev))
        e = since[i]["offset"] - m
        row = {"name": since[i]["name"], "pred": m, "psd": psd, "err": e, "obs": since[i]["offset"],
               "t95": t_quantile(0.975, len(prev) - 1)}
        for lv in LEVELS:
            row[lv] = abs(e) <= t_quantile(0.5 + lv / 2, len(prev) - 1) * psd
        out.append(row)
    return out


def _variant(since: List[dict]) -> Dict[str, object]:
    """Every oracle statistic for one version of the since-K50 data."""
    by = {r["name"]: r for r in since}
    offs = [r["offset"] for r in since]
    chipc = [by[n]["offset"] for n in CHIP_C if n in by]
    frozen = mean(chipc)
    fd = [n for n in FINAL_DAY if n in by]
    errs = {n: by[n]["offset"] - frozen for n in fd}
    seq3 = _expanding(since, 3)
    return {"n": len(offs), "mean": mean(offs), "sd": sd(offs), "frozen": frozen, "errs": errs,
            "f_mae": mean(abs(e) for e in errs.values()), "f_bias": mean(errs.values()),
            "seq3": seq3, "seq5": _expanding(since, 5)}


def run(N: Numbers) -> Dict[str, object]:
    rows = load_scores()
    scored = [r for r in rows if r["official"] is not None]
    cal = [r for r in scored if r["rehearsal"] is not None]
    for r in cal:
        r["offset"] = r["official"] - r["rehearsal"]
    by = {r["name"]: r for r in cal}
    k50 = next(i for i, r in enumerate(cal) if r["name"] == "K50")
    since = cal[k50:]
    early = cal[:k50]
    src = "results/official-scores.csv"

    # ---------------------------------------------------------------- headline facts
    first = scored[0]
    best = min(scored, key=lambda r: r["official"])
    k44 = next(r for r in scored if r["name"] == "K44")
    k82s7 = next(r for r in scored if r["name"] == "K82s7")
    N.add("n-scored", str(len(scored)), src)
    N.add("n-rehearsed", str(len(cal)), src)
    N.add("n-official-tests", str(sum(1 for r in scored if r["rehearsal"] is None)), src)
    N.add("first-official", num(first["official"], 4), src)
    N.add("first-name", first["name"], src)
    N.add("best-official", num(best["official"], 5), src)
    N.add("best-official-long", num(best["official"], 7), src)
    N.add("best-name", best["name"], src)
    N.add("best-rehearsal", num(best["rehearsal"], 6), src)
    N.add("best-steps", intc(best["steps"]), src)
    N.add("kfortyfour-official", num(k44["official"], 4), src)
    N.add("salt7-official", num(k82s7["official"], 5), src)
    N.add("salt7-official-long", num(k82s7["official"], 7), src)
    N.add("salt7-rehearsal", num(k82s7["rehearsal"], 6), src)
    N.add("salt7-steps", intc(k82s7["steps"]), src)
    N.add("descent-total", num(first["official"] - best["official"], 4), src)
    N.add("descent-from-kfortyfour", num(k44["official"] - best["official"], 4), src)
    k24 = next(r for r in cal if r["name"] == "K24")
    N.add("steps-ktwentyfour", intc(k24["steps"]), src)
    N.add("steps-ratio-best-ktwentyfour", num(best["steps"] / k24["steps"], 2), src)

    # ---------------------------------------------------------------- offset periods
    periods = [
        ("chip-C calibration uploads, K51--K60", [by[n] for n in CHIP_C], "chipc"),
        ("final day, chips C/E/G/H, K63--K82s7", [by[n] for n in FINAL_DAY], "final"),
        ("since K50 (all chips)", since, "since"),
        ("early, chips A/B, F2--K46", early, "early"),
        ("every rehearsed upload", cal, "all"),
    ]
    body = []
    for label, rs, key in periods:
        offs = [r["offset"] for r in rs]
        m, s = mean(offs), sd(offs)
        lo_r = min(rs, key=lambda r: r["offset"])
        hi_r = max(rs, key=lambda r: r["offset"])
        lo_s, hi_s = offset_str(lo_r["official"], lo_r["rehearsal"]), offset_str(hi_r["official"], hi_r["rehearsal"])
        N.add(f"off-n-{key}", str(len(offs)), src)
        N.add(f"off-mean-{key}", num(m, 5, sign=True), src)
        N.add(f"off-mean-{key}-short", num(m, 4, sign=True), src)
        N.add(f"off-sd-{key}", num(s, 5), src)
        N.add(f"off-sd-{key}-short", num(s, 4), src)
        N.add(f"off-min-{key}", lo_s, src + " (half-up)")
        N.add(f"off-max-{key}", hi_s, src + " (half-up)")
        tq = t_quantile(0.975, len(offs) - 1)
        half = tq * s / math.sqrt(len(offs))
        N.add(f"off-ci-{key}", num(half, 5), src)
        lo_v, hi_v = _halfup(lo_r), _halfup(hi_r)
        body.append(f"{label} & {len(offs)} & {milli(m, 2, sign=True)} & {milli(s, 2)} & "
                    f"{milli(lo_v, 1, sign=True)} to {milli(hi_v, 1, sign=True)} & $\\pm${milli(half, 2)} \\\\")
    table(TABLES / "offset_periods.tex", body, "@{}p{5.6cm}rrrrr@{}",
          "Uploads & $n$ & Mean & SD & Range & 95\\% CI",
          "offset = official - rehearsal by period, 1e-3 bpb (oracle.py)")

    # ---------------------------------------------------------------- per-chip offsets ("text + chip")
    chips = collections.OrderedDict()
    for r in since:
        chips.setdefault(r["chip"], []).append(r)
    frozen = mean([by[n]["offset"] for n in CHIP_C])
    body = []
    for c, rs in chips.items():
        offs = [r["offset"] for r in rs]
        fd = [r["offset"] - frozen for r in rs if r["name"] in FINAL_DAY]
        names = ", ".join(tex_escape(r["name"].replace("_min", "")) + _mark(r["name"]) for r in rs)
        body.append(f"{c} & {names} & {len(offs)} & {milli(mean(offs), 2, sign=True)} & "
                    f"{milli(sd(offs), 2) if len(offs) > 1 else '--'} & "
                    f"{milli(mean(fd), 2, sign=True) if fd else '--'} \\\\")
        N.add(f"chip-off-{c.lower()}", num(mean(offs), 5, sign=True), src)
        N.add(f"chip-off-n-{c.lower()}", str(len(offs)), src)
        if fd:
            N.add(f"chip-ferr-{c.lower()}", num(mean(fd), 5, sign=True), src + ": frozen chip-C error, mean by chip")
    table(TABLES / "offset_chips.tex", body, "@{}lp{5.4cm}rrrr@{}",
          "Chip & Uploads since K50 & $n$ & Mean & SD & Frozen-fit error",
          "offset by rehearsal chip since K50, 1e-3 bpb (oracle.py)")
    # within-chip pooled SD (chips with >= 2 uploads since K50)
    num_ss = sum((len(rs) - 1) * sd([r["offset"] for r in rs]) ** 2 for rs in chips.values() if len(rs) > 1)
    den = sum(len(rs) - 1 for rs in chips.values() if len(rs) > 1)
    s_within = math.sqrt(num_ss / den)
    N.add("off-sd-within", num(s_within, 5), src + ": pooled within-chip SD since K50")
    # 95% chi-square interval for the pooled SD (den degrees of freedom)
    N.add("off-sd-within-lo", num(s_within * math.sqrt(den / chi2_quantile(0.975, den)), 5),
          src + f": chi-square 95% CI of the pooled within-chip SD, {den} df")
    N.add("off-sd-within-hi", num(s_within * math.sqrt(den / chi2_quantile(0.025, den)), 5), src)
    # leave-one-out: pooled mean vs chip mean (text + chip), on the uploads that share a chip with another upload
    pooled_err, chip_err = [], []
    for r in since:
        others = [o for o in since if o is not r]
        same = [o["offset"] for o in others if o["chip"] == r["chip"]]
        if not same:
            continue
        pooled_err.append(abs(r["offset"] - mean(o["offset"] for o in others)))
        chip_err.append(abs(r["offset"] - mean(same)))
    N.add("loo-n", str(len(pooled_err)), src)
    N.add("loo-pooled-mae", num(mean(pooled_err), 5), src + ": leave-one-out, pooled mean")
    N.add("loo-chip-mae", num(mean(chip_err), 5), src + ": leave-one-out, same-chip mean")
    # resolution of a paired difference between two uploads
    s_since = sd([r["offset"] for r in since])
    N.add("diff-sd", num(math.sqrt(2) * s_since, 5), "sqrt(2) x since-K50 offset SD")
    N.add("diff-95", num(1.96 * math.sqrt(2) * s_since, 4), "1.96 sqrt(2) x since-K50 offset SD")
    N.add("diff-sd-within", num(math.sqrt(2) * s_within, 5), "sqrt(2) x within-chip offset SD")
    N.add("pi-half", num(t_quantile(0.975, len(since) - 1) * s_since * math.sqrt(1 + 1 / len(since)), 4),
          "95% prediction half-width at n=15")

    # ---------------------------------------------------------------- winner's curse heuristic for salt lotteries
    salt_sd = 0.00035
    sd_final = sd([by[n]["offset"] for n in FINAL_DAY])
    N.add("salt-sd-mid", num(salt_sd, 5), "docs/FINDINGS.md s3: salt SD 0.0003-0.0004, midpoint")
    lam = lambda s_eta: salt_sd ** 2 / (salt_sd ** 2 + s_eta ** 2)  # noqa: E731
    N.add("salt-shrink", num(lam(sd_final), 2), "reliability ratio (oracle.py)")
    # sensitivity: the final-day SD includes chip-to-chip variation; the within-chip SD is the other natural choice
    N.add("salt-shrink-within", num(lam(s_within), 2), "reliability ratio at the within-chip offset SD")
    N.add("salt-shrink-lo", num(min(lam(sd_final), lam(s_within), lam(s_since)), 2), "range over s_eta choices")
    N.add("salt-shrink-hi", num(max(lam(sd_final), lam(s_within), lam(s_since), lam(0.0002)), 2),
          "range over s_eta choices incl. 0.0002")

    # ---------------------------------------------------------------- frozen chip-C fit -> final day
    N.add("frozen-offset", num(frozen, 5, sign=True), src)
    errs = []
    body = []
    for n in FINAL_DAY:
        r = by[n]
        e = r["offset"] - frozen
        errs.append(e)
        body.append(f"{tex_escape(n)}{_mark(n)} & {r['chip']} & {num(r['rehearsal'], 6)} & {num(frozen + r['rehearsal'], 5)} & "
                    f"{num(r['official'], 4)} & {milli(e, 2, sign=True)} \\\\")
    mae = mean(abs(e) for e in errs)
    bias = mean(errs)
    for n, e in zip(FINAL_DAY, errs):
        if n in ("K82s4", "K82s7"):
            N.add(f"ferr-{n.lower()}", num(e, 5, sign=True), src + ": frozen chip-C error")
    N.add("t-two", num(t_quantile(0.975, 2), 1), "Student t quantile, 2 df")
    N.add("holdout-n", str(len(errs)), src)
    N.add("holdout-mae", num(mae, 5), src)
    N.add("holdout-bias", num(bias, 5, sign=True), src)
    N.add("holdout-maxerr", num(max(abs(e) for e in errs), 5), src)
    # percentile bootstrap of the frozen MAE over the final-day uploads (fixed RNG)
    import numpy as np
    rng = np.random.default_rng(0)
    ae = np.abs(np.asarray(errs))
    boot = ae[rng.integers(0, len(ae), (4000, len(ae)))].mean(axis=1)
    N.add("holdout-mae-lo", num(float(np.percentile(boot, 2.5)), 5),
          src + ": percentile bootstrap over the final-day uploads, 4000 draws, rng 0")
    N.add("holdout-mae-hi", num(float(np.percentile(boot, 97.5)), 5), src + ": percentile bootstrap")
    npos = sum(1 for e in errs if e > 0)
    N.add("holdout-npos", str(npos), src)
    N.add("holdout-signp", num(binom_tail(npos, len(errs)), 2), "one-sided sign test, Binomial(n, 1/2)")
    groups = {"direct": [n for n in FINAL_DAY if n in SALT_DIRECT],
              "inherited": [n for n in FINAL_DAY if n in SALT_INHERITED],
              "none": [n for n in FINAL_DAY if n not in SALT_DIRECT | SALT_INHERITED]}
    for g, names in groups.items():
        ge = [by[n]["offset"] - frozen for n in names]
        N.add(f"holdout-mae-{g}", num(mean(abs(e) for e in ge), 5), src)
        N.add(f"holdout-n-{g}", str(len(ge)), src)
    body.append("\\midrule")
    body.append(f"MAE / bias & & & & & {milli(mae, 2)} / {milli(bias, 2, sign=True)} \\\\")
    table(TABLES / "offset_holdout.tex", body, "@{}lcrrrr@{}",
          "Upload & Chip & Rehearsal & Predicted & Official & Error ($10^{-3}$)",
          "frozen chip-C offset applied to the final-day uploads (oracle.py)")

    # ---------------------------------------------------------------- pseudo-prospective expanding window
    seq = _expanding(since, 3)
    N.add("seq-n", str(len(seq)), src)
    N.add("seq-mae", num(mean(abs(s["err"]) for s in seq), 5), src)
    N.add("seq-bias", num(mean(s["err"] for s in seq), 5, sign=True), src)
    N.add("seq-inside", str(sum(s[0.95] for s in seq)), src)
    N.add("seq-maxerr", num(max(abs(s["err"]) for s in seq), 5), src)
    N.add("seq-first", seq[0]["name"].replace("_min", ""), src)
    N.add("seq-expected", num(0.95 * len(seq), 1), "0.95 x n")
    N.add("nooffset-mae", num(mean(abs(r["offset"]) for r in since), 4), src)

    # ---------------------------------------------------------------- K70a sensitivity (as used / raw / excluded)
    note = by["K70a"]["note"]
    m = re.search(r"raw chip E ([0-9.]+)", note)
    k70a_raw = float(m.group(1))
    N.add("k70a-raw", num(k70a_raw, 6), src + " (note column)")
    N.add("k70a-norm", num(by["K70a"]["rehearsal"], 6), src)
    N.add("k70a-adj", num(k70a_raw - by["K70a"]["rehearsal"], 5), src + ": raw - normalised")
    N.add("k70a-adj-pct", num(100 * (k70a_raw - by["K70a"]["rehearsal"]) / 0.057, 1) + "\\%",
          "adjustment expressed in % steps at 0.00057 per 1%")
    N.add("k70a-raw-offset", num(by["K70a"]["official"] - k70a_raw, 4, sign=True), src)
    variants = []
    raw_since = [dict(r, offset=(r["official"] - k70a_raw) if r["name"] == "K70a" else r["offset"]) for r in since]
    for label, data, key in (("K70a normalised to chip-C speed (as used)", since, "used"),
                             ("K70a raw chip-E rehearsal", raw_since, "raw"),
                             ("K70a excluded", [r for r in since if r["name"] != "K70a"], "excl")):
        v = _variant(data)
        variants.append((label, v, key))
        N.add(f"sens-{key}-sd", num(v["sd"], 5), src)
        N.add(f"sens-{key}-mae", num(v["f_mae"], 5), src)
        N.add(f"sens-{key}-inside", f"{sum(s[0.95] for s in v['seq3'])}/{len(v['seq3'])}", src)
        if "K70a" in v["errs"]:
            N.add(f"sens-{key}-k70a-err", milli(v["errs"]["K70a"], 2, sign=True), src)
    body = []
    for label, v, key in variants:
        body.append(f"{label} & {v['n']} & {milli(v['mean'], 2, sign=True)} & {milli(v['sd'], 2)} & "
                    f"{milli(v['f_mae'], 2)} & {milli(v['f_bias'], 2, sign=True)} & "
                    f"{sum(s[0.95] for s in v['seq3'])}/{len(v['seq3'])} & {milli(mean(abs(s['err']) for s in v['seq3']), 2)} \\\\")
    table(TABLES / "oracle_sensitivity.tex", body, "@{}p{4.9cm}rrrrrrr@{}",
          "Data since K50 & $n$ & Mean & SD & Frozen MAE & Frozen bias & In 95\\% PI & Seq.\\ MAE",
          "oracle statistics with K70a as used, raw and excluded, 1e-3 bpb (oracle.py)", size="\\footnotesize")

    # ---------------------------------------------------------------- interval coverage at 50/80/95%
    body = []
    for lv in LEVELS:
        cells = []
        for start in (3, 5):
            s_ = _expanding(since, start)
            k = sum(x[lv] for x in s_)
            cells.append(f"{k}/{len(s_)} ({num(lv * len(s_), 1)})")
            N.add(f"cov-{int(lv * 100)}-b{start}", f"{k}/{len(s_)}", src)
        body.append(f"{int(lv * 100)}\\% & " + " & ".join(cells) + " \\\\")
    table(TABLES / "oracle_coverage.tex", body, "@{}lcc@{}",
          "Nominal level & Burn-in 3 uploads & Burn-in 5 uploads",
          "observed (expected) count of uploads inside the expanding-window interval (oracle.py)")

    # ---------------------------------------------------------------- is the offset constant?
    reg = _ols([r["rehearsal"] for r in since], [r["official"] for r in since])
    N.add("reg-slope", num(reg["slope"], 3), src)
    N.add("reg-lo", num(reg["lo"], 3), src)
    N.add("reg-hi", num(reg["hi"], 3), src)
    N.add("reg-r", num(reg["r"], 4), src)
    trend = _ols(list(range(len(since))), [r["offset"] for r in since])
    N.add("trend-slope", num(trend["slope"] * 1e5, 2, sign=True), src)     # in 1e-5 bpb per upload
    N.add("trend-lo", num(trend["lo"] * 1e5, 2, sign=True), src)
    N.add("trend-hi", num(trend["hi"] * 1e5, 2, sign=True), src)
    span = max(r["rehearsal"] for r in since) - min(r["rehearsal"] for r in since)
    N.add("since-rehearsal-span", num(span, 4), src)

    # ---------------------------------------------------------------- where the steps came from (within chip)
    steps_decomp(N, scored)

    # ---------------------------------------------------------------- appendix ledger
    body = []
    for r in scored:
        off = "" if r["rehearsal"] is None else offset_str(r["official"], r["rehearsal"])
        reh = "" if r["rehearsal"] is None else num(r["rehearsal"], 5)
        steps = "" if r["steps"] is None else intc(r["steps"])
        star = "$\\bullet$" if r["best"] else ""
        date = r["date"].replace("2026-", "").replace("~", "$\\sim$")
        body.append(f"{tex_escape(r['name'])}{_mark(r['name'])} & {date} & {r['chip']} & {reh} & {steps} & "
                    f"{num(r['official'], 4)} & {off} & {star} & {tex_escape(_plain(r['note']))} \\\\")
    table(TABLES / "uploads.tex", body, "@{}llcrrrrcp{8.4cm}@{}",
          "Upload & Date & Chip & Rehearsal & Steps & Official & Offset & Best & What changed",
          "every scored upload, results/official-scores.csv (oracle.py)", size="\\scriptsize", tabcolsep="3pt")

    _figure(since, seq, frozen)
    return {"since": since, "seq": seq, "frozen": frozen}


# Internal shorthand in the ledger's "what changed" column, spelled out for readers (results/official-scores.csv is
# not edited; this only changes how the appendix prints it).
PLAIN = [
    ("0-dim lambdas", "scalar parameters as 0-dim tensors"),
    ("static loss", "loss computed in a static graph"),
    ("async sync", "asynchronous host synchronisation"),
    ("(FF_KV_SDPA)", ""),
    ("TT 1780", "time target 1,780 s"),
    ("TT 1795", "time target 1,795 s"),
    ("XU 48", "execution-queue depth 48"),
    ("c_proj LR 0.8", "MLP output-projection LR x0.8"),
    ("two insurance levels", "two EMA/checkpoint fallback levels"),
    ("(eval-layout finding)", "(rows laid out as the evaluation reads them)"),
    ("OPT_FUSE 2", "fused optimizer update, level 2"),
    ("OPT_FUSE 3", "fused optimizer update, level 3"),
    ("XSHIFT", "attention-input shift (XSHIFT)"),
    ("(re-rolled the warm-up path)", "(may have re-rolled the warm-up path)"),
]


def _plain(note: str) -> str:
    for a, b in PLAIN:
        note = note.replace(a, b)
    return note.replace("  ", " ").strip()


def steps_decomp(N: Numbers, scored: List[dict]) -> None:
    """Optimizer steps between uploads rehearsed on the SAME chip, grouped by what changed (results CSV)."""
    s = {r["name"]: r for r in scored}
    rows = [
        ("A", "K24", "K38", "stream rows, value embeddings, GQA (one KV head), head size 256, scalar stack",
         "FLOPs and dispatch", "depth 12"),
        ("A", "K40", "K41", "depth 10 $\\to$ 9", "model size", "constant batch"),
        ("B", "K43", "K46", "scalar constants, async optimizer, Muon views, fused update", "dispatch", "depth 9"),
        ("B", "K46", "K50", "batch warm-up 131k$\\to$262k (+ MTP, Muon schedule)", "batch schedule", "depth 9"),
        ("C", "K51", "K56", "ZeRO Muon/AdamW, 3-phase batch ramp, accumulation in the graph", "dispatch and schedule",
         "depth 9"),
        ("C", "K56", "K63", "time target 1,795 s, EMA off, cooldown 0.7", "budget", "depth 9"),
    ]
    body = []
    for chip, a, b, what, kind, ctx in rows:
        ra, rb = s[a], s[b]
        assert ra["chip"] == chip and rb["chip"] == chip, (a, b)
        ratio = rb["steps"] / ra["steps"] - 1.0
        key = f"sd-{a.lower()}-{b.lower()}".replace("_", "")
        N.add(key, num(100 * ratio, 1, sign=True) + "\\%", "results/official-scores.csv, same chip")
        body.append(f"{chip} & \\upl{{{a}}} $\\to$ \\upl{{{b}}} & {intc(ra['steps'])} $\\to$ {intc(rb['steps'])} & "
                    f"{num(100 * ratio, 1, sign=True)}\\% & {kind} & {what} \\\\")
    table(TABLES / "steps_decomp.tex", body, "@{}llrrlp{5.6cm}@{}",
          "Chip & Uploads & Steps & Change & Kind & What changed",
          "steps between same-chip uploads (oracle.py)", size="\\footnotesize")


def _figure(since: List[dict], seq: List[dict], frozen: float) -> None:
    from style import BLUE, CHIP_MARKER, GREY, ORANGE, SURFACE, BLUE_WASH, TEXTWIDTH, INK_2, chip_s, plt, save_pdf
    import numpy as np
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.7))
    names = [r["name"].replace("_min", "") for r in since]
    x = np.arange(len(since))
    obs = np.array([r["offset"] for r in since]) * 1e3
    idx0 = len(since) - len(seq)
    pred = np.array([s["pred"] for s in seq]) * 1e3
    band = np.array([s["t95"] * s["psd"] for s in seq]) * 1e3
    xs = x[idx0:]
    ax.fill_between(xs, pred - band, pred + band, step="mid", color=BLUE_WASH, lw=0, zorder=0)
    ax.step(xs, pred, where="mid", color=BLUE, lw=1.2, zorder=2)
    ax.axhline(frozen * 1e3, color=ORANGE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for i, r in enumerate(since):
        burn = i < idx0
        ax.scatter([i], [obs[i]], s=chip_s(r["chip"], 26), marker=CHIP_MARKER.get(r["chip"], "o"),
                   color=GREY if burn else BLUE,
                   edgecolor=SURFACE, linewidth=0.8, zorder=4)
    ax.set_xticks(x, names, rotation=55, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("offset ($10^{-3}$ bpb)")
    ax.set_ylim(4.4, 9.0)
    chips = sorted({r["chip"] for r in since})
    handles = [Line2D([], [], color=BLUE, lw=1.2, label="mean of earlier uploads"),
               mpatches.Patch(color=BLUE_WASH, label="95% prediction interval"),
               Line2D([], [], color=ORANGE, lw=1.0, ls=(0, (4, 3)), label="frozen chip-C fit")]
    handles += [Line2D([], [], lw=0, marker=CHIP_MARKER.get(c, "o"), ms=6 if CHIP_MARKER.get(c) == "*" else 5,
                       mfc=INK_2, mec=INK_2, label=f"chip {c}")
                for c in chips]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, handlelength=1.6,
              borderaxespad=0.2, columnspacing=1.2)
    ax.tick_params(axis="x", length=0)
    save_pdf(fig, FIGURES / "offset_prospective.pdf")
