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

Revision 2: the 2M-to-20M text gap from the runs.jsonl records scored on both texts, and a split of the within-chip
offset SD into that gap, rounding and a remainder; baselines for the final-day uploads (no offset, the mean of all
earlier uploads, the mean of the early uploads); pairwise rank concordance of rehearsal and official scores; offset
against model quality with and without an indicator for instances G and H, with an exact permutation test; and the
winner's-curse shrinkage with lambda = 1 - s_eta^2 / s_a^2 (``winners_curse``, called by make_all.py after seeds.py).
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
    N.add("diff-95-within", num(1.96 * math.sqrt(2) * s_within, 4), "1.96 sqrt(2) x within-chip offset SD")
    N.add("pi-half", num(t_quantile(0.975, len(since) - 1) * s_since * math.sqrt(1 + 1 / len(since)), 4),
          "95% prediction half-width at n=15")

    # ---------------------------------------------------------------- the 2M-to-20M text gap and the within-chip SD
    gap = text_gap(N)
    # split the within-chip offset variance into the text gap's run-to-run SD, the rounding of official scores to
    # four decimals (uniform on +-0.00005) and a remainder on the official side (host speed, step jitter, order)
    s_round = 0.0001 / math.sqrt(12.0)
    N.add("offdec-round", num(s_round, 5), "SD of rounding to 4 decimals, 0.0001/sqrt(12)")
    N.add("offdec-text", num(gap["sd"], 5), "research/sim-data/runs.jsonl: SD of bpb_20m - bpb_2m (= gap-sd)")

    def _rest(s: float) -> float:
        return math.sqrt(max(0.0, s ** 2 - gap["sd"] ** 2 - s_round ** 2))
    s_lo = s_within * math.sqrt(den / chi2_quantile(0.975, den))
    s_hi = s_within * math.sqrt(den / chi2_quantile(0.025, den))
    N.add("offdec-rest", num(_rest(s_within), 5), "sqrt(within-chip SD^2 - text-gap SD^2 - rounding SD^2)")
    N.add("offdec-rest-lo", num(_rest(s_lo), 5), "remainder at the lower chi-square limit of the within-chip SD")
    N.add("offdec-rest-hi", num(_rest(s_hi), 5), "remainder at the upper chi-square limit of the within-chip SD")
    N.add("offdec-text-share", num(100 * gap["sd"] ** 2 / s_within ** 2, 0) + "\\%",
          "text-gap variance as a share of the within-chip offset variance")
    N.add("offdec-rest-share", num(100 * _rest(s_within) ** 2 / s_within ** 2, 0) + "\\%",
          "remainder variance as a share of the within-chip offset variance")

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
    # baselines for the final-day uploads, each scored on the same seven uploads as the frozen chip-C fit
    fd_rows = [by[n] for n in FINAL_DAY]
    i_fd = min(cal.index(r) for r in fd_rows)
    prior_all = [r["offset"] for r in cal[:i_fd]]
    prior_since = [r["offset"] for r in cal[k50:i_fd]]
    early_offs = [r["offset"] for r in early]

    def _fd_mae(pred: float) -> float:
        return mean(abs(r["offset"] - pred) for r in fd_rows)
    N.add("nooffset-mae", num(mean(abs(r["offset"]) for r in fd_rows), 4), src + ": final-day uploads, offset 0")
    N.add("nooffset-mae-since", num(mean(abs(r["offset"]) for r in since), 4), src + ": uploads since K50, offset 0")
    N.add("priormean-mae", num(_fd_mae(mean(prior_all)), 5),
          src + ": final day, predicted by the mean of every earlier rehearsed upload")
    N.add("priormean-n", str(len(prior_all)), src)
    N.add("priormean-offset", num(mean(prior_all), 5, sign=True), src)
    N.add("priorsince-mae", num(_fd_mae(mean(prior_since)), 5),
          src + ": final day, predicted by the mean of the earlier uploads since K50")
    N.add("priorsince-n", str(len(prior_since)), src)
    N.add("earlymean-mae", num(_fd_mae(mean(early_offs)), 5),
          src + ": final day, predicted by the mean of the early uploads (chips A/B, F2-K46)")
    N.add("earlymean-n", str(len(early_offs)), src)
    N.add("earlymean-offset", num(mean(early_offs), 5, sign=True), src)

    # pairwise rank concordance since K50: does the rehearsal order two uploads as the official scores do?
    conc = {"n": 0, "agree": 0, "tie": 0, "close_n": 0, "close_agree": 0}
    close_thr = 0.0005
    for i in range(len(since)):
        for j in range(i + 1, len(since)):
            a, b = since[i], since[j]
            dr, do = a["rehearsal"] - b["rehearsal"], a["official"] - b["official"]
            ok = do != 0 and (dr > 0) == (do > 0)
            conc["n"] += 1
            conc["agree"] += ok
            conc["tie"] += do == 0
            if abs(dr) < close_thr:
                conc["close_n"] += 1
                conc["close_agree"] += ok
    N.add("conc-n", str(conc["n"]), src + ": pairs of uploads since K50")
    N.add("conc-agree", str(conc["agree"]), src + ": pairs ordered the same by rehearsal and official score")
    N.add("conc-ties", str(conc["tie"]), src + ": pairs with equal official scores")
    N.add("conc-disagree", str(conc["n"] - conc["agree"] - conc["tie"]), src)
    N.add("conc-close-thr", num(close_thr, 4), "rehearsal difference below which a pair counts as close")
    N.add("conc-close-n", str(conc["close_n"]), src + ": pairs whose rehearsals differ by less than 0.0005")
    N.add("conc-close-agree", str(conc["close_agree"]), src)

    # ---------------------------------------------------------------- chip or model quality? (since K50)
    chip_vs_quality(N, since)

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
    table(TABLES / "uploads.tex", body, "@{}llcrrrrcp{7.9cm}@{}",
          "Upload & Date & Chip & Rehearsal & Steps & Official & Offset & Best & What changed",
          "every scored upload, results/official-scores.csv (oracle.py)", size="\\scriptsize", tabcolsep="3pt")

    _figure(since, seq, frozen)
    return {"since": since, "seq": seq, "frozen": frozen, "s_within": s_within, "s_since": s_since,
            "s_between": _between_chip_sd(since), "gap_sd": gap["sd"]}


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


# ------------------------------------------------------------------------------------------- revision 2 analyses
def text_gap(N: Numbers) -> Dict[str, float]:
    """The 2M-to-20M text gap: runs.jsonl records scored on both the first 2M and the first ~20M public validation
    tokens (bpb_20m - bpb_2m), and whether the two texts order the runs the same way."""
    import itertools
    import json
    from common import RUNS
    recs = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]
    both = [r for r in recs if r.get("bpb_20m") is not None and r.get("bpb_2m") is not None]
    gaps = [float(r["bpb_20m"]) - float(r["bpb_2m"]) for r in both]
    src = "research/sim-data/runs.jsonl: bpb_20m - bpb_2m"
    chips = sorted({r.get("chip") or "?" for r in both})
    N.add("gap-n", str(len(gaps)), src)
    N.add("gap-chips", ", ".join(chips), src + ": chips of those runs")
    N.add("gap-mean", num(mean(gaps), 5, sign=True), src)
    N.add("gap-sd", num(sd(gaps), 5), src)
    N.add("gap-min", num(min(gaps), 5, sign=True), src)
    N.add("gap-max", num(max(gaps), 5, sign=True), src)
    k = n = 0
    for a, b in itertools.combinations(both, 2):
        n += 1
        k += (float(a["bpb_2m"]) - float(b["bpb_2m"])) * (float(a["bpb_20m"]) - float(b["bpb_20m"])) > 0
    N.add("gap-order-k", str(k), src + ": run pairs ordered the same on both texts")
    N.add("gap-order-pairs", str(n), src)
    N.add("gap-order-agree", num(100.0 * k / n, 0) + "\\%", src + ": share of run pairs ordered the same")
    x = [float(r["bpb_2m"]) for r in both]
    y = [float(r["bpb_20m"]) for r in both]
    mx, my = mean(x), mean(y)
    r_xy = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) *
                                                                   sum((b - my) ** 2 for b in y))
    N.add("gap-r", num(r_xy, 3), src + ": Pearson r between the two scores")
    return {"mean": mean(gaps), "sd": sd(gaps), "n": len(gaps)}


def _between_chip_sd(since: List[dict]) -> float:
    """Between-chip SD of the offset since K50: one-way random-effects method of moments, (MSB - MSW) / n0."""
    groups: Dict[str, List[float]] = collections.OrderedDict()
    for r in since:
        groups.setdefault(r["chip"], []).append(r["offset"])
    n_tot, k = len(since), len(groups)
    grand = mean(r["offset"] for r in since)
    msb = sum(len(v) * (mean(v) - grand) ** 2 for v in groups.values()) / (k - 1)
    msw = sum(sum((x - mean(v)) ** 2 for x in v) for v in groups.values()) / (n_tot - k)
    n0 = (n_tot - sum(len(v) ** 2 for v in groups.values()) / n_tot) / (k - 1)
    return math.sqrt(max(0.0, (msb - msw) / n0))


def _t_two_sided(t: float, df: int) -> float:
    from common import _betainc
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def chip_vs_quality(N: Numbers, since: List[dict]) -> None:
    """Is the offset a property of the rehearsal instance or of model quality? Regress the offset on the rehearsal
    score, with and without an indicator for the final-day instances G and H, and test the G/H difference by an
    exact permutation of which uploads carry the label (all C(n, k) assignments)."""
    import itertools
    import numpy as np
    src = "results/official-scores.csv, uploads since K50"
    y = np.array([r["offset"] for r in since])
    x = np.array([r["rehearsal"] for r in since])
    gh = np.array([1.0 if r["chip"] in ("G", "H") else 0.0 for r in since])
    n = len(y)

    def ols(cols):
        X = np.column_stack([np.ones(n)] + cols)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ beta
        dfree = n - X.shape[1]
        cov = (res @ res / dfree) * np.linalg.inv(X.T @ X)
        return beta, np.sqrt(np.diag(cov)), dfree

    b1, se1, df1 = ols([x])
    b2, se2, df2 = ols([x, gh])
    t1, t2 = t_quantile(0.975, df1), t_quantile(0.975, df2)
    N.add("cq-n", str(n), src)
    N.add("cq-gh-n", str(int(gh.sum())), src + ": uploads rehearsed on instances G and H")
    N.add("cq-slope", num(b1[1], 3, sign=True), src + ": d offset / d rehearsal bpb, no instance term")
    N.add("cq-slope-lo", num(b1[1] - t1 * se1[1], 3, sign=True), src + ": 95% t interval")
    N.add("cq-slope-hi", num(b1[1] + t1 * se1[1], 3, sign=True), src)
    N.add("cq-slope-p", num(_t_two_sided(b1[1] / se1[1], df1), 2), src + ": two-sided t test of the slope")
    span = float(x.max() - x.min())
    N.add("cq-slope-span", num(abs(b1[1]) * span, 5), src + ": |slope| x the since-K50 rehearsal span")
    N.add("cq-slope-adj", num(b2[1], 3, sign=True), src + ": slope with a G/H indicator")
    N.add("cq-slope-adj-lo", num(b2[1] - t2 * se2[1], 3, sign=True), src)
    N.add("cq-slope-adj-hi", num(b2[1] + t2 * se2[1], 3, sign=True), src)
    N.add("cq-gh-coef", num(b2[2], 5, sign=True), src + ": G/H indicator, given the rehearsal score")
    N.add("cq-gh-lo", num(b2[2] - t2 * se2[2], 5, sign=True), src + ": 95% t interval")
    N.add("cq-gh-hi", num(b2[2] + t2 * se2[2], 5, sign=True), src)
    diff = float(y[gh == 1].mean() - y[gh == 0].mean())
    N.add("cq-gh-diff", num(diff, 5, sign=True), src + ": mean offset, G/H minus the rest")
    # exact permutation over every assignment of the G/H label to the same number of uploads
    m = int(gh.sum())
    t_obs = abs(b2[2] / se2[2])
    tot = ex_d = ex_t = 0
    for comb in itertools.combinations(range(n), m):
        z = np.zeros(n)
        z[list(comb)] = 1.0
        tot += 1
        ex_d += abs(float(y[z == 1].mean() - y[z == 0].mean())) >= abs(diff) - 1e-15
        bz, sz, _ = ols([x, z])
        ex_t += abs(bz[2] / sz[2]) >= t_obs - 1e-12
    N.add("cq-perm-n", intc(tot), src + ": label assignments enumerated")
    N.add("cq-perm-p", num(ex_d / tot, 3), src + ": exact two-sided permutation p, G/H mean difference")
    N.add("cq-perm-p-adj", num(ex_t / tot, 3),
          src + ": exact two-sided permutation p, G/H coefficient given the rehearsal score")


def winners_curse(N: Numbers, o: Dict[str, float], jitter_rel: List[float], kappa: float) -> None:
    """Shrinkage of a salt lottery's rehearsal edge. A salt's rehearsal score varies across salts with SD s_a (the
    documented order re-draw SD, 0.0003-0.0004); write it as a = theta + eta, with eta the parts that are specific to
    the rehearsal and do not reach the official score. Then E[theta | a] = lambda a with
    lambda = max(0, 1 - s_eta^2 / s_a^2). On one chip, eta is the run-to-run text-gap SD plus the rehearsal's own
    step jitter (kappa x SD(steps) / steps); across chips it also carries the between-chip offset SD s_b, which
    widens the spread of a as well: share lost = (s_eta^2 + s_b^2) / (s_a^2 + s_b^2)."""
    src = "winner's curse: s_a from docs/FINDINGS.md s3, s_eta from runs.jsonl (oracle.py)"
    s_a_lo, s_a_hi = 0.0003, 0.0004
    jit = sorted(kappa * s for s in jitter_rel)
    jit_lo, jit_hi = jit[0], jit[-1]
    eta_lo = math.sqrt(o["gap_sd"] ** 2 + jit_lo ** 2)
    eta_hi = math.sqrt(o["gap_sd"] ** 2 + jit_hi ** 2)
    sb = o["s_between"]

    def lost(eta: float, s_a: float, s_b: float = 0.0) -> float:
        return min(1.0, (eta ** 2 + s_b ** 2) / (s_a ** 2 + s_b ** 2))
    N.add("wc-jit-lo", num(jit_lo, 5), src + ": kappa x within-sweep SD of steps / mean steps, smallest sweep SD")
    N.add("wc-jit-hi", num(jit_hi, 5), src + ": largest sweep SD")
    N.add("wc-eta-lo", num(eta_lo, 5), src + ": sqrt(gap-sd^2 + jitter^2)")
    N.add("wc-eta-hi", num(eta_hi, 5), src)
    N.add("wc-between", num(sb, 5), "results/official-scores.csv: between-chip offset SD since K50 (method of moments)")
    one = [lost(e, s) for e in (eta_lo, eta_hi) for s in (s_a_lo, s_a_hi)]
    cross = [lost(e, s, sb) for e in (eta_lo, eta_hi) for s in (s_a_lo, s_a_hi)]
    N.add("wc-shrink-lo", num(min(one), 2), src + ": share of the edge lost, one chip, minimum over s_a and jitter")
    N.add("wc-shrink-hi", num(max(one), 2), src + ": share lost, one chip, maximum")
    N.add("wc-lam-lo", num(1.0 - max(one), 2), src + ": lambda, one chip, minimum")
    N.add("wc-lam-hi", num(1.0 - min(one), 2), src + ": lambda, one chip, maximum")
    N.add("wc-shrink-cross-lo", num(min(cross), 2), src + ": share lost, draws on different chips, minimum")
    N.add("wc-shrink-cross-hi", num(max(cross), 2), src + ": share lost, draws on different chips, maximum")
    N.add("wc-lam-cross-lo", num(1.0 - max(cross), 2), src + ": lambda across chips, minimum")
    N.add("wc-lam-cross-hi", num(1.0 - min(cross), 2), src + ": lambda across chips, maximum")
