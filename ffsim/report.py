"""ffsim.report: markdown / CSV / JSON writers and the one-screen CLI summaries.

`results` is the dict `ffsim.search.evaluate` returns: {"meta", "base", "rows", "errors"}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from ffsim.schema import SimResult
from ffsim.simulate import BEST_OFFICIAL
from ffsim.search import ROW_COLUMNS, env_overrides, top_by

CONFIRM_SEEDS = (73, 58, 67)


# --- helpers ---------------------------------------------------------------------------------------
def _f(x: Any, nd: int = 5) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _changes_str(row: Dict[str, Any]) -> str:
    ch = row.get("changes") or {}
    if not ch:
        return "(base)"
    return " ".join(f"{k}={v}" for k, v in sorted(ch.items()))


def _support_flag(row: Dict[str, Any]) -> str:
    if row.get("n_changes", 0) == 0:
        return "base"
    return "ok" if row.get("in_support") else "EXTRAP"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _row_cells(row: Dict[str, Any], best: float) -> List[str]:
    return [str(row.get("rank", "")), f"`{_changes_str(row)}`", f"{row['steps_mean']:.0f}±{row['steps_sd']:.0f}",
            f"{_f(row['bpb_2m_mean'])}±{_f(row['bpb_2m_sd'], 4)}",
            f"{_f(row['official_p10'], 4)} / {_f(row['official_p50'], 4)} / {_f(row['official_p90'], 4)}",
            f"{row['p_beat_best']:.2f}", f"{row['delta_vs_base']:+.5f}", f"{row['p_beat_base']:.2f}", _support_flag(row)]


TABLE_HEADER = ("#", "changes vs base", "steps", "bpb_2m ± sd", "official p10 / p50 / p90", "P(beat best)",
                "Δ vs base", "P(< base)", "support")


# --- writers -----------------------------------------------------------------------------------------
def write_json(results: Dict[str, Any], path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    return p


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(results: Dict[str, Any], path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(ROW_COLUMNS) + ["rank_pbeat"]
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results["rows"]:
            out = []
            for c in cols:
                v = r.get(c)
                if c == "changes":
                    v = _changes_str(r) if r.get("n_changes") else ""
                elif isinstance(v, bool):
                    v = int(v)
                elif isinstance(v, float):
                    v = f"{v:.6g}" if c.startswith("step") else f"{v:.6f}"
                out.append("" if v is None else v)
            w.writerow(out)
    return p


def markdown_lines(results: Dict[str, Any], top: int = 20, seeds: Sequence[int] = CONFIRM_SEEDS) -> List[str]:
    meta = results["meta"]
    base = results.get("base")
    best = float(meta.get("best_official", BEST_OFFICIAL))
    b = meta.get("base", {})
    L: List[str] = []
    L.append(f"# ffsim search: {meta.get('space')}")
    L.append("")
    if meta.get("description"):
        L.append(meta["description"])
        L.append("")
    L.append(f"- base: **{b.get('name')}** ({b.get('code_version')}, chip {b.get('chip')}, time_target {b.get('time_target')})")
    L.append(f"- generator `{meta.get('gen')}`, {meta.get('n_candidates')} candidates "
             f"({meta.get('n_in_support')} inside the fitted support), {meta.get('n_sims')} draws each, "
             f"rng seed {meta.get('rng_seed')}, training seed {meta.get('seed') if meta.get('seed') is not None else 'marginal'}, "
             f"{meta.get('elapsed_s')} s, workers {meta.get('workers')}")
    L.append(f"- best official to beat: **{best:.4f}**; models: {meta.get('models') or 'fitted'}")
    if meta.get("n_errors"):
        L.append(f"- **{meta['n_errors']} candidates failed** (see the JSON `errors` list)")
    if base:
        L.append(f"- base prediction: {base['steps_mean']:.0f} steps, bpb_2m {_f(base['bpb_2m_mean'])} ± {_f(base['bpb_2m_sd'], 4)}, "
                 f"official p10/p50/p90 {_f(base['official_p10'], 4)} / {_f(base['official_p50'], 4)} / {_f(base['official_p90'], 4)}, "
                 f"P(beat {best:.4f}) {base['p_beat_best']:.2f}")
    dims = meta.get("dims") or []
    if dims:
        L.append("- dims: " + "; ".join(f"{d['knob']} (base {d['base']}): {', '.join(d['values'])}" for d in dims))
    L.append("")
    L.append(f"## Top {top} by predicted official bpb (lower is better)")
    L.append("")
    L += _md_table(TABLE_HEADER, [_row_cells(r, best) for r in top_by(results, "official_mean", top)])
    L.append("")
    L.append(f"## Top {min(top, 10)} by P(beat {best:.4f})")
    L.append("")
    L += _md_table(TABLE_HEADER, [_row_cells(r, best) for r in top_by(results, "p_beat_best", min(top, 10))])
    L.append("")
    L.append("## Next Trainium confirmations")
    L.append("")
    L.append(f"Top {min(top, 10)} by expected improvement over the base (Δ official mean), inside the fitted support only. "
             f"Run each on seeds {', '.join(str(s) for s in seeds)} (one pair rejects a configuration, never a family; "
             "effects under 0.0003 need all three seeds).")
    L.append("")
    conf = [r for r in top_by(results, "delta_vs_base", min(top, 10), in_support=True) if r["delta_vs_base"] < 0]
    if not conf:
        L.append("_No in-support candidate is predicted to improve on the base._")
    for i, r in enumerate(conf, 1):
        L.append(f"{i}. `{_changes_str(r)}` — Δ {r['delta_vs_base']:+.5f} (P(< base) {r['p_beat_base']:.2f}, "
                 f"P(beat best) {r['p_beat_best']:.2f}, {r['steps_mean']:.0f} steps); support: {r.get('support_note')}")
        for s in seeds:
            L.append(f"   - `{env_overrides(r, s)}`")
    extrap = [r for r in top_by(results, "delta_vs_base", 5, in_support=False) if r["delta_vs_base"] < 0]
    if extrap:
        L.append("")
        L.append("Extrapolations that look good (outside the fitted support: screen before trusting):")
        L.append("")
        for r in extrap:
            L.append(f"- `{_changes_str(r)}` — Δ {r['delta_vs_base']:+.5f}; {r.get('support_note')}")
    L.append("")
    L.append("## What the numbers mean")
    L.append("")
    L.append("- **steps**: optimizer steps the step-time model predicts in the charged time (time_target − 3 s save reserve), "
             "with a per-run chip-jitter draw (default sd 0.4%). 1% of steps ≈ 0.00057 bpb.")
    L.append("- **bpb_2m**: predicted public-shard rehearsal bpb (trained weights, first 2M tokens); sd includes seed luck "
             "(≈0.0006) unless a training seed was given.")
    L.append("- **official**: bpb_2m + the rehearsal→official offset (+0.0066 ± 0.0004). p10/p50/p90 are Monte Carlo percentiles.")
    L.append("- **P(beat best)**: fraction of draws with official < the best official score.")
    L.append("- **Δ vs base / P(< base)**: paired with the base under common random numbers, so shared noise cancels.")
    L.append("- **support**: `ok` = every changed knob value was seen (or lies between seen values) in the fitted runs; "
             "`EXTRAP` = the surrogate is guessing.")
    return L


def write_markdown(results: Dict[str, Any], path: Union[str, Path], top: int = 20,
                   seeds: Sequence[int] = CONFIRM_SEEDS) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines(results, top, seeds)) + "\n")
    return p


# --- one-screen summaries ---------------------------------------------------------------------------------
def format_summary(results: Dict[str, Any], top: int = 10) -> str:
    meta = results["meta"]
    base = results.get("base")
    best = float(meta.get("best_official", BEST_OFFICIAL))
    L = [f"ffsim search {meta.get('space')}: {meta.get('n_candidates')} candidates x {meta.get('n_sims')} draws "
         f"({meta.get('gen')}, {meta.get('elapsed_s')} s, {meta.get('n_in_support')} in support"
         + (f", {meta['n_errors']} errors" if meta.get("n_errors") else "") + ")"]
    if base:
        L.append(f"base {meta['base'].get('name')}: {base['steps_mean']:.0f} steps, bpb_2m {_f(base['bpb_2m_mean'])}, "
                 f"official {_f(base['official_p50'], 4)} [{_f(base['official_p10'], 4)}, {_f(base['official_p90'], 4)}], "
                 f"P(beat {best:.4f}) {base['p_beat_best']:.2f}")
    L.append(f"{'#':>3} {'official':>8} {'d.base':>9} {'P<base':>6} {'Pbest':>5} {'steps':>5} {'sup':>6}  changes")
    for r in top_by(results, "official_mean", top):
        L.append(f"{r['rank']:>3} {r['official_mean']:>8.4f} {r['delta_vs_base']:>+9.5f} {r['p_beat_base']:>6.2f} "
                 f"{r['p_beat_best']:>5.2f} {r['steps_mean']:>5.0f} {_support_flag(r):>6}  {_changes_str(r)}")
    conf = [r for r in top_by(results, "delta_vs_base", 3, in_support=True) if r["delta_vs_base"] < 0]
    if conf:
        L.append("confirm next (seeds 73, 58, 67): " + "; ".join(_changes_str(r) for r in conf))
    return "\n".join(L)


def format_simresult(r: SimResult, best: float = BEST_OFFICIAL) -> str:
    st = " ".join(f"{k} {v:.4f}s" for k, v in sorted(r.step_time.items()))
    L = [f"recipe {r.recipe}: n={r.n}",
         f"  steps     {r.steps_mean:.0f} +- {r.steps_sd:.0f}   step time {st}",
         f"  bpb_2m    {r.bpb_2m_mean:.5f} +- {r.bpb_2m_sd:.5f}   (rehearsal, 2M public tokens)",
         f"  official  {r.official_mean:.5f} +- {r.official_sd:.5f}   p10/p50/p90 {r.official_p10:.4f} / {r.official_p50:.4f} / {r.official_p90:.4f}",
         f"  P(official < {best:.4f}) = {r.p_beat_best:.2f}"]
    if r.notes:
        L.append(f"  notes: {r.notes}")
    return "\n".join(L)


def format_pair(pr: Any) -> str:
    return (f"P({pr.a.recipe} < {pr.b.recipe}) = {pr.p_a_better:.3f} (official, common random numbers); "
            f"on bpb_2m {pr.p_a_better_2m:.3f}\n"
            f"  diff official mean {pr.diff_mean:+.5f} +- {pr.diff_sd:.5f}, p10/p50/p90 "
            f"{pr.diff_p10:+.5f} / {pr.diff_p50:+.5f} / {pr.diff_p90:+.5f}\n"
            f"  {pr.a.recipe}: {pr.a.steps_mean:.0f} steps, official {pr.a.official_mean:.5f}; "
            f"{pr.b.recipe}: {pr.b.steps_mean:.0f} steps, official {pr.b.official_mean:.5f}")
