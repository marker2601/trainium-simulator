"""Facts the paper quotes, registered once so the prose cannot drift from the repository.

Two kinds:
* DERIVED from results/official-scores.csv (official deltas between named uploads, rehearsal step counts by chip);
* QUOTED from the campaign documents, which record chip-log measurements that are not in a machine-readable file
  (each carries its document and section in numbers.tex). Change a quoted value here and in the document together.
"""
from __future__ import annotations

from typing import Dict

from common import Numbers, intc, load_scores, mean, num

FIND = "docs/FINDINGS.md"
ORACLE = "docs/EXACT-ORACLE.md"
GAP = "docs/GAP-ANALYSIS.md"
SIM = "docs/SIMULATOR.md"
REPORT = "docs/simulator-report-20260929.md"

QUOTED = [
    # key, value, source
    ("rowpool-chip", "\\ensuremath{-}0.0023", f"{FIND} s3: chip E pair"),
    ("rowpool-chip-treat", "0.957259", f"{FIND} s3"),
    ("rowpool-chip-ctrl", "0.959523", f"{FIND} s3"),
    ("rowpool-steps-treat", "2{,}318", f"{FIND} s3"),
    ("rowpool-steps-ctrl", "2{,}329", f"{FIND} s3"),
    ("rowpool-stepnorm", "\\ensuremath{-}0.0026", f"{FIND} s3"),
    ("rowpool-step-cost", "0.5\\%", f"{FIND} s3"),
    ("rowpool-from", "96", f"{FIND} s2 (FF_ROW_SHUFFLE_FROM)"),
    ("rowpool-size", "256", f"{FIND} s2 (FF_ROW_SHUFFLE)"),
    ("rowpool-null", "0.0003", f"{FIND} s5: other order tricks 0 +- 0.0003"),
    ("asr-pairs", "\\ensuremath{-}0.00181, \\ensuremath{-}0.00136, \\ensuremath{-}0.00106, \\ensuremath{-}0.00173",
     f"{FIND} s3"),
    ("asr-mean", num(mean([-0.00181, -0.00136, -0.00106, -0.00173]), 5), f"{FIND} s3, mean of four pairs"),
    ("asr-step-cost", "0.1\\%", f"{FIND} s3"),
    ("ema-blend-chip", "\\ensuremath{-}0.0005", f"{FIND} s3, one chip pair"),
    ("ema-compile-steps", "15--30", f"{FIND} s3 / {ORACLE} traps"),
    ("ema-warm-bias", "0.0002--0.0006", f"{ORACLE} traps"),
    ("cooldown-stepnorm", "\\ensuremath{-}0.0006", f"{FIND} s3, three chip pairs"),
    ("cproj-stepnorm", "\\ensuremath{-}0.0004", f"{FIND} s3"),
    ("optfuse-steps", "0.8\\%", f"{FIND} s3"),
    ("salt-sd", "0.0003--0.0004", f"{FIND} s3"),
    ("salt-span", "0.0008", f"{FIND} s3: four salts spanned 0.0008 on chip"),
    ("seed-sd-doc", "0.0010--0.0012", f"{FIND} s4"),
    ("spike-penalty-doc", "\\ensuremath{+}0.003 to \\ensuremath{+}0.0045", f"{FIND} s4"),
    ("spike-frac-doc", "40\\%", f"{FIND} s4"),
    ("harvested-runs-doc", "430", f"{FIND} s4"),
    ("lottery-new-seeds", "6", f"{FIND} s4: 1 of about 6 new seeds landed on the good path"),
    ("det-cold", "1.041525", f"{ORACLE} why it works (19 Sep)"),
    ("det-warm", "1.041520", f"{ORACLE} why it works"),
    ("det-diff", "5\\ensuremath{\\times}10\\textsuperscript{\\ensuremath{-}6}", ORACLE),
    ("text-gap", "\\ensuremath{+}0.0070", f"{ORACLE}: 20M vs first 2M public tokens"),
    ("chip-spread", "3.5\\%", f"{ORACLE} traps"),
    ("chip-old-runtime", "7--12\\%", f"{ORACLE} traps"),
    ("headroom", "6--8", f"{ORACLE} traps: seconds of headroom under the 1,800 s cap"),
    ("rerun-steps", "2.3\\%", f"{ORACLE}: organiser-side K44 rerun"),
    ("rerun-delta", "\\ensuremath{+}0.0014", f"{ORACLE}: organiser-side K44 rerun"),
    ("k73s6-projected", "0.9619", f"{ORACLE}: largest miss against a written projection"),
    ("k82s4-projected", "0.9614--0.9615", f"{ORACLE} reproducing it"),
    ("k82-proj-offset", "\\ensuremath{+}0.0069 to \\ensuremath{+}0.0070", f"{ORACLE} reproducing it: final-day offset"),
    ("k82s7-projected", "0.9614", f"{FIND} s1: K82s7 projected about 0.9614"),
    ("asr-overestimate", "0.0005", f"{REPORT} s5: attention-source pairs over-predicted by about 0.0005"),
    ("gpu-cd-pairs", "\\ensuremath{+}0.09 and \\ensuremath{-}0.37$\\times10^{-3}$", f"{SIM} route 2: cooldown 0.7 chip pairs"),
    ("k82s4-projected-range", "0.9611--0.9619", f"{ORACLE} reproducing it"),
    ("tokens-per-s", "285k", f"{GAP} where the compute goes"),
    ("mfu", "30\\%", f"{GAP}: roughly 30% MFU"),
    ("idle", "19\\%", f"{GAP}: idle on host dispatch"),
    ("params", "124M", f"{FIND} s2"),
    ("win-scalar", "9.1\\%", f"{FIND} s6"),
    ("win-fusedce", "2.8\\%", f"{FIND} s6"),
    ("win-muonshard", "2.6\\%", f"{FIND} s6"),
    ("win-optfuse", "0.7\\%", f"{FIND} s6"),
    ("late-change", "0.0005", f"{GAP}: recipe changes on the final day"),
    ("lever-min", "0.0013", f"{FIND} s1: each structural change worth 0.0013 or more"),
    ("pool-large", "1{,}024", f"{FIND} s3: a pool of 1,024 gave the same result as 256"),
    ("pool-small", "64", f"{FIND} s3: a pool of 64 helped less"),
    ("scalar-keys", "61", f"{FIND} s5"),
    ("depth10-steps", "6--11\\%", f"{FIND} s5"),
    ("depth8-steps", "10.6\\%", f"{FIND} s5"),
    ("st-qkv", "3.2\\%", f"{SIM} route 1: QKV fusion slower"),
    ("st-gate", "7.6\\%", f"{SIM} route 1: a 96-parameter gate"),
    ("st-onepass", "1.8\\%", f"{SIM} route 1: one fewer elementwise pass"),
    ("st-novel-prior", "3\\%", f"{SIM} route 1: unknown-mechanism prior"),
    ("search-k60-cands", "282", f"{SIM}: local candidates around K60"),
    ("leaderboard-time", "7:40~AM CDT on 1~October", f"{GAP}: final snapshot"),
    ("eval-tokens", "2{,}097{,}152", f"{ORACLE}: first public tokens used by the rehearsal"),
    ("budget-s", "1{,}800", "contest rules (trainiumfrontier2026rules): 30 minutes excluding start-up/compilation"),
    ("rules-shard-tokens", "about 20M", "contest rules: pinned validation shard, ~20M tokens"),
    ("rules-deadline", "11:59~PM PT on 30~September 2026", "contest rules: Round 1 deadline"),
    ("k82s7-upload", "1:41~AM CDT on 1~October, i.e.\\ 11:41~PM PT on 30~September", f"{FIND} s1"),
    ("round-two", "7~October to 4~November 2026", "contest rules"),
    ("finals", "6--12~December 2026", "contest rules: NeurIPS 2026 competition event"),
    ("official-rounding", "0.00005", "official scores are published to 4 decimals (results CSV)"),
    ("anchor-date", "24~September", f"{GAP}"),
    ("qv-records", "1{,}188", "research/sim-data/quality-validation.md header: run records when the report was written"),
]


def run(N: Numbers) -> Dict[str, object]:
    for k, v, src in QUOTED:
        N.add(k, v, src + " (doc)")
    s = {r["name"]: r for r in load_scores()}
    src = "results/official-scores.csv"

    def d(a: str, b: str, nd: int = 4) -> str:
        return num(s[a]["official"] - s[b]["official"], nd, sign=True)

    N.add("rowpool-official", d("K70a", "K63"), src + ": K70a - K63")
    N.add("ema-blend-official", d("K77a", "K73s4"), src + ": K77a - K73s4")
    N.add("prewarm-official", d("K82s4", "K77a"), src + ": K82s4 - K77a")
    N.add("cooldown-official", d("K63", "K60"), src + ": K63 - K60")
    N.add("asr-official", d("K60", "K59"), src + ": K60 - K59")
    N.add("leaky-official", d("K59", "K57"), src + ": K59 - K57")
    N.add("salt7-vs-salt4", d("K82s7", "K82s4", 5), src + ": K82s7 - K82s4")
    N.add("salt6-vs-salt4", d("K73s6", "K73s4"), src + ": K73s6 - K73s4")
    N.add("salt6-reh-vs-salt4", num(s["K73s6"]["rehearsal"] - s["K73s4"]["rehearsal"], 5, sign=True),
          src + ": K73s6 - K73s4 rehearsal")
    N.add("xshift-official", d("K65a", "K63"), src + ": K65a - K63")
    N.add("latek8-official", d("K65b", "K63"), src + ": K65b - K63")
    N.add("nesterov-official", d("K65c", "K63", 5), src + ": K65c - K63")
    N.add("splitadamw-official", d("K65d", "K63"), src + ": K65d - K63")
    for name in ("K73s6", "K77a", "K82s7", "K73s4", "K82s4", "K70a", "K63", "K60"):
        N.add(f"steps-{name.lower()}", intc(s[name]["steps"]), src)
        N.add(f"chip-{name.lower()}", s[name]["chip"], src)
    for name in ("K70a", "K63", "K73s4", "K73s6", "K77a", "K60", "K59", "K57", "K65a"):
        N.add(f"official-{name.lower()}", num(s[name]["official"], 4), src)
    N.add("prewarm-g-drop", str(int(s["K73s6"]["steps"] - s["K77a"]["steps"])), src + ": chip G, no EMA - EMA without prewarm")
    N.add("prewarm-g-gain", num(s["K82s7"]["steps"] - s["K73s6"]["steps"], 0, sign=True), src + ": chip G, prewarm - no EMA")
    N.add("prewarm-h-gain", num(s["K82s4"]["steps"] - s["K73s4"]["steps"], 0, sign=True), src + ": chip H, prewarm - no EMA")
    N.add("first-prewarm", "K53", src + ": first upload whose change list names the EMA prewarm")
    N.add("prewarm-dropped", "K63", src + ": EMA turned off")
    N.add("leaky-chip-c", "\\ensuremath{-}0.00105", "docs/SIMULATOR.md route 2: chip C pair vs ReLU^2 (doc)")
    N.add("k73s6-miss", num(s["K73s6"]["official"] - 0.9619, 4), src + " vs the written projection 0.9619")
    return {}
