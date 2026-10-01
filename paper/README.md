# The paper

**Rehearse, Don't Guess: Deterministic Rehearsal, a Run Simulator and the Price of a Step on AWS Trainium.** The LaTeX source of the full-length paper on the FrontierForge
Phase 1 campaign. It is venue-agnostic (11 pt `article`, `natbib` + `plainnat`). The plan it follows is
[`OUTLINE.md`](OUTLINE.md), and where to submit it is covered in [`VENUES.md`](VENUES.md).

## Layout

| path | what it is |
|---|---|
| `main.tex` | preamble, title, the order of the sections. Author: Jagadeesh Kumar Gorla (independent researcher, team FrontierForge). |
| `sections/*.tex` | one file per section plus `appendix.tex` and `statements.tex` (reproducibility, ethics, acknowledgements) |
| `numbers.tex` | **generated**: every number the prose prints, as `\ffdef{key}{value}`, each with its source in a comment |
| `tables/*.tex` | **generated**: every data table |
| `figures/*.pdf` | **generated**: every figure, drawn at its printed width (6.5 in text width, 8 pt text) |
| `analysis/*.py` | the scripts that generate all of the above from the repository's data |
| `refs.bib` | 87 verified references; each entry's `verification` field says how it was checked (BibTeX ignores it) |
| `check_tex.py` | static checks that need no TeX installation, including estimated table widths and figure scales (`-v` lists every table) |
| `RESPONSE.md` | revision 1: point-by-point response to the three internal reviews |
| `make_bundle.py` | a flat, self-contained source folder and zip for Overleaf or arXiv |

The prose never types a data-derived number. It writes `\val{key}`, and `numbers.tex` defines the key. An undefined
key prints as a bold `??key??` and raises a LaTeX warning. `check_tex.py` catches it before that.

## Regenerate numbers, tables and figures

From the repository root, with `pip install -r requirements.txt matplotlib`:

```bash
python paper/analysis/make_all.py             # paper/numbers.tex, paper/tables/*.tex, paper/figures/*.pdf
                                              # (FF_PREVIEW_DIR=<dir> also writes PNG previews of the figures)
python paper/check_tex.py                     # labels/refs, cites, \input files, figures, \val keys, braces, widths
```

Everything is deterministic: fixed random seeds and no clock. The scripts read `results/official-scores.csv`,
`research/sim-data/{runs.jsonl,validation-pairs.json,official-uploads.csv}`, `ffsim/examples/recipe-K82s4.json` and
the shipped `ffsim` code. A few chip-log measurements exist only in the campaign documents
(`docs/FINDINGS.md`, `docs/EXACT-ORACLE.md`, `docs/GAP-ANALYSIS.md`, `docs/SIMULATOR.md`). They are registered once in
`analysis/docfacts.py` and `analysis/simulator.py`, and are marked `(doc)` in `numbers.tex`.

| script | produces |
|---|---|
| `analysis/oracle.py` | offset statistics by period and by chip, the frozen-calibration and expanding-window checks, the K70a sensitivity, interval coverage, the step-count decomposition, the upload ledger, `figures/offset_prospective.pdf` |
| `analysis/figs.py` | `figures/score_history.pdf`, `step_counts.pdf`, `oracle_calibration.pdf` |
| `analysis/simulator.py` | dataset counts, LOPO / LOKVO / temporal hold-out of the quality surrogate, step-time leave-one-lineage-out, offset model, coefficient and feature tables, GPU-proxy parity |
| `analysis/seeds.py` | warm-up spike-path statistics and `figures/warmup_paths.pdf` |
| `analysis/exchange.py` | the exchange rate, the gap table, levers priced in steps, the negative-results table |
| `analysis/contrib_guard.py` | the refit-guard stress test (honest records and coordinated attacks, in memory) |
| `analysis/docfacts.py` | official deltas between named uploads and the document-quoted facts |
| `analysis/make_all.py` | runs all of them and writes `numbers.tex` and the hyperparameter table |

## Build the PDF

**Locally** (TeX Live 2023 or newer, or MiKTeX):

```bash
cd paper
latexmk -pdf main.tex          # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`main.tex` finds figures through `\graphicspath{{figures/}}`. The README figures in `docs/figures/` are separate and
are not used by the paper.

**Overleaf.** Run `python paper/make_bundle.py`, then upload `paper/build/paper-source.zip` with *New project ->
Upload project*. The bundle is flat: all figures are in one `figures/` folder.

**arXiv.** Build the bundle, compile it once, copy the generated `main.bbl` into `paper/build/paper-source/` and zip
the folder. arXiv expects the `.bbl` to sit next to `main.tex`. Use the CC BY 4.0 licence, which matches TMLR's.

**CI.** [`.github/workflows/paper.yml`](../.github/workflows/paper.yml) runs on every push or pull request that
touches `paper/`, `docs/figures/`, the data or `ffsim/`. It regenerates the numbers and tables and fails if they
differ from the committed ones. It then runs `check_tex.py`, compiles the PDF with `xu-cheng/latex-action` (pinned
to the 4.1.0 release commit) and uploads it as the `paper-pdf` artifact.

## Moving it into a venue template

Keep `sections/`, `tables/`, `numbers.tex`, `refs.bib` and these three preamble lines:

```latex
\newcommand{\ffdef}[2]{\expandafter\def\csname ffv@#1\endcsname{#2}}
\DeclareRobustCommand{\val}[1]{\ifcsname ffv@#1\endcsname\csname ffv@#1\endcsname
  \else\textbf{??#1??}\PackageWarning{ffnumbers}{Undefined value `#1'}\fi}
\input{numbers}
```

plus the notation macros (`\valbpb`, `\ffsim`, `\knob`, `\upl`, `\docnum`), the `\ifanonymous` switch and the
packages `booktabs`, `longtable`, `amssymb`, `url`, `cleveref` and `placeins`. `\knob` is a url-style command:
keep it out of captions and section titles. Two-column templates (MLSys) need `figure*` / `table*` for the wide tables
(`uploads`, `features`, `sim_calibration`) and a tighter page budget. For a double-blind venue, set `\anonymoustrue` in `main.tex`:
it replaces the author block, hides the acknowledgements and replaces the repository URL.

## Before submitting

- Fill in authors and affiliations (`main.tex`), and confirm the acknowledgement names (`sections/statements.tex`).
- Confirm the compute-funding sentence in `sections/statements.tex` (`TODO(owner)`).
- Challenge terms: checked on 1 Oct 2026, no publication restriction found (`VENUES.md`).
- Compile once (CI artifact `paper-pdf` or Overleaf) and read the log for overfull boxes; `check_tex.py` only
  estimates widths.
- Rerun the three regeneration commands above and `python -m pytest -q tests`, then rebuild.
- Read `sections/limitations.tex` once more against any new results. The follow-up experiments listed in
  `OUTLINE.md` section 5 (E1, E2, E3, E5, E6, E7) are not in this version; pre-register them in the repository before
  running them.
