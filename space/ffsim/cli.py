"""python -m ffsim: harvest | build-dataset | fit | validate | simulate | search | report | gpu <sub>.

Every subcommand runs with numpy + stdlib only and prints a one-screen summary. The modules other
agents own (dataset, parse_*, steptime, quality, offset, harvest) are imported lazily inside the
subcommands, so `simulate` and `search` work with `--models stub` (the knob-blind anchor stubs)
or a fitted `models.pkl` even before those modules exist.

`validate`, `simulate` and `search` need a fitted `models.pkl` (default research/sim-data/models.pkl,
NOT shipped: a pickle of the fitted classes). When it is missing they fit it on the fly from
`--runs/--pairs/--uploads` (about 1.5 s on the full dataset) and save it to the requested path, so
the README quickstart runs out of the box once `build-dataset` has produced runs.jsonl. `search`
writes its md/csv/json next to the data by default (research/sim-data/search-<space>.*: that is the
git-tracked evidence directory of CONTRACT.md); pass `--out <dir>/<stem>` to write elsewhere.

`gpu <sub> [args]` is the route-2 group (ffsim/gpu, the equal-steps GPU proxy): plan / schedule / run /
report / verdict / from-search / prices go to ffsim.gpu.fleet.main, calibrate to ffsim.gpu.calibrate.main,
make-train to ffsim.gpu.make_train_gpu.main and launch to `bash ffsim/gpu/launch.sh`. Everything after the
sub name is passed through unchanged (`--config X` is accepted as `--configs X`), and the modules are
imported only when the group runs: the base CLI needs numpy + stdlib, never torch (torch is needed only
inside the generated train_gpu.py on the GPU box).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
SIM_DATA = REPO / "research" / "sim-data"
DEFAULT_RUNS = SIM_DATA / "runs.jsonl"
DEFAULT_PAIRS = SIM_DATA / "validation-pairs.json"
DEFAULT_UPLOADS = SIM_DATA / "official-uploads.csv"
DEFAULT_MODELS = SIM_DATA / "models.pkl"
DEFAULT_MONITOR = SIM_DATA / "monitor-runs.csv"
DEFAULT_EXPERIMENTS = REPO / "research" / "experiments.csv"
EXAMPLES = Path(__file__).resolve().parent / "examples"

# Function names tried, in order, on the modules other agents own (first callable wins).
NAMES = {
    "parse_csv": ("load_experiments_csv", "parse_experiments_csv", "parse_csv", "parse_experiments", "parse", "load", "main"),
    "parse_monitor": ("load_monitor_csv", "parse_monitor_csv", "parse_monitor", "parse", "load", "main"),
    "uploads": ("load_official_uploads", "parse_official_uploads", "load_uploads"),
    "parse_logs": ("parse_run_dir", "parse_run", "parse_dir", "parse", "load"),
    "merge": ("merge_runs",),
    "save": ("save_runs", "save", "write_runs", "write"),
    "load": ("load_runs", "load", "read_runs", "read"),
}


def _resolve(path_str: str) -> Path:
    """The given path, or - when it does not exist - a sibling `<stem>*<suffix>` variant (e.g. the
    builder's monitor-runs.b.csv when monitor-runs.csv is asked for). Prints which one it took."""
    p = Path(path_str)
    if p.exists():
        return p
    if p.parent.is_dir():
        alts = sorted(q for q in p.parent.glob(f"{p.stem}*{p.suffix}") if q.is_file())
        if alts:
            _p(f"note: {p.name} not found, using {alts[0].name}")
            return alts[0]
    return p


# --- small helpers -------------------------------------------------------------------------------
def _p(*a: Any) -> None:
    print(*a, flush=True)


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise SystemExit(f"{name} is not available yet ({e}). It is owned by another ffsim module; "
                         "see ffsim/CONTRACT.md.") from e


def _first(mod: Any, names: Sequence[str], what: str) -> Callable[..., Any]:
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    raise SystemExit(f"{mod.__name__} exposes none of {list(names)} ({what}); update ffsim/cli.py NAMES")


def _call_flex(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call with kwargs, then positionally without them (signatures of foreign modules vary)."""
    try:
        return fn(*args, **kwargs)
    except TypeError:
        if kwargs:
            return fn(*args)
        raise


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, dict):
        return list(x.values())
    return list(x)


def _print_report(title: str, rep: Any) -> None:
    _p(f"--- {title}")
    if rep is None:
        _p("  (no report)")
    elif isinstance(rep, str):
        for line in rep.rstrip().splitlines():
            _p("  " + line)
    elif isinstance(rep, dict):
        for k, v in rep.items():
            if isinstance(v, float):
                _p(f"  {k}: {v:.6g}")
            elif isinstance(v, (list, tuple)) and len(v) > 12:
                _p(f"  {k}: [{len(v)} items] {list(v)[:6]} ...")
            else:
                _p(f"  {k}: {v}")
    elif isinstance(rep, (list, tuple)):
        for item in rep[:40]:
            _p(f"  {item}")
        if len(rep) > 40:
            _p(f"  ... {len(rep) - 40} more")
    else:
        _p(f"  {rep}")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fit_model(module: str, cls_name: str, *args: Any) -> Any:
    """Model.fit(...) as an instance method returning self (ffsim.steptime/quality/offset), or as a
    classmethod / staticmethod returning the fitted model."""
    import inspect
    mod = _import(module)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise SystemExit(f"{module} has no {cls_name}")
    raw = inspect.getattr_static(cls, "fit", None)
    if raw is None:
        raise SystemExit(f"{module}.{cls_name} has no fit()")
    if isinstance(raw, (classmethod, staticmethod)):
        out = cls.fit(*args)
        if out is not None and (hasattr(out, "predict") or hasattr(out, "sample")):
            return out
        raise SystemExit(f"{module}.{cls_name}.fit returned {type(out).__name__}, not a model")
    inst = cls()
    out = inst.fit(*args)
    return out if (out is not None and (hasattr(out, "predict") or hasattr(out, "sample"))) else inst


def _models(args: argparse.Namespace) -> Any:
    """The models `--models` names: 'stub' -> the anchor stubs; an existing pickle -> loaded; a missing
    pickle -> fitted on the fly from --runs/--pairs/--uploads and saved there (the inputs are kept on
    `args._fit_inputs` so `validate` does not load them twice)."""
    from ffsim.simulate import anchor_models, load_models
    spec = getattr(args, "models", None) or str(DEFAULT_MODELS)
    if spec == "stub":
        _p("models: anchor stubs (knob-blind; pipeline check only, NOT a surrogate)")
        return anchor_models()
    p = Path(spec)
    if not p.exists():
        runs = _resolve(str(getattr(args, "runs", None) or DEFAULT_RUNS))
        if not runs.exists():
            raise SystemExit(f"{p} not found, and there is no {runs} to fit it from. Run `python -m ffsim build-dataset` "
                             "then `python -m ffsim fit`, or pass --models stub to exercise the pipeline with the "
                             "knob-blind anchor stubs.")
        _p(f"note: {p} not found: fitting it on the fly from {runs} (same as `python -m ffsim fit --out {p}`)")
        models, inputs = fit_models(args, p)
        args._fit_inputs = inputs
        return models
    m = load_models(p)
    meta = {k: v for k, v in (m.meta or {}).items() if k not in ("support", "runs", "pairs", "uploads")}
    n_sup = len((m.meta or {}).get("support") or {})
    _p(f"models: {p.name} {meta} support: {n_sup} knobs")
    return m


def _report_of(model: Any, names: Sequence[str], *args: Any) -> Any:
    """model.<name>(*args), or model.<name>() when the method takes no data (loo_report)."""
    for n in names:
        fn = getattr(model, n, None)
        if not callable(fn):
            continue
        for call in ((lambda: fn(*args)) if args else (lambda: fn()), lambda: fn()):
            try:
                return call()
            except TypeError:
                continue
            except Exception as e:  # noqa: BLE001 - a broken report must not hide the others
                return f"{n} failed: {type(e).__name__}: {e}"
    return None


# --- subcommands ---------------------------------------------------------------------------------
def cmd_harvest(args: argparse.Namespace) -> int:
    mod = _import("ffsim.harvest")
    main = getattr(mod, "main", None)
    if main is None:
        raise SystemExit("ffsim.harvest has no main(argv)")
    rc = main(list(args.rest))
    return int(rc or 0)


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from ffsim.schema import RunRecord
    sources: List[List[Any]] = []          # one list per source: dataset.merge_runs(list_of_lists)
    counts: Dict[str, int] = {}
    exp = _resolve(args.experiments)
    if exp.exists():
        fn = _first(_import("ffsim.parse_csv"), NAMES["parse_csv"], "experiments.csv -> RunRecords")
        got = _as_list(_call_flex(fn, str(exp)))
        counts["experiments.csv"] = len(got)
        sources.append(got)
    else:
        _p(f"skip experiments: {exp} not found")
    mon = _resolve(args.monitor)
    if mon.exists():
        fn = _first(_import("ffsim.parse_monitor"), NAMES["parse_monitor"], "monitor csv -> RunRecords")
        got = _as_list(_call_flex(fn, str(mon)))
        counts["monitor"] = len(got)
        sources.append(got)
    else:
        _p(f"skip monitor: {mon} not found")
    chips_dir = Path(args.chips_dir)
    chip_records: List[Any] = []
    n_fail = 0
    if chips_dir.exists():
        fn = _first(_import("ffsim.parse_logs"), NAMES["parse_logs"], "run dir -> RunRecord")
        for chip in ("A", "B", "C", "D"):
            d = chips_dir / f"chip{chip}"
            if not d.is_dir():
                continue
            for run_dir in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")):
                try:
                    rec = _call_flex(fn, str(run_dir), chip=chip)
                except Exception as e:  # noqa: BLE001 - one broken run dir must not stop the build
                    n_fail += 1
                    _p(f"  parse failed {run_dir.name}: {type(e).__name__}: {e}")
                    continue
                got = _as_list(rec) if isinstance(rec, (list, tuple)) else [rec]
                chip_records += [r for r in got if r is not None]
    counts["chip_log"] = len(chip_records)
    if chip_records:
        sources.append(chip_records)
    records = [r for src in sources for r in src]
    ds = _import("ffsim.dataset")
    merge = _first(ds, NAMES["merge"], "merge RunRecords by (chip, run)")
    try:
        merged = _as_list(merge(sources))
    except (TypeError, AttributeError):
        merged = _as_list(merge(records))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = None
    for n in NAMES["save"]:
        if callable(getattr(ds, n, None)):
            save = getattr(ds, n)
            break
    if save is not None:
        _call_flex(save, merged, str(out))
    else:
        with open(out, "w", encoding="utf-8") as f:
            for r in merged:
                f.write((r.to_json() if isinstance(r, RunRecord) else json.dumps(r, sort_keys=True)) + "\n")
    by_chip: Dict[str, int] = {}
    full = 0
    for r in merged:
        chip = getattr(r, "chip", None) if not isinstance(r, dict) else r.get("chip")
        by_chip[str(chip)] = by_chip.get(str(chip), 0) + 1
        if not (getattr(r, "is_screen", False) if not isinstance(r, dict) else r.get("is_screen")):
            full += 1
    _p(f"build-dataset: {len(records)} source records {counts} -> {len(merged)} merged runs -> {out}")
    _p(f"  by chip {by_chip}; full runs {full}, screens {len(merged) - full}; parse failures {n_fail}")
    return 0


def _load_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    runs = _resolve(str(getattr(args, "runs", None) or DEFAULT_RUNS))
    if not runs.exists():
        raise SystemExit(f"{runs} not found: run `python -m ffsim build-dataset` first")
    ds = _import("ffsim.dataset")
    load = _first(ds, NAMES["load"], "load runs.jsonl")
    records = _as_list(load(str(runs)))
    pairs: Any = []
    pp = _resolve(str(getattr(args, "pairs", None) or DEFAULT_PAIRS))
    if pp.exists():
        with open(pp, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        if isinstance(pairs, dict):
            pairs = pairs.get("pairs", pairs)
    else:
        _p(f"note: {pp} not found: quality model fitted without validation pairs")
    uploads: Any = []
    up = _resolve(str(getattr(args, "uploads", None) or DEFAULT_UPLOADS))
    if up.exists():
        uploads = None
        try:
            fn = _first(_import("ffsim.parse_monitor"), NAMES["uploads"], "official uploads table")
            uploads = _as_list(fn(str(up)))
        except SystemExit:
            uploads = None
        if uploads is None:
            uploads = _read_csv_rows(up)
    else:
        _p(f"note: {up} not found: offset model falls back to the contract's six K51-K59 uploads if it can")
    return {"records": records, "pairs": pairs, "uploads": uploads, "uploads_path": str(up) if up.exists() else None,
            "runs_path": str(runs), "pairs_path": str(pp)}


def derive_support(records: Sequence[Any]) -> Dict[str, List[str]]:
    """knob -> distinct values set explicitly in full runs that have a bpb_2m: the knobs that varied
    in the fitted data. Used when the quality model exposes no `support` of its own."""
    seen: Dict[str, set] = {}
    for r in records:
        if getattr(r, "is_screen", False) or getattr(r, "bpb_2m", None) is None:
            continue
        for k, v in (getattr(r, "knobs", None) or {}).items():
            if v is None or v == "":
                continue
            seen.setdefault(str(k), set()).add(str(v))
    return {k: sorted(v) for k, v in seen.items()}


def _print_validation(models: Any, inputs: Optional[Dict[str, Any]]) -> None:
    pairs = inputs["pairs"] if inputs else []
    _print_report("step-time model: leave-one-code-version-out", _report_of(models.steptime, ("loo_report",)))
    _print_report("quality model: paired validation (sign agreement, MAE)",
                  _report_of(models.quality, ("pair_validation", "validation_report"), pairs))
    off = models.offset
    _p("--- offset model (rehearsal -> official)")
    _p(f"  mean {float(getattr(off, 'mean', float('nan'))):+.5f}  sd {float(getattr(off, 'sd', float('nan'))):.5f}")
    from ffsim.search import quality_support
    sup = quality_support(models)
    src = "quality model" if getattr(models.quality, "support", None) else "derived from the fitted records"
    _p(f"--- knob support ({src}): {len(sup)} knobs varied: {', '.join(sorted(sup))[:600]}")


def _anchor_check(models: Any, best: float) -> None:
    from ffsim.simulate import load_recipe, simulate
    from ffsim.report import format_simresult
    rp = EXAMPLES / "recipe-K59.json"
    if not rp.exists():
        return
    r = simulate(load_recipe(rp), *models.as_tuple(), n=2000, rng=0, best_official=best, seed=73)
    _p(f"--- anchor check: K59 seed 73 (measured: C40 0.96092 @2357 steps, official 0.9671; best to beat {best:.4f})")
    _p(format_simresult(r, best))


def fit_models(args: argparse.Namespace, out: Any) -> Any:
    """Fit the step-time, quality and offset models from --runs/--pairs/--uploads, save them to `out`
    (models.pkl) and return (models, inputs). Used by `fit` and by the on-the-fly fit of `_models`."""
    from ffsim.simulate import Models, save_models
    t0 = time.perf_counter()
    inputs = _load_inputs(args)
    records, pairs, uploads = inputs["records"], inputs["pairs"], inputs["uploads"]
    _p(f"fit: {len(records)} records, {len(pairs) if hasattr(pairs, '__len__') else '?'} validation pairs, "
       f"{len(uploads)} uploads")
    steptime = _fit_model("ffsim.steptime", "StepTimeModel", records)
    quality = _fit_model("ffsim.quality", "QualityModel", records, pairs)
    try:
        offset = _fit_model("ffsim.offset", "OffsetModel", uploads)
    except (TypeError, ValueError, KeyError):
        offset = _fit_model("ffsim.offset", "OffsetModel", inputs["uploads_path"] or uploads)
    support = getattr(quality, "support", None) or derive_support(records)
    models = Models(steptime, quality, offset, meta={
        "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_records": len(records),
        "runs": inputs["runs_path"], "pairs": inputs["pairs_path"], "uploads": inputs["uploads_path"] or "",
        "support": dict(support)})
    save_models(models, out)
    _p(f"saved {out} ({time.perf_counter() - t0:.1f} s)")
    return models, inputs


def cmd_fit(args: argparse.Namespace) -> int:
    models, inputs = fit_models(args, args.out)
    _print_validation(models, inputs)
    _anchor_check(models, args.best)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    models = _models(args)
    inputs = getattr(args, "_fit_inputs", None)     # set when _models fitted on the fly
    if inputs is None and Path(args.runs).exists():
        try:
            inputs = _load_inputs(args)
        except SystemExit as e:
            _p(str(e))
    _print_validation(models, inputs)
    _anchor_check(models, args.best)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from ffsim.simulate import load_recipe, simulate, simulate_pair
    from ffsim.report import format_pair, format_simresult
    models = _models(args)
    recipe = load_recipe(args.recipe)
    seed = None if args.seed is None or args.seed < 0 else int(args.seed)
    res = simulate(recipe, *models.as_tuple(), n=args.n, rng=args.rng, best_official=args.best, seed=seed,
                   jitter_sd=args.jitter_sd)
    _p(f"training seed: {seed if seed is not None else 'marginal over seeds'}; rng {args.rng}")
    _p(format_simresult(res, args.best))
    if args.vs:
        other = load_recipe(args.vs)
        pr = simulate_pair(recipe, other, *models.as_tuple(), n=args.n, rng=args.rng, best_official=args.best,
                           seed=seed, jitter_sd=args.jitter_sd)
        _p(format_pair(pr))
    if args.json:
        from ffsim.simulate import simresult_to_dict
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(simresult_to_dict(res), f, indent=1)
        _p(f"wrote {args.json}")
    return 0


def _out_paths(out: Optional[str], space_name: str) -> Dict[str, Path]:
    base = Path(out) if out else SIM_DATA / f"search-{space_name}.md"
    if base.suffix in (".md", ".csv", ".json"):
        stem = base.with_suffix("")
    else:
        stem = base
    return {"md": stem.with_suffix(".md"), "csv": stem.with_suffix(".csv"), "json": stem.with_suffix(".json")}


def cmd_search(args: argparse.Namespace) -> int:
    from ffsim.search import evaluate, load_space
    from ffsim.report import format_summary, write_csv, write_json, write_markdown
    models = _models(args)
    space = load_space(args.space)
    seed = None if args.seed is None or args.seed < 0 else int(args.seed)
    results = evaluate(space, models, n_sims=args.n_sims, gen=args.gen, n_random=args.n_random, rng_seed=args.rng,
                       workers=args.workers, best_official=args.best, seed=seed, jitter_sd=args.jitter_sd)
    paths = _out_paths(args.out, space.name)
    write_json(results, paths["json"])
    write_csv(results, paths["csv"])
    write_markdown(results, paths["md"], top=args.top, seeds=tuple(args.confirm_seeds))
    _p(format_summary(results, top=min(args.top, 15)))
    _p(f"wrote {paths['md']}  {paths['csv']}  {paths['json']}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ffsim.report import format_summary, load_json, write_csv, write_markdown
    results = load_json(args.results)
    paths = _out_paths(args.out, results["meta"].get("space", "search")) if args.out else _out_paths(args.results, "")
    write_markdown(results, paths["md"], top=args.top, seeds=tuple(args.confirm_seeds))
    write_csv(results, paths["csv"])
    _p(format_summary(results, top=min(args.top, 15)))
    _p(f"wrote {paths['md']}  {paths['csv']}")
    return 0


# --- gpu group (route 2) -------------------------------------------------------------------------
GPU_FLEET_SUBS = ("plan", "schedule", "run", "report", "verdict", "from-search", "prices")
GPU_SUBS = GPU_FLEET_SUBS + ("calibrate", "make-train", "launch")
GPU_DIR = Path(__file__).resolve().parent / "gpu"


def gpu_argv(rest: Sequence[str]) -> List[str]:
    """The pass-through args of `gpu <sub>`: a leading `--` (argparse's own separator) dropped and the
    singular `--config X` / `--config=X` accepted for fleet's `--configs`."""
    out: List[str] = []
    for i, a in enumerate(rest):
        if i == 0 and a == "--":
            continue
        if a == "--config":
            out.append("--configs")
        elif a.startswith("--config="):
            out.append("--configs=" + a[len("--config="):])
        else:
            out.append(a)
    return out


def cmd_gpu(args: argparse.Namespace) -> int:
    sub = args.gpu_cmd
    rest = gpu_argv(list(args.rest))
    if sub in GPU_FLEET_SUBS:
        return int(_import("ffsim.gpu.fleet").main([sub] + rest) or 0)
    if sub == "calibrate":
        return int(_import("ffsim.gpu.calibrate").main(rest) or 0)
    if sub == "make-train":
        return int(_import("ffsim.gpu.make_train_gpu").main(rest) or 0)
    if sub == "launch":
        import shutil
        import subprocess
        bash = shutil.which("bash")
        if bash is None:
            raise SystemExit("`gpu launch` needs bash on PATH (Git Bash on Windows): run "
                             f"`bash {GPU_DIR / 'launch.sh'} {' '.join(rest)}` yourself")
        cmd = [bash, str(GPU_DIR / "launch.sh")] + rest
        _p("$ " + " ".join(cmd[1:]))
        return int(subprocess.call(cmd, cwd=str(REPO)))
    raise SystemExit(f"unknown gpu subcommand {sub!r}; one of {', '.join(GPU_SUBS)}")


# --- parser ----------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ffsim", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="pull run dirs off chips C/D (delegates to ffsim.harvest.main)")
    h.add_argument("rest", nargs=argparse.REMAINDER)
    h.set_defaults(fn=cmd_harvest)

    b = sub.add_parser("build-dataset", help="experiments.csv + monitor csv + harvested chip dirs -> runs.jsonl")
    b.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS))
    b.add_argument("--monitor", default=str(DEFAULT_MONITOR))
    b.add_argument("--chips-dir", default=str(SIM_DATA), help="directory holding chipC/<run>/ and chipD/<run>/")
    b.add_argument("--out", default=str(DEFAULT_RUNS))
    b.set_defaults(fn=cmd_build_dataset)

    def fit_args(p: argparse.ArgumentParser) -> None:
        """Inputs of `fit`, and of the on-the-fly fit that runs when `--models` names a missing pickle."""
        p.add_argument("--runs", default=str(DEFAULT_RUNS), help="runs.jsonl from build-dataset")
        p.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="validation-pairs.json")
        p.add_argument("--uploads", default=str(DEFAULT_UPLOADS), help="official-uploads.csv")

    def data_args(p: argparse.ArgumentParser) -> None:
        fit_args(p)
        p.add_argument("--best", type=float, default=0.9655, help="best official score to beat (K60 0.9655)")

    f = sub.add_parser("fit", help="fit the step-time, quality and offset models -> models.pkl, print validation")
    data_args(f)
    f.add_argument("--out", default=str(DEFAULT_MODELS))
    f.set_defaults(fn=cmd_fit)

    v = sub.add_parser("validate", help="print the validation reports of a fitted models.pkl (fitted on the fly when missing)")
    data_args(v)
    v.add_argument("--models", default=str(DEFAULT_MODELS),
                   help="models.pkl (fitted from --runs/--pairs/--uploads and saved there when missing), or 'stub'")
    v.set_defaults(fn=cmd_validate)

    def sim_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--models", default=str(DEFAULT_MODELS),
                       help="models.pkl (fitted from --runs/--pairs/--uploads and saved there when missing), "
                            "or 'stub' for the knob-blind anchor stubs")
        fit_args(p)
        p.add_argument("--seed", type=int, default=None, help="training seed for the quality model (omit = marginal)")
        p.add_argument("--rng", type=int, default=0, help="Monte Carlo rng seed (common random numbers)")
        p.add_argument("--best", type=float, default=0.9655, help="best official score to beat (K60 0.9655)")
        p.add_argument("--jitter-sd", type=float, default=None, help="override the step-time jitter sd (default 0.004)")

    s = sub.add_parser("simulate", help="Monte Carlo prediction of one recipe json")
    s.add_argument("--recipe", required=True)
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--vs", default=None, help="second recipe: also print P(recipe < vs) with common random numbers")
    s.add_argument("--json", default=None, help="write the SimResult here")
    sim_args(s)
    s.set_defaults(fn=cmd_simulate)

    q = sub.add_parser("search", help="score every candidate of a space json; write md/csv/json")
    q.add_argument("--space", required=True)
    q.add_argument("--n-sims", type=int, default=1000)
    q.add_argument("--gen", choices=("local", "oat", "random", "grid"), default="local")
    q.add_argument("--n-random", type=int, default=200, help="candidates for --gen random")
    q.add_argument("--top", type=int, default=20)
    q.add_argument("--out", default=None,
                   help="output stem or .md path for the md/csv/json (default research/sim-data/search-<space>.*, "
                        "the git-tracked evidence directory; give a stem under another directory to keep them out of it)")
    q.add_argument("--workers", type=int, default=1, help="processes over candidates (>1 uses multiprocessing)")
    q.add_argument("--confirm-seeds", type=int, nargs="+", default=[73, 58, 67])
    sim_args(q)
    q.set_defaults(fn=cmd_search)

    r = sub.add_parser("report", help="re-render md/csv from a saved search json")
    r.add_argument("--results", required=True, help="search-<name>.json written by `search`")
    r.add_argument("--out", default=None)
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--confirm-seeds", type=int, nargs="+", default=[73, 58, 67])
    r.set_defaults(fn=cmd_report)

    g = sub.add_parser("gpu", help="route 2, the equal-steps GPU proxy (ffsim/gpu): plan | schedule | run | report | "
                                   "verdict | from-search | prices | calibrate | make-train | launch <args>",
                       description="Delegates to ffsim.gpu.fleet (plan, schedule, run, report, verdict, from-search, "
                                   "prices), ffsim.gpu.calibrate, ffsim.gpu.make_train_gpu or bash ffsim/gpu/launch.sh; "
                                   "every argument after the sub name is passed through (`<sub> --help` shows its own "
                                   "options). Only `launch up` spends money and only `launch down` destroys, and both "
                                   "refuse without --yes.")
    g.add_argument("gpu_cmd", choices=GPU_SUBS, metavar="sub", help="one of " + ", ".join(GPU_SUBS))
    g.add_argument("rest", nargs=argparse.REMAINDER, help="passed through to the delegate")
    g.set_defaults(fn=cmd_gpu)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.fn(args) or 0)
    except SystemExit as e:
        if e.code not in (0, None) and not isinstance(e.code, int):
            _p(f"error: {e.code}")
            return 2
        raise
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
