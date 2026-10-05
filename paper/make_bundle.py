"""Make a self-contained source bundle for Overleaf or arXiv.

    python paper/make_bundle.py                    # writes paper/build/paper-source/ and paper/build/paper-source.zip
    python paper/make_bundle.py --bbl              # the same, with paper/main.bbl (from a local build) for arXiv
    python paper/make_bundle.py --out DIR --bbl FILE   # writes DIR/ and DIR.zip, with FILE as DIR/main.bbl

The bundle holds main.tex, numbers.tex, refs.bib, sections/, tables/ and every figure the paper uses, in one flat
``figures/`` folder, with ``\\graphicspath`` rewritten to it. Upload the zip to Overleaf ("New project -> Upload
project"). For arXiv, ship the ``main.bbl`` of a compiled build, so that arXiv uses it instead of running BibTeX
itself (it must be named after main.tex): ``--bbl`` copies it next to main.tex. CI (.github/workflows/paper.yml)
builds the arXiv bundle this way, compiles it without BibTeX and uploads the zip as the ``arxiv-source`` artifact.

The output folder is deleted and rebuilt on every run, so the script refuses any folder that might hold something it
did not write: the paper folder or one of its parents, a folder inside paper/ but outside paper/build/, and, anywhere
else, a non-empty folder without the ``.make_bundle`` marker that every run leaves in its folder. It replaces
``DIR.zip`` only when that file is an earlier bundle. paper/build/ is git-ignored.
"""
from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT = BUILD / "paper-source"
MARKER = ".make_bundle"   # left in every output folder (never zipped); marks the folder as this script's own


def _show(p: Path) -> str:
    try:
        return p.relative_to(HERE.parent).as_posix()
    except ValueError:
        return str(p)


def _check_out(out: Path) -> None:
    """Exit unless ``out`` can be deleted without touching anything this script did not write."""
    if out == HERE or out in HERE.parents or out.parent == out:
        raise SystemExit(f"--out {_show(out)} is the paper folder, one of its parents or a drive root; "
                         "choose a folder of its own")
    if HERE in out.parents and BUILD not in out.parents:
        raise SystemExit(f"--out {_show(out)} is inside paper/ but not under paper/build/")
    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"--out {_show(out)} exists and is not a folder")
        if any(out.iterdir()) and not (out / MARKER).is_file() and BUILD not in out.parents:
            raise SystemExit(f"refusing to delete {_show(out)}: it is not empty and was not written by make_bundle.py "
                             f"(no {MARKER} file); pass a new or empty folder")


def _check_zip(zpath: Path) -> None:
    """Exit unless ``zpath`` is absent or an earlier bundle (a zip with main.tex at its top level)."""
    if not zpath.exists():
        return
    try:
        with zipfile.ZipFile(zpath) as z:
            if "main.tex" in z.namelist():
                return
    except (OSError, zipfile.BadZipFile):
        pass
    raise SystemExit(f"refusing to overwrite {_show(zpath)}: it is not an earlier bundle")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make the paper's flat source bundle (folder and zip).")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output folder, rebuilt on every run; the zip is written next to it as <folder>.zip "
                         "(default: paper/build/paper-source)")
    ap.add_argument("--bbl", type=Path, nargs="?", const=HERE / "main.bbl", default=None,
                    help="copy this compiled bibliography into the bundle as main.bbl, as arXiv needs "
                         "(no value: paper/main.bbl)")
    args = ap.parse_args(argv)

    out = args.out.resolve()
    _check_out(out)
    zpath = out.with_name(out.name + ".zip")
    _check_zip(zpath)
    bbl = None
    if args.bbl is not None:
        bbl_path = args.bbl.resolve()
        if not bbl_path.is_file():
            raise SystemExit(f"--bbl {_show(bbl_path)} not found; compile the paper first (latexmk -pdf main.tex)")
        bbl = bbl_path.read_bytes()   # read before the output folder is cleared, in case it lives there
        if b"thebibliography" not in bbl:
            raise SystemExit(f"--bbl {_show(bbl_path)} holds no thebibliography environment")

    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    (out / MARKER).write_text("Written by paper/make_bundle.py, which deletes and rebuilds this folder.\n",
                              encoding="utf-8")
    for name in ("numbers.tex", "refs.bib"):
        shutil.copy2(HERE / name, out / name)
    for d in ("sections", "tables"):
        shutil.copytree(HERE / d, out / d)
    if bbl is not None:
        (out / "main.bbl").write_bytes(bbl)
    main_tex = (HERE / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\graphicspath\{((?:\{[^}]*\})+)\}", main_tex)
    dirs = [HERE / d for d in re.findall(r"\{([^}]*)\}", m.group(1))] if m else [HERE]
    used = set()
    for tex in [HERE / "main.tex"] + list((HERE / "sections").glob("*.tex")):
        used.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8")))
    for g in sorted(used):
        src = next((d / g for d in dirs if (d / g).exists()), None)
        if src is None:
            raise SystemExit(f"figure not found: {g}")
        shutil.copy2(src, out / "figures" / Path(g).name)
    main_tex = main_tex.replace(m.group(0), "\\graphicspath{{figures/}}") if m else main_tex
    (out / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != MARKER:
                z.write(p, p.relative_to(out).as_posix())
    extra = " and main.bbl" if bbl is not None else " (no main.bbl: Overleaf only)"
    print(f"{_show(out)}: {len(used)} figures{extra}; {zpath.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
