"""Make a self-contained source bundle for Overleaf or arXiv.

    python paper/make_bundle.py        # writes paper/build/paper-source/ and paper/build/paper-source.zip

The bundle holds main.tex, numbers.tex, refs.bib, sections/, tables/ and every figure the paper uses, in one flat
``figures/`` folder, with ``\\graphicspath`` rewritten to it. Upload the zip to Overleaf ("New project -> Upload
project"). For arXiv, compile once and add the generated main.bbl to the folder before zipping, since arXiv
does not run BibTeX on uploaded .bib files in every configuration. paper/build/ is git-ignored.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "paper-source"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    for name in ("numbers.tex", "refs.bib"):
        shutil.copy2(HERE / name, OUT / name)
    for d in ("sections", "tables"):
        shutil.copytree(HERE / d, OUT / d)
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
        shutil.copy2(src, OUT / "figures" / Path(g).name)
    main_tex = main_tex.replace(m.group(0), "\\graphicspath{{figures/}}") if m else main_tex
    (OUT / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")
    zpath = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    print(f"{OUT.relative_to(HERE.parent).as_posix()}: {len(used)} figures; {zpath.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
