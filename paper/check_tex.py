"""Static checks for the paper's LaTeX sources (no TeX installation needed).

    python paper/check_tex.py            # exit 0 when clean, 1 on any error; warnings never fail

Checks, starting from paper/main.tex and following every \\input / \\include:
  * every \\input file exists;
  * braces balance in every file (escaped \\{ \\} and comments ignored), and \\begin / \\end environments nest;
  * every \\ref / \\eqref / \\cref / \\Cref / \\autoref / \\pageref target has exactly one \\label, and no label is
    defined twice (unused labels are warnings);
  * every \\cite / \\citep / \\citet / ... key exists in refs.bib (uncited bib entries are listed as information);
  * refs.bib parses: unique keys, balanced braces, ASCII only;
  * every \\includegraphics file exists on the \\graphicspath (with .pdf / .png / .jpg if no extension is given);
  * every \\val{key} is defined by \\ffdef in numbers.tex (unused values are information);
  * no unescaped _ ^ # & outside math, tables and verbatim-like arguments (heuristic), and no stray "??";
  * no control characters in any source file, and \\knob never inside a caption or sectioning title;
  * layout estimates (revision 1): every tabular's width from Computer Modern metrics against the 469.75 pt text
    width (an error above it; ``-v`` lists every table), and every figure's placed vs natural width (a warning when
    it is shrunk below 0.9, which would print 8 pt text below 7.2 pt). Estimates only: read the compiled log too.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.tex"
BIB = HERE / "refs.bib"
NUMBERS = HERE / "numbers.tex"

errors: List[str] = []
warnings: List[str] = []
info: List[str] = []


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        m = re.search(r"(?<!\\)%", line)
        out.append(line[:m.start()] if m else line)
    return "\n".join(out)


def load(path: Path) -> str:
    return strip_comments(path.read_text(encoding="utf-8"))


def resolve_input(name: str) -> Path:
    p = HERE / name
    return p if p.suffix == ".tex" else p.with_suffix(p.suffix + ".tex") if p.suffix else p.with_name(p.name + ".tex")


def collect_files() -> List[Path]:
    seen: List[Path] = []
    stack = [MAIN]
    while stack:
        f = stack.pop(0)
        if f in seen:
            continue
        if not f.exists():
            errors.append(f"missing \\input file: {f.relative_to(HERE).as_posix()}")
            continue
        seen.append(f)
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", load(f)):
            stack.append(resolve_input(m.group(1).strip()))
    return seen


def check_braces(path: Path, text: str) -> None:
    depth = 0
    line = 1
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "\n":
            line += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"{path.name}:{line}: unbalanced '}}'")
                depth = 0
        i += 1
    if depth:
        errors.append(f"{path.name}: {depth} unclosed '{{'")
    envs: List[Tuple[str, int]] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        ln = text.count("\n", 0, m.start()) + 1
        if m.group(1) == "begin":
            envs.append((m.group(2), ln))
        else:
            if not envs:
                errors.append(f"{path.name}:{ln}: \\end{{{m.group(2)}}} without \\begin")
            elif envs[-1][0] != m.group(2):
                errors.append(f"{path.name}:{ln}: \\end{{{m.group(2)}}} closes \\begin{{{envs[-1][0]}}} "
                              f"(line {envs[-1][1]})")
                envs.pop()
            else:
                envs.pop()
    for name, ln in envs:
        errors.append(f"{path.name}:{ln}: \\begin{{{name}}} never closed")


def bib_keys() -> Set[str]:
    if not BIB.exists():
        errors.append("refs.bib is missing")
        return set()
    raw = BIB.read_bytes()
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as e:
        errors.append(f"refs.bib is not pure ASCII (byte {e.start})")
    text = raw.decode("utf-8", errors="replace")
    keys = re.findall(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", text, flags=re.M)
    dup = {k for k in keys if keys.count(k) > 1}
    for k in sorted(dup):
        errors.append(f"refs.bib: duplicate key {k}")
    depth = 0
    for i, c in enumerate(text):
        if c == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                errors.append(f"refs.bib: unbalanced '}}' near line {text.count(chr(10), 0, i) + 1}")
                depth = 0
    if depth:
        errors.append(f"refs.bib: {depth} unclosed '{{'")
    return set(keys)


def number_keys() -> Set[str]:
    if not NUMBERS.exists():
        errors.append("numbers.tex is missing: run python paper/analysis/make_all.py")
        return set()
    return set(re.findall(r"^\\ffdef\{([^}]+)\}", NUMBERS.read_text(encoding="utf-8"), flags=re.M))


def graphics_paths(main_text: str) -> List[Path]:
    m = re.search(r"\\graphicspath\{((?:\{[^}]*\})+)\}", main_text)
    dirs = re.findall(r"\{([^}]*)\}", m.group(1)) if m else []
    return [HERE] + [(HERE / d).resolve() for d in dirs]


_ARG_CMDS = ("url", "href", "knob", "texttt", "label", "ref", "eqref", "cref", "Cref", "autoref", "pageref",
             "input", "include", "includegraphics", "cite", "citep", "citet", "citealp", "citeauthor", "citeyear",
             "val", "ffdef", "graphicspath", "bibliography", "bibliographystyle", "detokenize", "usepackage",
             "newcommand", "DeclareRobustCommand", "renewcommand", "begin", "end", "PackageWarning")


def check_specials(path: Path, text: str) -> None:
    t = text
    # the preamble holds definitions (#1 etc.): only the document body is prose
    if path == MAIN:
        t = t.split("\\begin{document}", 1)[-1]
    # drop display math and tables (& and _ are legal there), then inline math
    t = re.sub(r"\\begin\{(equation|align|gather|multline)(\*?)\}.*?\\end\{\1\2\}", " ", t, flags=re.S)
    t = re.sub(r"\\begin\{(tabular|tabularx|longtable)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"(?<!\\)\$[^$]*(?<!\\)\$", " ", t)
    t = re.sub(r"\\ensuremath\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
    # drop arguments of commands whose arguments may hold _ or # legitimately
    t = re.sub(r"\\(?:" + "|".join(_ARG_CMDS) + r")\*?(?:\[[^\]]*\])*\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
    for ch, name in (("_", "underscore"), ("^", "caret"), ("#", "hash"), ("&", "ampersand")):
        for m in re.finditer(r"(?<!\\)" + re.escape(ch), t):
            ctx = t[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            errors.append(f"{path.name}: unescaped {name} outside math/tables: ...{ctx}...")
    if "??" in t:
        errors.append(f"{path.name}: stray '??' in the text")


# ---------------------------------------------------------------------------------------------- layout estimates
# Revision 1: the paper had never been compiled on the authoring machine (no TeX). These checks estimate, with the
# Computer Modern metrics that ship with matplotlib (cmr10 / cmtt10 / cmss10 / cmmi10), whether every tabular fits the
# text width and at what scale every figure is placed. They are estimates (kerning, ligatures and math spacing are
# approximated); confirm with the compiled log ("Overfull \hbox"), which the paper CI prints.
TEXTWIDTH_PT = 469.75          # letter paper, 1 in margins (geometry), in TeX points
SIZES = {"normalsize": 10.95, "small": 10.0, "footnotesize": 9.0, "scriptsize": 8.0, "tiny": 6.0}
LENGTH_UNITS = {"pt": 1.0, "cm": 28.4528, "mm": 2.84528, "in": 72.27, "em": 10.95, "bp": 1.00375}


def _length(s: str) -> float:
    m = re.match(r"\s*([0-9.]+)\s*(pt|cm|mm|in|em|bp)", s)
    return float(m.group(1)) * LENGTH_UNITS[m.group(2)] if m else 0.0


def _fonts():
    from matplotlib import get_data_path
    from matplotlib.ft2font import FT2Font, LoadFlags
    base = Path(get_data_path()) / "fonts" / "ttf"
    fonts = {k: FT2Font(str(base / f"{n}.ttf")) for k, n in (("rm", "cmr10"), ("tt", "cmtt10"), ("ss", "cmss10"))}

    def width(text: str, size: float, face: str = "rm") -> float:
        # the BaKoMa CM fonts lack a few Unicode glyphs: substitute ones of (nearly) the same width
        text = text.replace("−", "+").replace("–", "--").replace("×", "+").replace("→", "+")
        text = "".join(c if 32 <= ord(c) < 127 else "o" for c in text).strip()
        if not text:
            return 0.0
        f = fonts[face]
        f.set_size(size, 72)
        f.set_text(text, 0.0, flags=LoadFlags.NO_HINTING)
        return f.get_width_height()[0] / 64.0
    return width


def _numbers() -> Dict[str, str]:
    out = {}
    for line in NUMBERS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\ffdef\{([^}]+)\}\{(.*)\}(?:\s*%.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _plain(cell: str, vals: Dict[str, str]) -> List[Tuple[str, str, float]]:
    """A LaTeX cell as [(text, face, relative size)] runs, roughly as TeX would set it."""
    s = cell
    for _ in range(3):
        s = re.sub(r"\\val\{([^}]+)\}", lambda m: vals.get(m.group(1), m.group(1)), s)
    s = s.replace("\\ensuremath{-}", "\u2212").replace("\\ensuremath{+}", "+").replace("{,}", ",")
    s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\#", "#").replace("\\$", "$")
    s = s.replace("~", " ").replace("\\,", " ").replace("\\ ", " ").replace("--", "\u2013")
    runs: List[Tuple[str, str, float]] = []
    pos = 0
    pat = re.compile(r"\\(texttt|upl|textsf|textsuperscript|textbf|emph|checkmark|docnum|knob)\b\s*(\{((?:[^{}]|\{[^{}]*\})*)\})?")
    for m in pat.finditer(s):
        runs.append((s[pos:m.start()], "rm", 1.0))
        cmd, arg = m.group(1), m.group(3) or ""
        if cmd in ("texttt", "knob"):
            runs.append((arg, "tt", 1.0))
        elif cmd in ("upl", "textsf"):
            runs.append((arg, "ss", 1.0))
        elif cmd == "textsuperscript":
            runs.append((arg, "rm", 0.7))
        elif cmd in ("checkmark", "docnum"):
            runs.append(("x", "rm", 0.8))
        else:
            runs.append((arg, "rm", 1.0))
        pos = m.end()
    runs.append((s[pos:], "rm", 1.0))
    out = []
    for text, face, rel in runs:
        text = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\\[a-zA-Z]+|[{}^_]", "", m.group(1)), text)
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", text).replace("{", "").replace("}", "")
        out.append((text, face, rel))
    return out


def _colspec(spec: str) -> List[Tuple[str, float]]:
    """[('l'|'c'|'r'|'p'|'@', width)]; '@' marks an @{} that removes the adjacent \tabcolsep."""
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in "lcr":
            out.append((c, 0.0))
            i += 1
        elif c in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            j = spec.index("}", i)
            out.append(("p", _length(spec[i + 2:j])))
            i = j + 1
        elif c == "@" and spec[i + 1:i + 3] == "{}":
            out.append(("@", 0.0))
            i += 3
        else:
            i += 1
    return out


def check_layout(texts: Dict[Path, str]) -> None:
    try:
        width = _fonts()
    except Exception as exc:  # matplotlib missing: skip, never fail the static check for it
        info.append(f"layout estimates skipped ({exc})")
        return
    vals = _numbers()
    report = []
    for f, t in texts.items():
        for m in re.finditer(r"\\begin\{(tabular|longtable)\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{\1\}", t, re.S):
            before = t[:m.start()]
            size = 10.95
            for name, pt in SIZES.items():
                k = before.rfind("\\" + name)
                if k >= 0 and k > before.rfind("\\begin{table}") and (f.parent.name == "tables" or k > before.rfind("\\end{table}")):
                    size = pt
            if f.parent.name == "tables":
                head = t[:m.start()]
                for name, pt in SIZES.items():
                    if "\\" + name in head:
                        size = pt
            sep = 6.0
            ms = re.search(r"\\setlength\{\\tabcolsep\}\{([^}]+)\}", t[:m.start()][-200:])
            if ms:
                sep = _length(ms.group(1))
            cols = _colspec(m.group(2))
            real = [c for c in cols if c[0] != "@"]
            body = re.sub(r"\\(toprule|midrule|bottomrule|endfirsthead|endhead|endlastfoot)", "", m.group(3))
            body = re.sub(r"\\caption\{.*?\}\\label\{[^}]*\}", "", body, flags=re.S)
            nat = [0.0] * len(real)
            for row in re.split(r"\\\\", body):
                if "\\multicolumn" in row:
                    continue
                cells = re.split(r"(?<!\\)&", row)
                for j, cell in enumerate(cells[:len(real)]):
                    if real[j][0] == "p":
                        continue
                    w = sum(width(txt, size * rel, face) for txt, face, rel in _plain(cell.strip(), vals))
                    nat[j] = max(nat[j], w)
            total = 0.0
            for j, (kind, w) in enumerate(real):
                total += w if kind == "p" else nat[j]
            # \tabcolsep on both sides of every column, minus the sides an @{} removes
            total += 2 * sep * len(real)
            total -= sep * sum(1 for c in cols if c[0] == "@")
            name = f.relative_to(HERE).as_posix()
            report.append((total, name))
            if total > TEXTWIDTH_PT + 1.0:
                errors.append(f"{name}: estimated table width {total:.0f} pt > text width {TEXTWIDTH_PT} pt")
    if "-v" in sys.argv:
        for total, name in sorted(report, reverse=True):
            print(f"  table {name}: {total:.0f} pt")
    widest = max(report) if report else (0, "")
    print(f"tables: {len(report)} measured; widest estimate {widest[0]:.0f} pt ({widest[1]}) of {TEXTWIDTH_PT} pt")
    # figures: placed width vs natural (MediaBox) width
    main_text = texts[MAIN]
    dirs = graphics_paths(main_text)
    for f, t in texts.items():
        for m in re.finditer(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}", t):
            opts, g = m.group(1) or "", m.group(2)
            cand = [d / g for d in dirs] + [d / (g + ".pdf") for d in dirs]
            pdf = next((c for c in cand if c.exists() and c.suffix == ".pdf"), None)
            if pdf is None:
                continue
            mb = re.search(rb"/MediaBox\s*\[\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*\]", pdf.read_bytes())
            if not mb:
                continue
            nat = (float(mb.group(3)) - float(mb.group(1))) * 72.27 / 72.0
            mw = re.search(r"width\s*=\s*([0-9.]*)\\(linewidth|textwidth)", opts)
            placed = (float(mw.group(1) or 1.0) * TEXTWIDTH_PT) if mw else nat
            scale = placed / nat
            if nat > TEXTWIDTH_PT + 1.0 and not mw:
                errors.append(f"{f.name}: figure {g} is {nat:.0f} pt wide at natural size > text width")
            if scale < 0.9:
                warnings.append(f"{f.name}: figure {g} placed at {scale:.2f} of its natural size: 8 pt text prints at "
                                f"{8 * scale:.1f} pt")
            print(f"figure {g}: natural {nat:.0f} pt, placed {placed:.0f} pt, scale {scale:.2f}")


def check_moving_args(texts: Dict[Path, str]) -> None:
    """\\knob (a url-style command) must stay out of moving arguments: captions and sectioning titles."""
    for f, t in texts.items():
        for m in re.finditer(r"\\(caption|section|subsection|paragraph)\*?\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", t):
            if "\\knob" in m.group(2):
                errors.append(f"{f.name}: \\knob inside \\{m.group(1)}{{...}} (a moving argument)")
            if m.group(1) != "caption" and re.search(r"\\(ffsim|texttt)\b", m.group(2)) and "texorpdfstring" not in m.group(2):
                warnings.append(f"{f.name}: \\{m.group(1)} title uses \\texttt without \\texorpdfstring")


def check_control_chars(files: List[Path]) -> None:
    for f in files:
        raw = f.read_text(encoding="utf-8")
        bad = [i for i, c in enumerate(raw) if ord(c) < 32 and c not in "\n\t\r"]
        if bad:
            errors.append(f"{f.relative_to(HERE).as_posix()}: control character at offset {bad[0]}")


def main() -> int:
    files = collect_files()
    texts: Dict[Path, str] = {f: load(f) for f in files}
    for f, t in texts.items():
        check_braces(f, t)
        check_specials(f, t)

    labels: Dict[str, List[str]] = {}
    refs: Set[str] = set()
    cites: Set[str] = set()
    vals: Set[str] = set()
    graphics: List[Tuple[Path, str]] = []
    for f, t in texts.items():
        for m in re.finditer(r"\\label\{([^}]+)\}", t):
            labels.setdefault(m.group(1).strip(), []).append(f.name)
        for m in re.finditer(r"\\(?:ref|eqref|pageref|autoref|nameref|cref|Cref|crefrange|Crefrange)\*?\{([^}]+)\}", t):
            refs.update(k.strip() for k in m.group(1).split(",") if k.strip())
        for m in re.finditer(r"\\cite(?:p|t|alp|alt|author|year|yearpar)?\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", t):
            cites.update(k.strip() for k in m.group(1).split(",") if k.strip())
        vals.update(m.group(1) for m in re.finditer(r"\\val\{([^}]+)\}", t))
        graphics += [(f, m.group(1)) for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)]

    for k, where in labels.items():
        if len(where) > 1:
            errors.append(f"label defined {len(where)} times: {k} ({', '.join(where)})")
    for k in sorted(refs - set(labels)):
        errors.append(f"reference to an undefined label: {k}")
    for k in sorted(set(labels) - refs):
        warnings.append(f"label never referenced: {k}")

    keys = bib_keys()
    for k in sorted(cites - keys):
        errors.append(f"citation key not in refs.bib: {k}")
    unused_bib = sorted(keys - cites)

    defined = number_keys()
    for k in sorted(vals - defined):
        errors.append(f"\\val{{{k}}} is not defined in numbers.tex")
    unused_vals = sorted(defined - vals)

    dirs = graphics_paths(texts[MAIN])
    for f, g in graphics:
        cands = [d / g for d in dirs]
        if not Path(g).suffix:
            cands = [d / (g + ext) for d in dirs for ext in (".pdf", ".png", ".jpg")]
        if not any(c.exists() for c in cands):
            errors.append(f"{f.name}: figure not found on the graphicspath: {g}")

    check_moving_args(texts)
    check_control_chars(files)
    check_layout(texts)

    print(f"files checked: {len(files)} ({', '.join(f.relative_to(HERE).as_posix() for f in files)})")
    print(f"labels: {len(labels)}, references: {len(refs)}, citations: {len(cites)} distinct keys "
          f"({len(keys)} in refs.bib, {len(unused_bib)} not cited), figures: {len(graphics)}, "
          f"values used: {len(vals)} of {len(defined)} defined")
    if unused_bib:
        info.append("bib entries not cited: " + ", ".join(unused_bib))
    if unused_vals:
        info.append(f"{len(unused_vals)} numbers.tex values not used in the text (tables carry their own copies)")
    for w in warnings:
        print("warning:", w)
    for i in info:
        print("info:", i)
    for e in errors:
        print("ERROR:", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
