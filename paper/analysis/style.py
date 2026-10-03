"""The figure style of docs/figures/make_figures.py, reused so the paper's own figures match it.

Importing make_figures applies its matplotlib rcParams (Agg backend, fonts, colours); nothing is drawn or written.
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import ROOT

sys.path.insert(0, str(ROOT / "docs" / "figures"))
import make_figures as _mf  # noqa: E402

plt = _mf.plt
BLUE, ORANGE, AQUA, RED, GREY = _mf.BLUE, _mf.ORANGE, _mf.AQUA, _mf.RED, _mf.GREY
INK, INK_2, INK_3, GRID, SURFACE = _mf.INK, _mf.INK_2, _mf.INK_3, _mf.GRID, _mf.SURFACE
BLUE_WASH, TIE_WASH = _mf.BLUE_WASH, _mf.TIE_WASH

# The paper's figures are drawn at their final printed size (TEXTWIDTH = 6.5 in for letter paper with 1 in margins), so
# the sizes below are the printed sizes: 8 pt text, close to the 9-10 pt caption size.
TEXTWIDTH = 6.5
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.titlepad": 4, "lines.markersize": 4,
})
# Marker per chip, so that chip identity survives grayscale printing (colour is redundant with the marker).
CHIP_MARKER = {"A": "s", "B": "D", "C": "o", "D": "v", "E": "^", "G": "*", "H": "X", "F": "p"}
CHIP_AREA = {"*": 2.4, "X": 1.3, "^": 1.3, "v": 1.3}   # visual-size correction per marker (scatter area factor)


def chip_s(chip: str, base: float) -> float:
    """Scatter area for a chip's marker, so that every marker reads at about the same size."""
    return base * CHIP_AREA.get(CHIP_MARKER.get(chip, "o"), 1.0)


def save_pdf(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None})
    import os
    preview = os.environ.get("FF_PREVIEW_DIR")   # optional PNG previews for proofreading (not used by the paper)
    if preview:
        fig.savefig(Path(preview) / (path.stem + ".png"), dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path
