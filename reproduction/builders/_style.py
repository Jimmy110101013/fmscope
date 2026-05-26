"""Shared matplotlib style for JNE/IOP submission.

Import at the top of every build_fig*.py:

    from _jne_style import apply_jne_style, JNE_SINGLE_COL_CM, JNE_DOUBLE_COL_CM
    apply_jne_style()

This sets:
- pdf.fonttype = 42 / ps.fonttype = 42 (TrueType — fixes Type 3 prepress rejection)
- font.family = sans-serif with Liberation Sans (metric-compatible with Helvetica;
  IOP's font rule is Times/Helvetica/Courier/Symbol at final size)
- 8 pt floor for axis ticks / 8-9 pt for labels (JNE minimum at final figure size)

JNE final figure widths:
- single-column: 8.5 cm
- double-column (full page width): 15 cm
Design each figure at one of these widths so labels printed at native size
are already at the JNE minimum (8 pt).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path

JNE_SINGLE_COL_CM = 8.5
JNE_DOUBLE_COL_CM = 15.0
CM = 1.0 / 2.54

_FONT_DIR = Path.home() / ".local/share/fonts/jne"


def _register_liberation_sans() -> str:
    """Register Liberation Sans with matplotlib if installed in ~/.local/share/fonts/jne.

    Returns the resolved font family name. Falls back to DejaVu Sans if
    Liberation isn't available.
    """
    if _FONT_DIR.is_dir():
        for ttf in _FONT_DIR.glob("LiberationSans*.ttf"):
            fm.fontManager.addfont(str(ttf))
    available = {f.name for f in fm.fontManager.ttflist}
    if "Liberation Sans" in available:
        return "Liberation Sans"
    return "DejaVu Sans"


def apply_jne_style() -> None:
    family = _register_liberation_sans()
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
