"""Output-directory + bundled-data helpers for the paper-reproduction scripts.

All paper figure / table builders should resolve filesystem paths
through these helpers so the public-release install location stays
self-contained.

* :func:`output_dir` — where to write figures and tables. Defaults to
  ``./paper_figures`` (or ``./paper_tables`` for tables) relative to the
  current working directory; override with the ``FMSCOPE_OUTPUT_DIR``
  environment variable.
* :func:`paper_data_path` — resolve a path inside the bundled
  ``reproduction/data/`` package.
"""

from __future__ import annotations

import os
from pathlib import Path


_DATA = Path(__file__).resolve().parent.parent / "data"


def paper_data_path(*parts: str) -> Path:
    """Resolve ``parts`` against ``reproduction/data/``."""
    return _DATA.joinpath(*parts)


def output_dir(subdir: str = "paper_figures") -> Path:
    """Resolve the output directory for figures/tables.

    Reads ``FMSCOPE_OUTPUT_DIR`` if set; otherwise uses ``./<subdir>``
    relative to the current working directory. The directory is created
    on first call.
    """
    override = os.environ.get("FMSCOPE_OUTPUT_DIR")
    base = Path(override) if override else Path.cwd() / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base
