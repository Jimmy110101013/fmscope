"""End-to-end paper-reproducibility smoke tests.

For every figure and table builder, invoke ``main()`` against bundled
``paper_data`` and assert the output artifacts land on disk. Catches
silent regressions where a refactor breaks a builder's data path.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


BUILDER_MODULES = [
    ("reproduction.builders.fig2_variance", ["fig2_variance_2x2.pdf", "fig2_variance_2x2.png"]),
    ("reproduction.builders.fig3_layerwise", ["fig3_layerwise_probe.pdf", "fig3_layerwise_probe.png"]),
    ("reproduction.builders.fig4a_psd_fooof", ["fig4a_psd_fooof_fit.pdf", "fig4a_psd_fooof_fit.png"]),
    ("reproduction.builders.fig4b_fooof_scatter", ["fig4b_fooof_scatter.pdf", "fig4b_fooof_scatter.png"]),
    (
        "reproduction.builders.fig5_direction",
        [
            "fig5ab_rose_combined.pdf",
            "fig5c_dir_consistency.pdf",
            "fig5d_snr_ratio.pdf",
        ],
    ),
]

TABLE_MODULES = [
    ("reproduction.builders.tab1_master", "paper_tables",
     ["master_results_table.md", "table2_master_performance.tex"]),
]


@pytest.mark.parametrize("module_name,expected", BUILDER_MODULES)
def test_figure_builder_runs(tmp_path: Path, module_name: str, expected: list[str], monkeypatch):
    """Every figure builder must produce its named outputs end-to-end."""
    monkeypatch.setenv("FMSCOPE_OUTPUT_DIR", str(tmp_path))
    module = importlib.import_module(module_name)
    # Some builders run at import time (fig2/fig5) — reload to invoke main()
    importlib.reload(module)
    if hasattr(module, "main"):
        module.main()
    for fname in expected:
        assert (tmp_path / fname).exists(), f"{module_name}: missing {fname}"


@pytest.mark.parametrize("module_name,subdir,expected", TABLE_MODULES)
def test_table_builder_runs(tmp_path: Path, module_name: str, subdir: str, expected: list[str], monkeypatch):
    """Table builders must produce Markdown + LaTeX outputs."""
    monkeypatch.setenv("FMSCOPE_OUTPUT_DIR", str(tmp_path))
    module = importlib.import_module(module_name)
    importlib.reload(module)
    if hasattr(module, "main"):
        module.main()
    for fname in expected:
        assert (tmp_path / fname).exists(), f"{module_name}: missing {fname}"


def test_verdict_matrix_matches_paper_tab3():
    """The verdict matrix must reproduce paper Table 3 outcomes exactly.

    Paper Table 3 (``\\label{tab:verdict_matrix}``) assigns one outcome per
    cell, derived mechanically from four diagnostic signals. Bundled
    paper_data is a snapshot of the same JSONs the paper used, so the
    outcome column should match byte-for-byte. Disagreement means either
    the bundled JSON drifted from the paper snapshot or the rubric
    thresholds need recalibration — investigate before shipping.
    """
    from reproduction.builders.tab3_verdict import build_verdict_matrix

    table = build_verdict_matrix()
    assert len(table) == 4, "expected one row per cell (4 cells total)"
    assert list(table["cell"]) == ["EEGMAT", "ADFTD", "SleepDep", "Stress"], (
        f"Cell order drift: {list(table['cell'])}"
    )

    expected = {
        "EEGMAT":   ("+", "+",       "+", "state signal",     "Cross-subject-aligned"),
        "ADFTD":    ("+", "+ early", "0", "subject axis",     "Label-subject coupled"),
        "SleepDep": ("-", "-",       "-", "subject confound", "Idiosyncratic within-subject"),
        "Stress":   ("-", "- deep",  "0", "subject axis",     "Below linear-probe resolution"),
    }
    for _, row in table.iterrows():
        expect = expected[row["cell"]]
        observed = (row["delta_f_label"], row["layer_probe"], row["c_bar"],
                    row["oneoverf_role"], row["outcome"])
        assert observed == expect, (
            f"{row['cell']} drifted from paper Tab 3.\n"
            f"  expected: {expect}\n"
            f"  observed: {observed}"
        )


def test_tab2_master_md5_matches(tmp_path: Path, monkeypatch):
    """Tab 2 master must produce byte-identical output to the paper reference.

    The bundled paper_data is a snapshot of the same JSONs the paper used,
    so the regenerated Markdown should be byte-identical. If this test
    fails, either the bundled JSONs drifted from the paper snapshot or
    the builder logic changed — investigate before shipping.
    """
    import hashlib

    monkeypatch.setenv("FMSCOPE_OUTPUT_DIR", str(tmp_path))
    from reproduction.builders import tab1_master
    importlib.reload(tab1_master)
    tab1_master.main()
    md = (tmp_path / "master_results_table.md").read_bytes()
    digest = hashlib.md5(md).hexdigest()
    # Pinned at the snapshot taken on 2026-05-21 from
    # ``docs/master_results_table.md`` in the development tree (this is
    # paper Table 2: the per-cell BA matrix).
    expected = "31b3f0fdaa3e1954aa7f9a322ce6a38e"
    assert digest == expected, (
        f"Tab 2 md5 drifted from paper snapshot.\n"
        f"  expected: {expected}\n"
        f"  observed: {digest}\n"
        f"Either re-bundle paper_data or update the pinned digest."
    )
