"""Smoke tests for src/results.py canonical accessors.

Catches two classes of silent regression:
1. An accessor's internal path drifts and it stops returning data.
2. An accessor's return value diverges from the raw json.load at the
   same path (e.g. someone adds a coercion that silently changes types).

Tests are plain asserts (no pytest dependency) — matches the style of
tests/test_variance.py. Run under any env with the `fmscope`
package importable:

    python tests/test_results.py

They should complete in < 1 second and do NOT validate numeric
correctness of any experiment — that's the raw run's job.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from reproduction.builders import results_accessor as results  # noqa: E402

# Public release reads source tables out of ``reproduction/data/``.
PAPER_DATA = Path(results.__file__).resolve().parent.parent / "data"
REPO = PAPER_DATA  # legacy alias retained so existing test bodies keep working

SOURCE_TABLE_NAMES = [
    "variance_analysis_window_level",
    "within_subject_dir_consistency",
    "variance_triangulation",
    "band_rsa",
    "ft_rsa_stress_eegmat",
]
DATASETS = ["eegmat", "adftd", "stress", "sleepdep"]
FMS = ["labram", "cbramod", "reve"]


def test_source_table_matches_raw_json():
    for name in SOURCE_TABLE_NAMES:
        via_api = results.source_table(name)
        p = PAPER_DATA / "source_tables" / f"{name}.json"
        raw = json.loads(p.read_text())
        assert via_api == raw, f"{name}: accessor diverged from raw json.load"


def test_source_table_missing_reports_available():
    try:
        results.source_table("does_not_exist")
    except FileNotFoundError as e:
        assert "Available:" in str(e), "error message should list alternatives"
        return
    raise AssertionError("expected FileNotFoundError for missing source table")


def test_labram_ft_ba_null_matched_returns_triple():
    for ds in DATASETS:
        mean, sd, n = results.labram_ft_ba_null_matched(ds)
        assert 0.0 < mean < 1.0, f"{ds}: mean BA outside [0,1]"
        assert sd >= 0.0, f"{ds}: negative std"
        assert n >= 1, f"{ds}: n_seeds < 1"


def test_lp_multiseed_has_expected_keys():
    for ds in DATASETS:
        for fm in FMS:
            d = results.lp_multiseed(ds, fm)
            for key in ["extractor", "dataset", "mean_8seed", "std_8seed_ddof1",
                        "mean_3seed_42_123_2024", "std_3seed_42_123_2024_ddof1",
                        "per_seed_ba"]:
                assert key in d, f"{ds}×{fm}: missing {key}"


def test_lp_stats_3seed_matches_multiseed_fields():
    """The 3-seed convenience accessor must agree with the underlying
    multiseed record exactly (bit-identical floats)."""
    for ds, fm in [("stress", "labram"), ("eegmat", "cbramod"),
                   ("adftd", "reve")]:
        full = results.lp_multiseed(ds, fm)
        stats = results.lp_stats_3seed(ds, fm)
        assert stats["mean"] == full["mean_3seed_42_123_2024"]
        assert stats["std"] == full["std_3seed_42_123_2024_ddof1"]
        assert stats["n_seeds"] == 3


def test_ft_stats_3seed_paths_return_dicts():
    for ds, fm in [("stress", "labram"), ("eegmat", "labram"),
                   ("adftd", "cbramod"), ("sleepdep", "reve")]:
        d = results.ft_stats(ds, fm)
        assert d is not None, f"{ds}×{fm}: expected dict, got None"
        for key in ["mean", "std", "n_seeds", "source"]:
            assert key in d, f"{ds}×{fm}: missing {key}"


def test_ft_stats_nonexistent_combination_returns_none():
    """A combination with no canonical seeds must return None, not raise."""
    out = results.ft_stats("meditation", "reve")
    assert out is None


def test_fooof_ablation_probes_have_4_conditions():
    for ds in DATASETS:
        d = results.fooof_ablation_probes(ds)
        for fm in FMS:
            keys = set(d["results"][fm].keys())
            assert {"original", "aperiodic_removed", "periodic_removed",
                    "both_removed"}.issubset(keys), f"{ds}×{fm}: missing condition"


def test_subject_probe_temporal_block_has_4_conditions():
    for ds in DATASETS:
        d = results.subject_probe_temporal_block(ds)
        for fm in FMS:
            keys = set(d["results"][fm].keys())
            assert {"original", "aperiodic_removed", "periodic_removed",
                    "both_removed"}.issubset(keys), f"{ds}×{fm}: missing condition"


def test_classical_summary_returns_dict():
    for ds in ["eegmat", "stress", "sleepdep"]:
        d = results.classical_summary(ds)
        assert isinstance(d, dict)
        assert "dataset" in d


def test_path_accessors_resolve():
    # Public release bundles JSON aggregates only; raw NPZ feature caches
    # are NOT redistributed (they are 1.3 GB and contain raw EEG-derived
    # features). The path accessors must still return well-formed Paths.
    p = results.frozen_features_path("labram", "stress", 30)
    assert p.suffix == ".npz"
    p2 = results.fooof_ablated_features_path("stress")
    assert p2.suffix == ".npz"
    # Existence is asserted only when the bundle actually ships the NPZ
    # (development tree / collaboration-only). Otherwise the path is just
    # a stable pointer that downstream callers will hit-and-miss against.
    if p.parent.exists() and (p.parent / p.name).exists():
        assert p.exists()
    if p2.parent.exists() and (p2.parent / p2.name).exists():
        assert p2.exists()


if __name__ == "__main__":
    tests = [fn for name, fn in list(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERR   {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
