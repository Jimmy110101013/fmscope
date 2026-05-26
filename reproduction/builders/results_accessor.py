"""Canonical accessors for paper-relevant results bundled in ``paper_data/``.

Paper figure/table builders import from this module so that any change
in the underlying file layout is invisible to callers. Numbers are
loaded from the JSONs bundled at install time under
``reproduction/data/``; the layout mirrors the original
``results/final/`` tree on the development side.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

# Resolve relative to this file; ``data/`` is a sibling of ``builders/``
# inside the top-level ``reproduction/`` directory.
_PAPER_DATA = Path(__file__).resolve().parents[1] / "data"

# Retained as module-level aliases so existing call sites keep working.
# The ``STUDIES`` alias points at the same bundle — the public release
# ships only finalized JSONs, no scratchpad branch.
FINAL = _PAPER_DATA
STUDIES = _PAPER_DATA


# ---------------------------------------------------------------------------
# Source tables (aggregated JSONs at results/final/source_tables/)
# ---------------------------------------------------------------------------

def source_table(name: str) -> dict:
    """Load ``results/final/source_tables/<name>.json``.

    Raises FileNotFoundError with a discovery-friendly message if the table
    doesn't exist.
    """
    p = FINAL / "source_tables" / f"{name}.json"
    if not p.exists():
        available = sorted(p.name for p in (FINAL / "source_tables").glob("*.json"))
        raise FileNotFoundError(
            f"No source table at {p}. Available: {available}"
        )
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# LaBraM FT balanced accuracy (canonical 3-seed refresh)
# ---------------------------------------------------------------------------

def labram_ft_ba_null_matched(dataset: str) -> tuple[float, float, int]:
    """Return (mean, std, n) of LaBraM FT BA under the canonical refresh,
    which shares the same training recipe as the exp27 perm-null chain
    (lr=5e-4, encoder_lr_scale=1.0, llrd=0.65, head=0, norm=none, ep=200).

    Single source of truth: ``results/final/<ds>/ft/labram/seed{42,123,2024}/
    summary.json``. Returns sample std (ddof=1).
    """
    return _ft_ba_canonical(dataset, "labram")


def _ft_ba_canonical(dataset: str, fm: str,
                     seeds=(42, 123, 2024)) -> tuple[float, float, int]:
    """Read 3-seed FT BA from the canonical refresh layout.

    Path: ``results/final/<ds>/ft/<fm>/seed<S>/summary.json``.
    Returns ``(mean, sample_std_ddof1, n)``. Errors loudly if any seed is
    missing — the canonical layout is supposed to be complete.
    """
    ds = dataset.lower()
    bas = []
    for s in seeds:
        p = FINAL / ds / "ft" / fm / f"seed{s}" / "summary.json"
        if not p.exists():
            raise FileNotFoundError(f"Canonical FT seed missing: {p}")
        bas.append(float(json.loads(p.read_text())["subject_bal_acc"]))
    sd = statistics.stdev(bas) if len(bas) > 1 else 0.0
    return statistics.mean(bas), sd, len(bas)


# ---------------------------------------------------------------------------
# Linear probing (per-window LP, 8-seed output from train_lp.py)
# ---------------------------------------------------------------------------

def lp_multiseed(dataset: str, fm: str) -> dict:
    """Return the per-window LP multiseed JSON for (dataset, fm).

    Path: ``results/final/<ds>/lp/<fm>.json``.
    """
    p = FINAL / dataset.lower() / "lp" / f"{fm}.json"
    if not p.exists():
        raise FileNotFoundError(f"No LP multiseed at {p}")
    return json.loads(p.read_text())


def lp_stats_3seed(dataset: str, fm: str) -> dict:
    """3-seed (seeds 42/123/2024) LP statistics matching the paper's Table 1.

    Returns ``{mean, std, n_seeds, source}``. Std is sample std (ddof=1).
    """
    d = lp_multiseed(dataset, fm)
    return {
        "mean": float(d["mean_3seed_42_123_2024"]),
        "std":  float(d["std_3seed_42_123_2024_ddof1"]),
        "n_seeds": 3,
        "source": f"results/final/{dataset.lower()}/lp/{fm}.json",
    }


# ---------------------------------------------------------------------------
# Fine-tuning per-seed outputs (scattered across exp dirs — for now)
# ---------------------------------------------------------------------------

def ft_stats(dataset: str, fm: str, *, seeds=(42, 123, 2024)) -> dict | None:
    """Return ``{mean, std, n_seeds, source}`` for FT balanced accuracy from
    the canonical refresh layout.

    Path: ``results/final/<ds>/ft/<fm>/seed<S>/summary.json``.
    std is sample std (ddof=1). Returns ``None`` if no seeds are present.
    """
    ds = dataset.lower()
    vals, paths = [], []
    for s in seeds:
        p = FINAL / ds / "ft" / fm / f"seed{s}" / "summary.json"
        if p.exists():
            vals.append(float(json.loads(p.read_text())["subject_bal_acc"]))
            paths.append(str(p.relative_to(_PAPER_DATA)))
    if not vals:
        return None
    return {
        "mean":   statistics.mean(vals),
        "std":    statistics.stdev(vals) if len(vals) > 1 else None,
        "n_seeds": len(vals),
        "source": f"results/final/{ds}/ft/{fm}/seed{{{','.join(str(s) for s in seeds)}}}/summary.json",
    }


# ---------------------------------------------------------------------------
# FOOOF ablation probes (state + subject-ID), Fig 5
# ---------------------------------------------------------------------------

def fooof_ablation_probes(dataset: str) -> dict:
    """Return the FOOOF probes JSON (state-probe variant, Fig 5b).

    Reads from ``results/final/<dataset>/fooof_ablation/probes.json`` if
    present (the snapshotted form with ``provenance``); falls back to
    ``results/studies/fooof_ablation/<dataset>_probes.json`` otherwise.
    Both shapes have the same ``results: {<fm>: {<condition>: ...}}``
    payload at top level.
    """
    final = FINAL / dataset.lower() / "fooof_ablation" / "probes.json"
    if final.exists():
        return json.loads(final.read_text())
    studies = STUDIES / "fooof_ablation" / f"{dataset.lower()}_probes.json"
    if not studies.exists():
        raise FileNotFoundError(f"No FOOOF probes at {final} or {studies}")
    return json.loads(studies.read_text())


def subject_probe_temporal_block(dataset: str) -> dict:
    """Return the temporal-block subject-ID probe JSON (Fig 5b subject axis).

    Reads from bundled
    ``paper_data/<dataset>/subject_probe_temporal_block/probes.json``.
    """
    final = (FINAL / dataset.lower() / "subject_probe_temporal_block"
             / "probes.json")
    if final.exists():
        return json.loads(final.read_text())
    raise FileNotFoundError(f"No temporal-block probe at {final}")


def classical_summary(dataset: str) -> dict:
    """Return the classical-baseline summary.

    Schema: per-method (logreg/svm/rf/xgb) per-seed BA, mean, std.
    """
    final = FINAL / dataset.lower() / "classical" / "summary.json"
    if not final.exists():
        raise FileNotFoundError(f"No classical summary at {final}")
    return json.loads(final.read_text())


# ---------------------------------------------------------------------------
# Feature cache paths (loader is caller's job — features are large)
# ---------------------------------------------------------------------------

def frozen_features_path(fm: str, dataset: str, channels: int) -> Path:
    """Path to ``results/features_cache/frozen_<fm>_<dataset>_<channels>ch.npz``.

    Returns the path; caller invokes ``np.load(...)``. Kept as a path-only
    accessor because feature arrays are large and loading is the caller's
    responsibility (and they often don't need every key).
    """
    return _PAPER_DATA / f"features_cache/frozen_{fm}_{dataset}_{channels}ch.npz"


def fooof_ablated_features_path(dataset: str, *, w5: bool = False) -> Path:
    """Path to FOOOF-ablated frozen features.

    ``w5=True`` returns the 5 s window variant
    (``results/features_cache/fooof_ablation/<dataset>_norm_none_w5.npz``);
    default returns the standard window length
    (``results/features_cache/fooof_ablation/<dataset>_norm_none.npz``).
    """
    suffix = "_w5" if w5 else ""
    return _PAPER_DATA / f"features_cache/fooof_ablation/{dataset}_norm_none{suffix}.npz"


def within_subject_dc_snr(dataset: str, fm: str, source: str = "frozen") -> dict:
    """Per-(dataset, fm) DC + SNR + magnitude statistics.

    Reads ``results/final/source_tables/within_subject_dir_consistency.json``
    and returns the entry for ``(dataset, fm)`` under ``source``.

    Source values:
      - ``"frozen"``: deterministic single value
      - ``"ft"``: legacy single-seed FT (seed 42)
      - ``"ft_multiseed"``: 3-seed aggregate with mean/std for dc and snr,
        plus the per-seed lists

    Schema for frozen / ft single:
      ``{"dir_consistency": float, "n_subj": int,
         "mean_diff_mag": float, "std_diff_mag": float, "snr": float}``

    Schema for ft_multiseed:
      ``{"seeds": [int...], "dir_consistency": [float...], "snr": [float...],
         "mean_diff_mag": [...], "std_diff_mag": [...],
         "dc_mean": float, "dc_std": float, "snr_mean": float, "snr_std": float,
         "mean_diff_mag_mean": float, "std_diff_mag_mean": float,
         "n_subj": int}``

    Scope: defined for within-subject paired cells only (eegmat, sleepdep).
    """
    table = source_table("within_subject_dir_consistency")
    if dataset not in table:
        raise KeyError(f"dataset '{dataset}' not in within_subject_dir_consistency "
                       f"(available: {sorted(table.keys())})")
    if source not in table[dataset]:
        raise KeyError(f"source '{source}' not in {dataset} "
                       f"(available: {sorted(table[dataset].keys())})")
    if fm not in table[dataset][source]:
        raise KeyError(f"fm '{fm}' not in {dataset}/{source} "
                       f"(available: {sorted(table[dataset][source].keys())})")
    return table[dataset][source][fm]




_LAYERWISE_PROBES_PATH = _PAPER_DATA / "layerwise_probe/probes.json"


def layerwise_probes(dataset: str | None = None,
                     fm: str | None = None) -> dict:
    """Per-(dataset, fm) layer-wise subject + label probe sweep.

    Source: bundled ``paper_data/layerwise_probe/probes.json``.

    With both ``dataset`` and ``fm`` provided, returns the per-FM block:
      ``{"n_subjects": int, "n_recordings": int, "embed_dim": int,
         "per_depth": [{"depth_label": int, "depth_fraction": float,
                        "subject_ba": float, "subject_chance": float,
                        "label_ba_mean": float, "label_ba_std": float,
                        "label_ba_per_seed": {seed: ba}}, ... 8 depths]}``

    With neither argument, returns the full file dict.
    """
    with open(_LAYERWISE_PROBES_PATH) as f:
        full = json.load(f)
    if dataset is None and fm is None:
        return full
    if dataset is None or fm is None:
        raise ValueError("Pass both dataset and fm, or neither.")
    return full["results"][dataset][fm]
