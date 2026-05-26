"""Live-data audit — numeric diagnostic row for any cohort × extractor pair.

This is the public entry point researchers use to audit their own dataset
with their own FM (wrap it via :mod:`fmscope.api`'s ``FMExtractor`` /
``CohortAdapter`` protocols). It composes the FMScope diagnostics into one
call and returns **numbers only** — variance fractions, null-calibration
excess, the c̄ direction-consistency value, and the subject-axis erasure
result. There is no verdict, glyph, or outcome string: the consumer reads
the numbers directly.

The function runs:

1. Extract per-window features from every recording in the cohort.
2. Pool windows to recording-level features.
3. Crossed-SS fractions on recording-level features (variance tool).
4. Random-Gaussian null calibration on window-level features.
5. c̄ direction-consistency on within-subject paired cells.
6. Subject-axis linear erasure (LEACE).

Layer-wise probe and aperiodic (FOOOF) ablation are not run here — they
need a multi-layer sweep / signal re-extraction. Pass pre-computed
summaries via :class:`AuditConfig` to surface their numbers in the row.

Usage::

    from fmscope.verdict import audit_cell, AuditConfig
    row = audit_cell(my_cohort, my_extractor, config=AuditConfig("MyCohort"))
    print(row["label_frac"], row["erasure_label_ba_delta"])
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fmscope.api import CohortAdapter, FMExtractor
from fmscope.diagnostics.erasure import DEFAULT_LABEL_SEEDS, subject_axis_erasure
from fmscope.diagnostics.null_control import null_control
from fmscope.diagnostics.variance import crossed_ss_fractions


@dataclass
class AuditConfig:
    """Knobs controlling :func:`audit_cell`.

    Parameters
    ----------
    cell_name : str
        Display label for the row (e.g. ``"Meditation"``).
    cell_layout : str or None
        Optional display/metadata tag (``"W,C"`` / ``"T,C"`` / ``"W,N"`` /
        ``"T,N"``). When it starts with ``"T,"`` (trait — one label per
        subject) the c̄ direction-consistency value is skipped. It carries
        no rubric meaning; it is metadata only.
    batch_size : int
        Mini-batch size for the FM forward pass. Default 16.
    device : str
        Torch device for the extractor (``"cpu"`` or ``"cuda:0"``).
    n_null_seeds : int
        Number of random-Gaussian null draws for null calibration.
        Default 10 — paper uses 20 for the bundled cells.
    run_erasure : bool
        When True (default), run the subject-axis linear-erasure (LEACE)
        diagnostic and add its numeric columns (``erasure_*``) to the row.
        Set False to skip its probe cost.
    erasure_gate : float
        Pre-erasure label-BA threshold below which ``Δ_erase`` is marked
        ``erasure_interpretable = False``. Default 0.55.
    erasure_label_seeds : tuple
        Seeds for the erasure label probe. Default ``(42, 123, 2024)``.
    layer_probe : dict or None
        Optional pre-computed layer-probe summary
        (``label_ba_first`` / ``label_ba_last`` / ``label_ba_max`` /
        ``argmax_depth``). Surfaced as ``layer_*`` columns.
    oneoverf : dict or None
        Optional pre-computed FOOOF-ablation summary
        (``state_drop_mean`` / ``subject_drop_mean``). Surfaced as
        ``state_drop`` / ``subject_drop`` columns.
    """

    cell_name: str
    cell_layout: Optional[str] = None  # display/metadata only — no rubric
    batch_size: int = 16
    device: str = "cpu"
    n_null_seeds: int = 10
    run_erasure: bool = True
    erasure_gate: float = 0.55
    erasure_label_seeds: tuple = DEFAULT_LABEL_SEEDS
    layer_probe: Optional[dict] = None
    oneoverf: Optional[dict] = None


def _extract_features(
    extractor: FMExtractor,
    cohort: CohortAdapter,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the extractor against every window in the cohort.

    Returns ``(features, subject_ids, labels)`` aligned per window.
    """
    import torch  # local import — extractors are torch-backed but the
    #                  rest of the diagnostic stack is pure numpy.

    if isinstance(extractor, torch.nn.Module):
        extractor.eval().to(device)

    feats_list: list[np.ndarray] = []
    sids: list[int] = []
    labels: list[int] = []
    t0 = time.time()
    n_rec = 0
    n_win = 0
    for sid, label, windows in cohort.iter_recordings():
        windows_t = torch.from_numpy(np.asarray(windows)).float()
        n_rec += 1
        for i in range(0, windows_t.shape[0], batch_size):
            batch = windows_t[i:i + batch_size].to(device)
            with torch.no_grad():
                out = extractor(batch)
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
            feats_list.append(np.asarray(out, dtype=np.float32))
        sids.extend([sid] * windows_t.shape[0])
        labels.extend([label] * windows_t.shape[0])
        n_win += windows_t.shape[0]
    feats = np.concatenate(feats_list, axis=0)
    elapsed = time.time() - t0
    return feats, np.asarray(sids), np.asarray(labels), {
        "n_recordings": n_rec, "n_windows": n_win, "elapsed_s": elapsed,
    }


def _recording_mean_pool(
    feats: np.ndarray, sids: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool per-window features into one feature vector per recording."""
    pooled_feats, pooled_sids, pooled_labels = [], [], []
    start = 0
    for i in range(1, len(sids) + 1):
        if i == len(sids) or sids[i] != sids[start] or labels[i] != labels[start]:
            pooled_feats.append(feats[start:i].mean(axis=0))
            pooled_sids.append(sids[start])
            pooled_labels.append(labels[start])
            start = i
    return (np.stack(pooled_feats),
            np.asarray(pooled_sids),
            np.asarray(pooled_labels))


def audit_cell(
    cohort: CohortAdapter,
    extractor: FMExtractor,
    *,
    config: AuditConfig,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Audit one (cohort, FM) cell and return a numeric diagnostic row.

    Parameters
    ----------
    cohort : CohortAdapter
        Any object satisfying :class:`fmscope.CohortAdapter`. See
        :doc:`docs/byo_dataset.md`.
    extractor : FMExtractor
        Any object satisfying :class:`fmscope.FMExtractor`. See
        :doc:`docs/byo_fm.md`.
    config : AuditConfig
        Cell metadata + diagnostic knobs.
    rng : numpy.random.Generator, optional
        Random source for the null calibration.

    Returns
    -------
    dict
        Numeric row: variance fractions (``label_frac`` / ``subject_frac``
        / ``residual_frac``), null excess ratios, c̄ value, optional
        ``layer_*`` / ``state_drop`` / ``subject_drop`` columns, and
        ``erasure_*`` columns when ``run_erasure`` is True. No glyphs, no
        outcome — the consumer interprets the numbers directly.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # 1) Feature extraction.
    feats_w, sids_w, labels_w, ext_stats = _extract_features(
        extractor, cohort,
        batch_size=config.batch_size, device=config.device,
    )
    feats_r, sids_r, labels_r = _recording_mean_pool(feats_w, sids_w, labels_w)

    # 2) Crossed SS fractions on recording-level features.
    ss = crossed_ss_fractions(feats_r, sids_r, labels_r)

    # 3) Null calibration on window-level features.
    null = null_control(feats_w, sids_w, labels_w,
                        n_null_seeds=config.n_null_seeds, rng=rng)

    # 4) c̄ — within-subject direction consistency. Trait cells (one label
    #    per subject) have no within-subject contrast; report NaN.
    is_trait = config.cell_layout is not None and config.cell_layout.startswith("T,")
    c_bar_value = None
    c_bar_numeric = float("nan")
    if not is_trait:
        from fmscope.diagnostics.direction_consistency import direction_consistency
        dc = direction_consistency(feats_w, sids_w, labels_w)
        if np.isfinite(dc.c_bar):
            c_bar_value = {
                "iqr_low": dc.iqr_low, "iqr_high": dc.iqr_high,
                "n_subjects_paired": dc.n_subjects_paired,
            }
            c_bar_numeric = float(dc.c_bar)

    # 5) Subject-axis linear erasure (LEACE). Erase the linear subject
    #    axis, then re-probe identity (collapses toward chance) and the
    #    label (Δ_erase = post − pre). Skipped when run_erasure is False.
    er = (subject_axis_erasure(
              feats_w, sids_w, labels_w,
              gate=config.erasure_gate,
              label_seeds=config.erasure_label_seeds,
          ) if config.run_erasure else None)

    # 6) Numeric diagnostic row.
    layout_str = (f"({config.cell_layout.replace(',', ', ')})"
                  if config.cell_layout is not None else None)
    row: dict = {
        "cell": config.cell_name,
        "layout": layout_str,
        "label_frac": ss["label_frac"],
        "subject_frac": ss["subject_frac"],
        "residual_frac": ss["residual_frac"],
        "null_label_frac_mean": null["null_label_frac"]["mean"],
        "null_subject_frac_mean": null["null_subject_frac"]["mean"],
        "excess_label_ratio": null["excess_label"],
        "excess_subject_ratio": null["excess_subject"],
        "c_bar_value": c_bar_numeric,
        "c_bar_iqr_low": (c_bar_value or {}).get("iqr_low", float("nan")),
        "c_bar_iqr_high": (c_bar_value or {}).get("iqr_high", float("nan")),
        "c_bar_n_subjects_paired": (c_bar_value or {}).get("n_subjects_paired", 0),
        "layer_label_ba_first": (config.layer_probe or {}).get("label_ba_first",
                                                                float("nan")),
        "layer_label_ba_last":  (config.layer_probe or {}).get("label_ba_last",
                                                                float("nan")),
        "layer_label_ba_max":   (config.layer_probe or {}).get("label_ba_max",
                                                                float("nan")),
        "layer_argmax_depth":   (config.layer_probe or {}).get("argmax_depth",
                                                                float("nan")),
        "state_drop":   (config.oneoverf or {}).get("state_drop_mean",
                                                     float("nan")),
        "subject_drop": (config.oneoverf or {}).get("subject_drop_mean",
                                                     float("nan")),
        "extraction": ext_stats,
        "layer_probe_supplied": config.layer_probe is not None,
        "oneoverf_supplied": config.oneoverf is not None,
    }

    # Subject-axis erasure columns (erasure_*). NaN/absent when run_erasure
    # is False or the cohort has no binary within-subject label.
    if er is not None:
        row.update({
            "erasure_rank_subject_axis": er.rank_subject_axis,
            "erasure_degenerate": er.degenerate,
            "erasure_cond_shrunk": er.cond_shrunk,
            "erasure_subj_ba_linear_pre": er.subj_ba_linear_pre,
            "erasure_subj_ba_linear_post": er.subj_ba_linear_post,
            "erasure_subj_ba_mlp_post": er.subj_ba_mlp_post,
            "erasure_label_ba_raw": er.label_ba_raw,
            "erasure_label_ba_erased": er.label_ba_erased,
            "erasure_label_ba_delta": er.label_ba_delta,
            "erasure_label_ba_delta_std": er.label_ba_delta_std,
            "erasure_gate": er.gate,
            "erasure_interpretable": er.interpretable,
        })
    row["erasure_supplied"] = er is not None

    return row
