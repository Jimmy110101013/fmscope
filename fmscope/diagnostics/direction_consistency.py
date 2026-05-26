"""Within-subject label-direction consistency (the c̄ signal).

For each subject with both labels present, the **label direction** in
feature space is the unit vector pointing from the mean of label-0
windows to the mean of label-1 windows. The signal we care about is
whether these directions agree across subjects.

``c̄`` (paper notation) is the median pairwise cosine similarity of
those unit vectors across subjects. ``c̄ > 0`` means subjects' label
axes point the same way; ``c̄ ≈ 0`` means each subject has their own
idiosyncratic direction.

This is the toolkit's clean wrapper for what was inlined inside
``notebooks/audit_demo.ipynb``. Trait cells (one label per subject)
have no within-subject contrast — :func:`direction_consistency` skips
them and returns ``c_bar = NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DirectionConsistency:
    """Output of :func:`direction_consistency`.

    Attributes
    ----------
    c_bar : float
        Median pairwise cosine of per-subject unit directions. NaN if
        fewer than 2 subjects have both labels.
    iqr_low, iqr_high : float
        25th / 75th percentile of pairwise cosines.
    n_subjects_paired : int
        Number of subjects with at least 2 windows in each label.
    directions : np.ndarray, shape (n_paired, embed_dim)
        Per-subject unit vectors (label-1 mean − label-0 mean, normalized).
    """

    c_bar: float
    iqr_low: float
    iqr_high: float
    n_subjects_paired: int
    directions: np.ndarray


def direction_consistency(
    features: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    *,
    min_windows_per_label: int = 2,
) -> DirectionConsistency:
    """Compute c̄ — median pairwise cosine of per-subject label directions.

    Parameters
    ----------
    features : np.ndarray, shape (N, embed_dim)
        Per-window features from a frozen FM extractor.
    subject : np.ndarray, shape (N,)
        Subject id per window.
    label : np.ndarray, shape (N,)
        Binary label per window (0/1).
    min_windows_per_label : int, default 2
        Minimum windows per subject per label to include the subject's
        direction. Skip subjects that don't meet this threshold.

    Returns
    -------
    :class:`DirectionConsistency`
    """
    feats = np.asarray(features, dtype=np.float64)
    sids = np.asarray(subject)
    labs = np.asarray(label)

    directions = []
    for sid in np.unique(sids):
        m = sids == sid
        m0 = m & (labs == 0)
        m1 = m & (labs == 1)
        if m0.sum() < min_windows_per_label or m1.sum() < min_windows_per_label:
            continue
        d = feats[m1].mean(axis=0) - feats[m0].mean(axis=0)
        n = np.linalg.norm(d)
        if n > 1e-12:
            directions.append(d / n)

    if len(directions) < 2:
        return DirectionConsistency(
            c_bar=float("nan"), iqr_low=float("nan"), iqr_high=float("nan"),
            n_subjects_paired=len(directions),
            directions=np.zeros((len(directions), feats.shape[1])) if directions
                       else np.zeros((0, feats.shape[1])),
        )

    dirs = np.stack(directions)
    pairs = dirs @ dirs.T
    upper = pairs[np.triu_indices(len(dirs), k=1)]
    return DirectionConsistency(
        c_bar=float(np.median(upper)),
        iqr_low=float(np.percentile(upper, 25)),
        iqr_high=float(np.percentile(upper, 75)),
        n_subjects_paired=len(directions),
        directions=dirs,
    )
