"""Random-Gaussian null control for the variance-decomposition claim.

The paper reports ``f_subj > f_label`` in 12 of 12 frozen (cell, FM)
pairs. Devil's-advocate critique: when the number of subjects is much
larger than the number of labels, the inequality is a structural
consequence of ANOVA combinatorics and is *not* by itself diagnostic
of foundation-model behaviour (Edwards 2007). This module generates a
matched random-Gaussian null embedding of the same shape and reports
how far the real FM exceeds the null.

Headline test used by the paper's verdict rubric (reproduction-only)::

    excess = real_subject_frac / null_subject_frac.mean()

A real FM is flagged "trait-leak suspect" when ``excess`` is small
(close to 1) while ``label_frac`` is near zero — that combination means
the apparent subject structure could have come from any random
embedding of the same shape.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fmscope.diagnostics.variance import crossed_ss_fractions


def null_control(
    features: np.ndarray,
    subject: np.ndarray,
    label: np.ndarray,
    *,
    n_null_seeds: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Compare real (subject, label) variance against a random-Gaussian null.

    For ``n_null_seeds`` independent draws of a Gaussian matrix shaped
    like ``features``, run :func:`crossed_ss_fractions` against the same
    ``(subject, label)`` annotations and aggregate the resulting
    ``label_frac`` / ``subject_frac`` distributions.

    Parameters
    ----------
    features : (N, D) ndarray
        Real embedding from the FM (per-window or per-recording).
    subject : (N,) ndarray
        Subject id per row.
    label : (N,) ndarray
        Label per row.
    n_null_seeds : int, default 20
        Number of random-Gaussian draws.
    rng : numpy.random.Generator, optional
        Source of randomness. ``None`` → :func:`numpy.random.default_rng`.

    Returns
    -------
    dict with keys ``real``, ``null_label_frac``, ``null_subject_frac``,
    ``excess_label``, ``excess_subject``, ``n_null_seeds``, ``df_pred``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)
    n, d = f.shape

    real = crossed_ss_fractions(f, s, y)

    null_label_frac = np.empty(n_null_seeds, dtype=np.float64)
    null_subject_frac = np.empty(n_null_seeds, dtype=np.float64)
    for k in range(n_null_seeds):
        rand = rng.standard_normal((n, d))
        out = crossed_ss_fractions(rand, s, y)
        null_label_frac[k] = out["label_frac"]
        null_subject_frac[k] = out["subject_frac"]

    n_subjects = int(np.unique(s).size)
    n_labels = int(np.unique(y).size)
    df_subj_pred = (n_subjects - 1) / max(n - 1, 1)
    df_label_pred = (n_labels - 1) / max(n - 1, 1)

    def _safe(real, null_mean):
        return float(real / null_mean) if null_mean > 1e-18 else float("inf")

    return {
        "real": real,
        "null_label_frac": {
            "mean": float(null_label_frac.mean()),
            "std": float(null_label_frac.std(ddof=1)) if n_null_seeds > 1 else 0.0,
            "samples": null_label_frac.tolist(),
        },
        "null_subject_frac": {
            "mean": float(null_subject_frac.mean()),
            "std": float(null_subject_frac.std(ddof=1)) if n_null_seeds > 1 else 0.0,
            "samples": null_subject_frac.tolist(),
        },
        "excess_label": _safe(real["label_frac"], null_label_frac.mean()),
        "excess_subject": _safe(real["subject_frac"], null_subject_frac.mean()),
        "df_label_pred": df_label_pred,
        "df_subject_pred": df_subj_pred,
        "n_null_seeds": n_null_seeds,
        "n": n,
        "n_subjects": n_subjects,
        "n_labels": n_labels,
    }
