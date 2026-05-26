"""Unit tests for the subject-axis linear-erasure (LEACE) diagnostic.

Pure numpy — no FM weights, no torch. Builds a small synthetic cohort
where a strong per-subject identity offset is layered over a weak shared
label direction, so erasing the linear subject axis should collapse the
subject probe toward chance while leaving (or helping) the label probe.
"""

from __future__ import annotations

import numpy as np

from fmscope.diagnostics import (
    ErasureResult,
    apply_eraser,
    subject_axis_erasure,
    subject_eraser,
    subject_probe,
    subspace_overlap,
    whiten,
)


def _identity_dominated_cohort(n_subjects=6, n_windows=40, dim=32,
                               offset_scale=3.0, label_scale=0.4, seed=0):
    """Per-subject offset (identity) + weak shared label direction."""
    rng = np.random.default_rng(seed)
    label_dir = rng.standard_normal(dim)
    label_dir /= np.linalg.norm(label_dir)
    feats, subj, lab = [], [], []
    for s in range(n_subjects):
        offset = offset_scale * rng.standard_normal(dim)
        for y in (0, 1):  # both labels per subject → within-subject paired
            block = (rng.standard_normal((n_windows, dim)) * 0.5
                     + offset + (label_scale * y) * label_dir)
            feats.append(block)
            subj += [s] * n_windows
            lab += [y] * n_windows
    return np.concatenate(feats), np.array(subj), np.array(lab)


def test_erasure_collapses_linear_identity_and_reports_delta():
    X, subj, lab = _identity_dominated_cohort(seed=0)
    r = subject_axis_erasure(X, subj, lab)

    assert isinstance(r, ErasureResult)
    # Centred one-hot subject design has rank n_subjects - 1.
    assert r.rank_subject_axis == r.n_subjects - 1
    assert not r.degenerate  # dim (32) >> rank (5)

    # Linear identity axis is removed: post-erasure subject probe drops
    # well below the (near-perfect) pre-erasure probe, toward chance.
    assert r.subj_ba_linear_pre > 0.9
    assert r.subj_ba_linear_post < r.subj_ba_linear_pre
    assert r.subj_ba_linear_post < 0.3  # chance for 6 subjects is ~0.17

    # Label probe is computed (binary within-subject label) and the gate
    # fires because the raw label is readable.
    assert np.isfinite(r.label_ba_raw)
    assert np.isfinite(r.label_ba_delta)
    assert r.gate == 0.55
    assert r.interpretable == (r.label_ba_raw >= r.gate)


def test_erasure_without_label_skips_label_probe():
    X, subj, _ = _identity_dominated_cohort(seed=1)
    r = subject_axis_erasure(X, subj, None)
    assert np.isnan(r.label_ba_raw)
    assert np.isnan(r.label_ba_delta)
    assert r.interpretable is False
    # Subject-axis fields are still populated.
    assert r.subj_ba_linear_post < r.subj_ba_linear_pre


def test_gate_threshold_controls_interpretability():
    """The gate alone decides interpretability for a fixed, readable label."""
    X, subj, lab = _identity_dominated_cohort(seed=2)
    readable = subject_axis_erasure(X, subj, lab, gate=0.0)
    assert readable.interpretable is True  # any finite raw BA clears gate 0.0

    gated = subject_axis_erasure(X, subj, lab, gate=0.999)
    assert np.isfinite(gated.label_ba_raw)
    assert gated.interpretable is False    # raw BA cannot clear gate 0.999


def test_primitives_eraser_reduces_linear_subject_probe():
    """apply_eraser removes the linear subject axis; subspace_overlap is sane."""
    X, subj, _ = _identity_dominated_cohort(seed=3)
    mu, Xc, W, W_plus, cond = whiten(X)
    S, P_perp, r = subject_eraser(Xc, W, W_plus, subj)
    assert np.isfinite(cond) and r == len(np.unique(subj)) - 1

    Xe = apply_eraser(X, mu, P_perp)
    ba_pre, _ = subject_probe(X, subj, kind="linear")
    ba_post, _ = subject_probe(Xe, subj, kind="linear")
    assert ba_pre > 0.9 and ba_post < ba_pre

    # subspace_overlap: identical basis → 1, orthogonal complement → 0.
    assert abs(subspace_overlap(S, S) - 1.0) < 1e-9
    q_full, _ = np.linalg.qr(S, mode="complete")  # (dim, dim)
    comp = q_full[:, r:]  # orthonormal basis of the complement of span(S)
    assert subspace_overlap(S, comp) < 1e-9
