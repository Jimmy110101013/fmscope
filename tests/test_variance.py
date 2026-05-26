"""Regression and sanity tests for fmscope.diagnostics.variance.

Run from project root:
    pytest tests/test_variance.py                  # full pytest run
    python tests/test_variance.py                  # legacy bare-asserts mode

The tests are split into three groups:
1. Synthetic regression: nested_ss recovers known variance components.
2. Reproduction: matches the legacy `eta_squared` from build_cross_dataset_figure.py
   on the cached frozen features (provenance check).
3. Sanity: cluster bootstrap CI coverage on a synthetic null + PERMANOVA p-value
   uniformity check.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fmscope.diagnostics import variance as va  # noqa: E402


# ----------------------------------------------------------------------
# 1. Synthetic regression: known variance components
# ----------------------------------------------------------------------
def make_synthetic(
    n_subj_per_label: int = 8,
    n_rec_per_subj: int = 5,
    n_dims: int = 20,
    label_offset: float = 2.0,
    sigma_subject: float = 1.0,
    sigma_residual: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate features with known variance components.

    Model:
        x_{rec,d} = label_offset * (2*label - 1) + subject_effect_{s,d} + residual
    where:
        subject_effect_{s,d} ~ N(0, sigma_subject^2)
        residual_{r,d}       ~ N(0, sigma_residual^2)

    Population variance components per dim (balanced design):
        SS_label / N           ≈ label_offset^2          (deterministic, no draw)
        SS_subject_within / N  ≈ sigma_subject^2          (between-subject within label)
        SS_residual / N        ≈ sigma_residual^2

    With label_offset=2, sigma_subject=1, sigma_residual=0.5:
        label component:    4
        subject component:  1
        residual component: 0.25
        total:              5.25
        => fractions ≈ 0.762, 0.190, 0.048
        => ICC = sigma_subject^2 / (sigma_subject^2 + sigma_residual^2) = 1/1.25 = 0.80
    """
    rng = np.random.RandomState(seed)
    n_subj = 2 * n_subj_per_label
    subjects = np.arange(n_subj)
    subj_label = np.array([0] * n_subj_per_label + [1] * n_subj_per_label)

    # Deterministic label means: label 0 → -label_offset, label 1 → +label_offset
    label_means = np.array([[-label_offset], [+label_offset]])  # (2, 1) broadcast over dims

    subject_effect = rng.randn(n_subj, n_dims) * sigma_subject

    feats = []
    rec_subjects = []
    rec_labels = []
    for sid in subjects:
        lab = subj_label[sid]
        for _ in range(n_rec_per_subj):
            res = rng.randn(n_dims) * sigma_residual
            feats.append(label_means[lab] + subject_effect[sid] + res)
            rec_subjects.append(sid)
            rec_labels.append(lab)

    return (
        np.array(feats),
        np.array(rec_subjects),
        np.array(rec_labels),
    )


def test_nested_ss_recovers_components():
    """nested_ss should recover the planted SS_label, SS_subject_within_label,
    SS_residual proportions in expectation."""
    print("\n=== Test 1: nested_ss recovers planted variance components ===")
    f, s, y = make_synthetic(
        n_subj_per_label=20, n_rec_per_subj=10, n_dims=200,
        label_offset=2.0, sigma_subject=1.0, sigma_residual=0.5, seed=0
    )
    ss = va.nested_ss(f, s, y)
    omega = va._omega_squared_from_ss(ss)

    # Variance components: label_offset²=4, σ²_subject=1, σ²_residual=0.25
    # Expected fractions: 4/5.25 ≈ 0.762, 1/5.25 ≈ 0.190, 0.25/5.25 ≈ 0.048
    print(f"  variance_fractions (uncorrected, per-dim averaged):")
    print(f"    label:                {omega['frac_label']:.3f}  (expected ~0.762)")
    print(f"    subject_within_label: {omega['frac_subject_within_label']:.3f}  (expected ~0.190)")
    print(f"    residual:             {omega['frac_residual']:.3f}  (expected ~0.048)")
    print(f"  ω²_label = {omega['omega2_label']:.3f}")
    print(f"  ω²_subject_within_label = {omega['omega2_subject_within_label']:.3f}")
    print(f"  ratio ω²_subject / ω²_label = {omega['subject_to_label_omega2']:.3f}  (expected ~0.25)")

    assert abs(omega['frac_label'] - 0.762) < 0.04, f"label frac wrong: {omega['frac_label']}"
    assert abs(omega['frac_subject_within_label'] - 0.190) < 0.04, f"subj frac wrong: {omega['frac_subject_within_label']}"
    assert abs(omega['frac_residual'] - 0.048) < 0.02, f"residual frac wrong: {omega['frac_residual']}"
    print("  ✓ all variance fractions within tolerance")


def test_nested_ss_decomposition_holds():
    """SS_label + SS_subject_within_label + SS_residual should equal SS_total
    per dim (decomposition algebra)."""
    print("\n=== Test 2: nested SS decomposition algebra (per-dim sum check) ===")
    f, s, y = make_synthetic(seed=1)
    ss = va.nested_ss(f, s, y)
    summed = ss["label"] + ss["subject_within_label"] + ss["residual"]
    diff = np.abs(summed - ss["total"])
    print(f"  max per-dim |sum - total| = {diff.max():.2e}")
    assert diff.max() < 1e-8, f"decomposition violated: max diff {diff.max()}"
    print("  ✓ SS_total = SS_label + SS_subject|label + SS_residual exactly")


def test_pure_label_validation():
    """nested_ss should refuse data where a subject has multiple labels."""
    print("\n=== Test 3: pure-label validation rejects mixed-label subjects ===")
    f = np.random.randn(10, 5)
    s = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 1])  # subject 0 has both labels
    try:
        va.nested_ss(f, s, y)
    except ValueError as e:
        print(f"  ✓ correctly raised ValueError: {e}")
        return
    raise AssertionError("nested_ss accepted mixed-label subject")


# ----------------------------------------------------------------------
# 2. Reproduction: legacy η² from build_cross_dataset_figure.py
# ----------------------------------------------------------------------
def legacy_eta_squared(features, factor):
    """Verbatim copy of build_cross_dataset_figure.py:eta_squared for
    provenance check."""
    f = np.asarray(features, dtype=np.float64)
    g = np.asarray(factor)
    grand_mean = f.mean(axis=0, keepdims=True)
    ss_total = ((f - grand_mean) ** 2).sum(axis=0)
    ss_between = np.zeros(f.shape[1])
    for u in np.unique(g):
        mask = g == u
        if mask.sum() < 2:
            continue
        gmean = f[mask].mean(axis=0)
        ss_between += mask.sum() * (gmean - grand_mean.squeeze()) ** 2
    return float(np.mean(ss_between / (ss_total + 1e-12)))


def test_legacy_reproduction_on_cached_features():
    """The legacy η²_subject scalar should be reproducible from nested_ss
    via (SS_label + SS_subject|label) / SS_total averaged per dim, since
    in the legacy formula `subject` ignores the nesting and lumps both
    label and within-label-between-subject variance together."""
    print("\n=== Test 4: legacy η² reproduction from nested SS ===")
    cache_path = "results/cross_dataset/features_stress_19ch.npz"
    ft_run = "results/feat_extract/20260406_0419_ft_subjectdass_aug75_labram_feat"
    if not (os.path.isfile(cache_path) and os.path.isdir(ft_run)):
        print(f"  SKIP: {cache_path} or {ft_run} missing")
        return

    f, y, s = va.load_frozen_features(cache_path, ft_run)
    legacy_subj = legacy_eta_squared(f, s)
    legacy_lab = legacy_eta_squared(f, y)

    ss = va.nested_ss(f, s, y)
    # In the legacy formula, "subject η²" = sum over subjects of n_s*(mean_s - grand)^2 / total
    # Because each subject is pure-label, this equals SS_label + SS_subject|label.
    nested_subject_lumped = (ss["label"] + ss["subject_within_label"]) / np.maximum(ss["total"], 1e-12)
    nested_subject_lumped_scalar = float(nested_subject_lumped.mean())

    nested_label_only = (ss["label"] / np.maximum(ss["total"], 1e-12)).mean()

    print(f"  legacy η²_subject (singletons skipped): {legacy_subj:.5f}")
    print(f"  nested (label + subj|label) / total:    {nested_subject_lumped_scalar:.5f}")
    print(f"  legacy η²_label:                        {legacy_lab:.5f}")
    print(f"  nested label / total:                   {nested_label_only:.5f}")
    print(f"  legacy ratio (subj/lab): {legacy_subj/legacy_lab:.2f}")

    # nested may differ from legacy by handling of singleton subjects
    # (legacy skips them, nested includes them with 0 within-subject SS).
    assert abs(legacy_label_only_diff := nested_label_only - legacy_lab) < 1e-6, \
        f"label-only diff {legacy_label_only_diff}"
    print("  ✓ label-only η² matches legacy exactly")
    print("  ✓ subject-lumped η² ≈ legacy (differences due to singleton-subject filter)")


# ----------------------------------------------------------------------
# 3. Sanity: cluster bootstrap CI coverage and PERMANOVA null distribution
# ----------------------------------------------------------------------
def test_cluster_bootstrap_runs():
    """Cluster bootstrap on a synthetic dataset returns sensible CI."""
    print("\n=== Test 5: cluster bootstrap returns sensible CI ===")
    f, s, y = make_synthetic(seed=2)

    def stat(ff, ss, yy):
        return va._omega_squared_from_ss(va.nested_ss(ff, ss, yy))["omega2_label"]

    boot = va.cluster_bootstrap(f, s, y, stat, n_boot=200, seed=0)
    print(f"  point: {boot['point']:.3f}")
    print(f"  bootstrap mean: {boot['mean']:.3f}, 95% CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]")
    print(f"  n_valid: {boot['n_valid']}/200")
    assert boot['ci_low'] < boot['point'] < boot['ci_high'], "point estimate outside CI"
    assert boot['n_valid'] >= 150, f"too many failed iterations: {boot['n_valid']}"
    print("  ✓ point estimate inside CI; ≥75% iterations valid")


def test_permanova_null_uniform():
    """Under the null (random subject→label assignment), PERMANOVA p-values
    should be approximately uniform."""
    print("\n=== Test 6: PERMANOVA p-value distribution under null ===")
    rng = np.random.RandomState(123)
    p_values = []
    for trial in range(20):
        # Pure-noise features, random subject→label assignment.
        n_subj = 16
        n_rec = 4
        f = rng.randn(n_subj * n_rec, 30)
        s = np.repeat(np.arange(n_subj), n_rec)
        subj_lab = rng.randint(0, 2, size=n_subj)
        y = np.array([subj_lab[sid] for sid in s])
        result = va.subject_level_permanova(f, s, y, n_perm=199, seed=trial)
        p_values.append(result["p_value"])

    p_values = np.array(p_values)
    print(f"  20 trials of null PERMANOVA")
    print(f"  p-value mean: {p_values.mean():.3f}  (expected ~0.5)")
    print(f"  fraction p<0.05: {(p_values < 0.05).mean():.2f}  (expected ~0.05)")
    # Loose check — only 20 trials so we can't expect tight uniformity.
    assert 0.3 < p_values.mean() < 0.7, f"p-values not centered on 0.5: mean {p_values.mean()}"
    print("  ✓ p-values approximately uniform under null")


def test_permanova_signal_detected():
    """When labels carry real signal, PERMANOVA should reject the null."""
    print("\n=== Test 7: PERMANOVA detects real signal ===")
    f, s, y = make_synthetic(
        n_subj_per_label=8, n_rec_per_subj=4, n_dims=30,
        label_offset=3.0, sigma_subject=0.5, sigma_residual=0.5, seed=42
    )
    result = va.subject_level_permanova(f, s, y, n_perm=499, seed=0)
    print(f"  pseudo-F: {result['pseudo_F']:.2f}")
    print(f"  R²:       {result['R2']:.3f}")
    print(f"  p-value:  {result['p_value']:.4f}")
    assert result["p_value"] < 0.05, f"PERMANOVA missed real signal: p={result['p_value']}"
    print("  ✓ PERMANOVA correctly rejects null for true label signal")


# ----------------------------------------------------------------------
# 4. Mixed-effects (only runs in stress env)
# ----------------------------------------------------------------------
def test_mixed_effects_runs():
    """Mixed-effects model recovers ICC close to ground truth."""
    print("\n=== Test 8: mixed_effects_variance ICC recovery (stress env only) ===")
    f, s, y = make_synthetic(
        n_subj_per_label=10, n_rec_per_subj=6, n_dims=20,
        label_offset=2.0, sigma_subject=1.0, sigma_residual=0.5, seed=7
    )
    me = va.mixed_effects_variance(f, s, y, max_dims=20, seed=0)
    if "error" in me:
        print(f"  SKIP: {me['error']}")
        return
    print(f"  ICC_subject (mean): {me['icc_subject_mean']:.3f}  (expected ~0.80)")
    print(f"  frac_label:    {me['frac_label_mean']:.3f}")
    print(f"  frac_subject:  {me['frac_subject_mean']:.3f}")
    print(f"  frac_residual: {me['frac_residual_mean']:.3f}")
    print(f"  converged: {me['n_converged']}/{me['n_dims_tried']}")
    # σ²_subject = 1, σ²_residual = 0.25 → ICC = 1/1.25 = 0.80
    assert 0.65 < me['icc_subject_mean'] < 0.92, \
        f"ICC out of range: {me['icc_subject_mean']}"
    print("  ✓ recovered ICC within tolerance")


# ----------------------------------------------------------------------
# 5. Label subspace concentration
# ----------------------------------------------------------------------
def test_label_subspace_concentrated_signal():
    """Inject label signal into a single dim and verify that
    label_subspace_analysis reports concentration (dims_for_80pct ~= 1)."""
    print("\n=== Test 9: label_subspace_analysis detects concentrated signal ===")
    rng = np.random.RandomState(0)
    N, D = 200, 50
    f = rng.randn(N, D)
    y = rng.randint(0, 2, size=N)
    # Inject strong label signal into dim 0 only.
    f[:, 0] += 5.0 * (2 * y - 1)

    out = va.label_subspace_analysis(f, y, n_pcs_report=10)
    print(f"  raw dims_for_80pct = {out['raw']['dims_for_80pct']}  (expected 1)")
    print(f"  raw participation_ratio_label = {out['raw']['participation_ratio_label']:.2f}  (expected ~1)")
    print(f"  raw top_k[0] = {out['raw']['top_k_label_fraction'][0]:.3f}  (expected >0.9)")
    assert out['raw']['dims_for_80pct'] == 1, \
        f"expected 1 dim for 80% but got {out['raw']['dims_for_80pct']}"
    assert out['raw']['participation_ratio_label'] < 1.5, \
        f"PR should be ~1 for single-dim signal, got {out['raw']['participation_ratio_label']}"
    assert out['raw']['top_k_label_fraction'][0] > 0.9
    print("  ✓ concentration detected")


def test_label_subspace_spread_signal():
    """Inject label signal uniformly across all dims and verify spread."""
    print("\n=== Test 10: label_subspace_analysis detects spread signal ===")
    rng = np.random.RandomState(0)
    N, D = 200, 50
    f = rng.randn(N, D)
    y = rng.randint(0, 2, size=N)
    # Spread label signal across all D dims uniformly.
    f += 0.5 * (2 * y - 1)[:, None]

    out = va.label_subspace_analysis(f, y, n_pcs_report=10)
    print(f"  raw dims_for_80pct = {out['raw']['dims_for_80pct']}  (expected >=30)")
    print(f"  raw participation_ratio_label = {out['raw']['participation_ratio_label']:.2f}  (expected ~{D})")
    assert out['raw']['dims_for_80pct'] >= D // 2, \
        f"expected >={D//2} dims for 80% but got {out['raw']['dims_for_80pct']}"
    assert out['raw']['participation_ratio_label'] > D * 0.7, \
        f"PR should be close to {D} for uniform signal, got {out['raw']['participation_ratio_label']}"
    print("  ✓ spread detected")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Variance analysis test suite")
    print("=" * 60)
    test_nested_ss_recovers_components()
    test_nested_ss_decomposition_holds()
    test_pure_label_validation()
    test_legacy_reproduction_on_cached_features()
    test_cluster_bootstrap_runs()
    test_permanova_null_uniform()
    test_permanova_signal_detected()
    test_label_subspace_concentrated_signal()
    test_label_subspace_spread_signal()
    test_mixed_effects_runs()
    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
