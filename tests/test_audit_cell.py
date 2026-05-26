"""End-to-end smoke test for fmscope.verdict.audit_cell().

Uses a tiny in-test ``StubExtractor`` (random linear projection, no
weights, CPU-only) so the test gate doesn't need a real foundation
model. The extractor is intentionally defined inside the test module —
mock/stub classes don't belong in the public package surface.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fmscope import CohortAdapter, FMExtractor
from fmscope.data.adapters import synthetic_cohort
from fmscope.verdict import AuditConfig, audit_cell


class StubExtractor(torch.nn.Module):
    """Random linear projection of flattened EEG → ``(B, embed_dim)``.

    Satisfies :class:`fmscope.FMExtractor` via structural typing
    (``embed_dim`` attribute + callable returning ``(B, embed_dim)``).
    Used only inside tests.
    """

    def __init__(self, *, n_channels: int = 19, n_samples: int = 1000,
                 embed_dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_dim = embed_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_channels * n_samples, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_audit_cell_returns_numeric_row():
    """A synthetic cohort + the in-test stub yields a numbers-only row.

    The numbers-only audit reports raw diagnostics — no rubric glyph or
    ``outcome`` string (that lives in reproduction, not the toolkit).
    """
    cohort = synthetic_cohort(
        n_subjects=8, n_recordings_per_subj=3, n_windows_per_rec=10,
        n_channels=19, n_samples=1000, leak_subject_into_label=True, seed=0,
    )
    assert isinstance(cohort, CohortAdapter)

    extractor = StubExtractor(n_channels=19, n_samples=1000)
    assert isinstance(extractor, FMExtractor)

    row = audit_cell(
        cohort, extractor,
        config=AuditConfig(
            cell_name="SyntheticTrait", cell_layout="T,C",
            batch_size=8, device="cpu", n_null_seeds=5, run_erasure=False,
        ),
    )

    # Numeric columns present.
    for k in ("cell", "layout", "label_frac", "subject_frac", "residual_frac",
              "excess_label_ratio", "excess_subject_ratio", "c_bar_value"):
        assert k in row, f"audit_cell row missing numeric key: {k}"
    # No rubric glyphs / outcome — numbers only.
    for k in ("outcome", "delta_f_label", "layer_probe", "c_bar", "oneoverf_role"):
        assert k not in row, f"numbers-only row must not carry rubric key: {k}"

    assert 0.0 <= row["label_frac"] <= 1.0
    assert 0.0 <= row["subject_frac"] <= 1.0
    assert row["extraction"]["n_recordings"] == 24
    # Trait cell (T,) → no within-subject contrast → c̄ is NaN.
    assert np.isnan(row["c_bar_value"])


def test_audit_cell_surfaces_supplied_layer_probe():
    """A supplied layer_probe summary is surfaced as layer_* numeric columns."""
    cohort = synthetic_cohort(
        n_subjects=6, n_recordings_per_subj=3, n_windows_per_rec=10,
        n_channels=19, n_samples=1000, leak_subject_into_label=True, seed=1,
    )
    extractor = StubExtractor(n_channels=19, n_samples=1000)
    row = audit_cell(
        cohort, extractor,
        config=AuditConfig(
            cell_name="SyntheticTrait", cell_layout="T,C",
            batch_size=8, device="cpu", n_null_seeds=5, run_erasure=False,
            layer_probe={
                "label_ba_first": 0.55, "label_ba_last": 0.55,
                "label_ba_max":   0.72, "argmax_depth":  0.5,
            },
        ),
    )
    assert row["layer_probe_supplied"] is True
    assert row["layer_label_ba_max"] == 0.72
    assert row["layer_argmax_depth"] == 0.5


def test_audit_cell_emits_erasure_columns_by_default():
    """run_erasure defaults True → numeric row carries erasure_* columns."""
    cohort = synthetic_cohort(
        n_subjects=6, n_recordings_per_subj=4, n_windows_per_rec=10,
        n_channels=19, n_samples=1000, leak_subject_into_label=False, seed=2,
    )
    extractor = StubExtractor(n_channels=19, n_samples=1000)
    row = audit_cell(
        cohort, extractor,
        config=AuditConfig(cell_name="SyntheticPaired", cell_layout=None,
                           batch_size=8, device="cpu", n_null_seeds=5),
    )
    assert row["erasure_supplied"] is True
    for k in ("erasure_rank_subject_axis", "erasure_subj_ba_linear_pre",
              "erasure_subj_ba_linear_post", "erasure_subj_ba_mlp_post",
              "erasure_label_ba_raw", "erasure_label_ba_delta",
              "erasure_gate", "erasure_interpretable"):
        assert k in row, f"missing erasure column: {k}"
    # Numbers only — no rubric glyph/outcome on the live (cell_layout=None) path.
    assert "outcome" not in row
    # Erasure removes the linear identity axis.
    assert row["erasure_subj_ba_linear_post"] <= row["erasure_subj_ba_linear_pre"]


def test_audit_cell_run_erasure_false_omits_columns():
    cohort = synthetic_cohort(
        n_subjects=6, n_recordings_per_subj=4, n_windows_per_rec=10,
        n_channels=19, n_samples=1000, leak_subject_into_label=False, seed=3,
    )
    extractor = StubExtractor(n_channels=19, n_samples=1000)
    row = audit_cell(
        cohort, extractor,
        config=AuditConfig(cell_name="SyntheticPaired", cell_layout=None,
                           batch_size=8, device="cpu", n_null_seeds=5,
                           run_erasure=False),
    )
    assert row["erasure_supplied"] is False
    assert "erasure_label_ba_delta" not in row


def test_audit_cell_rejects_non_cohort():
    """audit_cell must error on objects that fail the CohortAdapter protocol."""
    extractor = StubExtractor()
    with pytest.raises((AttributeError, TypeError)):
        audit_cell(
            "not a cohort",  # type: ignore[arg-type]
            extractor,
            config=AuditConfig(cell_name="bad", cell_layout="T,C"),
        )
