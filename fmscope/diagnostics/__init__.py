"""Diagnostic statistics — variance decomposition, null calibration, probes."""

from fmscope.diagnostics.direction_consistency import (
    DirectionConsistency,
    direction_consistency,
)
from fmscope.diagnostics.erasure import (
    ErasureResult,
    apply_eraser,
    subject_axis_erasure,
    subject_eraser,
    subject_probe,
    subspace_overlap,
    whiten,
)
from fmscope.diagnostics.layer_probe import layer_probe
from fmscope.diagnostics.null_control import null_control
from fmscope.diagnostics.variance import (
    cluster_bootstrap,
    crossed_ss_fractions,
    label_subspace_analysis,
    mixed_effects_variance,
    nested_ss,
    subject_level_permanova,
)

__all__ = [
    # variance.py
    "cluster_bootstrap",
    "crossed_ss_fractions",
    "label_subspace_analysis",
    "mixed_effects_variance",
    "nested_ss",
    "subject_level_permanova",
    # null_control.py
    "null_control",
    # direction_consistency.py
    "direction_consistency",
    "DirectionConsistency",
    # erasure.py
    "subject_axis_erasure",
    "ErasureResult",
    "whiten",
    "subject_eraser",
    "apply_eraser",
    "subspace_overlap",
    "subject_probe",
    # layer_probe.py
    "layer_probe",
]
