"""Verdict rubric — Tab 3 outcome decision logic.

The paper's Table 3 ("Cross-diagnostic results and per-cell assessment")
reports a single ``Outcome`` per cell, derived from the four
sign/role columns:

* ``Δf_label`` — sign of label-fraction change between frozen and
  fine-tuned representations (+ / − / 0).
* ``layer probe`` — direction of the label-vs-subject layer-wise
  probe (+ / − / 0), with optional qualifier ``early`` / ``deep``
  marking where the signal concentrates.
* ``c̄`` — sign of within-subject direction consistency
  (+ / − / 0). Defined only in within-subject cells; trait cells
  report ``0`` by construction.
* ``1/f role`` — qualitative role of the aperiodic component, derived
  from the FOOOF ablation drops (``state signal`` / ``subject confound``
  / ``subject axis``).

These four columns combine into the four outcomes that name the cell
layout: ``Cross-subject-aligned`` (W,C), ``Label–subject coupled``
(T,C), ``Idiosyncratic within-subject`` (W,N), ``Below linear-probe
resolution`` (T,N). Each outcome is the mechanical consequence of the
four signs — see :func:`classify` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Sign = Literal["+", "-", "0"]
OneOverFRole = Literal["state signal", "subject confound", "subject axis"]
Outcome = Literal[
    "Cross-subject-aligned",
    "Label-subject coupled",
    "Idiosyncratic within-subject",
    "Below linear-probe resolution",
    "Uncategorized",
]


@dataclass(frozen=True)
class Thresholds:
    """Sign-decision thresholds used by :func:`classify`.

    Defaults match the paper-locked rubric. Override if recalibrating
    against a new cohort set.

    The signs are 2-way (``+`` / ``-``) rather than 3-way: the paper's
    ``0`` glyph is reserved for "not applicable" (e.g. c̄ in trait
    cells), not for "noise". Anything that fails the positive threshold
    is read as ``-`` ("did not yield a positive signal"), matching the
    paper's reading of Tab 3.
    """

    # Δf_label: mean Δ above this counts as a positive change.
    delta_label_positive: float = 0.005

    # Layer probe: max(label_ba) at or above this is "robust above chance".
    layer_label_above_chance: float = 0.60

    # Layer probe "- deep" qualifier: last-depth label_ba below this is "below chance".
    layer_label_below_chance: float = 0.45

    # Layer probe "+ early" qualifier: max(label_ba) lives in the first
    # ``early_fraction`` of depth AND drops by at least ``early_drop_min``
    # to the final layer.
    layer_early_fraction: float = 0.35
    layer_early_drop_min: float = 0.04

    # c̄: median dir_consistency at or above this is a positive direction.
    c_bar_positive: float = 0.05

    # 1/f role: state-probe drop when aperiodic removed (mean across FMs).
    # If state_drop > this threshold, 1/f is part of the state signal.
    state_drop_threshold: float = 0.03

    # 1/f role: subject-probe drop when aperiodic removed.
    subject_drop_threshold: float = 0.05


DEFAULT = Thresholds()


def positive_sign(values: list[float], *, threshold: float) -> Sign:
    """Return ``+`` if mean(values) >= threshold; else ``-``.

    ``0`` is never returned by this helper — callers reserve ``0`` for
    "diagnostic does not apply to this cell" (e.g. trait cells have no
    within-subject paired contrast for c̄, so c̄ is forced to ``0`` at
    a higher level in :func:`build_verdict_matrix`).
    """
    if not values:
        return "-"
    return "+" if (sum(values) / len(values)) >= threshold else "-"


# Backwards-compatible name retained.
sign_of_median = positive_sign


def layer_sign(label_ba_first: float, label_ba_last: float, label_ba_max: float,
               argmax_depth: float, *, t: Thresholds = DEFAULT) -> str:
    """Compute the layer-probe sign + qualifier for one cell.

    The base sign is determined by whether ``label_ba_max`` clears the
    above-chance threshold. Qualifiers ``early`` / ``deep`` mark
    layer-depth concentration of the signal.
    """
    if label_ba_max >= t.layer_label_above_chance:
        # Robust above-chance signal exists somewhere along the depth.
        if (argmax_depth <= t.layer_early_fraction and
                (label_ba_max - label_ba_last) >= t.layer_early_drop_min):
            return "+ early"
        return "+"
    # Below-chance / weak signal.
    if label_ba_last <= t.layer_label_below_chance:
        return "- deep"
    return "-"


def oneoverf_role(state_drop_mean: float, subject_drop_mean: float,
                  *, has_within_subject_contrast: bool,
                  t: Thresholds = DEFAULT) -> OneOverFRole:
    """Classify the 1/f role from FOOOF aperiodic-ablation drops.

    Parameters
    ----------
    state_drop_mean : float
        Mean across FMs of ``original.state_probe - aperiodic_removed.state_probe``.
        Positive = label/state signal is partly in the 1/f component.
    subject_drop_mean : float
        Same, for the subject-ID probe.
    has_within_subject_contrast : bool
        True for W cells (EEGMAT, SleepDep). Trait cells (ADFTD, Stress)
        cannot disambiguate "subject confound" from "subject axis" since
        they have no within-subject paired contrast — report
        ``"subject axis"`` regardless.
    """
    if state_drop_mean > t.state_drop_threshold:
        return "state signal"
    if subject_drop_mean > t.subject_drop_threshold:
        if has_within_subject_contrast:
            return "subject confound"
        return "subject axis"
    return "subject axis"  # fallback for cells with no clear effect


def classify(
    *,
    delta_label_sign: Sign,
    layer_probe_sign: str,
    c_bar_sign: Sign,
    oneoverf: OneOverFRole,
    cell_layout: str,  # "W,C" / "T,C" / "W,N" / "T,N"
) -> Outcome:
    """Combine the four diagnostic signals into a Tab-3 outcome.

    The decision tree is anchored on cell layout because the within-vs-
    trait dichotomy determines whether ``c̄`` is informative. The four
    paper-locked outcomes are exhaustive over the cell layout × signal
    combinations the paper actually reports.
    """
    layer_base = layer_probe_sign.split()[0]  # strip "early" / "deep"
    if cell_layout == "W,C":
        # Within-subject + consensus marker. Expect aligned: label↑, layer↑, c̄↑, 1/f drives state.
        if delta_label_sign == "+" and layer_base == "+" and c_bar_sign == "+":
            return "Cross-subject-aligned"
    elif cell_layout == "T,C":
        # Trait + consensus marker: label↑, layer↑ (often early), c̄=0 by construction.
        if delta_label_sign == "+" and layer_base == "+":
            return "Label-subject coupled"
    elif cell_layout == "W,N":
        # Within-subject + no consensus marker: label↓, layer↓, c̄↓, 1/f drives subject.
        if delta_label_sign == "-" and layer_base == "-" and c_bar_sign in ("-", "0"):
            return "Idiosyncratic within-subject"
    elif cell_layout == "T,N":
        # Trait + no consensus marker: label↓, layer↓ (deep), c̄=0, no state signal.
        if delta_label_sign == "-" and layer_base == "-":
            return "Below linear-probe resolution"
    return "Uncategorized"


# Cell layout metadata (paper-locked).
CELL_LAYOUT: dict[str, str] = {
    "eegmat": "W,C",
    "adftd": "T,C",
    "sleepdep": "W,N",
    "stress": "T,N",
}
