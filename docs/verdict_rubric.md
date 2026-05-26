# Verdict rubric

> **Reproduction-only.** The rubric is **not** part of the FMScope
> toolkit — `audit_cell` reports numbers, not outcomes. This +/−/0
> classification lives under `reproduction/builders/_rubric.py` and exists
> solely to reproduce the paper's Table 3. It documents *the paper's*
> interpretation of the diagnostic numbers, not a toolkit feature.

FMScope's reproduction layer produces a per-cell verdict matrix matching
paper Table 3 (`tab:verdict_matrix`). Each cell receives one of four
outcomes, derived mechanically from four diagnostic signals.

## The four diagnostic columns

| Column | Source | Sign meaning |
|---|---|---|
| **Δf_label** | `variance_analysis_window_level.json` (`delta_label_frac`) | `+` if label-fraction rises from frozen → fine-tuned (mean across FMs ≥ 0.005). `−` otherwise. Reserve `0` for "not applicable". |
| **layer probe** | `exp35_layerwise_probe/probes.json` | `+` if max(`label_ba`) ≥ 0.60 along depth; `+ early` if the peak is in the first 35% of depth AND drops ≥ 0.04 to the final layer; `− deep` if final-layer `label_ba` ≤ 0.45; `−` otherwise. |
| **c̄** | `within_subject_dir_consistency.json` (`dir_consistency`) | `+` if mean ≥ 0.05; `−` otherwise. **Forced to `0` for trait cells (T,*) by construction** — no within-subject paired contrast exists. |
| **1/f role** | `fooof_ablation/probes.json` (state_probe and subject_probe drops) | `state signal` if `state_drop > 0.03`; `subject confound` if `subject_drop > 0.05` AND cell has within-subject contrast; `subject axis` otherwise. |

All thresholds are exposed as a `Thresholds` dataclass in
`reproduction.builders._rubric` — override them if recalibrating against
a new cohort set.

## Cell layout × outcome decision tree

Cell layout combines two binary axes:

* **W vs T** — Within-subject (paired contrast) vs Trait (one label per subject).
* **C vs N** — Consensus cross-subject marker present (e.g. theta-band
  during EEGMAT arithmetic load) vs no consensus marker.

The four bundled cells:

| Cell | Layout | Expected outcome |
|---|---|---|
| EEGMAT | (W, C) | **Cross-subject-aligned** |
| ADFTD | (T, C) | **Label–subject coupled** |
| SleepDep | (W, N) | **Idiosyncratic within-subject** |
| Stress | (T, N) | **Below linear-probe resolution** |

`classify(...)` in `reproduction.builders._rubric` implements the decision
logic. Trait cells (T,*) have `SS_label ⊆ SS_subject` by construction,
so both `Δf_label = +` and `layer probe = +` simply indicate that the
diagnostic signal exists at all — it does not break the trait
structure. That is why ADFTD reports `(+, + early, 0)` and lands at
"Label–subject coupled" rather than the W-cell "Cross-subject-aligned"
outcome.

## Reading a verdict

| Outcome | Operational meaning |
|---|---|
| **Cross-subject-aligned** | FM picks up a label-discriminative axis that generalizes across subjects. Linear probe + finetune both improve label decoding without confounding subject identity. |
| **Label–subject coupled** | Label is fully nested in subject; you cannot tell whether the FM learned the label or memorized subject identity. This is the structural ceiling of trait cells with consensus markers. |
| **Idiosyncratic within-subject** | Within-subject contrast exists but does not generalize. Each subject has their own label-axis direction; cross-subject pooling collapses it. |
| **Below linear-probe resolution** | No detectable label signal at the linear-probe layer. Either the signal is absent, too subtle, or buried under subject variance — your study design cannot disambiguate. |

## Calibration

The thresholds were locked against the paper cells before public
release. If you apply FMScope to a new dataset:

1. Run `reproduction.builders.tab3_verdict.build_verdict_matrix()` and
   read the column signs.
2. Cross-check the underscore-prefixed provenance columns
   (`_delta_label_values`, `_layer_probe_info`, `_c_bar_values`,
   `_oneoverf_info`) to see the underlying numbers.
3. If the signs disagree with what you expect from the raw numbers,
   recalibrate `Thresholds(...)` rather than reinterpreting the rubric.

The rubric is intentionally simple — four binary signs and one
3-valued role. Calibration drift is easier to debug than rubric drift.
