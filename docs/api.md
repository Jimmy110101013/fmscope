# FMScope API surface

FMScope exposes two layers:

1. **Public protocols** (`fmscope.api`) — duck-typed contracts for any EEG
   foundation model or cohort.
2. **Diagnostic library** (`fmscope.diagnostics`, `fmscope.preprocess`,
   `fmscope.training`, `fmscope.verdict`) — the five frozen-representation
   diagnostics plus the `audit_cell` orchestrator.

Every public entry point is a plain Python callable. The toolkit reports
**numbers**; it does not classify. The paper's verdict rubric (the +/−/0
outcome mapping) lives under `reproduction/`, not in the package.

## Top-level imports

```python
from fmscope import FMExtractor, CohortAdapter   # public protocols
from fmscope.data import (                        # cohort builders (BYO)
    InMemoryCohort, PyTorchDatasetAdapter, synthetic_cohort,
)
from fmscope.diagnostics import (                 # diagnostic stats
    crossed_ss_fractions, nested_ss, null_control,    # (1) variance + null
    subject_axis_erasure,                             # (2) LEACE erasure
    layer_probe,                                      # (4) layer-wise probe
    direction_consistency,                            # (5) c̄
    # auxiliary variance methods:
    cluster_bootstrap, mixed_effects_variance,
    subject_level_permanova, label_subspace_analysis,
)
from fmscope.preprocess import fooof_ablate       # (3) aperiodic ablation
from fmscope.training import run_canonical_lp      # linear-probe trainer
from fmscope.verdict import audit_cell, AuditConfig
```

## Protocol surface

### `FMExtractor`

```python
@runtime_checkable
class FMExtractor(Protocol):
    embed_dim: int
    def __call__(self, x): ...  # (B, C, T) -> (B, embed_dim)
```

Wrap any pretrained model into this protocol — see
[`docs/byo_fm.md`](byo_fm.md). FMScope ships no FM model code.

### `CohortAdapter`

```python
@runtime_checkable
class CohortAdapter(Protocol):
    n_channels: int
    sfreq: float
    def iter_recordings(self) -> Iterator[tuple[int, int, np.ndarray]]: ...
```

Wrap any cohort source — see [`docs/byo_dataset.md`](byo_dataset.md).

## The five diagnostics

| Function | Tool | Purpose |
|---|---|---|
| `crossed_ss_fractions(features, subject, label)` | 1 | Subject × label crossed SS fractions (any cell layout). |
| `nested_ss(features, subject, label)` | 1 | Subject-nested-in-label additive variance partition. |
| `null_control(features, subject, label, n_null_seeds=20)` | 1 | Random-Gaussian null calibration for the SS fractions. |
| `subject_axis_erasure(features, subject, label=None)` | 2 | LEACE subject-axis erasure; re-probes identity (linear + nonlinear MLP) and the label (`Δ_erase`, 3-seed) under a 0.55 gate. |
| `fooof_ablate(...)` (`fmscope.preprocess`) | 3 | Remove the aperiodic (1/f) component before re-extraction. |
| `layer_probe(...)` | 4 | Subject + label linear probe across FM depths. |
| `direction_consistency(features, subject, label)` | 5 | c̄ — median pairwise cosine of per-subject label directions. |

Auxiliary variance methods (used by the paper's robustness analyses, not
the core recipe): `cluster_bootstrap`, `mixed_effects_variance`,
`subject_level_permanova`, `label_subspace_analysis`.

## Audit orchestrator

```python
from fmscope.verdict import audit_cell, AuditConfig

row = audit_cell(
    my_cohort, my_extractor,
    config=AuditConfig(cell_name="MyCohort", device="cuda:0", n_null_seeds=20),
)
```

Returns a **numbers-only** dict: variance fractions (`label_frac`,
`subject_frac`, `residual_frac`), null excess ratios, `c_bar_value`, and
`erasure_*` columns (when `run_erasure=True`). Pass pre-computed
`layer_probe` / `oneoverf` summaries via `AuditConfig` to surface the
layer-probe / FOOOF columns. There is no `outcome` field — the consumer
interprets the numbers. See [`docs/byo_dataset.md`](byo_dataset.md).

## Linear probe

```python
from fmscope.training import run_canonical_lp

results = run_canonical_lp(extractor="labram", dataset="eegmat",
                           cv="stratified-kfold", n_splits=5)
```

Subject-level `StratifiedGroupKFold` linear probe with recording-level
mean-pooling — the same probe `subject_axis_erasure` uses for `Δ_erase`.

## Reproducing the paper

Paper figures, tables, and the verdict rubric live under
[`reproduction/`](../reproduction/) and read bundled JSON / frozen
features via `reproduction.builders.results_accessor`. See
[`docs/reproducing_paper.md`](reproducing_paper.md).
