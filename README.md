# FMScope

> Companion code for the paper
> **"The Identity Trap in EEG Foundation Models: A Diagnostic Audit"**.

> **Diagnostic toolkit for auditing EEG foundation models.**
> Measures whether a frozen EEG foundation model (FM) encodes the task
> label, or encodes subject identity that happens to correlate with it.

## Overview

FMScope answers a single, sharp question about any frozen EEG foundation
model: **is its apparent task skill real, or is it reading subject
identity that happens to correlate with the label?** Five
frozen-representation diagnostics, run against a principled null and an
explicit scope condition, turn that question into numbers for any
(cohort, FM) pair — so you can see exactly where a model's performance
comes from before you trust it.

Numbers in, numbers out: FMScope reports the measurements and leaves the
calls to you. The paper's interpretation layer (the verdict rubric that
maps these numbers to Table 3) lives separately under
[`reproduction/`](reproduction/), so the toolkit stays a clean,
dependency-light instrument you can drop onto your own data.

## Install

```bash
git clone https://github.com/Jimmy110101013/fmscope.git
cd fmscope
pip install -e .[dev]          # core + pytest / ruff / mypy / papermill
```

Python ≥3.10. For CUDA wheels of `torch`, follow the
[official PyTorch install matrix](https://pytorch.org/get-started/locally/)
first. FMScope ships **no** FM weights and **no** vendored model code —
you bring your own FM (see [Audit your own data](#audit-your-own-data)).

## The five diagnostics

Each answers one question and carries a **scope condition** — the cells
where it returns a defined answer. A cohort need not exercise all five.
Every diagnostic lives in `fmscope.diagnostics` (FOOOF ablation in
`fmscope.preprocess`) and returns numbers only.

| # | Diagnostic | Question | Published method |
|---|---|---|---|
| 1 | **Variance decomposition** (`crossed_ss_fractions`, `nested_ss`) + random-Gaussian **null** (`null_control`) | Does subject identity dominate the representation? | Crossed/nested SS (Edwards 2007); combinatorial null `E[f_subj]≈(S−1)/(N−1)` |
| 2 | **Subject-axis erasure** (`subject_axis_erasure`) | Is that dominance confined to a linearly removable axis — and does removing it *help* the label? | LEACE, Belrose et al. NeurIPS 2023 |
| 3 | **Aperiodic (FOOOF) ablation** (`fmscope.preprocess.fooof_ablate`) | Is the 1/f spectral component the carrier of identity? | FOOOF / specparam, Donoghue et al. 2020 |
| 4 | **Layer-wise probe** (`layer_probe`) | At what depth does the label become linearly separable? | Linear classifier probes, Alain & Bengio 2017 |
| 5 | **Direction consistency** c̄ (`direction_consistency`) | Do subjects encode the task contrast along a *shared* direction? | Median pairwise cosine of per-subject label directions |

**On diagnostic 2 vs concurrent work.** Tang et al. (*What Do EEG
Foundation Models Capture from Human Brain Signals?*, arXiv:2605.11410,
2026) apply LEACE-style erasure to EEG FMs in the **opposite direction**:
they erase hand-crafted neuro-features to show that removing them *hurts*
the probe. FMScope erases **subject identity** and reports the
**complementary** effect — removing the identity that correlates with the
label can *help* the label probe (`Δ_erase ≥ 0`). We erase only the
**linear** subject axis; the nonlinear residual is measured (an MLP
probe) and reported, never hidden.

## Audit your own data

```python
from fmscope.data import InMemoryCohort          # or wrap your torch Dataset
from fmscope.verdict import audit_cell, AuditConfig

# 1. Wrap your data as (subject_id, label, windows) recordings.
#    See docs/byo_dataset.md for the CohortAdapter protocol.
cohort = InMemoryCohort(recordings, n_channels=19, sfreq=200.0)

# 2. Wrap your frozen FM as a (B, C, T) -> (B, D) callable.
#    See docs/byo_fm.md and examples/byo_fm_minimal.py.
my_fm = MyExtractor()

# 3. Run the audit — it returns NUMBERS, not a verdict.
row = audit_cell(cohort, my_fm, config=AuditConfig("MyDataset", device="cuda:0"))

print(row["label_frac"], row["subject_frac"])           # variance decomposition (1)
print(row["erasure_subj_ba_linear_pre"],                # subject-axis erasure (2)
      row["erasure_subj_ba_linear_post"],
      row["erasure_label_ba_delta"])
print(row["c_bar_value"])                                # direction consistency (5)
```

`audit_cell` runs diagnostics 1, 2, 5 directly; pass pre-computed
layer-probe / FOOOF-ablation summaries via `AuditConfig` to surface
diagnostics 4 / 3 in the same row. It never returns an outcome glyph —
you read the numbers.

## Reproduce the paper

All paper-specific code, data, and tests — including the verdict **rubric**
that maps these numbers to the paper's Table 3 outcomes — live under
[`reproduction/`](reproduction/). Numbers come from JSON aggregates and
bundled frozen features. No raw EEG, no FM weights, no GPU.

```bash
pytest reproduction/tests/                            # paper repro tests
python -m reproduction.builders.tab3_verdict          # Table 3 (verdict matrix)
python -m reproduction.builders.tab1_master           # Table 2
python -m reproduction.builders.tab_leace             # Tab. subject-axis erasure
python -m reproduction.builders.tab_appendix_leace_gen  # erasure generalization
python -m reproduction.builders.fig2_variance         # ...fig3, fig4a/b, fig5_direction
```

See [`reproduction/README.md`](reproduction/README.md) for the full recipe.

## Bring your own FM / dataset

- `examples/byo_fm_minimal.py` — wrap any `(B, C, T) → (B, D)` callable as
  an `FMExtractor`.
- `examples/byo_dataset_minimal.py` — wrap any iterator of
  `(subject_id, label, windows)` recordings as a `CohortAdapter`.

Protocols: [`docs/byo_fm.md`](docs/byo_fm.md),
[`docs/byo_dataset.md`](docs/byo_dataset.md). API surface:
[`docs/api.md`](docs/api.md).

## Datasets

Public cohorts used in the paper's four-cell layout:

| Cohort | Source | License |
|---|---|---|
| EEGMAT (mental arithmetic) | [PhysioNet eegmat 1.0.0](https://physionet.org/content/eegmat/1.0.0/) | ODbL v1.0 |
| SleepDep (sleep deprivation) | [OpenNeuro ds004902](https://openneuro.org/datasets/ds004902) | CC0 |
| ADFTD (Alzheimer's / FTD) | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) | CC0 |

A fourth cohort — resting-state stress (Taiwan graduate-student dataset,
2020; see paper) — is **collaboration-only** and not redistributed; its
verdict-matrix row is reproducible from bundled cached JSON.

## License

[MIT](LICENSE).

## Citing

See [`CITATION.cff`](CITATION.cff). The paper is currently under review.
