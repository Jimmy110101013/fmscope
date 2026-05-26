# Contributing to FMScope

Thanks for your interest. FMScope is a small, focused diagnostic
toolkit accompanying a published paper, so the contribution surface is
intentionally narrow.

## Where contributions are most welcome

| Area | Examples |
|---|---|
| **New cohort adapters** | Wrappers around a public EEG dataset that satisfy `fmscope.CohortAdapter`. See `docs/byo_dataset.md`. |
| **New FM wrappers** | Wrappers around a published EEG foundation model that satisfy `fmscope.FMExtractor`. See `docs/byo_fm.md`. |
| **Bug reports** | Reproducible failures with minimal repro; please include `python -c "import fmscope; print(fmscope.__version__)"` and the smallest failing snippet. |

## Where contributions are out of scope

- **New diagnostic primitives** beyond the five the paper uses (variance
  decomposition, subject-axis erasure, 1/f ablation, layer-wise probe,
  direction consistency). We deliberately keep the diagnostic surface
  small. Generic representation-analysis tools belong upstream in
  EEG-FM-Bench's `baseline/analysis/`.
- **Fine-tuning recipes**. FMScope is an audit toolkit, not a training
  framework. Use the paper-bundled FT results in `reproduction/data/` or run
  your own training elsewhere and feed the features to `audit_cell()`.
- **Renaming / refactoring** of the public protocol surface
  (`FMExtractor`, `CohortAdapter`, `AuditConfig`). Stability matters
  more than aesthetics here.

## How to submit

1. Fork the repository.
2. Create a feature branch from `main`.
3. Run the full local gate before pushing:

   ```bash
   pytest
   ruff check fmscope/ tests/
   python scripts/scrub_check.py --strict
   ```

4. Open a PR with a description of *why* (the bug, the missing cohort,
   the design constraint) before *what* (the diff).

## Code style

- Follow `ruff` defaults; line length 100.
- Public functions get docstrings; private helpers (`_*`) don't need them.
- No type ignores without an inline reason.
- New diagnostics must have a regression test in `tests/`.

## Code of conduct

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
