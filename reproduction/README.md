# Paper reproduction

This folder contains everything needed to reproduce the figures and
tables in the JNE paper **"The Identity Trap in EEG Foundation Models:
A Diagnostic Audit"**. It is **separate from the FMScope library** at
`../fmscope/` — the library is general-purpose; this folder is
paper-specific.

```
reproduction/
├── data/                    # Bundled JSON aggregates + 12 NPZs (~55 MB)
├── builders/                # One script per paper figure / table
│   ├── tab1_master.py       # Paper Table 2 (per-cell BA matrix)
│   ├── tab_appendix_d.py    # Appendix D tables
│   ├── tab3_verdict.py      # Paper Table 3 (verdict matrix)
│   ├── fig2_variance.py
│   ├── fig3_layerwise.py
│   ├── fig4a_psd_fooof.py
│   ├── fig4b_fooof_scatter.py
│   ├── fig5_direction.py
│   └── results_accessor.py  # Canonical reader for data/
├── notebooks/               # Thin notebook orchestrators (papermill-testable)
└── tests/                   # Tab 2 md5 + Tab 3 outcome regression tests
```

## Quickstart — full reproduction in 5 minutes

```bash
git clone https://github.com/Jimmy110101013/fmscope.git
cd fmscope
pip install -e .[dev]               # installs the library; reproduction/ ships with the clone
pytest reproduction/tests/          # md5-pinned Tab 2 + 4/4 Tab 3 outcomes
```

If `pytest` is green, every paper number is byte-identical to the
manuscript snapshot. The pinned Tab 2 md5 is
`31b3f0fdaa3e1954aa7f9a322ce6a38e`.

## Per-figure / per-table builders

Run from the repo root. Outputs land in `./paper_figures/` by default
(override with `FMSCOPE_OUTPUT_DIR`).

| Builder | Produces |
|---|---|
| `python -m reproduction.builders.tab1_master` | `master_results_table.md`, `table2_master_performance.tex` |
| `python -m reproduction.builders.tab_appendix_d` | Appendix D tables |
| `python -m reproduction.builders.tab3_verdict` | Paper Table 3 (verdict matrix, Markdown + optional `--out file.tex`) |
| `python -m reproduction.builders.fig2_variance` | `fig2_variance_2x2.pdf` / `.png` |
| `python -m reproduction.builders.fig3_layerwise` | `fig3_layerwise_probe.pdf` / `.png` |
| `python -m reproduction.builders.fig4a_psd_fooof` | `fig4a_psd_fooof_fit.pdf` / `.png` |
| `python -m reproduction.builders.fig4b_fooof_scatter` | `fig4b_fooof_scatter.pdf` / `.png` |
| `python -m reproduction.builders.fig5_direction` | `fig5ab_rose_combined.pdf`, `fig5c_dir_consistency.pdf`, `fig5d_snr_ratio.pdf` |

Or run via notebooks under `reproduction/notebooks/`.

## Bundled data layout

```
reproduction/data/
├── source_tables/              # cross-cell aggregates (per-row per-FM)
├── layerwise_probe/            # exp35 — fig3, verdict
├── null_calibration/           # exp37 — fig2's random-Gaussian null
├── temporal_block_probe/       # exp33 — fig4b subject-axis probe
├── fooof_ablation/             # paper §4.3
├── eegmat/, sleepdep/, adftd/, stress/    # per-cell LP / FT / classical / fooof
├── features_cache/             # 12 frozen NPZs, ~50 MB, used by fig5
└── fig4a_psd_cache.json        # 35 KB pre-computed PSD cache
```

Raw EEG features (5 GB) are **not** bundled. The PSD cache and frozen
NPZ subsets carry exactly the numbers each figure needs. Live FT / LP
re-runs against raw EEG are out of scope here — see the main toolkit
documentation for how to wire up your own data.

## What if a number drifts?

If `pytest reproduction/tests/` fails:

1. **Bundled JSONs drifted from the paper snapshot.** The Tab 2 md5 is
   pinned in `tests/test_paper_reproduce.py::test_tab2_master_md5_matches`.
2. **Rubric thresholds changed.** Default thresholds live in
   `reproduction.builders._rubric.Thresholds` (paper-locked). The Tab 3 outcome
   strings are pinned in `tests/test_paper_reproduce.py::test_verdict_matrix_matches_paper_tab3`.
3. **Builder logic changed.** Check `git diff reproduction/builders/`
   against the pinned release.

The Tab 2 md5 gate is the EIC-critical check — do not ship a release
where it fails.

## Where this differs from the FMScope library

The library `fmscope` (at `../fmscope/`) is the toolkit:
`audit_cell(cohort, extractor)` runs the diagnostic stack on **any**
user-supplied cohort and FM. It does not depend on `reproduction/data/`.

`reproduction/builders/tab3_verdict.py` is the paper-specific applicator:
it aggregates the four diagnostic signals from the bundled four-cell
JSONs into the paper's exact Table 3 outcome strings. If you're
auditing a new cohort, use `audit_cell()` instead; if you're
reproducing the paper's exact numbers, use this folder.
