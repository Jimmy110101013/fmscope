# Reproducing paper results

All paper-specific code, data, tests, and the verdict rubric live under
[`reproduction/`](../reproduction/). The manuscript's tables and figures
reproduce from bundled JSON aggregates + frozen-feature caches — no raw
EEG, no FM weights, no GPU. See
[`reproduction/README.md`](../reproduction/README.md) for the full recipe;
this page is the short version.

## Fresh-clone reproduction

```bash
git clone https://github.com/Jimmy110101013/fmscope.git
cd fmscope
pip install -e .[dev]
pytest tests/ reproduction/tests/             # toolkit + reproduction tests
python -m reproduction.builders.tab3_verdict  # paper Table 3 (verdict matrix)
```

The Tab 2 byte-identical check pins md5 `31b3f0fdaa3e1954aa7f9a322ce6a38e`.

## Per-figure / table builders

Each artifact has one standalone builder under `reproduction.builders`.
Builders read via `reproduction.builders.results_accessor` and write to
`./paper_figures` / `./paper_tables` by default (override with
`FMSCOPE_OUTPUT_DIR`).

| Builder | Produces |
|---|---|
| `python -m reproduction.builders.tab1_master` | Table 2 (`master_results_table.md`, `.tex`) |
| `python -m reproduction.builders.tab3_verdict` | Table 3 (verdict matrix) |
| `python -m reproduction.builders.tab_leace` | Subject-axis erasure table |
| `python -m reproduction.builders.tab_appendix_leace_gen` | Erasure-generalization table |
| `python -m reproduction.builders.tab_appendix_d` | Appendix D tables |
| `python -m reproduction.builders.fig2_variance` | `fig2_variance_2x2.{pdf,png}` |
| `python -m reproduction.builders.fig3_layerwise` | `fig3_layerwise_probe.{pdf,png}` |
| `python -m reproduction.builders.fig4a_psd_fooof` | `fig4a_psd_fooof_fit.{pdf,png}` |
| `python -m reproduction.builders.fig4b_fooof_scatter` | `fig4b_fooof_scatter.{pdf,png}` |
| `python -m reproduction.builders.fig5_direction` | `fig5ab_rose_combined.pdf`, `fig5c_dir_consistency.pdf`, `fig5d_snr_ratio.pdf` |

## Bundled data + accessor

Bundled under `reproduction/data/` (`source_tables/`, `layerwise_probe/`,
`null_calibration/`, `temporal_block_probe/`, per-cell `fooof_ablation/`,
and a `features_cache/` of frozen NPZ subsets). Every builder reads
through `reproduction.builders.results_accessor`, so a re-bundle with a
different layout only touches the accessor:

```python
from reproduction.builders import results_accessor as results
results.source_table("master_results_table_ba")
results.lp_multiseed("eegmat", "labram")
```

The `tab_leace` builder is the exception — it live-computes the
subject-axis erasure table from the bundled per-window caches via
`fmscope.diagnostics.subject_axis_erasure`, reproducing the table
(including ±SD) and exercising the released diagnostic.
