"""Build Fig 4b — FOOOF ablation signature per FM (Δ BA scatter), 2-row.

Usage (from repo root):
    python -m reproduction.builders.fig4b_fooof_scatter

2×3 panel grid (3 FMs × 2 representation states).
  Top row    — frozen Δ probe BA: circle = remove aperiodic, square = remove
               periodic; line within each FM connects the two ablation
               conditions per dataset.
  Bottom row — FT Δ probe BA under both ablation conditions (same marker
               convention). Tests whether the frozen aperiodic-subject
               coupling persists when the input is altered upstream of FT.

Read **within-FM only** — cross-FM magnitudes are confounded by feature-dim
and architecture differences.

Top-row source: state probe = `results.fooof_ablation_probes(cell)`,
                 subject probe = `results.subject_probe_temporal_block(cell)`.
Bottom-row source: `results/final/source_tables/fooof_ft_probe_ba_3seed.json`
                 (3-seed mean Δsubj_pp / Δstate_pp computed by
                 `scripts/analysis/run_fooof_ft_probe_comparison.py`).

Output: paper_v4/figures/main/fig4b_fooof_scatter.{pdf,png}
        and paper/figures/fig5/ for legacy callers.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

from reproduction.builders._runtime import output_dir, paper_data_path
# fmscope is pip-installed; no sys.path hacks needed
# fmscope is pip-installed; no sys.path hacks needed

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from reproduction.builders import results_accessor as results  # noqa: E402
from reproduction.builders._style import (apply_jne_style, JNE_SINGLE_COL_CM,  # noqa: E402
                        JNE_DOUBLE_COL_CM, CM)

OUT_PRIMARY = output_dir("paper_figures")
OUT_LEGACY = output_dir("paper_figures")
for d in (OUT_PRIMARY, OUT_LEGACY):
    d.mkdir(parents=True, exist_ok=True)

apply_jne_style()

W_SINGLE = JNE_SINGLE_COL_CM * CM
W_DOUBLE = JNE_DOUBLE_COL_CM * CM
ROW_HEIGHT = 4.4 * CM   # 2-row total ~8.8cm at 15cm-wide double-col

FMS = ["labram", "cbramod", "reve"]
FM_COLOR = {"labram": "#1f3a5f", "cbramod": "#B8442C", "reve": "#2E8B57"}

DS_PROBE = ["eegmat", "sleepdep", "stress", "adftd"]
DS_SHORT = {"eegmat": "EEGMAT", "sleepdep": "SleepDep",
            "stress": "Stress", "adftd": "ADFTD"}
DS_CMAP = {"eegmat": "#2E8B8B", "sleepdep": "#7A4B9C",
           "stress": "#D55E00", "adftd": "#1F77B4"}

COND_MAP = [("aperiodic_removed", "remove aperiodic"),
            ("periodic_removed", "remove periodic")]
MARKERS = {"remove aperiodic": "o", "remove periodic": "s"}


def _load_ft_probe_canonical():
    p = paper_data_path("source_tables/fooof_ft_probe_ba_3seed.json")
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def main() -> None:
    F_STATE = {ds: results.fooof_ablation_probes(ds)["results"] for ds in DS_PROBE}
    F_SUBJ = {ds: results.subject_probe_temporal_block(ds)["results"] for ds in DS_PROBE}
    FT_PROBE = _load_ft_probe_canonical()
    have_ft = FT_PROBE is not None

    n_rows = 2 if have_ft else 1
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(W_DOUBLE, ROW_HEIGHT * n_rows),
        sharey=False, sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    # ── Top row: frozen probe Δ ────────────────────────────────────
    for ax, fm in zip(axes[0], FMS):
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        for ds in DS_PROBE:
            pts = []
            for cond, short in COND_MAP:
                dsub = (F_SUBJ[ds][fm][cond]["subject_probe_mean"]
                        - F_SUBJ[ds][fm]["original"]["subject_probe_mean"]) * 100
                dsta = (F_STATE[ds][fm][cond]["state_probe_mean"]
                        - F_STATE[ds][fm]["original"]["state_probe_mean"]) * 100
                pts.append((short, dsub, dsta))
            ax.plot([p[1] for p in pts], [p[2] for p in pts],
                    color=DS_CMAP[ds], lw=1.0, alpha=0.4, zorder=2)
            for short, dsub, dsta in pts:
                ax.scatter(dsub, dsta, s=45, color=DS_CMAP[ds],
                           marker=MARKERS[short], edgecolor="k", lw=0.4,
                           zorder=3)
        ax.set_title(fm.upper(), fontsize=10, color=FM_COLOR[fm], fontweight="bold")
        ax.tick_params(labelsize=7.5)
        ax.grid(True, ls=":", lw=0.3, alpha=0.5)
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.set_xlim(-45, 45)
        ax.set_ylim(-14, 14)

    axes[0][0].set_ylabel("Δ State probe BA (pp)\n[Frozen]", fontsize=8.5)

    # ── Bottom row: FT probe Δ under both ablations ────────────────
    if have_ft:
        for ax, fm in zip(axes[1], FMS):
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            for ds in DS_PROBE:
                key = f"{fm}_{ds}"
                entry = FT_PROBE.get(key)
                if entry is None:
                    continue
                pts = []
                for cond, short in COND_MAP:
                    sub = entry.get(cond) if isinstance(entry, dict) else None
                    # Back-compat: legacy flat schema (aperiodic-only)
                    if sub is None and cond == "aperiodic_removed" and \
                            isinstance(entry, dict) and "delta_subj_pp_mean" in entry:
                        sub = entry
                    if sub is None:
                        continue
                    pts.append((short,
                                sub["delta_subj_pp_mean"],
                                sub["delta_state_pp_mean"]))
                if len(pts) > 1:
                    ax.plot([p[1] for p in pts], [p[2] for p in pts],
                            color=DS_CMAP[ds], lw=1.0, alpha=0.4, zorder=2)
                for short, dsub, dsta in pts:
                    ax.scatter(dsub, dsta, s=45, color=DS_CMAP[ds],
                               marker=MARKERS[short], edgecolor="k", lw=0.4,
                               zorder=3)
            ax.tick_params(labelsize=7.5)
            ax.grid(True, ls=":", lw=0.3, alpha=0.5)
            ax.set_xticks([-40, -20, 0, 20, 40])
            ax.set_yticks([-10, -5, 0, 5, 10])
            ax.set_xlim(-45, 45)
            ax.set_ylim(-14, 14)
            ax.set_xlabel("Δ Subject probe BA (pp)", fontsize=8)

        axes[1][0].set_ylabel("Δ State probe BA (pp)\n[FT, intervention]", fontsize=8.5)
    else:
        for ax in axes[0]:
            ax.set_xlabel("Δ Subject probe BA (pp)", fontsize=8)

    # ── Legends ────────────────────────────────────────────────────
    cond_leg = [
        Line2D([], [], marker="o", ls="", color="gray",
               markeredgecolor="k", markersize=7, label="−aperiodic"),
        Line2D([], [], marker="s", ls="", color="gray",
               markeredgecolor="k", markersize=7, label="−periodic"),
    ]
    ds_leg = [
        Line2D([], [], marker="o", ls="", color=DS_CMAP[ds],
               markeredgecolor="k", markersize=7,
               label=DS_SHORT[ds])
        for ds in DS_PROBE
    ]
    fig.legend(handles=cond_leg + ds_leg, loc="lower center", ncol=6,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, handletextpad=0.5, columnspacing=1.4)

    plt.tight_layout(rect=[0, 0.06 if have_ft else 0.07, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_PRIMARY / f"fig4b_fooof_scatter.{ext}")
    plt.close(fig)
    rows_msg = "2 rows (frozen + FT)" if have_ft else "1 row (frozen only)"
    print(f"wrote fig4b_fooof_scatter.{{pdf,png}} → "
          f"{OUT_PRIMARY.name} ({rows_msg})")


if __name__ == "__main__":
    main()
