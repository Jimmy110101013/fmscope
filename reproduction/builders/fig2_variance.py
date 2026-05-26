"""Build Fig 2 — 2×2 grid of representation-structure panels, one per dataset.

Usage (from repo root):
    python -m reproduction.builders.fig2_variance

Layout (rows: within-subject paired vs subject-label trait;
       columns: prior cross-subject marker vs no prior cross-subject marker —
       axis fixed *a priori* from literature, see Methods §3.1.2):
  (0,0) EEGMAT   — within × prior-marker
  (0,1) SleepDep — within × no-prior-marker
  (1,0) ADFTD    — trait × prior-marker
  (1,1) Stress   — trait × no-prior-marker

Each panel shows variance decomposition bars: 3 FMs × {frozen, FT}, with
subject_frac (lower) + label_frac (upper, colored). Numerical callouts
(Δlabel_frac, dir_consistency) live in the figure caption / paper §4.2,
not on the panel.

Output: paper/figures/main/fig2_variance_2x2.{pdf,png}
"""
from __future__ import annotations
import sys
from pathlib import Path

from reproduction.builders._runtime import output_dir, paper_data_path
# fmscope is pip-installed; no sys.path hacks needed
# fmscope is pip-installed; no sys.path hacks needed

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from reproduction.builders._style import apply_jne_style, JNE_DOUBLE_COL_CM, CM  # noqa: E402

OUT  = output_dir("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

apply_jne_style()

FMS        = ["labram", "cbramod", "reve"]
FM_PRETTY  = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
FM_COLOR   = {"labram": "#1f3a5f", "cbramod": "#B8442C", "reve": "#2E8B57"}

# ── load source tables via canonical accessor ──────────────
import json  # noqa: E402

from reproduction.builders import results_accessor as results

# Unified window-level decomposition for all 4 cells. Window-level instead of
# recording-level because ADFTD has 1 session/subject, which makes the
# recording-level decomposition degenerate (n_rec == n_subj, 0 within-subject df).
va_win = results.source_table("variance_analysis_window_level")

# Matched random-Gaussian null reference. Used as a dashed horizontal
# overlay on each (cell, FM) bar to anchor the "f_subj exceeds combinatorial
# null by 13–84×" claim. Closed-form prediction E[f_subj] ≈ (S-1)/(N-1) matches
# empirical null mean to 3 decimal places.
NULL_VAR_PATH = paper_data_path("null_calibration", "null_vs_real.json")
with open(NULL_VAR_PATH) as _fh:
    NULL_VAR = json.load(_fh)


def null_subject_pct(fm: str, ds: str) -> float | None:
    """Mean null subject_frac for this (cell, FM) pair, in percent. None if missing."""
    entry = NULL_VAR.get(f"{fm}_{ds}")
    if entry is None or "null_subject_frac_mean" not in entry:
        return None
    return float(entry["null_subject_frac_mean"]) * 100.0


def variance_entry(fm: str, ds: str) -> dict | None:
    """Return a flat percent-based dict matching the legacy recording-level schema.

    Window-level table stores fractions in [0, 1] under nested `frozen` / `ft`
    keys; this adapter scales to percent and flattens for downstream code.
    """
    key = f"{fm}_{ds}"
    raw = va_win.get(key)
    if raw is None or "frozen" not in raw:
        return None
    fz, ft = raw["frozen"], raw.get("ft") or {}
    return {
        "frozen_label_frac":   fz["label_frac"]   * 100,
        "frozen_subject_frac": fz["subject_frac"] * 100,
        "ft_label_frac":       (ft.get("label_frac")   or 0) * 100 if ft else None,
        "ft_subject_frac":     (ft.get("subject_frac") or 0) * 100 if ft else None,
        "delta_label_frac":    (raw.get("delta_label_frac") or 0) * 100,
    }


# ── panel definitions ──────────────────────────────────────
PANELS = [
    ("eegmat",   "EEGMAT",        "Within-subject × consensus marker"),
    ("sleepdep", "SleepDep",      "Within-subject × no consensus marker"),
    ("adftd",    "ADFTD",         "Subject/label-trait × consensus marker"),
    ("stress",   "Stress (DASS)", "Subject/label-trait × no consensus marker"),
]


def draw_panel(ax, ds: str, pretty: str, quadrant: str):
    """Variance bars + callout for one dataset."""
    xticks = []
    xlabels = []
    for i, fm in enumerate(FMS):
        entry = variance_entry(fm, ds)
        if entry is None:
            xticks.extend([3*i, 3*i+1]); xlabels.extend([f"{FM_PRETTY[fm]}\nfrz", f"{FM_PRETTY[fm]}\nft"])
            continue
        fl = entry.get("frozen_label_frac")     or 0
        fs = entry.get("frozen_subject_frac")   or 0
        tl = entry.get("ft_label_frac")         or 0
        ts = entry.get("ft_subject_frac")       or 0

        # frozen bar (index 3*i)
        ax.bar(3*i,     fs, width=0.8, color=FM_COLOR[fm], alpha=0.35, edgecolor="k", lw=0.5)
        ax.bar(3*i,     fl, width=0.8, bottom=fs, color=FM_COLOR[fm], alpha=0.95, edgecolor="k", lw=0.5)
        # FT bar (index 3*i+1)
        ax.bar(3*i+1,   ts, width=0.8, color=FM_COLOR[fm], alpha=0.35, edgecolor="k", lw=0.5, hatch="///")
        ax.bar(3*i+1,   tl, width=0.8, bottom=ts, color=FM_COLOR[fm], alpha=0.95, edgecolor="k", lw=0.5, hatch="///")

        # null reference (matched random Gaussian). Same null for both
        # frozen and FT (depends only on N, D, S, L). Visually anchors that real
        # f_subj is 13–84× the combinatorial baseline.
        null_pct = null_subject_pct(fm, ds)
        if null_pct is not None:
            ax.hlines(null_pct, 3*i - 0.45, 3*i + 1.45,
                      colors="#B22222", linestyles="--", lw=1.3, zorder=5)

        # label_frac value above each stack (kept; callout box removed earlier)
        ax.text(3*i,   fs + fl + 2, f"{fl:.1f}", ha="center", fontsize=6.5, fontweight="bold")
        ax.text(3*i+1, ts + tl + 2, f"{tl:.1f}", ha="center", fontsize=6.5, fontweight="bold")

    # Put frz/FT labels inside the bars instead of as xticks, use xticks only for FM centers
    ax.set_xticks([3*i + 0.5 for i in range(len(FMS))])
    ax.set_xticklabels([FM_PRETTY[fm] for fm in FMS], fontsize=8, fontweight="bold")
    for i, fm in enumerate(FMS):
        ax.get_xticklabels()[i].set_color(FM_COLOR[fm])
    # frz/FT tiny labels inside bars (upper region, white on bar)
    for i, fm in enumerate(FMS):
        entry = variance_entry(fm, ds)
        if entry is None: continue
        ax.text(3*i,   4, "frz", ha="center", fontsize=6, color="white", fontweight="bold")
        ax.text(3*i+1, 4, "FT",  ha="center", fontsize=6, color="white", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, 8)
    ax.set_ylabel("variance explained (%)" if ds in ("eegmat", "adftd") else "")
    ax.set_title(f"{pretty}\n({quadrant})", fontsize=9, pad=4)
    ax.grid(axis="y", alpha=0.25, lw=0.4)

    # Per-panel annotation of the null subject_frac magnitude. We pull the
    # smallest and largest across the 3 FMs in this cell (in practice they
    # agree to 2 decimal places because they share N, S, L; D differs but
    # cancels for the marginal subject SS).
    null_vals = [null_subject_pct(fm, ds) for fm in FMS]
    null_vals = [v for v in null_vals if v is not None]
    if null_vals:
        nmean = sum(null_vals) / len(null_vals)
        ax.text(0.98, 0.96, f"null $f_{{\\mathrm{{subj}}}}$ ≈ {nmean:.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color="#B22222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#B22222", lw=0.5, alpha=0.95))


# ── figure ─────────────────────────────────────────────────
def main():
    W  = JNE_DOUBLE_COL_CM * CM
    fig, axes = plt.subplots(2, 2, figsize=(W, W*0.75), sharey=True)
    for ax, (ds, pretty, quadrant) in zip(axes.flat, PANELS):
        draw_panel(ax, ds, pretty, quadrant)

    # shared legend
    legend = [
        Patch(facecolor="#888", alpha=0.35, edgecolor="k", lw=0.5, label="subject_frac"),
        Patch(facecolor="#888", alpha=0.95, edgecolor="k", lw=0.5, label="label_frac"),
        Patch(facecolor="white", edgecolor="k", lw=0.5, label="frozen"),
        Patch(facecolor="white", edgecolor="k", lw=0.5, hatch="///", label="fine-tuned"),
        Line2D([0], [0], color="#B22222", ls="--", lw=1.3, label="null subject_frac"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    #fig.suptitle("Representation structure across the 2×2 factorial",
    #            fontsize=10, y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    fig.savefig(OUT / "fig2_variance_2x2.pdf")
    fig.savefig(OUT / "fig2_variance_2x2.png")
    print(f"saved → {(OUT/'fig2_variance_2x2.pdf').name} + .png")


if __name__ == "__main__":
    main()
