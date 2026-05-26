"""Build Fig 3 — layer-wise subject and label probes across the 2x2 sampling layout.

Layout matches Fig 2 / Fig 3 / Tab 1:
    rows    = subject relation of label  (within-subject paired vs trait)
    columns = consensus cross-subject marker (consensus vs no-consensus)

                consensus       no-consensus
    within      EEGMAT          SleepDep
    trait       ADFTD           Stress

Each panel contains 6 lines: 3 FMs (LaBraM/CBraMod/REVE) x 2 probes
(subject solid / label dashed). Shared legend at bottom.

Sources
-------
bundled ``paper_data/layerwise_probe/probes.json``.

Output
------
paper/figures/main/fig3_layerwise_probe.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reproduction.builders._runtime import output_dir, paper_data_path
# fmscope is pip-installed; no sys.path hacks needed
# fmscope is pip-installed; no sys.path hacks needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reproduction.builders._style import apply_jne_style, JNE_DOUBLE_COL_CM, CM  # noqa: E402

apply_jne_style()

PROBES_JSON = paper_data_path("layerwise_probe/probes.json")
OUT = output_dir("paper_figures") / "fig3_layerwise_probe.pdf"

# 2x2 grid: row = within/trait, col = consensus/no-consensus
PANEL_GRID = [
    [("eegmat", "EEGMAT"),   ("sleepdep", "SleepDep")],   # row 0: within-subject paired
    [("adftd",  "ADFTD"),    ("stress",   "Stress")],     # row 1: subject-label trait
]
ROW_LABELS = ["within-subject paired", "subject-label trait"]
COL_LABELS = ["consensus marker", "no consensus marker"]

FMS = [("labram",  "LaBraM",  "#1f77b4"),
       ("cbramod", "CBraMod", "#d62728"),
       ("reve",    "REVE",    "#2ca02c")]


def main():
    with open(PROBES_JSON) as f:
        d = json.load(f)

    fig, axes = plt.subplots(
        2, 2, figsize=(JNE_DOUBLE_COL_CM * CM, JNE_DOUBLE_COL_CM * CM * 0.85),
        sharey=True, sharex=True,
        gridspec_kw={"wspace": 0.10, "hspace": 0.32,
                     "left": 0.17, "right": 0.985,
                     "top": 0.86, "bottom": 0.20},
    )

    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            ds, ds_label = PANEL_GRID[r][c]
            chance_subj = None
            for fm_key, fm_label, color in FMS:
                row = d["results"][ds][fm_key]
                depths = [pd["depth_fraction"] for pd in row["per_depth"]]
                subj = [pd["subject_ba"] for pd in row["per_depth"]]
                label = [pd["label_ba_mean"] for pd in row["per_depth"]]
                chance_subj = row["per_depth"][0]["subject_chance"]

                ax.plot(depths, subj, color=color, linewidth=1.6,
                        marker="o", markersize=4, linestyle="-",
                        label=f"{fm_label} subject")
                ax.plot(depths, label, color=color, linewidth=1.6,
                        marker="s", markersize=4, linestyle="--",
                        label=f"{fm_label} label")

            ax.axhline(0.5, color="0.55", linestyle=":", linewidth=0.8)
            if chance_subj is not None:
                ax.axhline(chance_subj, color="0.55", linestyle=":", linewidth=0.8)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(0.0, 1.02)
            ax.set_title(ds_label, fontsize=9, pad=3)
            ax.tick_params(labelsize=7)
            if r == 1:
                ax.set_xlabel("Relative transformer depth", fontsize=8)
            if c == 0:
                ax.set_ylabel("Probe balanced accuracy", fontsize=8, labelpad=2)

    # Column headers (axis B = consensus marker) above the top row of titles.
    for c in range(2):
        bbox = axes[0][c].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.94,
                 COL_LABELS[c], ha="center", va="center",
                 fontsize=9, fontweight="bold")

    # Row labels (axis A = subject relation of label) on the far left.
    for r in range(2):
        bbox = axes[r][0].get_position()
        fig.text(0.06, (bbox.y0 + bbox.y1) / 2,
                 ROW_LABELS[r], ha="left", va="center",
                 fontsize=8, fontweight="bold", rotation=90)

    # Single shared legend across the bottom: 3 FMs x 2 probes = 6 entries.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, fontsize=7, frameon=False,
               handlelength=2.0, columnspacing=1.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
