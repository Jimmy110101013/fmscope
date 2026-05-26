"""Reproduce main-text Tab. (tab:leace) — subject-axis erasure on frozen features.

Computed live from the bundled per-window frozen-feature caches via
``fmscope.diagnostics.subject_axis_erasure`` (the same diagnostic the
public toolkit ships), so the table both reproduces the paper and
exercises the released code. Δ_erase (3-seed mean ± SD) is shown only in
the within-subject paired cells; trait cells (ADFTD, Stress) carry a
fixed per-subject label inside the subject subspace and are marked "---".

    python -m reproduction.builders.tab_leace
"""
from __future__ import annotations

import numpy as np

from fmscope.diagnostics import subject_axis_erasure
from reproduction.builders._runtime import paper_data_path

CELLS = ["eegmat", "adftd", "sleepdep", "stress"]  # paper Tab. order
CELL_LABEL = {"eegmat": "EEGMAT", "adftd": "ADFTD",
              "sleepdep": "SleepDep", "stress": "Stress"}
PAIRED = {"eegmat", "sleepdep"}  # within-subject contrast → Δ_erase defined
FMS = ["labram", "cbramod", "reve"]
FM_LABEL = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}


def _load(fm: str, dataset: str):
    f = paper_data_path("features_cache") / f"frozen_{fm}_{dataset}_perwindow.npz"
    if not f.exists():
        return None
    return np.load(f, allow_pickle=True)


def gather():
    rows = []
    for cell in CELLS:
        for fm in FMS:
            d = _load(fm, cell)
            if d is None:
                print(f"# missing cache for {fm}/{cell}")
                continue
            # Paired cells get the binary label (Δ_erase); trait cells skip
            # the label probe — the label lies inside the subject subspace.
            label = d["window_labels"] if cell in PAIRED else None
            er = subject_axis_erasure(
                d["features"], d["window_pids"], label,
            )
            rows.append({
                "cell": cell, "fm": FM_LABEL[fm],
                "subj_pre": 100 * er.subj_ba_linear_pre,
                "subj_post": 100 * er.subj_ba_linear_post,
                "nonlin": 100 * er.subj_ba_mlp_post,
                "delta": (100 * er.label_ba_delta if cell in PAIRED else None),
                "delta_sd": (100 * er.label_ba_delta_std if cell in PAIRED else None),
            })
    return rows


def print_latex(rows):
    print("=" * 70)
    print("Tab. tab:leace — subject-axis erasure (main text)")
    print("=" * 70)
    print(r"% Cell | FM | Subj. pre | Subj. post | Nonlin. | Δ_erase")
    by_cell = {c: [r for r in rows if r["cell"] == c] for c in CELLS}
    for ci, cell in enumerate(CELLS):
        crows = by_cell[cell]
        if not crows:
            continue
        if ci:
            print(r"\midrule")
        for i, r in enumerate(crows):
            head = (rf"\multirow{{{len(crows)}}}{{*}}{{{CELL_LABEL[cell]}}}"
                    if i == 0 else "")
            if r["delta"] is None:
                delta = "---"
            else:
                delta = rf"\(+{r['delta']:.1f} \pm {r['delta_sd']:.1f}\)" \
                    if r["delta"] >= 0 else \
                    rf"\({r['delta']:.1f} \pm {r['delta_sd']:.1f}\)"
            print(f"{head:28s} & {r['fm']:8s} & "
                  f"{r['subj_pre']:5.1f} & {r['subj_post']:4.1f} & "
                  f"{r['nonlin']:5.1f} & {delta} " + r"\\")


if __name__ == "__main__":
    print_latex(gather())
