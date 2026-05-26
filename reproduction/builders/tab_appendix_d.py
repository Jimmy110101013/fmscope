"""One-shot: print LaTeX-ready content for App. D tables (variance, anchor, direction)."""
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

from reproduction.builders._runtime import output_dir, paper_data_path

CELLS = ["eegmat", "adftd", "sleepdep", "stress"]
CELL_LABEL = {"eegmat": "EEGMAT", "adftd": "ADFTD",
              "sleepdep": "SleepDep", "stress": "Stress"}
FMS = ["labram", "cbramod", "reve"]
FM_LABEL = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}


# -----------------------------------------------------------------
# D.1 Variance decomposition
# -----------------------------------------------------------------
def gather_variance():
    v = json.load(open(paper_data_path("source_tables", "variance_analysis_window_level.json")))
    rows = []
    for cell in CELLS:
        for fm in FMS:
            key = f"{fm}_{cell}"
            if key not in v:
                # fallback for labelling differences
                continue
            blk = v[key]
            fr = blk["frozen"]
            ft = blk["ft"]
            rows.append({
                "cell": CELL_LABEL[cell],
                "fm": FM_LABEL[fm],
                "fr_label": fr["label_frac"],
                "fr_subj": fr["subject_frac"],
                "fr_resid": fr["residual_frac"],
                "ft_label": ft["label_frac"],
                "ft_subj": ft["subject_frac"],
                "ft_resid": ft["residual_frac"],
                "d_label": blk["delta_label_frac"],
                "d_subj": blk["delta_subject_frac"],
            })
    return rows


def print_variance_latex(rows):
    print("=" * 70)
    print("D.1 VARIANCE DECOMPOSITION (12 rows)")
    print("=" * 70)
    print(r"% Cell | FM | frozen: lab/subj/resid | FT: lab/subj/resid | Δlab | Δsubj")
    for r in rows:
        # values in % (multiply by 100)
        print(
            f"{r['cell']:9s} & {r['fm']:8s} & "
            f"{100*r['fr_label']:5.1f} & {100*r['fr_subj']:5.1f} & {100*r['fr_resid']:5.1f} & "
            f"{100*r['ft_label']:5.1f} & {100*r['ft_subj']:5.1f} & {100*r['ft_resid']:5.1f} & "
            f"{100*r['d_label']:+5.1f} & {100*r['d_subj']:+5.1f}"
            r" \\"
        )


# -----------------------------------------------------------------
# D.2 Anchor ablation (FOOOF)
# -----------------------------------------------------------------
def gather_anchor():
    """State probe from fooof_ablation/probes.json (per-window subject-stratified
    group K-fold, 8 seeds). Subject probe from subject_probe_temporal_block/
    probes.json (5-fold temporal-block, shrinkage LDA, 1 seed)."""
    rows = []
    for cell in CELLS:
        sp = paper_data_path(cell, "fooof_ablation", "probes.json")
        bp = paper_data_path(cell, "subject_probe_temporal_block", "probes.json")
        if not sp.exists() or not bp.exists():
            print(f"# missing probe file for {cell}")
            continue
        sd = json.load(open(sp))
        bd = json.load(open(bp))
        for fm in FMS:
            sb = sd["results"].get(fm)
            bb = bd["results"].get(fm)
            if sb is None or bb is None:
                continue
            rows.append({
                "cell": CELL_LABEL[cell],
                "fm": FM_LABEL[fm],
                "state_orig": sb["original"]["state_probe_mean"],
                "state_d_aper": (sb["aperiodic_removed"]["state_probe_mean"]
                                 - sb["original"]["state_probe_mean"]),
                "state_d_per": (sb["periodic_removed"]["state_probe_mean"]
                                - sb["original"]["state_probe_mean"]),
                "subj_orig": bb["original"]["subject_probe_mean"],
                "subj_d_aper": (bb["aperiodic_removed"]["subject_probe_mean"]
                                - bb["original"]["subject_probe_mean"]),
                "subj_d_per": (bb["periodic_removed"]["subject_probe_mean"]
                               - bb["original"]["subject_probe_mean"]),
            })
    return rows


def print_anchor_latex(rows):
    print("=" * 70)
    print("D.2 ANCHOR ABLATION (12 rows)")
    print("=" * 70)
    print(r"% Cell | FM | state-orig | Δstate(-aper) | Δstate(-per) | subj-orig | Δsubj(-aper) | Δsubj(-per)")
    for r in rows:
        print(
            f"{r['cell']:9s} & {r['fm']:8s} & "
            f"{100*r['state_orig']:5.1f} & {100*r['state_d_aper']:+5.1f} & {100*r['state_d_per']:+5.1f} & "
            f"{100*r['subj_orig']:5.1f} & {100*r['subj_d_aper']:+5.1f} & {100*r['subj_d_per']:+5.1f}"
            r" \\"
        )


# -----------------------------------------------------------------
# D.3 Direction consistency with bootstrap CIs
# -----------------------------------------------------------------
def per_subject_directions(X_pooled, pids, labels):
    """Return list of unit-norm contrast vectors, one per subject."""
    directions = []
    for pid in np.unique(pids):
        m = pids == pid
        x0 = X_pooled[m & (labels == 0)]
        x1 = X_pooled[m & (labels == 1)]
        if len(x0) == 0 or len(x1) == 0:
            continue
        diff = x1.mean(0) - x0.mean(0)
        norm = np.linalg.norm(diff)
        if norm > 0:
            directions.append(diff / norm)
    return np.array(directions)


def mean_pairwise_cosine(directions):
    n = len(directions)
    if n < 2:
        return float("nan")
    sims = [
        1.0 - cosine(directions[i], directions[j])
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(sims))


def bootstrap_ci(directions, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(directions)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Skip degenerate resamples (all same subject)
        if len(np.unique(idx)) < 2:
            continue
        boots.append(mean_pairwise_cosine(directions[idx]))
    boots = np.array(boots)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_frozen(fm, dataset):
    f = paper_data_path("features_cache") / f"frozen_{fm}_{dataset}_perwindow.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    win_feats = d["features"]
    win_rec_idx = d["window_rec_idx"]
    rec_labels = d["rec_labels"].astype(int)
    rec_pids = d["rec_pids"]
    pooled = np.stack([
        win_feats[win_rec_idx == i].mean(0) for i in range(len(rec_pids))
    ])
    return pooled, rec_pids, rec_labels


def gather_direction():
    rows = []
    # frozen + FT for both EEGMAT and SleepDep
    src = json.load(open(paper_data_path("source_tables", "within_subject_dir_consistency.json")))
    for cell in ["eegmat", "sleepdep"]:
        for fm in FMS:
            point = src[cell]["frozen"][fm]["dir_consistency"]
            n = src[cell]["frozen"][fm]["n_subj"]
            # bootstrap on frozen
            loaded = load_frozen(fm, cell)
            if loaded is None:
                ci_lo = ci_hi = float("nan")
            else:
                Xp, P, L = loaded
                directions = per_subject_directions(Xp, P, L)
                ci_lo, ci_hi = bootstrap_ci(directions)
            rows.append({
                "cell": CELL_LABEL[cell],
                "fm": FM_LABEL[fm],
                "dir": point,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": n,
            })
    return rows


def print_direction_latex(rows):
    print("=" * 70)
    print("D.3 DIRECTION CONSISTENCY (frozen, 6 rows)")
    print("=" * 70)
    print(r"% Cell | FM | c-bar | 95% CI lower | 95% CI upper | n_subj")
    for r in rows:
        print(
            f"{r['cell']:9s} & {r['fm']:8s} & "
            f"{r['dir']:+0.3f} & "
            f"[{r['ci_lo']:+0.3f}, {r['ci_hi']:+0.3f}] & "
            f"{r['n']}"
            r" \\"
        )


if __name__ == "__main__":
    print_variance_latex(gather_variance())
    print()
    print_anchor_latex(gather_anchor())
    print()
    print_direction_latex(gather_direction())
