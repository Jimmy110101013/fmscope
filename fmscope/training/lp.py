"""Canonical linear probing (LP) for EEG foundation-model features.

Protocol (codified as guardrail G-F10 in docs/methodology_notes.md):
  1. Per-window frozen features cached at
     results/features_cache/frozen_{extractor}_{dataset}_perwindow.npz
  2. CV at RECORDING level — StratifiedGroupKFold(n_splits) with
     groups=patient_id (default) or LeaveOneGroupOut (--cv loso).
  3. For each fold: all train-recording windows are training samples.
     Per-dim 1-99 percentile clip (train-fit) → StandardScaler →
     LogisticRegression(liblinear, class_weight=balanced, C=1.0).
  4. Test-set window probabilities mean-pooled per recording →
     threshold 0.5 → recording-level balanced accuracy.
  5. Default 8 seeds; report mean + std + per-seed BA.

CLI:
    python -m fmscope.training.lp --extractor labram --dataset eegmat
    python -m fmscope.training.lp --extractor reve --dataset tdbrain
    python -m fmscope.training.lp --extractor labram --dataset stress --cv loso

Library use:
    from fmscope.training import run_canonical_lp
    result = run_canonical_lp(extractor="labram", dataset="stress")

History: the pre-2026-04-25 body was a PyTorch pool-then-classify probe
(optionally with MLP head + mixup). That implementation is preserved at
git tag `lp-pool-then-classify-v1` for reproducibility of pre-migration
numbers. G-F12 in docs/methodology_notes.md records the migration rationale.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_SEEDS = (42, 123, 2024, 7, 0, 1, 99, 31337)
EXTRACTOR_CHOICES = ("labram", "cbramod", "reve")
DATASET_CHOICES = ("stress", "eegmat", "adftd", "tdbrain", "meditation", "sleepdep")
CV_CHOICES = ("stratified-kfold", "loso")
REPO = Path(__file__).resolve().parent


def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO
        ).decode().strip()
    except Exception:
        return "unknown"


def _fit_fold(X_tr, y_tr, X_te):
    """Train-set percentile clip → StandardScaler → LogReg → return test probs."""
    lo = np.percentile(X_tr, 1, axis=0)
    hi = np.percentile(X_tr, 99, axis=0)
    X_tr_c = np.clip(X_tr, lo, hi)
    X_te_c = np.clip(X_te, lo, hi)

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0,
            solver="liblinear", tol=1e-3,
        )),
    ])
    clf.fit(X_tr_c, y_tr)
    return clf.predict_proba(X_te_c)[:, 1]


def _pool_fold(rec_prob, rec_pred, test_rec, window_rec_idx_te, prob_te):
    for r in test_rec:
        m = window_rec_idx_te == r
        if m.any():
            rec_prob[r] = float(prob_te[m].mean())
            rec_pred[r] = int(rec_prob[r] >= 0.5)


def eval_seed(window_feats, window_rec_idx, rec_labels, rec_pids,
              seed, cv="stratified-kfold", n_splits=5):
    """Run per-window LP with prediction pooling for one seed.

    Returns (recording-level BA, per-recording pooled prob).

    cv: "stratified-kfold" (default) → StratifiedGroupKFold(n_splits) shuffled
        with random_state=seed. "loso" → LeaveOneGroupOut; seed is then
        nominal (LOSO is deterministic; multiple seeds kept for symmetry).
    """
    n_rec = len(rec_labels)
    rec_indices = np.arange(n_rec)
    rec_prob = np.full(n_rec, np.nan, dtype=float)
    rec_pred = np.zeros(n_rec, dtype=int)

    # Precompute rec → window-row indices once; per-fold masks become
    # set-based concatenation instead of O(n_rec * n_windows) np.isin scans.
    rec_to_rows: dict[int, np.ndarray] = {}
    for r in range(n_rec):
        rec_to_rows[r] = np.flatnonzero(window_rec_idx == r)

    if cv == "loso":
        splitter = LeaveOneGroupOut().split(rec_indices, rec_labels, groups=rec_pids)
    elif cv == "stratified-kfold":
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = kf.split(rec_indices, rec_labels, groups=rec_pids)
    else:
        raise ValueError(f"unknown cv={cv!r}; expected one of {CV_CHOICES}")

    for train_rec, test_rec in splitter:
        # Sort to preserve original window ordering (matches old np.isin mask),
        # so LogReg sees the same sample sequence and produces identical fits.
        train_idx = np.sort(np.concatenate([rec_to_rows[r] for r in train_rec]))
        test_idx = np.sort(np.concatenate([rec_to_rows[r] for r in test_rec]))

        prob_te = _fit_fold(
            window_feats[train_idx],
            rec_labels[window_rec_idx[train_idx]],
            window_feats[test_idx],
        )
        _pool_fold(rec_prob, rec_pred, test_rec,
                   window_rec_idx[test_idx], prob_te)

    return float(balanced_accuracy_score(rec_labels, rec_pred)), rec_prob


def run_canonical_lp(extractor, dataset, features_npz=None, out_path=None,
                     cv="stratified-kfold", n_splits=5, seeds=DEFAULT_SEEDS,
                     verbose=True):
    """Run the canonical LP and return the result dict.

    If out_path is provided (or None → default study path), the result is
    written to JSON. Returns the dict either way.
    """
    features_npz = features_npz or \
        f"results/features_cache/frozen_{extractor}_{dataset}_perwindow.npz"
    out_path = Path(out_path or
        f"results/final/{dataset}/lp/{extractor}.json")

    data = np.load(features_npz)
    window_feats = data["features"]
    window_rec_idx = data["window_rec_idx"]
    rec_labels = data["rec_labels"]
    rec_pids = data["rec_pids"]
    rec_n_epochs = data["rec_n_epochs"]

    n_rec = len(rec_labels)
    n_subj = len(np.unique(rec_pids))
    pos = int(rec_labels.sum())
    neg = int((1 - rec_labels).sum())

    # Drop n_splits to min class size if needed (stratified-kfold only).
    effective_n_splits = n_splits
    if cv == "stratified-kfold":
        min_class_count = min(pos, neg)
        if min_class_count < n_splits:
            effective_n_splits = min_class_count
            if verbose:
                print(f"  note: reducing n_splits {n_splits} → "
                      f"{effective_n_splits} (min class size = {min_class_count})")

    if verbose:
        cv_desc = "LOSO" if cv == "loso" else f"StratifiedGroupKFold({effective_n_splits})"
        print(f"{extractor} × {dataset} per-window LP ({cv_desc})")
        print(f"  features: {window_feats.shape}")
        print(f"  n_rec={n_rec}, n_subj={n_subj}, pos={pos}, neg={neg}")
        print(f"  total windows={window_feats.shape[0]} "
              f"(avg {rec_n_epochs.mean():.1f} per recording)")

    per_seed = {}
    for s in seeds:
        ba, _ = eval_seed(window_feats, window_rec_idx, rec_labels, rec_pids,
                          s, cv=cv, n_splits=effective_n_splits)
        per_seed[str(s)] = ba
        if verbose:
            print(f"  seed={s:>5}  BA={ba:.4f}")

    vals = np.array(list(per_seed.values()))
    cv_label = "LOSO" if cv == "loso" else f"StratifiedGroupKFold({effective_n_splits})"
    out = {
        "provenance": {
            "snapshot_date": str(date.today()),
            "commit": _current_commit(),
            "script": "train_lp.py",
            "notes": "Per-window frozen LP, 8-seed sklearn LogReg + test-prob mean-pool.",
        },
        "extractor": extractor,
        "dataset": dataset,
        "source_features": features_npz,
        "cv": cv,
        "protocol": (
            f"Per-window LogisticRegression (liblinear, C=1.0, "
            f"class_weight=balanced) on StandardScaler-normed per-window "
            f"features after per-dim 1-99 percentile clip (train-fit). "
            f"Recording-level CV = {cv_label} with groups=patient_id; "
            f"window-level training within each fold; test-set window probs "
            f"mean-pooled per recording; threshold 0.5; recording-level BA."
        ),
        "n_recordings": int(n_rec),
        "n_subjects": int(n_subj),
        "n_positive": pos,
        "n_negative": neg,
        "n_splits": effective_n_splits if cv == "stratified-kfold" else n_subj,
        "embed_dim": int(window_feats.shape[1]),
        "total_windows": int(window_feats.shape[0]),
        "avg_windows_per_rec": float(rec_n_epochs.mean()),
        "seeds": list(seeds),
        "per_seed_ba": per_seed,
        "mean_8seed": float(vals.mean()),
        "std_8seed_ddof1": float(vals.std(ddof=1)),
        "std_8seed_ddof0": float(vals.std(ddof=0)),
        "mean_3seed_42_123_2024": float(vals[:3].mean()),
        "std_3seed_42_123_2024_ddof1": float(vals[:3].std(ddof=1)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print(f"\n→ {out_path}")
        print(f"  8-seed mean={out['mean_8seed']:.4f} "
              f"std={out['std_8seed_ddof1']:.4f} (ddof=1)")

    return out


def _parse_seeds(s):
    return tuple(int(x) for x in s.split(","))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--extractor", required=True, choices=EXTRACTOR_CHOICES)
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--cv", default="stratified-kfold", choices=CV_CHOICES,
                   help="StratifiedGroupKFold(5) (default) or LeaveOneGroupOut.")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Ignored when --cv loso. Auto-reduced if < min class size.")
    p.add_argument("--features-npz", default=None,
                   help="Override auto path results/features_cache/frozen_{fm}_{ds}_perwindow.npz")
    p.add_argument("--out-path", default=None,
                   help="Override auto path results/final/{ds}/lp/{fm}.json")
    p.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS,
                   help="Comma-separated seed list (default 8 seeds).")
    args = p.parse_args()

    run_canonical_lp(
        extractor=args.extractor,
        dataset=args.dataset,
        features_npz=args.features_npz,
        out_path=args.out_path,
        cv=args.cv,
        n_splits=args.n_splits,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
