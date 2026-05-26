"""Subject-axis linear erasure (LEACE) diagnostic.

Borrowed method: **LEACE** — Least-squares Concept Erasure, the
closed-form linear concept eraser of Belrose et al., *LEACE: Perfect
linear concept erasure in closed form*, NeurIPS 2023. FMScope applies it
to the **subject-identity** axis of frozen FM features, then re-probes
both subject identity and the task label.

The diagnostic quantity is ``Δ_erase`` = (label BA after erasing the
subject axis) − (label BA before). The reading is deliberately the
*inverse* of work that erases neural features to measure their
contribution (erasing the feature *hurts* the probe): here we erase the
subject identity that happens to correlate with the label, so when the
FM was leaning on identity rather than the state, erasure can *help* the
label probe (``Δ_erase`` ≥ 0).

We erase only the **linear** subject axis; the nonlinear residual is
measured (``subj_ba_mlp_post``) and reported, never hidden. When the
subject subspace nearly fills the ambient feature space
(``rank ≥ degenerate_frac·dim``) erasure is degenerate and flagged.

Interpretability gate: ``Δ_erase`` is meaningful only when the FM can
read the label at all. We report ``interpretable=False`` when the
pre-erasure label BA is below ``gate`` (default 0.55) — below chance-plus
there is no label signal whose fate erasure could change.

Numbers only — this module returns ``Δ_erase``, pre/post subject BA, and
the gate flag. It never returns a verdict or a +/−/0 glyph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_LABEL_SEEDS = (42, 123, 2024)
DEGENERATE_FRAC = 0.95  # rank ≥ this · dim ⇒ subject subspace ~fills ambient


# --------------------------------------------------------------------------- #
# LEACE primitives                                                            #
# --------------------------------------------------------------------------- #

def whiten(features: np.ndarray, *, shrinkage: bool = True):
    """Whitening + de-whitening operators for centred features.

    Returns ``(mu, Xc, W, W_plus, cond)`` where ``mu`` is the feature
    mean, ``Xc`` the centred features, ``W = Σ^{-1/2}`` (pseudo-inverse
    of the symmetric square root), ``W_plus = Σ^{1/2}``, and ``cond`` the
    condition number of the (optionally Ledoit-Wolf shrunk) covariance.
    """
    X = np.asarray(features, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    n = len(X)
    if shrinkage:
        Sigma, _ = ledoit_wolf(Xc, assume_centered=True)
    else:
        Sigma = Xc.T @ Xc / n
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 0.0, None)
    sq = np.sqrt(evals)
    smax = sq.max() if sq.size else 0.0
    pos = sq > 1e-8 * smax if smax > 0 else np.zeros_like(sq, bool)
    inv = np.where(pos, 1.0 / np.where(pos, sq, 1.0), 0.0)
    W = (evecs * inv) @ evecs.T       # (Σ^{1/2})^+
    W_plus = (evecs * sq) @ evecs.T   # Σ^{1/2}
    cond = (float(evals[pos].max() / evals[pos].min())
            if pos.sum() >= 2 else float("inf"))
    return mu, Xc, W, W_plus, cond


def subject_eraser(Xc: np.ndarray, W: np.ndarray, W_plus: np.ndarray,
                   subject: np.ndarray):
    """Closed-form LEACE eraser for the subject-identity axis.

    Builds the centred one-hot subject design ``Z``, finds the
    whitened cross-covariance subspace between features and subject, and
    returns ``(S, P_perp, rank)``:

    - ``S`` : (dim, rank) orthonormal basis of the subject subspace.
    - ``P_perp`` : (dim, dim) oblique eraser; ``Xc @ P_perp.T`` removes the
      linear subject axis (Belrose et al. 2023, closed form).
    - ``rank`` : numerical rank of the subject subspace.
    """
    Xc = np.asarray(Xc, dtype=np.float64)
    pids = np.asarray(subject)
    n, d = Xc.shape
    subs = np.unique(pids)
    idx = {s: i for i, s in enumerate(subs)}
    Z = np.zeros((n, len(subs)))
    for i, s in enumerate(pids):
        Z[i, idx[s]] = 1.0
    Zc = Z - Z.mean(0)
    Sigma_XZ = Xc.T @ Zc / n
    U, s, _ = np.linalg.svd(W @ Sigma_XZ, full_matrices=False)
    r = int((s > 1e-6 * s.max()).sum()) if s.size and s.max() > 0 else 0
    Ur = U[:, :r]
    Q, _ = np.linalg.qr(W_plus @ Ur)
    S = Q[:, :r]
    P_perp = np.eye(d) - W_plus @ (Ur @ Ur.T) @ W
    return S, P_perp, r


def apply_eraser(features: np.ndarray, mu: np.ndarray,
                 P_perp: np.ndarray) -> np.ndarray:
    """Erase the subject axis from ``features`` using a fitted eraser.

    ``Xe = (X − mu) @ P_perp.T + mu`` — centre, project out the subject
    subspace, restore the mean.
    """
    X = np.asarray(features, dtype=np.float64)
    return (X - mu) @ P_perp.T + mu


def subspace_overlap(SA: np.ndarray, SB: np.ndarray) -> float:
    """Normalised principal-angle overlap of two orthonormal bases.

    ``‖SAᵀ SB‖_F² / min(rank_A, rank_B)`` ∈ [0, 1]; 1 = identical span,
    0 = orthogonal. Used by the mechanism analysis to compare subject
    subspaces across cohorts.
    """
    if SA.shape[1] == 0 or SB.shape[1] == 0:
        return 0.0
    M = SA.T @ SB
    return float((M ** 2).sum() / min(SA.shape[1], SB.shape[1]))


# --------------------------------------------------------------------------- #
# Probes                                                                      #
# --------------------------------------------------------------------------- #

def subject_probe(features: np.ndarray, subject: np.ndarray, *,
                  kind: str = "linear", cap: int = 100, n_splits: int = 5,
                  seed: int = 42) -> tuple[Optional[float], int]:
    """Cross-validated subject-identity probe (BA over a k-way subject task).

    Caps windows per subject at ``cap`` and drops subjects with fewer than
    ``n_splits`` windows so the stratified split is well-defined.
    ``kind="linear"`` uses logistic regression; ``kind="mlp"`` a small MLP
    (the nonlinear-residual check). Returns ``(balanced_accuracy, n_subjects)``;
    BA is ``None`` when fewer than two subjects survive.
    """
    X = np.asarray(features, dtype=np.float64)
    pids = np.asarray(subject)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for s in np.unique(pids):
        i = np.where(pids == s)[0]
        if len(i) > cap:
            i = rng.choice(i, cap, replace=False)
        keep.extend(i.tolist())
    keep = np.array(keep)
    Xs, ys = X[keep], pids[keep]
    cnt = {s: int((ys == s).sum()) for s in np.unique(ys)}
    mask = np.array([cnt[s] >= n_splits for s in ys])
    Xs, ys = Xs[mask], ys[mask]
    if len(np.unique(ys)) < 2:
        return None, len(np.unique(ys))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bas = []
    for tr, te in skf.split(Xs, ys):
        sc = StandardScaler().fit(Xs[tr])
        if kind == "linear":
            clf = LogisticRegression(max_iter=500, C=1.0)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                                early_stopping=True, random_state=seed)
        clf.fit(sc.transform(Xs[tr]), ys[tr])
        bas.append(balanced_accuracy_score(ys[te], clf.predict(sc.transform(Xs[te]))))
    return float(np.mean(bas)), int(len(np.unique(ys)))


def _segment_recordings(subject: np.ndarray, label: np.ndarray):
    """Reconstruct recording boundaries from per-window subject/label runs.

    Windows are emitted recording-by-recording, so each maximal contiguous
    run with constant (subject, label) is one recording — the same
    segmentation :func:`fmscope.verdict.audit._recording_mean_pool` uses.
    Returns ``(window_recording, rec_labels, rec_pids)``.
    """
    sids = np.asarray(subject)
    labs = np.asarray(label)
    n = len(sids)
    window_rec = np.zeros(n, dtype=int)
    rec_labels: list = []
    rec_pids: list = []
    start, rec = 0, 0
    for i in range(1, n + 1):
        if i == n or sids[i] != sids[start] or labs[i] != labs[start]:
            window_rec[start:i] = rec
            rec_labels.append(labs[start])
            rec_pids.append(sids[start])
            rec += 1
            start = i
    return window_rec, np.asarray(rec_labels), np.asarray(rec_pids)


def _label_bas(features, window_rec, rec_labels, rec_pids, seeds, n_splits):
    """Per-seed recording-level label BA via the canonical linear probe."""
    from fmscope.training.lp import eval_seed  # local: keeps training optional
    X = np.asarray(features, dtype=np.float32)
    return np.array([eval_seed(X, window_rec, rec_labels, rec_pids, s,
                               n_splits=n_splits)[0]
                     for s in seeds])


# --------------------------------------------------------------------------- #
# Diagnostic                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class ErasureResult:
    """Output of :func:`subject_axis_erasure`.

    Subject-axis fields are always populated. Label fields are ``NaN`` and
    ``interpretable=False`` when no binary label is supplied (a single
    label per subject, i.e. a trait cohort, has no within-subject contrast
    to erase toward).

    Attributes
    ----------
    embed_dim, n_subjects : int
        Feature dimension and number of subjects.
    rank_subject_axis : int
        Dimension of the erased linear subject subspace.
    degenerate : bool
        ``rank ≥ degenerate_frac·dim`` — erasure removes ~all variance.
    cond_shrunk : float
        Condition number of the shrunk covariance (whitening stability).
    chance : float
        ``1 / n_subjects`` — chance for the subject probe.
    subj_ba_linear_pre, subj_ba_linear_post : float
        Linear subject-probe BA before / after erasure (post → chance ⇒
        the linear identity axis was removed).
    subj_ba_mlp_post : float
        Nonlinear subject-probe BA after erasure (the residual we report).
    label_ba_raw, label_ba_erased, label_ba_delta : float
        Mean label BA before / after erasure and their paired difference
        ``Δ_erase`` (averaged over ``label_seeds``).
    label_ba_raw_std, label_ba_erased_std, label_ba_delta_std : float
        Across-seed standard deviations.
    gate : float
        Interpretability threshold applied to ``label_ba_raw``.
    interpretable : bool
        ``label_ba_raw ≥ gate`` — whether ``Δ_erase`` is interpretable.
    """

    embed_dim: int
    n_subjects: int
    rank_subject_axis: int
    degenerate: bool
    cond_shrunk: float
    chance: float
    subj_ba_linear_pre: float
    subj_ba_linear_post: float
    subj_ba_mlp_post: float
    label_ba_raw: float
    label_ba_erased: float
    label_ba_delta: float
    label_ba_raw_std: float
    label_ba_erased_std: float
    label_ba_delta_std: float
    gate: float
    interpretable: bool


def subject_axis_erasure(
    features: np.ndarray,
    subject: np.ndarray,
    label: Optional[np.ndarray] = None,
    *,
    window_recording: Optional[np.ndarray] = None,
    rec_labels: Optional[np.ndarray] = None,
    rec_pids: Optional[np.ndarray] = None,
    gate: float = 0.55,
    label_seeds: tuple = DEFAULT_LABEL_SEEDS,
    subject_cap: int = 100,
    degenerate_frac: float = DEGENERATE_FRAC,
    shrinkage: bool = True,
) -> ErasureResult:
    """Erase the linear subject axis and re-probe subject identity + label.

    Parameters
    ----------
    features : np.ndarray, shape (N, embed_dim)
        Per-window frozen FM features.
    subject : np.ndarray, shape (N,)
        Subject id per window.
    label : np.ndarray, shape (N,), optional
        Binary task label per window. When omitted (or not binary) the
        label probe is skipped and ``Δ_erase`` is reported as ``NaN``.
    window_recording, rec_labels, rec_pids : np.ndarray, optional
        Recording-level grouping for the subject-level label CV. If not
        given they are reconstructed from contiguous ``(subject, label)``
        runs (windows are emitted recording-by-recording).
    gate : float, default 0.55
        Minimum pre-erasure label BA for ``Δ_erase`` to be interpretable.
    label_seeds : tuple, default (42, 123, 2024)
        Seeds for the recording-level label probe.
    subject_cap : int, default 100
        Max windows per subject in the subject probe.
    degenerate_frac : float, default 0.95
        Degeneracy threshold on ``rank / dim``.
    shrinkage : bool, default True
        Ledoit-Wolf shrinkage of the covariance before whitening.

    Returns
    -------
    :class:`ErasureResult`
    """
    X = np.asarray(features, dtype=np.float64)
    pids = np.asarray(subject)
    dim = int(X.shape[1])
    n_subj = int(len(np.unique(pids)))

    mu, Xc, W, W_plus, cond = whiten(X, shrinkage=shrinkage)
    _, P_perp, r = subject_eraser(Xc, W, W_plus, pids)
    Xe = apply_eraser(X, mu, P_perp)
    degenerate = bool(r >= degenerate_frac * dim)

    lin_pre, _ = subject_probe(X, pids, kind="linear", cap=subject_cap)
    lin_post, _ = subject_probe(Xe, pids, kind="linear", cap=subject_cap)
    mlp_post, _ = subject_probe(Xe, pids, kind="mlp", cap=subject_cap)

    nan = float("nan")
    raw_mean = era_mean = delta_mean = nan
    raw_std = era_std = delta_std = nan
    interpretable = False

    binary = label is not None and len(np.unique(np.asarray(label))) == 2
    if binary:
        if window_recording is None or rec_labels is None or rec_pids is None:
            window_recording, rec_labels, rec_pids = _segment_recordings(pids, label)
        # Recording-level subject CV; cap folds at the minority-class
        # recording count (mirrors run_canonical_lp's auto-reduction).
        classes, counts = np.unique(rec_labels, return_counts=True)
        eff_splits = min(5, int(counts.min()))
        if len(classes) == 2 and eff_splits >= 2:
            raw = _label_bas(X, window_recording, rec_labels, rec_pids,
                             label_seeds, eff_splits)
            era = _label_bas(Xe, window_recording, rec_labels, rec_pids,
                             label_seeds, eff_splits)
            delta = era - raw  # paired per-seed
            raw_mean, era_mean, delta_mean = (float(raw.mean()),
                                              float(era.mean()),
                                              float(delta.mean()))
            raw_std, era_std, delta_std = (float(raw.std()),
                                           float(era.std()),
                                           float(delta.std()))
            interpretable = bool(raw_mean >= gate)

    return ErasureResult(
        embed_dim=dim, n_subjects=n_subj, rank_subject_axis=int(r),
        degenerate=degenerate, cond_shrunk=float(cond),
        chance=1.0 / n_subj if n_subj else nan,
        subj_ba_linear_pre=lin_pre if lin_pre is not None else nan,
        subj_ba_linear_post=lin_post if lin_post is not None else nan,
        subj_ba_mlp_post=mlp_post if mlp_post is not None else nan,
        label_ba_raw=raw_mean, label_ba_erased=era_mean, label_ba_delta=delta_mean,
        label_ba_raw_std=raw_std, label_ba_erased_std=era_std,
        label_ba_delta_std=delta_std,
        gate=gate, interpretable=interpretable,
    )
