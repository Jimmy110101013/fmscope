"""Depth-wise linear-probe diagnostic.

Probes both **label** and **subject identity** at every layer of an FM
extractor. The output is the per-depth balanced accuracy of each probe
— the signal that drives the "layer probe" column of the verdict
matrix (paper Tab 3).

Mechanism
---------
1. Resolve a list of transformer-block submodules (auto-detected via
   the extractor's class-level ``_layer_probe_path``, or passed in
   explicitly).
2. Register ``torch.nn.Module.register_forward_hook`` on each block to
   capture its output.
3. Run the cohort through the extractor once; for every window get a
   per-depth hidden state.
4. At each depth, mean-pool tokens to a vector and run two
   :class:`sklearn.linear_model.LogisticRegression` probes:
     - label probe (subject-level StratifiedGroupKFold, group by sid)
     - subject probe (StratifiedKFold)
5. Return per-depth balanced accuracies + the depth fraction.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
import torch

from fmscope.api import CohortAdapter, FMExtractor


def _resolve_block_list(extractor: Any, path: str) -> torch.nn.ModuleList:
    """Resolve a dotted attribute path to an ``nn.ModuleList`` on the extractor."""
    obj = extractor
    for part in path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.ModuleList):
        # Some FMs use nn.Sequential — also iterable.
        if isinstance(obj, torch.nn.Sequential):
            return obj
        raise TypeError(
            f"Resolved {path!r} to {type(obj).__name__}; expected nn.ModuleList "
            f"or nn.Sequential of transformer blocks."
        )
    return obj


def _flatten_hidden(h: Any) -> torch.Tensor:
    """Coerce a transformer block's output to a (B, ?, embed_dim) tensor.

    Different FMs return:
      - a plain Tensor (B, T, D)            → as-is
      - a tuple (Tensor, *aux)              → take first
      - a Tensor (B, D)                     → already pooled
    We always pool to (B, embed_dim) by mean-over-tokens at the caller.
    """
    if isinstance(h, tuple):
        h = h[0]
    if not isinstance(h, torch.Tensor):
        raise TypeError(f"hook output is not a tensor: {type(h)}")
    return h


def _pool_to_vector(h: torch.Tensor) -> torch.Tensor:
    """Mean-pool a hidden-state tensor to one vector per batch row.

    Handles common shapes:
      (B, D)           → unchanged
      (B, T, D)        → mean over T
      (B, C, T, D)     → mean over C, T
      (B, ..., D)      → mean over all but first and last axes
    """
    if h.dim() == 2:
        return h
    return h.flatten(start_dim=1, end_dim=-2).mean(dim=1)


def layer_probe(
    extractor: FMExtractor,
    cohort: CohortAdapter,
    *,
    block_list_path: Optional[str] = None,
    batch_size: int = 16,
    device: str = "cpu",
    n_folds: int = 3,
    seed: int = 0,
    max_windows_per_recording: Optional[int] = None,
    layers_subset: Optional[Sequence[int]] = None,
) -> dict:
    """Depth-wise label + subject linear probe.

    Parameters
    ----------
    extractor : FMExtractor
        A bundled FM (LaBraM / CBraMod / REVE) or any callable whose
        underlying module has ``_layer_probe_path`` set, or whose
        block list is reachable via ``block_list_path``.
    cohort : CohortAdapter
        The cohort to probe.
    block_list_path : str, optional
        Dotted path to a ``nn.ModuleList`` of transformer blocks. If
        ``None``, read from ``extractor._layer_probe_path``.
    batch_size, device : forward-pass knobs.
    n_folds : int
        K for subject-level (label probe) / stratified (subject probe) CV.
    seed : int
    max_windows_per_recording : int, optional
        Cap windows per recording to keep the probe fast on large
        cohorts. ``None`` = no cap.
    layers_subset : sequence of int, optional
        Probe only this subset of layer indices (negative indices
        accepted — ``-1`` means the last layer). ``None`` = probe every
        layer. The forward pass still runs the full network, but hooks
        and per-layer LogReg only fire for layers in this subset —
        saves the LogReg cost on long-recording cohorts where the
        bottleneck is the per-layer probe, not the forward.

    Returns
    -------
    dict
        ``{"per_depth": [{"depth": int, "depth_fraction": float,
            "label_ba_mean": float, "subject_ba_mean": float,
            "label_ba_std": float, "subject_ba_std": float}, ...],
        "n_layers": int, "elapsed_s": float}``. ``per_depth`` has one
        entry per probed layer (one per layer when ``layers_subset`` is
        ``None``); ``depth_fraction`` is the layer's depth in the full
        network, not the position within the subset.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    if isinstance(extractor, torch.nn.Module):
        extractor.eval().to(device)

    # Two ways to specify the per-depth module list:
    # 1. Dotted attribute path → ``_layer_probe_path`` (most FMs).
    # 2. Callable that returns ``list[nn.Module]`` → ``_layer_probe_modules``
    #    (escape hatch for FMs whose blocks aren't a flat nn.ModuleList,
    #    e.g. REVE's ``ModuleList[ModuleList[attn, ff]]`` structure).
    custom = getattr(extractor, "_layer_probe_modules", None)
    if callable(custom):
        block_list = list(custom())
    else:
        path = block_list_path or getattr(extractor, "_layer_probe_path", None)
        if path is None:
            raise ValueError(
                "Cannot auto-detect block list. Pass block_list_path "
                "explicitly, set _layer_probe_path on your extractor class, "
                "or define _layer_probe_modules() returning a list of modules."
            )
        block_list = _resolve_block_list(extractor, path)
    n_layers = len(block_list)

    # Resolve which layers to probe — None = all layers (default), else
    # the caller-supplied subset (negative indices allowed).
    if layers_subset is None:
        probe_set = set(range(n_layers))
    else:
        probe_set = {d % n_layers for d in layers_subset}
        if not probe_set:
            raise ValueError("layers_subset is empty.")

    # Register hooks ONLY on probed layers — saves both the
    # forward-time CPU copy and the post-pass memory for layers we'd
    # immediately discard.
    captured: list[Optional[torch.Tensor]] = [None] * n_layers

    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, out):
            h = _pool_to_vector(_flatten_hidden(out))
            captured[layer_idx] = h.detach().cpu()
        return _hook

    handles = [block_list[i].register_forward_hook(_make_hook(i))
               for i in range(n_layers) if i in probe_set]

    feats_per_layer: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    sids: list[int] = []
    labels: list[int] = []
    rec_ids: list[int] = []           # one int per window, increments per recording

    t0 = time.time()
    try:
        for rec_idx, (sid, label, windows) in enumerate(cohort.iter_recordings()):
            windows = np.asarray(windows, dtype=np.float32)
            if (max_windows_per_recording is not None
                    and windows.shape[0] > max_windows_per_recording):
                idx = np.linspace(0, windows.shape[0] - 1,
                                  max_windows_per_recording).astype(int)
                windows = windows[idx]
            t = torch.from_numpy(windows).float()
            for i in range(0, t.shape[0], batch_size):
                batch = t[i:i + batch_size].to(device)
                with torch.no_grad():
                    extractor(batch)
                for d in probe_set:
                    feats_per_layer[d].append(captured[d].numpy())
            sids.extend([sid] * windows.shape[0])
            labels.extend([label] * windows.shape[0])
            rec_ids.extend([rec_idx] * windows.shape[0])
    finally:
        for h in handles:
            h.remove()

    sids_arr = np.asarray(sids)
    labels_arr = np.asarray(labels)
    rec_arr = np.asarray(rec_ids)

    per_depth = []
    for d in sorted(probe_set):
        feats = np.concatenate(feats_per_layer[d], axis=0)

        # Subject-grouped label probe.
        try:
            cv_lab = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                          random_state=seed)
            lab_bas = []
            for tr, te in cv_lab.split(feats, labels_arr, groups=sids_arr):
                sc = StandardScaler().fit(feats[tr])
                clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                         C=1.0)
                clf.fit(sc.transform(feats[tr]), labels_arr[tr])
                pred = clf.predict(sc.transform(feats[te]))
                lab_bas.append(balanced_accuracy_score(labels_arr[te], pred))
            lab_ba_mean = float(np.mean(lab_bas))
            lab_ba_std = float(np.std(lab_bas))
        except ValueError:
            lab_ba_mean, lab_ba_std = float("nan"), float("nan")

        # Subject identity probe — recording-level GroupKFold.
        # Holding out RECORDINGS (not windows) of each subject avoids the
        # window-level leakage where the same recording's windows appear in
        # both train and test. With multiple recordings per subject this
        # measures "given new recordings of these subjects, can the FM
        # identify them?" — the re-identification reading.
        #
        # Single-recording subjects can't be re-identified across
        # recordings, so we exclude them from the probe (but the label
        # probe still uses all subjects).
        sub_rec_counts = {s: np.unique(rec_arr[sids_arr == s]).size
                          for s in np.unique(sids_arr)}
        keep_subjects = {s for s, n in sub_rec_counts.items() if n >= 2}

        if len(keep_subjects) < 2:
            sub_ba_mean, sub_ba_std = float("nan"), float("nan")
        else:
            mask = np.isin(sids_arr, list(keep_subjects))
            feats_kept = feats[mask]
            sids_kept = sids_arr[mask]
            rec_kept = rec_arr[mask]
            try:
                n_groups = int(np.unique(rec_kept).size)
                effective_folds = min(n_folds, n_groups)
                cv_sub = GroupKFold(n_splits=effective_folds)
                sub_bas = []
                for tr, te in cv_sub.split(feats_kept, sids_kept, groups=rec_kept):
                    # Only score on subjects present in train (re-id task).
                    train_subs = set(sids_kept[tr].tolist())
                    te = te[np.isin(sids_kept[te], list(train_subs))]
                    if te.size == 0:
                        continue
                    sc = StandardScaler().fit(feats_kept[tr])
                    clf = LogisticRegression(max_iter=1000,
                                             class_weight="balanced", C=1.0)
                    clf.fit(sc.transform(feats_kept[tr]), sids_kept[tr])
                    pred = clf.predict(sc.transform(feats_kept[te]))
                    sub_bas.append(balanced_accuracy_score(sids_kept[te], pred))
                sub_ba_mean = float(np.mean(sub_bas)) if sub_bas else float("nan")
                sub_ba_std = float(np.std(sub_bas)) if sub_bas else float("nan")
            except ValueError:
                sub_ba_mean, sub_ba_std = float("nan"), float("nan")

        per_depth.append({
            "depth": d,
            "depth_fraction": (d + 1) / n_layers,
            "label_ba_mean": lab_ba_mean,
            "label_ba_std": lab_ba_std,
            "subject_ba_mean": sub_ba_mean,
            "subject_ba_std": sub_ba_std,
        })

    return {
        "per_depth": per_depth,
        "n_layers": n_layers,
        "elapsed_s": time.time() - t0,
        "n_windows": int(len(sids)),
        "n_subjects": int(np.unique(sids_arr).size),
    }
