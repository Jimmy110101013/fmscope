"""Cohort adapter protocol + helpers.

The :class:`CohortAdapter` protocol is the public input contract for the
diagnostic pipeline. PyTorch-Dataset users wrap their loader with
:class:`PyTorchDatasetAdapter`; users with custom in-memory cohorts use
:class:`InMemoryCohort` directly.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class InMemoryCohort:
    """A cohort built from already-loaded ``(subject_id, label, windows)`` tuples.

    Mostly useful for tests, synthetic fixtures, and BYO custom data that
    is already in memory.
    """

    def __init__(
        self,
        recordings: list[tuple[int, int, np.ndarray]],
        *,
        n_channels: int,
        sfreq: float,
    ):
        self._recordings = recordings
        self.n_channels = n_channels
        self.sfreq = sfreq

    def iter_recordings(self) -> Iterator[tuple[int, int, np.ndarray]]:
        yield from self._recordings


class PyTorchDatasetAdapter:
    """Wrap an existing PyTorch Dataset returning ``(epochs, label, n_ep, sid)``.

    Matches the legacy 4-tuple ``__getitem__`` signature used by the
    bundled cohort loaders (EEGMAT, SleepDep, ADFTD). Stress dataset
    returns a 4-tuple variant ``(epochs, label, stress_score, n_ep)`` —
    use a custom adapter for that case.
    """

    def __init__(self, dataset, *, n_channels: int, sfreq: float):
        self.dataset = dataset
        self.n_channels = n_channels
        self.sfreq = sfreq

    def iter_recordings(self) -> Iterator[tuple[int, int, np.ndarray]]:
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if len(item) == 4:
                epochs, label, n_ep, sid = item
            else:
                raise ValueError(
                    f"Expected 4-tuple from dataset[{i}], got {len(item)}-tuple."
                )
            arr = epochs.detach().cpu().numpy() if hasattr(epochs, "detach") else np.asarray(epochs)
            yield int(sid), int(label), arr


def synthetic_cohort(
    *,
    n_subjects: int = 8,
    n_recordings_per_subj: int = 3,
    n_windows_per_rec: int = 20,
    n_channels: int = 19,
    n_samples: int = 1000,
    sfreq: float = 200.0,
    label_balance: float = 0.5,
    leak_subject_into_label: bool = False,
    seed: int = 0,
) -> InMemoryCohort:
    """Build a synthetic cohort for tutorials, smoke tests, and rubric demos.

    If ``leak_subject_into_label=True``, each subject is assigned a fixed
    label (creates the "trait-leak" failure mode the verdict rubric is
    designed to detect).
    """
    rng = np.random.default_rng(seed)
    recordings: list[tuple[int, int, np.ndarray]] = []
    for sid in range(n_subjects):
        subject_signature = rng.normal(0, 1, size=(n_channels,))
        subj_label = rng.choice([0, 1], p=[1 - label_balance, label_balance]) if leak_subject_into_label else None
        for _ in range(n_recordings_per_subj):
            label = subj_label if leak_subject_into_label else int(rng.choice([0, 1]))
            windows = rng.normal(0, 1, size=(n_windows_per_rec, n_channels, n_samples))
            windows += subject_signature[None, :, None] * 0.5  # subject DC offset
            if not leak_subject_into_label:
                windows += float(label) * rng.normal(0, 0.1, size=(n_channels, 1))
            recordings.append((sid, int(label), windows.astype(np.float32)))
    return InMemoryCohort(recordings, n_channels=n_channels, sfreq=sfreq)
