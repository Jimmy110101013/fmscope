"""Minimal Bring-Your-Own dataset example.

Wrap any source of EEG recordings into the :class:`fmscope.CohortAdapter`
protocol. Two routes shown:

1. :class:`InMemoryCohort` — already-loaded ``(sid, label, windows)`` tuples.
2. :class:`PyTorchDatasetAdapter` — wrap a ``torch.utils.data.Dataset``
   that returns the legacy 4-tuple ``(epochs, label, n_ep, sid)``.

Run::

    python examples/byo_dataset_minimal.py
"""

from __future__ import annotations

import numpy as np

from fmscope import CohortAdapter
from fmscope.data.adapters import (
    InMemoryCohort,
    PyTorchDatasetAdapter,
    synthetic_cohort,
)


def route_a_in_memory() -> InMemoryCohort:
    """Hand-built recordings — simplest possible adapter."""
    rng = np.random.default_rng(0)
    recordings: list[tuple[int, int, np.ndarray]] = []
    for sid in range(5):
        for _ in range(3):
            label = int(rng.choice([0, 1]))
            windows = rng.normal(size=(20, 19, 1000)).astype(np.float32)
            recordings.append((sid, label, windows))
    return InMemoryCohort(recordings, n_channels=19, sfreq=200.0)


def route_b_pytorch_dataset() -> PyTorchDatasetAdapter:
    """Wrap a PyTorch ``Dataset`` returning ``(epochs, label, n_ep, sid)``."""

    class ToyDataset:
        def __init__(self) -> None:
            self.rng = np.random.default_rng(1)

        def __len__(self) -> int:
            return 10

        def __getitem__(self, idx):
            n_ep = 20
            epochs = self.rng.normal(size=(n_ep, 19, 1000)).astype(np.float32)
            label = idx % 2
            sid = idx // 2
            return epochs, label, n_ep, sid

    return PyTorchDatasetAdapter(ToyDataset(), n_channels=19, sfreq=200.0)


def main() -> None:
    for name, cohort in (
        ("InMemoryCohort", route_a_in_memory()),
        ("PyTorchDatasetAdapter", route_b_pytorch_dataset()),
        ("synthetic_cohort", synthetic_cohort(n_subjects=4, seed=2)),
    ):
        assert isinstance(cohort, CohortAdapter), f"{name} failed protocol check."
        n_rec = sum(1 for _ in cohort.iter_recordings())
        print(f"[ok] {name}: {n_rec} recordings, "
              f"{cohort.n_channels}ch @ {cohort.sfreq} Hz")


if __name__ == "__main__":
    main()
