"""Public protocols defining the FMScope API surface.

End users wrapping their own pretrained model or dataset implement
these protocols by structural typing — no inheritance, no decorator,
no registration required.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FMExtractor(Protocol):
    """Anything callable as ``extractor(x) -> (B, D)`` with an ``embed_dim``.

    The contract:

    * ``x`` is a batch of EEG windows shaped ``(B, C, T)`` (channels × samples).
      Accept either ``torch.Tensor`` or ``np.ndarray``; FMScope will convert
      as needed.
    * Output is shaped ``(B, embed_dim)``.
    * ``embed_dim`` is a positive integer attribute.

    See ``examples/byo_fm_minimal.py`` for a 30-line wrapper around a
    HuggingFace model.
    """

    embed_dim: int

    def __call__(self, x):  # noqa: D401
        ...


@runtime_checkable
class CohortAdapter(Protocol):
    """Iterator over recordings within one cohort.

    The contract:

    * ``iter_recordings()`` yields ``(subject_id, label, windows)`` per
      recording, where ``windows`` is a ``np.ndarray`` of shape
      ``(n_windows, n_channels, n_samples)``.
    * ``n_channels`` and ``sfreq`` describe the windowed data.

    PyTorch-Dataset users get a 5-line shim via
    :class:`fmscope.data.adapters.PyTorchDatasetAdapter`.
    """

    n_channels: int
    sfreq: float

    def iter_recordings(self) -> Iterator[tuple[int, int, np.ndarray]]:  # noqa: D401
        ...
