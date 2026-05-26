"""Minimal Bring-Your-Own foundation model example.

Wrap any callable producing ``(B, embed_dim)`` from ``(B, C, T)`` input
into the :class:`fmscope.FMExtractor` protocol. No inheritance, no
decorator, no registry — pure structural typing.

Run::

    python examples/byo_fm_minimal.py
"""

from __future__ import annotations

import numpy as np
import torch

from fmscope import FMExtractor


class MyExtractor:
    """A 30-line wrapper around any ``torch.nn.Module``-shaped object.

    The contract :class:`fmscope.FMExtractor` requires:

    * attribute ``embed_dim: int``
    * ``__call__(x)`` returning ``(B, embed_dim)`` for input ``(B, C, T)``.
    """

    embed_dim: int = 64

    def __init__(self) -> None:
        torch.manual_seed(0)
        self._proj = torch.nn.Linear(19 * 1000, self.embed_dim)
        self._proj.eval()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self._proj(x.reshape(x.shape[0], -1)).cpu().numpy()


def main() -> None:
    extractor = MyExtractor()

    # ``isinstance`` checks against runtime_checkable Protocols — no
    # subclassing needed.
    assert isinstance(extractor, FMExtractor), "Did not match FMExtractor protocol."
    print(f"[ok] MyExtractor conforms to FMExtractor (embed_dim={extractor.embed_dim})")

    x = np.random.default_rng(0).normal(size=(4, 19, 1000)).astype(np.float32)
    y = extractor(x)
    print(f"[ok] Forward pass: input {x.shape} -> output {y.shape}")
    assert y.shape == (4, extractor.embed_dim)


if __name__ == "__main__":
    main()
