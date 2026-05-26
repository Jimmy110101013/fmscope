# Bring Your Own foundation model

FMScope diagnostics work on any callable that maps EEG windows to
embeddings. Wrap your model into the `FMExtractor` protocol — no
inheritance, no decorator, no registration.

## The contract

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class FMExtractor(Protocol):
    embed_dim: int                     # positive int
    def __call__(self, x): ...         # (B, C, T) -> (B, embed_dim)
```

* `x` is a batch of EEG windows shaped `(B, channels, samples)`. Accept
  either `torch.Tensor` or `np.ndarray` — FMScope will convert.
* The return must be shaped `(B, embed_dim)` as a numpy array or any
  array-like (the diagnostic pipeline will `np.asarray` it).
* `embed_dim` is a class or instance attribute.

That is the whole contract. `isinstance(my_extractor, FMExtractor)`
returns `True` if both attributes are present.

## Minimal wrapper (30 lines)

```python
import numpy as np
import torch

class MyExtractor:
    embed_dim: int = 256

    def __init__(self, ckpt_path: str):
        self.model = load_my_pretrained_model(ckpt_path)
        self.model.eval()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.model.encode(x).cpu().numpy()
```

That is it. Pass `MyExtractor()` anywhere FMScope accepts an
`FMExtractor` — linear probes, layer-wise probes, verdict-matrix
construction.

## Input normalization (read this)

Each bundled FM extractor declares an internal normalization
convention; passing the wrong norm silently destroys the run:

| FM | Required `--norm` | Why |
|---|---|---|
| LaBraM | `none` | The extractor internally divides by 100; passing z-scored input double-scales to ~0. |
| CBraMod | `none` | Same internal /100 scaling. |
| REVE | `none` | Same internal scaling (per-dataset `scale_factor` in upstream `task/*.yaml`). |
| EEGNet / ShallowConvNet / DeepConvNet / EEGConformer | `zscore` | Lawhern / Schirrmeister / Song conventions; early BatchNorm assumes zero-mean input. |

If your BYO model expects µV input, accept `(B, C, T)` directly and do
nothing extra. If it expects z-scored input, document the convention
on your class so callers know to z-score before passing data.

## Working examples

See [`examples/byo_fm_minimal.py`](../examples/byo_fm_minimal.py) for a
runnable 30-line wrapper around a trivial linear projection — passes
`isinstance(_, FMExtractor)` and produces correctly-shaped output.

For a real FM, wrap its frozen forward pass so a `(B, C, T)` batch maps
to `(B, embed_dim)` and expose an `embed_dim` attribute. Any
LaBraM / CBraMod / REVE checkpoint satisfies the protocol once wrapped
this way; FMScope ships no vendored model code, so the wrapper is yours.
