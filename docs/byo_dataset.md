# Bring Your Own cohort

FMScope diagnostics consume cohorts through the `CohortAdapter`
protocol. Three routes from easiest to most general:

1. **`InMemoryCohort`** — hand a list of `(subject_id, label, windows)`
   tuples. Best for already-loaded data.
2. **`PyTorchDatasetAdapter`** — wrap an existing `torch.utils.data.Dataset`
   returning the legacy 4-tuple `(epochs, label, n_ep, sid)`.
3. **Custom class** — implement the protocol directly. Use this if your
   data lives in a database / cloud bucket / streaming source.

## The contract

```python
@runtime_checkable
class CohortAdapter(Protocol):
    n_channels: int
    sfreq: float
    def iter_recordings(self) -> Iterator[tuple[int, int, np.ndarray]]: ...
```

Each `iter_recordings()` call must yield `(subject_id, label, windows)`
where `windows` is shaped `(n_windows, n_channels, n_samples)` and
`subject_id` / `label` are integers.

## Route 1: in-memory

```python
from fmscope.data.adapters import InMemoryCohort
import numpy as np

recordings = []
for sid in range(N_SUBJECTS):
    for label, windows in load_subject(sid):     # your loader
        recordings.append((sid, label, windows))

cohort = InMemoryCohort(recordings, n_channels=19, sfreq=200.0)
```

## Route 2: wrap a PyTorch Dataset

If your dataset already returns `(epochs, label, n_ep, sid)`, the
five-line shim is:

```python
from fmscope.data.adapters import PyTorchDatasetAdapter

cohort = PyTorchDatasetAdapter(my_torch_dataset, n_channels=19, sfreq=200.0)
```

For other tuple layouts, write a custom adapter (route 3).

## Route 3: custom class

```python
class StreamingCohort:
    n_channels = 19
    sfreq = 200.0

    def __init__(self, manifest_path):
        self.manifest = read_manifest(manifest_path)

    def iter_recordings(self):
        for entry in self.manifest:
            windows = fetch_and_window(entry.uri)
            yield entry.subject_id, entry.label, windows
```

`isinstance(StreamingCohort(...), CohortAdapter)` returns `True` —
that is all that is required for the diagnostic pipeline to accept it.

## Windowing convention

The diagnostic pipeline does not re-window. Pre-window your data
yourself:

* 5 s windows at 200 Hz → `n_samples = 1000`
* Channel order should match what your `FMExtractor` expects (most
  bundled FMs accept arbitrary channel counts; check the FM card if in
  doubt).
* `windows` should be float32 µV-scale unless you have z-scored
  upstream — see [`docs/byo_fm.md`](byo_fm.md) for per-FM norm rules.

## Working example

See [`examples/byo_dataset_minimal.py`](../examples/byo_dataset_minimal.py)
for all three routes running end-to-end against synthetic data.

## Running the audit on your cohort

Once your cohort satisfies `CohortAdapter`, the single-call audit path is:

```python
from fmscope.verdict import audit_cell, AuditConfig

# Bring your own frozen FM as a (B, C, T) -> (B, D) callable.
# See docs/byo_fm.md and examples/byo_fm_minimal.py.
extractor = MyExtractor()

row = audit_cell(
    my_cohort, extractor,
    config=AuditConfig(
        cell_name="MyDataset",
        device="cuda:0",
        n_null_seeds=20,
    ),
)
# Numbers, not a verdict:
print(row["label_frac"], row["subject_frac"])
print(row["erasure_label_ba_delta"])     # Δ_erase, if run_erasure (default)
```

The toolkit reports the diagnostic numbers; mapping them to the paper's
Table 3 outcomes is a reproduction-only step (see
[`reproduction/`](../reproduction/)).

For finer-grained diagnostic calls (just variance decomposition, just
direction consistency, just the null calibration), see
[`docs/api.md`](api.md) §"The five diagnostics".
