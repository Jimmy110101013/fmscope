"""Cohort adapters — bring your own dataset.

The toolkit operates on any cohort satisfying :class:`fmscope.api.CohortAdapter`.
Wrap an in-memory cohort or a torch ``Dataset`` with the helpers here;
dataset-specific loaders are not shipped (see ``examples/byo_dataset_minimal.py``).
"""

from fmscope.api import CohortAdapter
from fmscope.data.adapters import InMemoryCohort, PyTorchDatasetAdapter, synthetic_cohort

__all__ = [
    "CohortAdapter",
    "InMemoryCohort",
    "PyTorchDatasetAdapter",
    "synthetic_cohort",
]
