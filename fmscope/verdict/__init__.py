"""Live audit API — numbers-only diagnostic row for any cohort × extractor.

``audit_cell`` composes the FMScope diagnostics and returns raw numbers;
it does not classify. Paper Tab 3 reproduction (the rubric that maps the
numbers to +/−/0 outcomes) lives under
``reproduction/builders/`` outside this package.
"""

from fmscope.verdict.audit import AuditConfig, audit_cell

__all__ = ["audit_cell", "AuditConfig"]
