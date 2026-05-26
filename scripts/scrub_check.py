"""Anonymization scanner for the FMScope public release.

Scans tracked files for identifying information that must NOT appear in
the anonymous reviewer mirror:

* Personal identifiers: ``linjimmy``, ``jimmy``, ``UCSD``, ``Komarov``
  (collaborator), ``Wang`` (collaborator), ``Anonymous-author`` placeholders.
* Absolute paths from the development tree: ``/raid/jupyter-linjimmy1003.md10/``
  and any other ``/raid/`` path.
* Email addresses.
* Git config leaks: ``git config user.name``, ``user.email``.
* Jupyter notebook execution outputs that may carry environment paths
  or full tracebacks (anonymity audit treats outputs as suspect).
* ``Co-Authored-By:`` trailers from collaborative commits.

Usage::

    python scripts/scrub_check.py                # report only
    python scripts/scrub_check.py --strict       # exit 1 on any hit (for CI)
    python scripts/scrub_check.py --json         # machine-readable output

The scanner is opinionated about what counts as a hit. Common false
positives (e.g. ``Anonymous`` in CITATION.cff) are pre-allowlisted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]

# Patterns that ALWAYS count as a hit when found in tracked files.
HARD_PATTERNS: list[tuple[str, str]] = [
    ("dev_path", r"/raid/jupyter-linjimmy1003\.md10/"),
    ("dev_path_generic", r"/raid/(?!jupyter-linjimmy1003)[A-Za-z0-9_\-./]+"),
    ("identifier_linjimmy", r"\blinjimmy\b"),
    ("identifier_jimmy", r"\bjimmy\b"),
    ("identifier_ucsd", r"\bUCSD\b"),
    ("identifier_komarov", r"\bKomarov\b"),  # collaborator name
    # NB: 'Wang' is too common; only flag the bibliographic 'Wang et al' citations
    # which the public README must omit per anonymization rules.
    ("identifier_wang_etal", r"\bWang et al"),
    ("email_pattern", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("git_author_real", r"Co-Authored-By:\s*(?!Anonymous)"),
]

# Patterns that count as a hit unless paired with the strict allowlist.
SOFT_PATTERNS: list[tuple[str, str]] = [
    ("our_lab", r"\bour lab\b"),
    ("our_group", r"\bour group\b"),
    ("we_acknowledge", r"\bwe acknowledge\b"),  # often carries grant info
]

# Files we never scan (binary blobs, vendored dirs, cached builds).
EXCLUDE_DIRS = {
    ".git", ".github/workflows/_cache", "__pycache__", ".pytest_cache",
    ".mypy_cache", "node_modules", ".ipynb_checkpoints", "paper_figures",
    # Build artifacts — gitignored, never shipped.
    "build", "dist",
    # paper_data JSONs contain dataset names like 'eegmat' (fine) and
    # provenance fields that may carry source-tree paths (NOT fine).
    # We DO scan paper_data — anonymization must reach the bundled JSONs.
}
# Path-suffix patterns to skip (handles wildcarded names like *.egg-info).
EXCLUDE_PATH_SUFFIXES = (".egg-info",)
EXCLUDE_PATTERNS = {".pyc", ".pyo", ".so", ".pdf", ".png", ".jpg", ".jpeg",
                    ".npz", ".npy", ".pt", ".pth", ".bin"}

# Per-file allowlist: hit_kind -> set of file path suffixes where the hit
# is expected and OK. Use the suffix relative to REPO root.
ALLOWLIST: dict[str, set[str]] = {
    # Allow the scrub_check tool itself to mention the patterns it scans for.
    "dev_path": {"scripts/scrub_check.py"},
    "dev_path_generic": {"scripts/scrub_check.py"},
    "identifier_linjimmy": {"scripts/scrub_check.py"},
    "identifier_jimmy": {"scripts/scrub_check.py"},
    "identifier_ucsd": {"scripts/scrub_check.py"},
    "identifier_komarov": {"scripts/scrub_check.py"},
    "identifier_wang_etal": {"scripts/scrub_check.py"},
    "email_pattern": {"scripts/scrub_check.py"},
    "git_author_real": {"scripts/scrub_check.py"},
    "our_lab": {"scripts/scrub_check.py"},
    "our_group": {"scripts/scrub_check.py"},
    "we_acknowledge": {"scripts/scrub_check.py"},
}


@dataclass
class Hit:
    kind: str
    path: str
    line: int
    snippet: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path, "line": self.line, "snippet": self.snippet}


@dataclass
class Report:
    hits: list[Hit] = field(default_factory=list)
    files_scanned: int = 0

    def add(self, h: Hit) -> None:
        self.hits.append(h)


def iter_tracked_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if any(part.endswith(EXCLUDE_PATH_SUFFIXES) for part in rel.parts):
            continue
        if p.suffix in EXCLUDE_PATTERNS:
            continue
        yield p


def scan_text_file(path: Path, rel: str, report: Report) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    lines = text.splitlines()
    for kind, pattern in HARD_PATTERNS + SOFT_PATTERNS:
        if rel in ALLOWLIST.get(kind, set()):
            continue
        for lineno, line in enumerate(lines, start=1):
            if re.search(pattern, line):
                report.add(Hit(kind=kind, path=rel, line=lineno,
                               snippet=line.strip()[:200]))


def scan_notebook(path: Path, rel: str, report: Report) -> None:
    """Notebooks: scan source AND flag any cell that carries outputs."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    cells = data.get("cells", [])
    # Flag cell outputs unless this is one of the pre-rendered showcase
    # notebooks. Flagship demos ship with outputs on purpose so readers
    # can see the report without running anything.
    OUTPUT_OK = {"notebooks/audit_demo.ipynb"}
    if rel not in OUTPUT_OK:
        for i, cell in enumerate(cells):
            if cell.get("outputs"):
                report.add(Hit(kind="ipynb_outputs", path=rel, line=i + 1,
                               snippet="cell has outputs — strip before release"))
            if cell.get("execution_count") is not None:
                report.add(Hit(kind="ipynb_exec_count", path=rel, line=i + 1,
                               snippet="cell has execution_count — strip"))
    # Also scan the source text against the pattern list.
    flat = json.dumps(data, ensure_ascii=False)
    for kind, pattern in HARD_PATTERNS:
        if rel in ALLOWLIST.get(kind, set()):
            continue
        if re.search(pattern, flat):
            for lineno, line in enumerate(flat.splitlines(), start=1):
                if re.search(pattern, line):
                    report.add(Hit(kind=kind, path=rel, line=lineno,
                                   snippet=line.strip()[:200]))
                    break


def scan(root: Path) -> Report:
    report = Report()
    for path in iter_tracked_files(root):
        rel = str(path.relative_to(root))
        if path.suffix == ".ipynb":
            scan_notebook(path, rel, report)
        else:
            scan_text_file(path, rel, report)
        report.files_scanned += 1
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any hit is found (for CI gates)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--root", type=str, default=str(REPO),
                        help="repo root to scan (default: parent of this file)")
    args = parser.parse_args(argv)

    report = scan(Path(args.root))

    if args.json:
        print(json.dumps({
            "files_scanned": report.files_scanned,
            "n_hits": len(report.hits),
            "hits": [h.to_dict() for h in report.hits],
        }, indent=2))
    else:
        print(f"Scanned {report.files_scanned} files. Hits: {len(report.hits)}")
        if not report.hits:
            print("[ok] No anonymization hits found.")
        else:
            # Group by file for readability.
            by_file: dict[str, list[Hit]] = {}
            for h in report.hits:
                by_file.setdefault(h.path, []).append(h)
            for path, hits in sorted(by_file.items()):
                print(f"\n{path}:")
                for h in hits:
                    print(f"  L{h.line} [{h.kind}] {h.snippet}")

    if args.strict and report.hits:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
