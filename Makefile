.PHONY: install test lint typecheck scrub notebooks verdict clean help

# Default target — print available commands.
help:
	@echo "FMScope developer targets:"
	@echo "  make install     -- pip install -e .[dev]"
	@echo "  make test        -- pytest"
	@echo "  make lint        -- ruff check + ruff format --check"
	@echo "  make typecheck   -- mypy on public surface"
	@echo "  make scrub       -- anonymization gate (strict)"
	@echo "  make notebooks   -- papermill-execute the reproduction notebooks"
	@echo "  make verdict     -- reproduce paper Table 3"
	@echo "  make clean       -- remove caches and build artifacts"

install:
	pip install -e .[dev]

test:
	pytest tests/ reproduction/tests/ -q

lint:
	ruff check fmscope/ tests/
	ruff format --check fmscope/ tests/

typecheck:
	mypy fmscope/api.py fmscope/verdict fmscope/diagnostics

scrub:
	python scripts/scrub_check.py --strict

notebooks:
	for nb in reproduction/notebooks/*.ipynb; do \
		papermill --no-progress-bar --log-level WARNING "$$nb" "/tmp/_pm_$$(basename $$nb)" || exit 1; \
	done

verdict:
	python -m reproduction.builders.tab3_verdict

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
