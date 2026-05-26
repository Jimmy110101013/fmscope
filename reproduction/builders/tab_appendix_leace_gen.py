"""Reproduce appendix Tab. (tab:erasure_gen) — erasure beyond the EEGMAT demo.

Per-cohort raw label BA (range over the three FMs) and Δ_erase (post − pre,
3-seed mean, pp) per FM. EEGMAT / SAM40 / TDBRAIN-state are read from the
bundled erasure-probe checkpoint; the three external generalization cohorts
are read from their bundled ``fmscope_leace_ds*.json`` files. Top block:
established consensus markers; bottom block: non-established markers.

    python -m reproduction.builders.tab_appendix_leace_gen
"""
from __future__ import annotations

import json

from reproduction.builders._runtime import paper_data_path

FMS = ["labram", "cbramod", "reve"]


def _probes() -> dict:
    p = paper_data_path("source_tables", "fmscope_mechanism_erasure.json")
    return json.loads(p.read_text())


def _gen(cohort: str) -> dict:
    p = paper_data_path("source_tables", f"fmscope_leace_{cohort}.json")
    return json.loads(p.read_text())["results"]


# (display label with citep key, {fm: erasure record}) per row.
def _rows():
    pr = _probes()
    rows_top = [
        (r"EEGMAT \citep{eegmat}",
         {fm: pr[f"{fm}/eegmat"] for fm in FMS}),
        (r"Test--retest \citep{wang2022testretest}",
         {fm: _gen("ds004148")[f"{fm}/ds004148"] for fm in FMS}),
        (r"EEGMMIDB \citep{schalk2004bci2000}",
         {fm: _gen("ds004362")[f"{fm}/ds004362"] for fm in FMS}),
        (r"Aud.--Vis.\ Shift \citep{ceponiene2008}",
         {fm: _gen("ds002893")[f"{fm}/ds002893"] for fm in FMS}),
    ]
    rows_bottom = [
        (r"SAM40 \citep{ghosh2022sam40}",
         {fm: pr[f"{fm}/sam40"] for fm in FMS}),
        (r"TDBRAIN-state \citep{vandijk2022tdbrain}",
         {fm: pr[f"{fm}/tdbrain_state"] for fm in FMS}),
    ]
    return rows_top, rows_bottom


def _fmt_row(label, recs) -> str:
    raws = [100 * recs[fm]["label_ba_raw"] for fm in FMS]
    deltas = [100 * recs[fm]["label_ba_delta"] for fm in FMS]
    rng = f"{min(raws):.0f}--{max(raws):.0f}"
    dcells = " & ".join(rf"\({d:+.1f}\)" for d in deltas)
    return f"{label} & {rng} & {dcells} " + r"\\"


def print_latex():
    print("=" * 70)
    print("Tab. tab:erasure_gen — erasure generalization (appendix)")
    print("=" * 70)
    print(r"% Cohort | Raw BA (range over 3 FMs) | Δ_erase: LaBraM | CBraMod | REVE")
    top, bottom = _rows()
    for label, recs in top:
        print(_fmt_row(label, recs))
    print(r"\midrule")
    for label, recs in bottom:
        print(_fmt_row(label, recs))


if __name__ == "__main__":
    print_latex()
