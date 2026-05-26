"""Build the (cohort, FM) verdict matrix — paper Table 3 reproduction.

This module aggregates four diagnostic signals from bundled
``reproduction/data/`` aggregates and produces the per-cell outcome
matrix that maps onto the paper's Table 3
(``\\label{tab:verdict_matrix}``).

The verdict matrix has one row per cell (not per cell × FM pair).
``Δf_label`` / ``layer probe`` / ``c̄`` / ``1/f role`` are aggregated by
taking the **arithmetic mean across the three FMs** of the underlying
diagnostic numbers and then passing the mean through the rubric exactly
once (see :func:`reproduction.builders._rubric.positive_sign`,
:func:`reproduction.builders._rubric.layer_sign`,
:func:`reproduction.builders._rubric.oneoverf_role`). This is *mean→rubric*,
not a per-FM-verdict majority vote: a single FM whose number is far
from the others can pull the mean across the threshold even when the
other two FMs individually disagree with the cohort sign.

To make per-FM disagreement visible without changing the headline
column values, each row carries an ``_agreement`` dict with one entry
per diagnostic of the form ``(n_agree, n_total)``, where ``n_agree``
counts how many FMs' individual sign matches the cohort-level sign
reported in the headline column. A value of ``(3, 3)`` denotes
unanimous agreement; ``(2, 3)`` flags one dissenter. These counts are
exposed for downstream renderers (e.g. paper Tab 3 superscript
annotation) and never alter the visible Tab 3 columns themselves.
"""

from __future__ import annotations

import json

import pandas as pd

from reproduction.builders._runtime import paper_data_path
from reproduction.builders._rubric import (
    CELL_LAYOUT,
    DEFAULT,
    Thresholds,
    classify,
    layer_sign,
    oneoverf_role,
    positive_sign,
)

CELLS_ORDER = ["eegmat", "adftd", "sleepdep", "stress"]
CELL_PRETTY = {
    "eegmat": "EEGMAT",
    "adftd": "ADFTD",
    "sleepdep": "SleepDep",
    "stress": "Stress",
}
FMS = ["labram", "cbramod", "reve"]


def _delta_label_sign(
    cell: str, t: Thresholds
) -> tuple[str, list[float], tuple[int, int]]:
    va = json.loads(paper_data_path("source_tables", "variance_analysis_window_level.json").read_text())
    deltas: list[float] = []
    for fm in FMS:
        entry = va.get(f"{fm}_{cell}", {})
        d = entry.get("delta_label_frac")
        if d is not None:
            deltas.append(float(d))
    cohort_sign = positive_sign(deltas, threshold=t.delta_label_positive)
    per_fm_signs = [positive_sign([d], threshold=t.delta_label_positive)
                    for d in deltas]
    agreement = (sum(1 for s in per_fm_signs if s == cohort_sign), len(deltas))
    return cohort_sign, deltas, agreement


def _layer_probe_sign(
    cell: str, t: Thresholds
) -> tuple[str, dict, tuple[int, int]]:
    d = json.loads(paper_data_path("layerwise_probe", "probes.json").read_text())
    cell_block = d["results"].get(cell, {})
    firsts, lasts, maxes, argmax_depths = [], [], [], []
    for fm in FMS:
        per_depth = cell_block.get(fm, {}).get("per_depth", [])
        if not per_depth:
            continue
        firsts.append(per_depth[0]["label_ba_mean"])
        lasts.append(per_depth[-1]["label_ba_mean"])
        best = max(per_depth, key=lambda p: p["label_ba_mean"])
        maxes.append(best["label_ba_mean"])
        argmax_depths.append(best["depth_fraction"])
    if not firsts:
        return "0", {}, (0, 0)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    s = layer_sign(mean(firsts), mean(lasts), mean(maxes), mean(argmax_depths), t=t)
    info = {
        "label_ba_first_mean": mean(firsts),
        "label_ba_last_mean": mean(lasts),
        "label_ba_max_mean": mean(maxes),
        "argmax_depth_mean": mean(argmax_depths),
    }
    # Per-FM base signs (strip "early"/"deep" qualifier — agreement is on
    # direction only, not on the depth-concentration sub-qualifier).
    cohort_base = s.split()[0]
    per_fm_base = [
        layer_sign(f, l, m, d, t=t).split()[0]
        for f, l, m, d in zip(firsts, lasts, maxes, argmax_depths)
    ]
    agreement = (sum(1 for p in per_fm_base if p == cohort_base), len(per_fm_base))
    return s, info, agreement


def _c_bar_sign(
    cell: str, t: Thresholds
) -> tuple[str, list[float], tuple[int, int]]:
    p = paper_data_path("source_tables", "within_subject_dir_consistency.json")
    if not p.exists():
        return "0", [], (0, 0)
    d = json.loads(p.read_text())
    block = d.get(cell, {}).get("frozen") if cell in d else None
    if not block:
        return "0", [], (0, 0)
    vals = [float(v["dir_consistency"]) for v in block.values()
            if isinstance(v, dict) and v.get("dir_consistency") is not None]
    cohort_sign = positive_sign(vals, threshold=t.c_bar_positive)
    per_fm_signs = [positive_sign([v], threshold=t.c_bar_positive) for v in vals]
    agreement = (sum(1 for s in per_fm_signs if s == cohort_sign), len(vals))
    return cohort_sign, vals, agreement


def _oneoverf_role(
    cell: str, t: Thresholds
) -> tuple[str, dict, tuple[int, int]]:
    p = paper_data_path(cell, "fooof_ablation", "probes.json")
    if not p.exists():
        # Without the FOOOF ablation we cannot classify the 1/f role.
        return "subject axis", {}, (0, 0)
    d = json.loads(p.read_text())
    res = d.get("results", {})
    state_drops, subj_drops = [], []
    for fm in FMS:
        r = res.get(fm)
        if not r:
            continue
        o = r["original"]
        a = r["aperiodic_removed"]
        state_drops.append(o["state_probe_mean"] - a["state_probe_mean"])
        subj_drops.append(o["subject_probe_mean"] - a["subject_probe_mean"])
    if not state_drops:
        return "subject axis", {}, (0, 0)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    state_mean = mean(state_drops)
    subj_mean = mean(subj_drops)
    has_wsc = cell in {"eegmat", "sleepdep"}
    role = oneoverf_role(state_mean, subj_mean,
                         has_within_subject_contrast=has_wsc, t=t)
    per_fm_roles = [
        oneoverf_role(sd, jd, has_within_subject_contrast=has_wsc, t=t)
        for sd, jd in zip(state_drops, subj_drops)
    ]
    agreement = (sum(1 for r in per_fm_roles if r == role), len(per_fm_roles))
    return role, {"state_drop_mean": state_mean,
                  "subject_drop_mean": subj_mean}, agreement


def build_verdict_matrix(*, thresholds: Thresholds = DEFAULT) -> pd.DataFrame:
    """Reproduce paper Table 3 (``\\label{tab:verdict_matrix}``).

    Returns a DataFrame with one row per cell (4 rows) and columns:
    ``cell``, ``layout``, ``delta_f_label``, ``layer_probe``, ``c_bar``,
    ``oneoverf_role``, ``outcome``, plus diagnostic provenance fields.
    """
    rows = []
    for cell in CELLS_ORDER:
        layout = CELL_LAYOUT[cell]
        dl_sign, dl_values, dl_agreement = _delta_label_sign(cell, thresholds)
        lp_sign, lp_info, lp_agreement = _layer_probe_sign(cell, thresholds)
        cb_sign, cb_values, cb_agreement = _c_bar_sign(cell, thresholds)
        # Trait cells (T,*) report c̄ = "0" by construction — no within-subject
        # paired contrast exists. Agreement is not meaningful here: report
        # (n_total, n_total) so downstream renderers treat trait c̄ as
        # "agreement is N/A" rather than "0/n disagreement".
        if layout.startswith("T,"):
            cb_sign = "0"
            cb_agreement = (cb_agreement[1], cb_agreement[1])
        role, role_info, role_agreement = _oneoverf_role(cell, thresholds)
        outcome = classify(
            delta_label_sign=dl_sign,
            layer_probe_sign=lp_sign,
            c_bar_sign=cb_sign,
            oneoverf=role,
            cell_layout=layout,
        )
        # Numeric values (mean across the three FMs) — used by the paper
        # numeric Tab 3 renderer and by external audit consumers that
        # prefer raw quantities over rubric glyphs.
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
        cb_mean = mean(cb_values) if not layout.startswith("T,") else float("nan")
        rows.append({
            "cell": CELL_PRETTY[cell],
            "layout": f"({layout.replace(',', ', ')})",
            "delta_f_label": dl_sign,
            "layer_probe": lp_sign,
            "c_bar": cb_sign,
            "oneoverf_role": role,
            "outcome": outcome,
            "delta_f_label_value": mean(dl_values),
            "layer_max_ba": lp_info.get("label_ba_max_mean", float("nan")),
            "layer_last_ba": lp_info.get("label_ba_last_mean", float("nan")),
            "layer_argmax_depth": lp_info.get("argmax_depth_mean", float("nan")),
            "c_bar_value": cb_mean,
            "state_drop": role_info.get("state_drop_mean", float("nan")),
            "subject_drop": role_info.get("subject_drop_mean", float("nan")),
            "_delta_label_values": dl_values,
            "_layer_probe_info": lp_info,
            "_c_bar_values": cb_values,
            "_oneoverf_info": role_info,
            "_agreement": {
                "delta_f_label": dl_agreement,
                "layer_probe": lp_agreement,
                "c_bar": cb_agreement,
                "oneoverf_role": role_agreement,
            },
        })
    return pd.DataFrame(rows)


# Backwards-compatible aliases — the smoke tests and the verdict CLI
# referenced ``build_table`` previously; keep the name available.
build_table = build_verdict_matrix


GLYPH_COLUMNS = ["cell", "layout", "delta_f_label", "layer_probe",
                 "c_bar", "oneoverf_role", "outcome"]

NUMERIC_COLUMNS = ["cell", "layout", "delta_f_label_value",
                   "layer_max_ba", "layer_last_ba", "c_bar_value",
                   "state_drop", "subject_drop", "outcome"]


def render_markdown(table: pd.DataFrame, *, mode: str = "glyph") -> str:
    """Render a Tab-3 DataFrame as a Markdown table.

    ``mode="glyph"`` (default) emits the legacy +/-/0 columns matching
    paper drafts prior to the numeric pivot. ``mode="numeric"`` emits
    the cohort-mean numeric columns matching the current paper Tab 3.
    Provenance / agreement columns (underscore-prefixed) are always
    dropped from the rendered output.
    """
    cols = NUMERIC_COLUMNS if mode == "numeric" else GLYPH_COLUMNS
    visible = table[[c for c in cols if c in table.columns]]
    return visible.to_markdown(index=False, floatfmt="+.3f")


def render_latex(table: pd.DataFrame, *, mode: str = "glyph") -> str:
    """Render a Tab-3 DataFrame as LaTeX.

    See :func:`render_markdown` for ``mode`` semantics. The numeric
    render is what feeds the paper's Tab 3
    (``\\label{tab:verdict_matrix}``); the glyph render is retained
    for legacy comparison and the paper-reproduction smoke test.
    """
    cols = NUMERIC_COLUMNS if mode == "numeric" else GLYPH_COLUMNS
    visible = table[[c for c in cols if c in table.columns]]
    return visible.to_latex(index=False, escape=True, float_format="%+.3f")


def main(argv: list[str] | None = None) -> int:
    """CLI for ``python -m reproduction.builders.tab3_verdict`` — paper Table 3."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="python -m reproduction.builders.tab3_verdict",
        description="Build the (cell, FM) verdict matrix and reproduce paper Table 3.",
    )
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (.md or .tex). Default: stdout.")
    parser.add_argument("--show-provenance", action="store_true",
                        help="Include underscore-prefixed provenance columns.")
    parser.add_argument("--mode", choices=["glyph", "numeric"], default="numeric",
                        help="Render mode. 'numeric' (default) matches the current "
                             "paper Tab 3; 'glyph' emits the legacy +/-/0 columns.")
    args = parser.parse_args(argv)

    table = build_verdict_matrix()
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.suffix == ".tex":
            outp.write_text(render_latex(table, mode=args.mode))
        else:
            outp.write_text(render_markdown(table, mode=args.mode) + "\n")
        print(f"wrote {outp}", file=sys.stderr)
        return 0
    if args.show_provenance:
        print(table.to_string(index=False))
    else:
        print(render_markdown(table, mode=args.mode))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
