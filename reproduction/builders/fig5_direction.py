"""Build Fig 5 (direction panels) — within-subject directional coherence.

Usage (from repo root):
    python -m reproduction.builders.fig5_direction

Two panels saved as separate PDFs/PNGs (PPT-friendly composition):

  fig5ab_rose_combined  — combined polar rose, 1 row × 3 FMs.
                           EEGMAT (filled gray) vs SleepDep (outlined black)
                           on the same half-circle, per FM.
  fig5c_dir_consistency — cross-dataset dir_consistency bar.

Method (panel 4ab):
  • For each subject, direction v_i = (mean_class1 − mean_class0), normalised
    in the FM's full feature space (200 / 512 D — no projection).
  • Group consensus v_c = mean(v_i), normalised.
  • θ_i = arccos(⟨v_i, v_c⟩) in degrees.
  • Polar half-circle [0°, 180°]: 0° = aligned with consensus, 90° = isotropic
    null (high-D random vectors are nearly orthogonal), 180° = anti-aligned.
  • Coherent subjects pull θ toward 0; incoherent stay near 90°.

Replaces the prior UMAP-trajectory strip, which couldn't visually render
direction: UMAP preserves topology not direction, so dir=+0.149 and
dir=+0.022 looked equally chaotic.

Canonical source: ``results.source_table('within_subject_dir_consistency')``.
Note this differs from legacy ``exp11/within_subject_supplementary.json``
for REVE EEGMAT (0.192 vs 0.064) because the canonical table was recomputed
after the 2026-04-27 REVE scale_factor=100 alignment. LaBraM/CBraMod match.
"""
from __future__ import annotations
import sys
from pathlib import Path

from reproduction.builders._runtime import output_dir, paper_data_path
# fmscope is pip-installed; no sys.path hacks needed
# fmscope is pip-installed; no sys.path hacks needed

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from reproduction.builders import results_accessor as results  # noqa: E402
from reproduction.builders._style import apply_jne_style, JNE_DOUBLE_COL_CM, CM  # noqa: E402

OUT = output_dir("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = paper_data_path("features_cache")

apply_jne_style()

W_DOUBLE_IN = JNE_DOUBLE_COL_CM * CM  # 15 cm → ~5.91 in
# fig5c (\bar c bars) and fig5d (SNR bars) ship side-by-side as
# subfigures; design native to match the LaTeX display width (0.49
# \linewidth) so \includegraphics scales 1:1 and labels don't shrink.
W_HALF_IN  = 0.49 * JNE_DOUBLE_COL_CM * CM  # ≈ 7.35 cm → ~2.89 in

FMS = ["labram", "cbramod", "reve"]
FM_PRETTY = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
FM_COLOR = {"labram": "#1f3a5f", "cbramod": "#B8442C", "reve": "#2E7D5B"}

dc_table = results.source_table("within_subject_dir_consistency")


def load_perwindow(fm: str, ds: str):
    """Mean-pool per-window frozen features into per-recording features."""
    f = CACHE / f"frozen_{fm}_{ds}_perwindow.npz"
    d = np.load(f, allow_pickle=True)
    pooled = np.stack([
        d["features"][d["window_rec_idx"] == i].mean(0)
        for i in range(len(d["rec_pids"]))
    ])
    return pooled, d["rec_pids"], d["rec_labels"].astype(int)


def per_subject_angles(X: np.ndarray, pids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return angle (deg) of each subject's direction vector to the group consensus."""
    dirs = []
    for pid in np.unique(pids):
        m = pids == pid
        x0 = X[m & (labels == 0)]
        x1 = X[m & (labels == 1)]
        if len(x0) == 0 or len(x1) == 0:
            continue
        diff = x1.mean(0) - x0.mean(0)
        n = np.linalg.norm(diff)
        if n > 0:
            dirs.append(diff / n)
    dirs = np.stack(dirs)
    consensus = dirs.mean(0)
    consensus /= np.linalg.norm(consensus) + 1e-12
    cos = np.clip(dirs @ consensus, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def render_angle_rose_combined(out_stem: str) -> None:
    """Combined rose plot — both datasets on the same polar half-circle.

    EEGMAT: mid-gray filled wedges. SleepDep: near-black outline (no fill).
    Three panels (one per FM) for direct visual comparison.

    Style: Nature-tier minimal panel — titles carry only FM name; c̄ numeric
    callouts centred below title in uniform dark gray; reference lines
    de-emphasised; angle semantics demoted to figure-level legend.
    """
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE_IN, W_DOUBLE_IN * 0.32),
                             subplot_kw={"projection": "polar"})
    bin_edges_deg = np.linspace(0, 180, 19)
    bin_edges_rad = np.deg2rad(bin_edges_deg)
    bin_centers_rad = 0.5 * (bin_edges_rad[:-1] + bin_edges_rad[1:])
    bin_width_rad = bin_edges_rad[1] - bin_edges_rad[0]

    GRID = "#D8D8D8"
    TICK = "#888"
    DARK = "#222"        # near-black for both datasets (geometry distinguishes)
    FILL = "#666"        # mid-gray fill for EEGMAT (with alpha)
    LABEL = "#333"       # dark gray for c̄ numeric callouts

    for ax, fm in zip(axes, FMS):
        Xe, Pe, Le = load_perwindow(fm, "eegmat")
        ang_e = per_subject_angles(Xe, Pe, Le)
        cnt_e, _ = np.histogram(ang_e, bins=bin_edges_deg)
        Xs, Ps, Ls = load_perwindow(fm, "sleepdep")
        ang_s = per_subject_angles(Xs, Ps, Ls)
        cnt_s, _ = np.histogram(ang_s, bins=bin_edges_deg)

        rmax = max(cnt_e.max(), cnt_s.max()) * 1.20

        # EEGMAT: filled mid-gray (alpha ≈ 0.5)
        ax.bar(bin_centers_rad, cnt_e, width=bin_width_rad * 0.95,
               color=FILL, alpha=0.5,
               edgecolor=FILL, lw=0.3,
               align="center", zorder=2)
        # SleepDep: outlined near-black (no fill)
        ax.bar(bin_centers_rad, cnt_s, width=bin_width_rad * 0.95,
               facecolor="none",
               edgecolor=DARK, lw=1.2,
               align="center", zorder=3)

        # null reference (light dashed)
        ax.plot([np.pi / 2, np.pi / 2], [0, rmax],
                color=GRID, lw=0.7, ls="--", zorder=1)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
        ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°"],
                           fontsize=6.5, color=TICK)
        ax.set_yticks([])
        ax.set_ylim(0, rmax)
        ax.set_frame_on(False)  # turn off polar frame (incl. bottom diameter)
        ax.grid(alpha=1.0, lw=0.35, color=GRID)
        ax.tick_params(colors=TICK, pad=1)

        # θ̄ callouts: 將 Y 軸座標稍微往下拉 (從 1.10/1.02 改為 1.12/1.02)
        ax.text(0.5, 1.12,
                f"$\\bar\\theta_\\mathrm{{EEGMAT}}$ = {ang_e.mean():.0f}°",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.5, color=LABEL)
        ax.text(0.5, 1.02,
                f"$\\bar\\theta_\\mathrm{{Sleepdep}}$ = {ang_s.mean():.0f}°",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.5, color=LABEL)

        # Title: 將 pad 從 22 增加到 34，把模型名字往上推，讓出空間
        ax.set_title(FM_PRETTY[fm], fontsize=10,
                     color=FM_COLOR[fm], fontweight="bold", pad=34)

    # minimal legend: filled vs outlined squares (geometry distinguishes)
    legend_elems = [
        plt.matplotlib.patches.Patch(facecolor=FILL, alpha=0.5,
                                     edgecolor=FILL,
                                     label="EEGMAT"),
        plt.matplotlib.patches.Patch(facecolor="none",
                                     edgecolor=DARK, lw=1.2,
                                     label="SleepDep"),
    ]
    # Two centred rows below the panels:
    #   row 1 — dataset legend (colored squares)
    #   row 2 — angle-semantics annotation
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.04),
               handlelength=1.2, handletextpad=0.5, columnspacing=2.0)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{out_stem}.{ext}")
    plt.close(fig)
    print(f"wrote {out_stem}.{{pdf,png}}")


def render_bar(out_stem: str) -> None:
    """DC bar chart: frozen vs FT (3-seed mean ± std), parallel to SNR panel.

    Width 7.6 matches fig2ab so LaTeX stacks panels cleanly.
    Color encoding shared with rose panel (filled gray = EEGMAT,
    outlined dark = SleepDep); hatched bars indicate FT (3-seed).
    """
    fig, ax = plt.subplots(figsize=(W_HALF_IN, W_HALF_IN * 0.62))

    x = np.arange(len(FMS))
    bar_w = 0.18
    GAP = 0.04
    FILL = "#666"
    DARK = "#222"

    # Order along x for each FM (matches SNR panel for visual alignment):
    #   EEGMAT frozen | EEGMAT FT-3seed | SleepDep frozen | SleepDep FT-3seed
    offs = np.array([-1.5*bar_w - 1.5*GAP,
                     -0.5*bar_w - 0.5*GAP,
                     +0.5*bar_w + 0.5*GAP,
                     +1.5*bar_w + 1.5*GAP])

    for i, fm in enumerate(FMS):
        e_fz = dc_table["eegmat"]["frozen"][fm]["dir_consistency"]
        e_ft_mean = dc_table["eegmat"]["ft_multiseed"][fm]["dc_mean"]
        e_ft_std = dc_table["eegmat"]["ft_multiseed"][fm]["dc_std"]
        s_fz = dc_table["sleepdep"]["frozen"][fm]["dir_consistency"]
        s_ft_mean = dc_table["sleepdep"]["ft_multiseed"][fm]["dc_mean"]
        s_ft_std = dc_table["sleepdep"]["ft_multiseed"][fm]["dc_std"]

        # EEGMAT frozen + FT
        ax.bar(x[i] + offs[0], e_fz, bar_w,
               facecolor=FILL, alpha=0.5, edgecolor=FILL, lw=0.3)
        ax.bar(x[i] + offs[1], e_ft_mean, bar_w,
               facecolor=FILL, alpha=0.5, edgecolor=FILL, lw=0.3,
               hatch="//")
        # SleepDep frozen + FT
        ax.bar(x[i] + offs[2], s_fz, bar_w,
               facecolor="none", edgecolor=DARK, lw=1.2)
        ax.bar(x[i] + offs[3], s_ft_mean, bar_w,
               facecolor="none", edgecolor=DARK, lw=1.2,
               hatch="//")

    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([FM_PRETTY[fm] for fm in FMS],
                       fontsize=10, fontweight="bold")
    for tick, fm in zip(ax.get_xticklabels(), FMS):
        tick.set_color(FM_COLOR[fm])

    ax.set_ylabel(r"Directional Consistency $\bar{c}$", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, pad=5)

    # Compact legend: only frozen-vs-FT distinction
    # (EEGMAT/SleepDep encoding is shared with rose panel above).
    legend_elems = [
        plt.matplotlib.patches.Patch(facecolor="lightgray",
                                     edgecolor=DARK, lw=0.6,
                                     label="frozen"),
        plt.matplotlib.patches.Patch(facecolor="lightgray",
                                     edgecolor=DARK, lw=0.6,
                                     hatch="//",
                                     label="FT"),
    ]
    ax.legend(handles=legend_elems, loc="upper center",
              ncol=2, frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, 1.20),
              handlelength=1.0, handleheight=0.7,
              handletextpad=0.4, columnspacing=1.2)

    # Pin layout + save at exact figsize (override apply_jne_style's
    # savefig.bbox="tight") so fig5c and fig5d save at identical
    # dimensions regardless of differing ylabel widths.
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.20)
    with mpl.rc_context({"savefig.bbox": "standard"}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{out_stem}.{ext}")
    plt.close(fig)
    print(f"wrote {out_stem}.{{pdf,png}}")


def render_snr_bar(out_stem: str) -> None:
    """SNR ratio bar (frozen vs FT 3-seed mean ± std).

    Per (cell × FM), shows frozen SNR (filled gray for EEGMAT, outlined
    black for SleepDep) and FT 3-seed mean SNR with std error bars.
    Same width / X-axis layout as render_bar so fig2c and fig2d stack
    cleanly in LaTeX.
    """
    fig, ax = plt.subplots(figsize=(W_HALF_IN, W_HALF_IN * 0.62))

    x = np.arange(len(FMS))
    bar_w = 0.18
    GAP = 0.04
    FILL = "#666"
    DARK = "#222"

    # Order along x for each FM:
    #   EEGMAT frozen | EEGMAT FT-3seed | SleepDep frozen | SleepDep FT-3seed
    offs = np.array([-1.5*bar_w - 1.5*GAP,
                     -0.5*bar_w - 0.5*GAP,
                     +0.5*bar_w + 0.5*GAP,
                     +1.5*bar_w + 1.5*GAP])

    for i, fm in enumerate(FMS):
        e_fz = dc_table["eegmat"]["frozen"][fm]["snr"]
        e_ft_mean = dc_table["eegmat"]["ft_multiseed"][fm]["snr_mean"]
        e_ft_std = dc_table["eegmat"]["ft_multiseed"][fm]["snr_std"]
        s_fz = dc_table["sleepdep"]["frozen"][fm]["snr"]
        s_ft_mean = dc_table["sleepdep"]["ft_multiseed"][fm]["snr_mean"]
        s_ft_std = dc_table["sleepdep"]["ft_multiseed"][fm]["snr_std"]

        # EEGMAT frozen + FT
        ax.bar(x[i] + offs[0], e_fz, bar_w,
               facecolor=FILL, alpha=0.5, edgecolor=FILL, lw=0.3)
        ax.bar(x[i] + offs[1], e_ft_mean, bar_w,
               facecolor=FILL, alpha=0.5, edgecolor=FILL, lw=0.3,
               hatch="//")
        # SleepDep frozen + FT
        ax.bar(x[i] + offs[2], s_fz, bar_w,
               facecolor="none", edgecolor=DARK, lw=1.2)
        ax.bar(x[i] + offs[3], s_ft_mean, bar_w,
               facecolor="none", edgecolor=DARK, lw=1.2,
               hatch="//")

    # SNR=1 reference
    ax.axhline(1.0, color="#888", lw=0.7, ls="--", alpha=0.7,
               label="SNR=1 (signal ~ noise)")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([FM_PRETTY[fm] for fm in FMS],
                       fontsize=10, fontweight="bold")
    for tick, fm in zip(ax.get_xticklabels(), FMS):
        tick.set_color(FM_COLOR[fm])
    ax.set_ylabel(r"Per-subject SNR  $\overline{\sigma}_s\,/\,\mathrm{SD}_s(\sigma_s)$",
                  fontsize=9)

    # Compact legend: only frozen-vs-FT distinction (color/fill encoding
    # for EEGMAT vs SleepDep is shared with the rose panel above and
    # repeated in panel c, so we do not duplicate it here).
    legend_elems = [
        plt.matplotlib.patches.Patch(facecolor="lightgray",
                                     edgecolor=DARK, lw=0.6,
                                     label="frozen"),
        plt.matplotlib.patches.Patch(facecolor="lightgray",
                                     edgecolor=DARK, lw=0.6,
                                     hatch="//",
                                     label="FT"),
    ]
    ax.legend(handles=legend_elems, loc="upper center",
              ncol=2, frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, 1.20),
              handlelength=1.0, handleheight=0.7,
              handletextpad=0.4, columnspacing=1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, pad=5)

    # Pin layout + save at exact figsize (override apply_jne_style's
    # savefig.bbox="tight") so fig5c and fig5d save at identical
    # dimensions regardless of differing ylabel widths.
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.20)
    with mpl.rc_context({"savefig.bbox": "standard"}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{out_stem}.{ext}")
    plt.close(fig)
    print(f"wrote {out_stem}.{{pdf,png}}")


def main() -> None:
    render_angle_rose_combined("fig5ab_rose_combined")
    render_bar("fig5c_dir_consistency")
    render_snr_bar("fig5d_snr_ratio")


if __name__ == "__main__":
    main()
