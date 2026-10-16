"""
Figure B: Cascade Gain vs. Stale Semantic Accuracy (E8)
-------------------------------------------------------
Scatter plot: x = stale semantic accuracy, y = cascade gain over stale semantic.
4 points from E8 (SDXL and FLUX x SigLIP2 / DINOv2 -- CLIP excluded, see below).
Includes OLS trendline and a horizontal zero-gain reference line. All 4 points now
show positive gain (no negative-gain point remains once the unreconciled FLUX+CLIP
cell -- the only negative-gain cell in the old 6-point version -- is excluded).

All numbers are read LIVE from CSVs at plot time (no hand-copied constants).

Data sources:
  SigLIP2/DINOv2 (SDXL/FLUX) — routing/pipeline/confusion_matrices/
    sdxl_flux_confusion_matrices.csv (stale vs. cascade overall_accuracy; gain =
    cascade - stale). This is the reconciled, reproducible snapshot — see
    RECONCILIATION_REPORT.md in that directory. The old hardcoded values sourced
    from e8_modern_diffusion.csv were STALE (probes were retrained after that run):
      SDXL  SigLIP2: stale 0.8465->0.86375,  gain 0.0340->0.0535
      SDXL  DINOv2:  stale 0.85475->0.851,   gain 0.0360->0.0678 (approx)
      FLUX  SigLIP2: stale 0.90925->0.90625, gain 0.0038->0.0265
      FLUX  DINOv2:  stale 0.815->0.81775,   gain 0.0488->0.0660 (approx)
  CLIP (SDXL/FLUX) — EXCLUDED, same reasoning as plot_fig_a_semantic_degradation.py:
    the paper's own Methods text already excludes CLIP from the SDXL/FLUX cascade
    analysis, and its stale cross-family number cannot currently be verified against
    the corrected probe snapshot. Only 4 points (2 datasets x SigLIP2/DINOv2) are
    plotted now, not 6.

Output: routing/plot_fig_b_cascade_gain_scatter.png
        routing/plot_fig_b_cascade_gain_scatter.pdf
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
CONFUSION_DIR = os.path.join(HERE, "..", "pipeline", "confusion_matrices")

E8_MODERN_PATH = os.path.join(RESULTS_DIR, "e8_modern_diffusion.csv")
CONFUSION_PATH = os.path.join(CONFUSION_DIR, "sdxl_flux_confusion_matrices.csv")


def load_reconciled_points(path):
    """SigLIP2/DINOv2 points: stale & cascade overall_accuracy from the reconciled
    confusion-matrix CSV, read live. gain = cascade_acc - stale_acc."""
    stale, cascade = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["backbone"] not in ("SigLIP2", "DINOv2"):
                continue
            key = (row["dataset"], row["backbone"])
            if row["stage"] == "stale":
                stale[key] = float(row["overall_accuracy"])
            elif row["stage"] == "cascade":
                cascade[key] = float(row["overall_accuracy"])
    points = []
    for key in stale:
        dataset, backbone = key
        s = stale[key]
        c = cascade[key]
        points.append((dataset, backbone, s, c - s))
    return points


# CLIP is deliberately excluded -- see the module docstring. No load_clip_points()
# call; only the reconciled SigLIP2/DINOv2 points are plotted.
# ---------------------------------------------------------------------------
# Points: 4 total (SDXL and FLUX x SigLIP2 / DINOv2)
# ---------------------------------------------------------------------------
POINTS = load_reconciled_points(CONFUSION_PATH)
# Fix a stable, readable order: dataset then backbone (SigLIP2, DINOv2)
_backbone_order = {"SigLIP2": 0, "DINOv2": 1}
_dataset_order = {"SDXL": 0, "FLUX": 1}
POINTS.sort(key=lambda p: (_dataset_order[p[0]], _backbone_order[p[1]]))

DATASET_MARKER = {"SDXL": "o", "FLUX": "s"}
BACKBONE_COLOR = {
    "SigLIP2": "#FF9800",
    "DINOv2":  "#4CAF50",
}
MARKER_SIZE = 110

fig, ax = plt.subplots(figsize=(6.5, 4.5))

xs = np.array([p[2] for p in POINTS])
ys = np.array([p[3] for p in POINTS])

# ── Plot points ──────────────────────────────────────────────────────────────
for (dataset, backbone, x, y) in POINTS:
    ax.scatter(
        x, y,
        s=MARKER_SIZE,
        marker=DATASET_MARKER[dataset],
        color=BACKBONE_COLOR[backbone],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

# ── Annotations ──────────────────────────────────────────────────────────────
offsets_label = {
    ("SDXL", "CLIP"):    (-0.002, +0.0025),
    ("SDXL", "SigLIP2"): (-0.0065, -0.004),
    ("SDXL", "DINOv2"):  (+0.002, +0.0020),
    ("FLUX", "CLIP"):    (+0.001, -0.0045),
    ("FLUX", "SigLIP2"): (+0.0015, +0.0025),
    ("FLUX", "DINOv2"):  (-0.005, +0.002),
}
for (dataset, backbone, x, y) in POINTS:
    dx, dy = offsets_label[(dataset, backbone)]
    label = f"{dataset}\n{backbone}"
    ax.annotate(
        label, xy=(x, y), xytext=(x + dx, y + dy),
        fontsize=7.5, ha="center", va="center",
        color="#333333",
    )

# ── OLS trendline ─────────────────────────────────────────────────────────────
coeffs = np.polyfit(xs, ys, 1)
x_line = np.linspace(xs.min() - 0.005, xs.max() + 0.005, 200)
y_line = np.polyval(coeffs, x_line)
ax.plot(x_line, y_line, color="#9E9E9E", linestyle="--",
        linewidth=1.4, zorder=1, label="OLS trendline")

# ── Zero-gain reference ───────────────────────────────────────────────────────
ax.axhline(0, color="#E53935", linestyle="-", linewidth=1.0,
           alpha=0.7, zorder=1, label="No gain (break-even)")

# ── Shaded gain region (padded to the data's actual range, not hardcoded) ───────
y_pad = 0.008
y_hi = max(ys.max() + y_pad, 0.0)
y_lo = min(ys.min() - y_pad, 0.0)
ax.axhspan(0, y_hi, alpha=0.06, color="#4CAF50")
ax.axhspan(y_lo, 0, alpha=0.06, color="#E53935")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("Stale Semantic Accuracy (before cascade)", fontsize=11)
ax.set_ylabel("Cascade Gain over Stale Semantic", fontsize=11)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:+.1%}"))
x_pad = 0.012
ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
ax.set_ylim(y_lo, y_hi)
ax.grid(linestyle="--", alpha=0.35, linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
backbone_patches = [
    mpatches.Patch(color=BACKBONE_COLOR[b], label=b) for b in ["SigLIP2", "DINOv2"]
]
sdxl_marker = mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                             markersize=7, label="SDXL")
flux_marker  = mlines.Line2D([], [], color="grey", marker="s", linestyle="None",
                             markersize=7, label="FLUX")
trendline_h  = mlines.Line2D([], [], color="#9E9E9E", linestyle="--",
                              linewidth=1.4, label="OLS trendline")
zero_h       = mlines.Line2D([], [], color="#E53935", linestyle="-",
                              linewidth=1.0, label="Break-even (0%)")
ax.legend(
    handles=backbone_patches + [sdxl_marker, flux_marker, trendline_h, zero_h],
    loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2
)

ax.set_title(
    "Cascade Gain vs. Stale Semantic Accuracy (E8: SDXL & FLUX)",
    fontsize=12, pad=10
)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "..")
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"plot_fig_b_cascade_gain_scatter.{ext}")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
