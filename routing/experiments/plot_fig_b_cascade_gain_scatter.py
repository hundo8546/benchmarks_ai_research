"""
Figure B: Cascade Gain vs. Stale Semantic Accuracy (E8)
-------------------------------------------------------
Scatter plot: x = stale semantic accuracy, y = cascade gain over stale semantic.
6 points from E8 (SDXL and FLUX × CLIP / SigLIP2 / DINOv2).
Includes OLS trendline and a horizontal zero-gain reference line.
FLUX+CLIP is annotated as the boundary condition.

Data source: routing/results/e8_modern_diffusion.csv

Output: routing/plot_fig_b_cascade_gain_scatter.png
        routing/plot_fig_b_cascade_gain_scatter.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Data from e8_modern_diffusion.csv
#   sem_stale_acc, cascade_gain_over_stale_sem
# ---------------------------------------------------------------------------
POINTS = [
    # (dataset, backbone, stale_sem_acc, cascade_gain)
    ("SDXL", "CLIP",    0.87225,  0.0205),
    ("SDXL", "SigLIP2", 0.8465,   0.034),
    ("SDXL", "DINOv2",  0.85475,  0.036),
    ("FLUX", "CLIP",    0.91975, -0.006),
    ("FLUX", "SigLIP2", 0.90925,  0.00375),
    ("FLUX", "DINOv2",  0.815,    0.04875),
]

DATASET_MARKER = {"SDXL": "o", "FLUX": "s"}
BACKBONE_COLOR = {
    "CLIP":    "#2196F3",
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

# ── Shaded gain region ────────────────────────────────────────────────────────
ax.axhspan(0, 0.06, alpha=0.06, color="#4CAF50")
ax.axhspan(-0.015, 0, alpha=0.06, color="#E53935")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("Stale Semantic Accuracy (before cascade)", fontsize=11)
ax.set_ylabel("Cascade Gain over Stale Semantic", fontsize=11)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:+.1%}"))
ax.set_xlim(0.795, 0.938)
ax.set_ylim(-0.016, 0.058)
ax.grid(linestyle="--", alpha=0.35, linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
backbone_patches = [
    mpatches.Patch(color=BACKBONE_COLOR[b], label=b) for b in ["CLIP", "SigLIP2", "DINOv2"]
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
