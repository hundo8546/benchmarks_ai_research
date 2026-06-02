"""
Figure A: Semantic Detector Degradation Across Datasets
--------------------------------------------------------
Bar chart showing CLIP / SigLIP2 / DINOv2 accuracy on five dataset families.
In-domain bars (GenBuster, SD14, BigGAN) use full training data (train_fraction=0.5).
Cross-family bars (SDXL, FLUX) use stale SD14-trained probes.

Data sources:
  In-domain  — routing/results/e1_semantic_results.csv
  Cross-family — routing/results/e8_modern_diffusion.csv

Output: routing/plot_fig_a_semantic_degradation.png
        routing/plot_fig_a_semantic_degradation.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data — in-domain accuracy at train_fraction=0.5 (from e1_semantic_results)
# ---------------------------------------------------------------------------
INDOMAIN = {
    # dataset: {backbone: accuracy}
    "GenBuster": {"CLIP": 0.9383, "SigLIP2": 0.9783, "DINOv2": 0.9550},
    "SD1.4":     {"CLIP": 0.9022, "SigLIP2": 0.9108, "DINOv2": 0.8589},
    "BigGAN":    {"CLIP": 0.9856, "SigLIP2": 0.9903, "DINOv2": 0.9811},
}

# Cross-family accuracy (stale SD14-trained probes on SDXL / FLUX)
# sem_stale_acc column from e8_modern_diffusion.csv
CROSSFAMILY = {
    "SDXL": {"CLIP": 0.87225, "SigLIP2": 0.8465,  "DINOv2": 0.85475},
    "FLUX": {"CLIP": 0.91975, "SigLIP2": 0.90925, "DINOv2": 0.815},
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
DATASETS   = ["GenBuster", "SD1.4", "BigGAN", "SDXL", "FLUX"]
BACKBONES  = ["CLIP", "SigLIP2", "DINOv2"]
COLORS     = {"CLIP": "#2196F3", "SigLIP2": "#FF9800", "DINOv2": "#4CAF50"}
HATCH_STALE = "///"          # diagonal hatching for cross-family bars

n_datasets  = len(DATASETS)
n_backbones = len(BACKBONES)
x           = np.arange(n_datasets)
bar_width   = 0.22
offsets     = np.array([-1, 0, 1]) * bar_width

fig, ax = plt.subplots(figsize=(9, 4.5))

for i, backbone in enumerate(BACKBONES):
    acc_vals = []
    hatch_vals = []
    for ds in DATASETS:
        if ds in INDOMAIN:
            acc_vals.append(INDOMAIN[ds][backbone])
            hatch_vals.append(False)
        else:
            acc_vals.append(CROSSFAMILY[ds][backbone])
            hatch_vals.append(True)

    for j, (ds_x, acc, is_stale) in enumerate(zip(x, acc_vals, hatch_vals)):
        h = HATCH_STALE if is_stale else ""
        ax.bar(
            ds_x + offsets[i], acc,
            width=bar_width,
            color=COLORS[backbone],
            alpha=0.85 if not is_stale else 0.60,
            hatch=h,
            edgecolor="white" if not is_stale else COLORS[backbone],
            linewidth=0.6,
            label=backbone if j == 0 else "_nolegend_",
        )

# Divider between in-domain and cross-family
ax.axvline(x=2.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(1.0, 0.64, "In-domain", ha="center", va="bottom",
        fontsize=9, color="grey", style="italic")
ax.text(3.5, 0.64, "Cross-family\n(stale SD1.4 probes)", ha="center", va="bottom",
        fontsize=9, color="grey", style="italic")

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, fontsize=11)
ax.set_ylabel("Detection Accuracy", fontsize=11)
ax.set_ylim(0.62, 1.02)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:.0%}"))
ax.set_yticks(np.arange(0.65, 1.01, 0.05))
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
backbone_patches = [
    mpatches.Patch(facecolor=COLORS[b], label=b, alpha=0.85) for b in BACKBONES
]
stale_patch = mpatches.Patch(
    facecolor="grey", hatch=HATCH_STALE, alpha=0.4,
    label="Stale probe (zero-shot)", edgecolor="grey"
)
ax.legend(handles=backbone_patches + [stale_patch],
          loc="lower right", fontsize=9, framealpha=0.9)

ax.set_title(
    "Semantic Detector Accuracy: In-Domain vs. Cross-Family Deployment",
    fontsize=12, pad=10
)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "..")
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"plot_fig_a_semantic_degradation.{ext}")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
