"""
Figure C: Disagreement Rate Across Datasets
-------------------------------------------
Grouped bar chart showing FFT+Semantic disagreement rate per dataset family.
In-domain (GenBuster, SD14, BigGAN) uses FFT+CLIP / FFT+SigLIP2 / FFT+DINOv2
  disagreement rates from e3_disagreement_matrix.csv.
Cross-family (SDXL, FLUX) uses disagree_rate from e8_modern_diffusion.csv.

The chart shows disagreement rising under cross-family generator shift,
confirming that both detectors are stale on SDXL / FLUX.

Output: routing/plot_fig_c_disagreement_rate.png
        routing/plot_fig_c_disagreement_rate.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# e3_disagreement_matrix.csv (FFT rows, in-domain)
# disagreement_rate = 1 - agreement_rate
INDOMAIN = {
    #           CLIP         SigLIP2      DINOv2
    "GenBuster": [1-0.7167, 1-0.7283, 1-0.7083],
    "SD1.4":     [1-0.6958, 1-0.7333, 1-0.7125],
    "BigGAN":    [1-0.9875, 1-0.9979, 1-0.9875],
}

# e8_modern_diffusion.csv (disagree_rate column, stale probes)
CROSSFAMILY = {
    "SDXL": [0.352,  0.366,  0.386 ],
    "FLUX": [0.403,  0.414,  0.414 ],
}

DATASETS   = ["BigGAN", "GenBuster", "SD1.4", "SDXL", "FLUX"]
BACKBONES  = ["CLIP", "SigLIP2", "DINOv2"]
COLORS     = {"CLIP": "#2196F3", "SigLIP2": "#FF9800", "DINOv2": "#4CAF50"}
HATCH_STALE = "///"

n_datasets  = len(DATASETS)
n_backbones = len(BACKBONES)
x           = np.arange(n_datasets)
bar_width   = 0.22
offsets     = np.array([-1, 0, 1]) * bar_width

fig, ax = plt.subplots(figsize=(9, 4.5))

# Build data lookup
all_data = {}
all_data.update(INDOMAIN)
all_data.update(CROSSFAMILY)
stale_datasets = set(CROSSFAMILY.keys())

for i, backbone in enumerate(BACKBONES):
    bi = i  # backbone index 0,1,2
    for j, ds in enumerate(DATASETS):
        rate = all_data[ds][bi]
        is_stale = ds in stale_datasets
        h = HATCH_STALE if is_stale else ""
        ax.bar(
            x[j] + offsets[i], rate,
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
ax.text(1.0, 0.445, "In-domain", ha="center", va="bottom",
        fontsize=9, color="grey", style="italic")
ax.text(3.5, 0.445, "Cross-family\n(stale SD1.4 probes)", ha="center", va="bottom",
        fontsize=9, color="grey", style="italic")

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(DATASETS, fontsize=11)
ax.set_ylabel("Disagreement Rate", fontsize=11)
ax.set_ylim(0.0, 0.47)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:.0%}"))
ax.set_yticks(np.arange(0.0, 0.46, 0.05))
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
          loc="upper left", fontsize=9, framealpha=0.9)

ax.set_title(
    "FFT–Semantic Disagreement Rate Across Datasets",
    fontsize=12, pad=10
)

# Annotation: "Near-zero — FFT near-perfect on BigGAN"
ax.annotate(
    "FFT near-perfect\non BigGAN → near-zero\ndisagreement",
    xy=(x[0], 0.016), xytext=(x[0] + 0.5, 0.12),
    arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
    fontsize=7.5, color="#555", ha="left"
)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "..")
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"plot_fig_c_disagreement_rate.{ext}")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
