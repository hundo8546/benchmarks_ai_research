"""
Figure A: Semantic Detector Degradation Across Datasets
--------------------------------------------------------
Bar chart showing CLIP / SigLIP2 / DINOv2 accuracy on five dataset families.
In-domain bars (GenBuster, SD14, BigGAN) use full training data (train_fraction=0.5).
Cross-family bars (SDXL, FLUX) use stale SD1.4-trained probes applied zero-shot.

All numbers are read LIVE from CSVs at plot time (no hand-copied constants) so this
figure can never again silently drift from the pipeline's current source of truth.

Data sources:
  In-domain (GenBuster/SD14/BigGAN, train_fraction=0.5)
    — routing/results/tab_main_results.csv (in-domain, canonical evaluation sets)
  Cross-family SigLIP2/DINOv2 (SDXL/FLUX), reconciled "stale" stage
    — routing/pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv
      (see RECONCILIATION_REPORT.md in that directory: the SD1.4-trained probes were
      retrained between the original E8 run and this reconciled snapshot, so the old
      e8_modern_diffusion.csv sem_stale_acc values for SigLIP2/DINOv2 were STALE —
      0.8465/0.85475 (SDXL) and 0.90925/0.815 (FLUX) vs. the correct 0.86375/0.851 and
      0.90625/0.81775. Fixed here.)
  Cross-family CLIP (SDXL/FLUX) — EXCLUDED. The paper's own Methods text already
      excludes CLIP from the SDXL/FLUX cascade analysis (the raw SD1.4 CLIP features
      needed to reconstruct the stale zero-shot probe are unavailable), so showing a
      CLIP bar here would be inconsistent with that stated scope and, per
      RECONCILIATION_REPORT.md, cannot currently be verified against the corrected
      probe snapshot anyway. CLIP is still shown for the three in-domain datasets,
      where it is unaffected by this issue.

Output: routing/plot_fig_a_semantic_degradation.png
        routing/plot_fig_a_semantic_degradation.pdf
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results")
CONFUSION_DIR = os.path.join(HERE, "..", "pipeline", "confusion_matrices")

E1_PATH = os.path.join(RESULTS_DIR, "tab_main_results.csv")
E8_MODERN_PATH = os.path.join(RESULTS_DIR, "e8_modern_diffusion.csv")
CONFUSION_PATH = os.path.join(CONFUSION_DIR, "sdxl_flux_confusion_matrices.csv")


def load_indomain(path):
    """In-domain semantic-probe accuracy, read live from tab_main_results.csv (the canonical
    evaluation sets of Table main_results)."""
    display_to_csv_name = {"GenBuster": "GenBuster", "SD1.4": "SD14", "BigGAN": "BigGAN"}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    indomain = {display: {} for display in display_to_csv_name}
    for display, csv_name in display_to_csv_name.items():
        for row in rows:
            if row["dataset"] == csv_name and row["method"] in ("CLIP", "SigLIP2", "DINOv2"):
                indomain[display][row["method"]] = float(row["accuracy"])
    return indomain


def load_crossfamily_reconciled(path):
    """SigLIP2/DINOv2 cross-family accuracy, read live from the reconciled
    confusion-matrix CSV (stale stage = zero-shot SD1.4-trained probe on SDXL/FLUX)."""
    data = {"SDXL": {}, "FLUX": {}}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["stage"] == "stale" and row["backbone"] in ("SigLIP2", "DINOv2"):
                data[row["dataset"]][row["backbone"]] = float(row["overall_accuracy"])
    return data


# CLIP is deliberately NOT read for the cross-family (SDXL/FLUX) panel -- see the
# module docstring. The paper's own scope already excludes it there, and its stale
# cross-family number cannot currently be verified against the corrected probe
# snapshot the way SigLIP2/DINOv2 were. CLIP remains available for the in-domain
# panel (INDOMAIN), where this issue does not apply.

INDOMAIN = load_indomain(E1_PATH)

_recon = load_crossfamily_reconciled(CONFUSION_PATH)
CROSSFAMILY = {
    "SDXL": {**_recon["SDXL"]},
    "FLUX": {**_recon["FLUX"]},
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

legend_drawn = set()
for i, backbone in enumerate(BACKBONES):
    for ds_x, ds in zip(x, DATASETS):
        if ds in INDOMAIN:
            acc, is_stale = INDOMAIN[ds][backbone], False
        elif backbone in CROSSFAMILY[ds]:
            acc, is_stale = CROSSFAMILY[ds][backbone], True
        else:
            continue  # CLIP excluded from the cross-family (SDXL/FLUX) panel -- leave a gap
        h = HATCH_STALE if is_stale else ""
        ax.bar(
            ds_x + offsets[i], acc,
            width=bar_width,
            color=COLORS[backbone],
            alpha=0.85 if not is_stale else 0.60,
            hatch=h,
            edgecolor="white" if not is_stale else COLORS[backbone],
            linewidth=0.6,
            label=backbone if backbone not in legend_drawn else "_nolegend_",
        )
        legend_drawn.add(backbone)

# Divider between in-domain and cross-family
ax.axvline(x=2.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(1.0, 0.64, "In-domain", ha="center", va="bottom",
        fontsize=9, color="grey", style="italic")
ax.text(3.5, 0.64, "Cross-family\n(stale SD1.4 probes, CLIP excluded)", ha="center", va="bottom",
        fontsize=8.5, color="grey", style="italic")

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
          loc="upper right", fontsize=9, framealpha=0.9)

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
