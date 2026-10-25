"""
Figure E: The BigGAN Resolution Shortcut
------------------------------------------
This revision's headline new finding: a classifier trained on nothing but image
width/height (no pixel content at all) matches the real FFT detector's accuracy on
BigGAN -- and forcibly resolution-matching real/fake images before running the real
FFT detector collapses its accuracy to chance. This shows the "real" detector was
substantially exploiting a resolution shortcut, not genuine forensic signal.

Three bars per dataset:
  1. Real FFT detector (original)            -- normal train/test, no intervention
  2. Resolution-only control (no pixel content) -- trained on width/height alone
  3. Real FFT detector (resolution-matched)   -- real & fake resized to a common
                                                  resolution before train/test

BigGAN is the headline case (bars 1 & 2 both near-ceiling; bar 3 collapses to
chance). SD1.4 is a partial-confound comparison (bar 3 drops but not to chance).
GenBuster is the clean negative control: real/fake images are already the same
native resolution (1024x1024 for both), so there is no resolution shortcut to
exploit -- the resolution-only control sits at chance (bar 2) while the real
detector's genuine, non-trivial accuracy (bar 1) shows real pixel-level signal.
Because GenBuster already has no resolution confound, there is no separate
"resolution-matched" ablation to run for it (bar 3 is not applicable).

Data is read LIVE (no hardcoding) from:
  routing/results/resolution_only_control.csv    (bar 2, all 3 datasets)
  routing/results/resolution_matched_ablation.csv (bars 1 & 3, BigGAN + SD1.4 only)
  routing/results/e0_fft_table.csv                (bar 1 for GenBuster -- the
      resolution_matched_ablation.csv ablation was never run for GenBuster since it
      has no resolution confound to match; its original FFT-only accuracy is instead
      read from the FFT-only row of e0_fft_table.csv, the same live source used by
      the ablation study's own "original" condition for the other two datasets --
      confirmed consistent: BigGAN/SD14 FFT-only accuracy there equals the
      resolution_matched_ablation.csv "original" condition bit-for-bit.)

Output: routing/plot_fig_e_resolution_shortcut.png
        routing/plot_fig_e_resolution_shortcut.pdf
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

RESOLUTION_ONLY_PATH = os.path.join(RESULTS_DIR, "resolution_only_control.csv")
RESOLUTION_MATCHED_PATH = os.path.join(RESULTS_DIR, "resolution_matched_ablation.csv")
E0_FFT_TABLE_PATH = os.path.join(RESULTS_DIR, "e0_fft_table.csv")

# ---------------------------------------------------------------------------
# Load data live
# ---------------------------------------------------------------------------
resolution_only = {}
with open(RESOLUTION_ONLY_PATH, newline="") as f:
    for row in csv.DictReader(f):
        resolution_only[row["dataset"]] = float(row["accuracy"])

original_acc = {}
matched_acc = {}
with open(RESOLUTION_MATCHED_PATH, newline="") as f:
    for row in csv.DictReader(f):
        if row["condition"] == "original":
            original_acc[row["dataset"]] = float(row["fft_accuracy"])
        elif row["condition"] == "resolution-matched":
            matched_acc[row["dataset"]] = float(row["fft_accuracy"])

# GenBuster has no resolution-matched ablation row (no confound to match); pull its
# "original" FFT-only accuracy from e0_fft_table.csv instead (live read, not hardcoded).
with open(E0_FFT_TABLE_PATH, newline="") as f:
    for row in csv.DictReader(f):
        if row["dataset"] == "GenBuster" and row["method"] == "FFT-only":
            original_acc.setdefault("GenBuster", float(row["accuracy"]))
            break

# ---------------------------------------------------------------------------
# Layout: BigGAN first (headline), SD14 (partial confound), GenBuster (clean control)
# ---------------------------------------------------------------------------
DATASET_ORDER = ["BigGAN", "SD14", "GenBuster"]
DATASET_LABEL = {"BigGAN": "BigGAN", "SD14": "SD1.4", "GenBuster": "GenBuster"}
DATASET_SUBTITLE = {
    "BigGAN": "headline case",
    "SD14": "partial confound",
    "GenBuster": "clean negative control",
}

BAR_LABELS = [
    "Real FFT detector (original)",
    "Resolution-only control\n(no pixel content)",
    "Real FFT detector\n(resolution-matched)",
]
# Validated categorical palette (dataviz skill): slots 1, 3, 8 -- blue, aqua, red.
# Red (slot 8) is used deliberately for the "resolution-matched" bar: it is the one
# that is expected to collapse, and this reserves red for that collapse story without
# overloading it as a semantic "bad" status color elsewhere in the figure.
BAR_COLORS = ["#2a78d6", "#1baf7a", "#e34948"]

n_groups = len(DATASET_ORDER)
n_bars = len(BAR_LABELS)
bar_width = 0.24
group_gap = 0.04
x = np.arange(n_groups)
offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * (bar_width + group_gap / n_bars)

fig, ax = plt.subplots(figsize=(10.5, 5.2))

for i, (label, color) in enumerate(zip(BAR_LABELS, BAR_COLORS)):
    for gi, dataset in enumerate(DATASET_ORDER):
        if i == 0:
            val = original_acc.get(dataset)
        elif i == 1:
            val = resolution_only.get(dataset)
        else:
            val = matched_acc.get(dataset)

        bx = x[gi] + offsets[i]
        if val is None:
            # Not applicable (GenBuster resolution-matched: no confound to match)
            ax.text(bx, 0.52, "N/A", ha="center", va="bottom", fontsize=8.5,
                     color="#8a8a86", style="italic", rotation=90)
            continue

        ax.bar(bx, val, width=bar_width, color=color, alpha=0.92,
               edgecolor="white", linewidth=0.6, zorder=3,
               label=label if gi == 0 else "_nolegend_")
        ax.text(bx, val + 0.015, f"{val:.1%}", ha="center", va="bottom",
                 fontsize=8.3, color="#3a3a38")

# Chance-rate reference line
ax.axhline(0.5, color="#6b6a66", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)
ax.text(x[-1] + 0.62, 0.5, "chance (50%)", ha="left", va="center",
         fontsize=8.5, color="#6b6a66", style="italic")

# Group labels (dataset + role subtitle)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{DATASET_LABEL[d]}\n({DATASET_SUBTITLE[d]})" for d in DATASET_ORDER],
    fontsize=10.5,
)

ax.set_ylabel("Detection accuracy", fontsize=11)
ax.set_ylim(0.40, 1.08)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_yticks(np.arange(0.4, 1.01, 0.1))
ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(-0.6, n_groups - 0.4 + 0.9)

handles = [mpatches.Patch(facecolor=c, label=l, alpha=0.92)
           for c, l in zip(BAR_COLORS, BAR_LABELS)]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
          fontsize=8.8, framealpha=0.95, frameon=False)

ax.set_title(
    "A Width/Height-Only Classifier Matches BigGAN's Real Detector\n"
    "Resolution-Matching Collapses It to Chance",
    fontsize=12.5, pad=12,
)

plt.tight_layout()

out_dir = os.path.join(HERE, "..")
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"plot_fig_e_resolution_shortcut.{ext}")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
