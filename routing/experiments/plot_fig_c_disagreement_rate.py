"""
Figure C: artifact-semantic disagreement rate per dataset and semantic backbone.

Read live from:
  routing/results/tab_main_results.csv  in-domain escalation_rate of Casc.(C/S/D)
  routing/results/tab_e8_cascade.csv    SDXL/FLUX esc_rate with the stale SD 1.4 probes
                                        (no CLIP: SD 1.4 lacks raw CLIP features)

Output: routing/plot_fig_c_disagreement_rate.png / .pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTING = os.path.dirname(HERE)
RESULTS = os.path.join(ROUTING, "results")

BACKBONES = ["CLIP", "SigLIP2", "DINOv2"]
COLOR = {"CLIP": "#2a78d6", "SigLIP2": "#eb6834", "DINOv2": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
DATASETS = ["GenBuster", "SD14", "BigGAN", "SDXL", "FLUX"]
LABEL = {"GenBuster": "GenBuster", "SD14": "SD 1.4", "BigGAN": "BigGAN", "SDXL": "SDXL", "FLUX": "FLUX"}


def rates():
    main = pd.read_csv(os.path.join(RESULTS, "tab_main_results.csv"))
    e8 = pd.read_csv(os.path.join(RESULTS, "tab_e8_cascade.csv"))
    out = {}
    for ds in ["GenBuster", "SD14", "BigGAN"]:
        for bb in BACKBONES:
            out[(ds, bb)] = float(main[(main.dataset == ds) & (main.method == f"Casc.({bb[0]})")].escalation_rate.iloc[0])
    for r in e8.itertuples():
        out[(r.dataset, r.semantic)] = float(r.esc_rate)
    return out


def main():
    r = rates()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.26
    x = np.arange(len(DATASETS))
    for j, bb in enumerate(BACKBONES):
        vals = [r.get((ds, bb), np.nan) for ds in DATASETS]
        ax.bar(x + (j - 1) * width, vals, width, color=COLOR[bb], edgecolor="white", linewidth=1.5, label=bb)
    ax.axvspan(2.5, 4.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(3.5, 0.47, "Deployment shift\n(stale SD 1.4 probes)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[d] for d in DATASETS])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Disagreement rate", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=COLOR[b], label=b) for b in BACKBONES]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", title="Semantic probe (with FFT)",
              title_fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROUTING, f"plot_fig_c_disagreement_rate.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
