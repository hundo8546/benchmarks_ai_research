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
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # the submitted figures' font
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTING = os.path.dirname(HERE)
RESULTS = os.path.join(ROUTING, "results")

BACKBONES = ["CLIP", "SigLIP2", "DINOv2"]
# Original figure style (pre-revision layout), numbers read live below.
COLORS = {"CLIP": "#2196F3", "SigLIP2": "#FF9800", "DINOv2": "#4CAF50"}
HATCH_STALE = "///"
DATASETS = ["BigGAN", "GenBuster", "SD14", "SDXL", "FLUX"]
LABEL = {"GenBuster": "GenBuster", "SD14": "SD1.4", "BigGAN": "BigGAN", "SDXL": "SDXL", "FLUX": "FLUX"}
STALE = {"SDXL", "FLUX"}


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
    x = np.arange(len(DATASETS))
    bar_width = 0.22
    offsets = np.array([-1, 0, 1]) * bar_width
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, bb in enumerate(BACKBONES):
        for j, ds in enumerate(DATASETS):
            rate = r.get((ds, bb))
            if rate is None:  # no stale CLIP probe on SDXL/FLUX
                continue
            stale = ds in STALE
            ax.bar(x[j] + offsets[i], rate, width=bar_width, color=COLORS[bb],
                   alpha=0.60 if stale else 0.85, hatch=HATCH_STALE if stale else "",
                   edgecolor=COLORS[bb] if stale else "white", linewidth=0.6)

    ax.axvline(x=2.5, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(1.0, 0.445, "In-domain", ha="center", va="bottom", fontsize=9, color="grey", style="italic")
    ax.text(3.5, 0.445, "Cross-family\n(stale SD1.4 probes)", ha="center", va="bottom",
            fontsize=9, color="grey", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[d] for d in DATASETS], fontsize=11)
    ax.set_ylabel("Disagreement Rate", fontsize=11)
    ax.set_ylim(0.0, 0.47)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_yticks(np.arange(0.0, 0.46, 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(facecolor=COLORS[b], label=b, alpha=0.85) for b in BACKBONES]
    handles.append(mpatches.Patch(facecolor="grey", hatch=HATCH_STALE, alpha=0.4,
                                  label="Stale probe (zero-shot)", edgecolor="grey"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("FFT\u2013Semantic Disagreement Rate Across Datasets", fontsize=12, pad=10)
    ax.annotate("FFT near-perfect\non BigGAN \u2192 near-zero\ndisagreement",
                xy=(x[0], 0.016), xytext=(x[0] + 0.5, 0.12),
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
                fontsize=7.5, color="#555", ha="left")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROUTING, f"plot_fig_c_disagreement_rate.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
