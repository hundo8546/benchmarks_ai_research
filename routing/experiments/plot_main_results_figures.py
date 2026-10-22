"""
Figures accuracy_comparison and cost_accuracy, read live from:
  routing/results/tab_main_results.csv   (in-domain accuracy and latency)
  routing/results/tab_e8_cascade.csv     (SDXL/FLUX stale-semantic vs. GPT-5.5 cascade)

Output: routing/plot_accuracy_comparison.png, routing/plot_cost_accuracy_all.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTING = os.path.dirname(HERE)
RESULTS = os.path.join(ROUTING, "results")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"

METHODS = ["FFT", "CNNSpot", "UnivFD", "CLIP", "SigLIP2", "DINOv2",
           "Entropy", "Casc.(C)", "Casc.(S)", "Casc.(D)", "Always esc."]
FAMILY = {"FFT": "Artifact detector", "CNNSpot": "Artifact detector", "UnivFD": "Artifact detector",
          "CLIP": "Semantic probe", "SigLIP2": "Semantic probe", "DINOv2": "Semantic probe",
          "Casc.(C)": "Disagreement cascade", "Casc.(S)": "Disagreement cascade",
          "Casc.(D)": "Disagreement cascade", "Entropy": "Reference", "Always esc.": "Reference"}
FAMILY_COLOR = {"Artifact detector": BLUE, "Semantic probe": ORANGE,
                "Disagreement cascade": AQUA, "Reference": YELLOW}
DATASET_TITLE = {"GenBuster": "GenBuster", "SD14": "SD 1.4", "BigGAN": "BigGAN"}
BACKBONE_COLOR = {"CLIP": BLUE, "SigLIP2": ORANGE, "DINOv2": AQUA}
DATASET_MARKER = {"GenBuster": "o", "SD14": "s", "BigGAN": "^", "SDXL": "D", "FLUX": "v"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_accuracy_comparison(main):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, ds in zip(axes, ["GenBuster", "SD14", "BigGAN"]):
        d = main[main.dataset == ds].set_index("method").loc[METHODS]
        ax.barh(range(len(METHODS)), d["accuracy"], height=0.7,
                color=[FAMILY_COLOR[FAMILY[m]] for m in METHODS], edgecolor="white", linewidth=2)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([m.replace("Casc.", "Cascade ") for m in METHODS])
        ax.invert_yaxis()
        ax.set_xlim(0.4, 1.0)
        ax.set_title(DATASET_TITLE[ds], fontsize=10, color=INK)
        ax.set_xlabel("Accuracy", fontsize=9, color=INK)
        _style(ax)
        ax.grid(axis="y", visible=False)
    handles = [mlines.Line2D([], [], color=c, marker="s", linestyle="", markersize=9, label=f)
               for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(ROUTING, "plot_accuracy_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _pair(ax, x0, y0, x1, y1, color, marker):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=5, shrinkB=5))
    ax.plot([x0], [y0], marker=marker, markersize=8, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2, linestyle="")
    ax.plot([x1], [y1], marker=marker, markersize=8, color=color, markeredgecolor="white",
            markeredgewidth=1.5, linestyle="")


def plot_cost_accuracy(main, e8):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ds in ["GenBuster", "SD14", "BigGAN"]:
        d = main[main.dataset == ds].set_index("method")
        for bb in ["CLIP", "SigLIP2", "DINOv2"]:
            sem, casc = d.loc[bb], d.loc[f"Casc.({bb[0]})"]
            _pair(left, sem.latency_ms, sem.accuracy, casc.latency_ms, casc.accuracy,
                  BACKBONE_COLOR[bb], DATASET_MARKER[ds])
    left.set_xlabel("Mean latency per input (ms)", fontsize=9, color=INK)
    left.set_ylabel("Accuracy", fontsize=9, color=INK)
    left.set_title("In-domain: semantic probe → cascade (Qwen2.5-VL-7B)", fontsize=10, color=INK)
    _style(left)

    for _, r in e8.iterrows():
        _pair(right, 0.0, r.stale_sem_acc, r.esc_rate, r.cascade_acc_gpt55,
              BACKBONE_COLOR[r.semantic], DATASET_MARKER[r.dataset])
    right.set_xlabel("Share of inputs escalated to GPT-5.5", fontsize=9, color=INK)
    right.set_title("Deployment shift: stale probe → cascade (GPT-5.5)", fontsize=10, color=INK)
    right.set_xlim(-0.03, 0.5)
    _style(right)

    hollow = mlines.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=MUTED,
                           markeredgewidth=2, linestyle="", markersize=8, label="Hollow: probe alone")
    filled = mlines.Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=8, label="Filled: cascade")
    colors = [mlines.Line2D([], [], color=c, lw=3, label=b) for b, c in BACKBONE_COLOR.items()]
    markers = [mlines.Line2D([], [], marker=m, color=MUTED, linestyle="", markersize=7,
                             label=DATASET_TITLE.get(ds, ds)) for ds, m in DATASET_MARKER.items()]
    fig.legend(handles=[hollow, filled] + colors + markers, loc="lower center", ncol=10,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(ROUTING, "plot_cost_accuracy_all.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    main_df = pd.read_csv(os.path.join(RESULTS, "tab_main_results.csv"))
    e8 = pd.read_csv(os.path.join(RESULTS, "tab_e8_cascade.csv"))
    print(plot_accuracy_comparison(main_df))
    print(plot_cost_accuracy(main_df, e8))


if __name__ == "__main__":
    main()
