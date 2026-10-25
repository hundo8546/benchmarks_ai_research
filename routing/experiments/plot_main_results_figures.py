"""
Figures accuracy_comparison and cost_accuracy, read live from:
  routing/results/tab_main_results.csv   (in-domain accuracy and latency)
  routing/results/tab_e8_cascade.csv     (SDXL/FLUX stale-semantic vs. GPT-5.5 cascade)

Output: routing/plot_accuracy_comparison.png, routing/plot_cost_accuracy_all.png
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"  # the submitted figures' font
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTING = os.path.dirname(HERE)
RESULTS = os.path.join(ROUTING, "results")

# Original (submitted-version) figure styles; numbers are read live from the CSVs.
GREY, NAVY, BLUE, ORANGE, GREEN, RED, PURPLE = "#aaaaaa", "#2d3e50", "#2f77c6", "#f0932e", "#3fa75e", "#d9372e", "#8e44ad"
pct = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
CHEAP_LATENCY_MS = 2.5 + 10.4  # FFT + one semantic probe


def _acc(main, ds, method):
    return float(main[(main.dataset == ds) & (main.method == method)].accuracy.iloc[0])


def _lat(main, ds, method):
    return float(main[(main.dataset == ds) & (main.method == method)].latency_ms.iloc[0])


def plot_accuracy_comparison(main):
    bars = [("CNNSpot", "CNNSpot", GREY), ("FFT", "FFT", NAVY), ("CLIP", "CLIP", BLUE),
            ("SigLIP2", "SigLIP2", ORANGE), ("DINOv2", "DINOv2", GREEN),
            ("Casc.(S)", "Cascade (FFT+SigLIP2+Qwen)", RED)]
    datasets = [("GenBuster", "GenBuster"), ("SD14", "SD 1.4"), ("BigGAN", "BigGAN")]
    width = 0.13
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k, (method, label, color) in enumerate(bars):
        xs = [i + (k - (len(bars) - 1) / 2) * width for i in range(len(datasets))]
        ax.bar(xs, [_acc(main, ds, method) for ds, _ in datasets], width, color=color,
               edgecolor="white", linewidth=0.5, label=label)
    ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([t for _, t in datasets], fontsize=12)
    ax.set_ylim(0.4, 1.03)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Detector accuracy by dataset (in-domain)", fontsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(ROUTING, "plot_accuracy_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _gpt55_latency_ms(dataset, semantic):
    """Mean GPT-5.5 wall-clock per call on that configuration's disagreement subset."""
    preds = pd.read_csv(os.path.join(ROUTING, f"{dataset.lower()}_gpt55_{semantic.lower()}_disagree_preds.csv"))
    return float(preds.latency_ms.mean())


def plot_cost_accuracy(main, e8):
    sem_color = {"CLIP": ORANGE, "SigLIP2": GREEN, "DINOv2": PURPLE}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2))

    ds = "GenBuster"
    left.scatter(_lat(main, ds, "FFT"), _acc(main, ds, "FFT"), marker="s", s=180, color=NAVY, zorder=3)
    left.scatter(_lat(main, ds, "CNNSpot"), _acc(main, ds, "CNNSpot"), marker="s", s=180, color=GREY, zorder=3)
    for sem, color in sem_color.items():
        left.scatter(_lat(main, ds, sem), _acc(main, ds, sem), marker="o", s=180, color=color, zorder=3)
        casc = f"Casc.({sem[0]})"
        left.scatter(_lat(main, ds, casc), _acc(main, ds, casc), marker="^", s=180, color=color, zorder=3)
    left.scatter(_lat(main, ds, "Always esc."), _acc(main, ds, "Always esc."), marker="X", s=180, color=RED, zorder=3)
    left.set_ylim(0.5, 1.02)
    left.set_xlim(-8, 222)
    left.set_ylabel("Accuracy", fontsize=11)
    left.set_title("In-domain (GenBuster)", fontsize=12)

    for r in e8[e8.dataset == "SDXL"].itertuples():
        color = sem_color[r.semantic]
        right.scatter(10.4, r.stale_sem_acc, marker="o", s=180, color=color, zorder=3)
        lat = CHEAP_LATENCY_MS + r.esc_rate * _gpt55_latency_ms("SDXL", r.semantic)
        right.scatter(lat, r.cascade_acc_gpt55, marker="^", s=180, color=color, zorder=3)
    right.set_xlim(-50, 1400)
    right.set_title("Deployment shift (SDXL)", fontsize=12)

    for ax in (left, right):
        ax.yaxis.set_major_formatter(pct)
        ax.set_xlabel("Mean latency per sample (ms)", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def m(marker, color, label, size=9):
        return mlines.Line2D([], [], marker=marker, color=color, linestyle="", markersize=size, label=label)
    handles = [m("s", "grey", "Artifact detector", 7), m("o", "grey", "Semantic-only", 7),
               m("^", "grey", "Cascade (disagree routing)", 7),
               m("X", RED, "Always-escalate"), m("s", ORANGE, "CLIP"), m("s", GREEN, "SigLIP2"),
               m("s", PURPLE, "DINOv2"), m("s", NAVY, "FFT"), m("s", GREY, "CNNSpot")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
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
