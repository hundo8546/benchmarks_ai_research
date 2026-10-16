"""
Figure D: Cost-Accuracy Tradeoff — Cheap-Only vs. Cascade vs. Uniform GPT-5.5
------------------------------------------------------------------------------
Reviewer 1, point #10, asked for a direct three-way comparison of accuracy vs.
inference cost: (1) the cheap semantic probe alone, (2) the cascade (cheap probe +
GPT-5.5 escalation on disagreement), and (3) always calling GPT-5.5 (uniform
verifier, no routing). One panel per dataset (SDXL, FLUX); within each panel, one
line per backbone (SigLIP2, DINOv2) connecting its three cost/accuracy points in
increasing-cost order, so the reader sees the full tradeoff curve the cascade sits
on between the two extremes.

Data is read LIVE (no hardcoding) from:
  routing/results/tab_cost_accuracy_three_way.csv

Output: routing/plot_fig_d_cost_accuracy_tradeoff.png
        routing/plot_fig_d_cost_accuracy_tradeoff.pdf
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "..", "results", "tab_cost_accuracy_three_way.csv")

# ---------------------------------------------------------------------------
# Load data live
# ---------------------------------------------------------------------------
rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        rows.append({
            "dataset": row["dataset"].upper(),
            "backbone": row["backbone"],
            "cheap_acc": float(row["cheap_only_acc"]),
            "cheap_cost": float(row["cheap_only_cost_usd"]),
            "cascade_acc": float(row["cascade_acc"]),
            "cascade_cost": float(row["cascade_cost_usd"]),
            "uniform_acc": float(row["uniform_gpt55_acc"]),
            "uniform_cost": float(row["uniform_gpt55_cost_usd"]),
        })

_dataset_order = {"SDXL": 0, "FLUX": 1}
DATASETS = sorted({r["dataset"] for r in rows}, key=lambda d: _dataset_order.get(d, 99))
BACKBONES = sorted({r["backbone"] for r in rows})         # DINOv2, SigLIP2

# ---------------------------------------------------------------------------
# Style — validated categorical palette (dataviz skill: slots 1 & 2, blue/orange)
# ---------------------------------------------------------------------------
BACKBONE_COLOR = {"SigLIP2": "#2a78d6", "DINOv2": "#eb6834"}
STAGE_MARKER = {"cheap": "o", "cascade": "s", "uniform": "^"}
STAGE_LABEL = {
    "cheap": "Cheap-only (probe alone)",
    "cascade": "Cascade (routed escalation)",
    "uniform": "Uniform GPT-5.5 (always escalate)",
}
STAGE_ORDER = ["cheap", "cascade", "uniform"]

fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.6), sharey=True)
if len(DATASETS) == 1:
    axes = [axes]

for ax, dataset in zip(axes, DATASETS):
    for r in [r for r in rows if r["dataset"] == dataset]:
        backbone = r["backbone"]
        color = BACKBONE_COLOR[backbone]
        xs = [r["cheap_cost"], r["cascade_cost"], r["uniform_cost"]]
        ys = [r["cheap_acc"], r["cascade_acc"], r["uniform_acc"]]

        # Tradeoff line connecting the three stages
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.85, zorder=2)

        for stage, x, y in zip(STAGE_ORDER, xs, ys):
            ax.scatter(
                x, y, s=120, marker=STAGE_MARKER[stage],
                facecolor=color, edgecolor="white", linewidths=1.0, zorder=3,
            )

        # Direct label the cascade point (the paper's recommended operating point).
        # Offset up for SigLIP2, down for DINOv2 so close-together labels don't collide.
        label_dy = 10 if backbone == "SigLIP2" else -22
        ax.annotate(
            f"{backbone}: ${r['cascade_cost']:.2f}, {r['cascade_acc']:.1%}",
            xy=(r["cascade_cost"], r["cascade_acc"]),
            xytext=(8, label_dy), textcoords="offset points",
            fontsize=8, color="#3a3a38",
        )

    ax.set_title(dataset, fontsize=12, pad=8)
    ax.set_xlabel("Inference cost (USD, full eval set)", fontsize=10)
    ax.grid(linestyle="--", alpha=0.35, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=-1.0)

axes[0].set_ylabel("Detection accuracy", fontsize=11)
for ax in axes:
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
ymin = min(min(r["cheap_acc"], r["cascade_acc"], r["uniform_acc"]) for r in rows)
ymax = max(max(r["cheap_acc"], r["cascade_acc"], r["uniform_acc"]) for r in rows)
axes[0].set_ylim(ymin - 0.05, min(ymax + 0.03, 1.01))

# ---------------------------------------------------------------------------
# Shared legend: backbone (color) + stage (marker shape)
# ---------------------------------------------------------------------------
backbone_handles = [
    mlines.Line2D([], [], color=BACKBONE_COLOR[b], linewidth=2.5, label=b)
    for b in BACKBONES
]
stage_handles = [
    mlines.Line2D([], [], color="#6b6a66", marker=STAGE_MARKER[s], linestyle="None",
                  markersize=8, label=STAGE_LABEL[s])
    for s in STAGE_ORDER
]
fig.legend(
    handles=backbone_handles + stage_handles,
    loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=8.8,
    frameon=False,
)

fig.suptitle(
    "Cost vs. Accuracy: Cheap-Only, Cascade, and Uniform GPT-5.5 Verification",
    fontsize=13, y=1.02,
)

plt.tight_layout()

out_dir = os.path.join(HERE, "..")
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"plot_fig_d_cost_accuracy_tradeoff.{ext}")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
