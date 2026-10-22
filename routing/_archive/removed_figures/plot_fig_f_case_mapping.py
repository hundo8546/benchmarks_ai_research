"""
Figure F: Reviewer-2 Case Mapping — Case 1/2/3/Breakdown Across All Configurations
-------------------------------------------------------------------------------------
Reviewer 2, point 4, asked for a unified summary mapping the paper's Case 1/2/3/
Breakdown framework (Sec. III.D) onto every empirical configuration tested. That
currently exists only as a 36-row table (routing/results/tab_case_mapping.csv). This
figure is a compact pictogram grid: one row per experiment group, one colored chip
per individual configuration within that group (verifier / backbone / train-fraction
variant), colored by its Case. The reader sees group-level patterns in a few seconds
(e.g. "cross-family transfer is uniformly Case 2", "the SD1.4 probe-degradation sweep
flips from Case 3 to Case 2 as training data grows") that would take much longer to
extract from the raw table.

Case semantics (Sec. III.D):
  Case 1      — cheap path near-ceiling; verifier rarely invoked (benign)
  Case 2      — cascade does not clearly beat the best cheap detector (neutral/flat)
  Case 3      — cascade significantly outperforms the best cheap detector (the win)
  Breakdown   — correlated cheap-detector errors never enter the disagreement set,
                so no verifier can fix them regardless of quality (blind spot)

Grouping and row labels are derived programmatically from the `configuration` column
text (regex), not hand-typed per row, so this figure stays in sync if the table
grows. All data is read LIVE from:
  routing/results/tab_case_mapping.csv

Output: routing/plot_fig_f_case_mapping.png
        routing/plot_fig_f_case_mapping.pdf
"""

import os
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "..", "results", "tab_case_mapping.csv")

# ---------------------------------------------------------------------------
# Load data live
# ---------------------------------------------------------------------------
with open(CSV_PATH, newline="") as f:
    records = list(csv.DictReader(f))


def pretty_dataset(name):
    return "SD1.4" if name == "SD14" else name


def abbrev_dataset(name):
    return {"GenBuster": "GB", "SD14": "SD", "SD1.4": "SD", "BigGAN": "BG"}.get(name, name)


def abbrev_verifier(name):
    return (name.replace("Qwen2.5-VL-", "Qwen")
                .replace("GPT-5.5", "GPT5.5"))


def classify(rec):
    """Return (group_key, group_order, chip_sublabel) for one row of the table."""
    ds = pretty_dataset(rec["dataset"])
    cfg = rec["configuration"]

    m = re.match(r"cross-family (\S+)->(\S+) \(", cfg)
    if m:
        src, dst = abbrev_dataset(m.group(1)), abbrev_dataset(m.group(2))
        return ("Cross-family transfer\n(6 dataset pairs)", 3, f"{src}→{dst}")

    m = re.match(r"(.+) distant shift \(.*verifier=([^)]+)\)", cfg)
    if m:
        return (f"{pretty_dataset(m.group(1))} distant shift", 4, abbrev_verifier(m.group(2)))

    m = re.match(r"(.+) deployment shift, stale (\w+) probe, verifier=(.+)", cfg)
    if m:
        ds_name = pretty_dataset(m.group(1))
        return (f"{ds_name} deployment shift\n(stale probe → SDXL/FLUX)", 2,
                 f"{m.group(2)}\n{abbrev_verifier(m.group(3))}")

    m = re.match(r"(.+) probe-degradation sweep, (\w+), frac=(\S+) \(verifier=([^)]+)\)", cfg)
    if m:
        backbone = m.group(2)
        return (f"SD1.4 degradation sweep\n({backbone})", 5, m.group(3))

    m = re.match(r"(.+) in-domain \(.*verifier=([^)]+)\)", cfg)
    if m:
        return (f"{pretty_dataset(m.group(1))} in-domain", 1, abbrev_verifier(m.group(2)))

    if "agreement-path correlated errors" in cfg:
        return (f"{ds} agreement-path\ncorrelated errors", 6, "never\nescalated")

    return ("Other", 9, cfg[:12])


rows_data = []
for rec in records:
    group, order, sub = classify(rec)
    rows_data.append({
        "group": group, "order": order, "sub": sub,
        "case": rec["case"].strip(),
    })

# ---------------------------------------------------------------------------
# Assemble rows: preserve group order (as first seen under `order`), then the
# original table order within a group (verifier sweep order etc.)
# ---------------------------------------------------------------------------
groups_seen = {}
for r in rows_data:
    groups_seen.setdefault((r["order"], r["group"]), []).append(r)

group_keys = sorted(groups_seen.keys())

# ---------------------------------------------------------------------------
# Style — validated categorical palette (dataviz skill, fixed hue order)
# ---------------------------------------------------------------------------
CASE_COLOR = {
    "1": "#1baf7a",          # aqua  — benign, cheap path suffices
    "2": "#eb6834",          # orange — cascade flat vs. best cheap
    "3": "#2a78d6",          # blue  — cascade significantly wins
    "Breakdown": "#e34948",  # red   — correlated blind spot
}
CASE_LABEL = {
    "1": "Case 1 — cheap path suffices, verifier rarely invoked",
    "2": "Case 2 — cascade flat vs. best cheap detector (not significant)",
    "3": "Case 3 — cascade significantly beats best cheap detector",
    "Breakdown": "Breakdown — correlated errors, never escalated",
}

n_rows = len(group_keys)
max_cols = max(len(v) for v in groups_seen.values())

chip = 0.5            # chip size (data units, square — aspect is forced equal below)
col_pitch = 1.0
row_h = 1.35           # extra headroom per row for two-line sublabels
label_margin = 4.6     # x-units reserved on the left for row labels

fig_w = label_margin + max_cols * col_pitch + 1.0
fig_h = n_rows * row_h + 2.0
fig, ax = plt.subplots(figsize=(fig_w * 0.62, fig_h * 0.62))
ax.set_aspect("equal", adjustable="box")

for ri, key in enumerate(group_keys):
    y = (n_rows - ri) * row_h
    items = groups_seen[key]
    group_label = key[1]
    ax.text(-0.3, y, group_label, ha="right", va="center", fontsize=9.2,
             color="#222221", linespacing=1.15)
    for ci, item in enumerate(items):
        xc = ci * col_pitch
        color = CASE_COLOR.get(item["case"], "#b0b0ac")
        rect = mpatches.FancyBboxPatch(
            (xc - chip / 2, y - chip / 2), chip, chip,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color, edgecolor="white", linewidth=1.0, zorder=3,
        )
        ax.add_patch(rect)
        case_txt = "B" if item["case"] == "Breakdown" else item["case"]
        ax.text(xc, y, case_txt, ha="center", va="center", fontsize=8.0,
                 color="white", fontweight="bold", zorder=4)
        ax.text(xc, y - chip / 2 - 0.08, item["sub"], ha="center", va="top",
                 fontsize=5.4, color="#8a8a86", linespacing=0.9)

ax.set_xlim(-label_margin, (max_cols - 1) * col_pitch + chip / 2 + 0.3)
ax.set_ylim(0.15, n_rows * row_h + 0.85)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Row separators
for ri in range(n_rows + 1):
    ax.axhline((n_rows - ri) * row_h + row_h / 2, color="#ececea", linewidth=0.8, zorder=0)

handles = [mpatches.Patch(facecolor=CASE_COLOR[c], label=CASE_LABEL[c])
           for c in ["1", "2", "3", "Breakdown"]]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
           ncol=2, fontsize=8.4, frameon=False, handlelength=1.2, columnspacing=1.2)

fig.suptitle(
    "Case 1/2/3/Breakdown Framework Mapped onto Every Tested Configuration",
    fontsize=12.5, y=1.015,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

out_dir = os.path.join(HERE, "..")
for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"plot_fig_f_case_mapping.{ext}")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close()
