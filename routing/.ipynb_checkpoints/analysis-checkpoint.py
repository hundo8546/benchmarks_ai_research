import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import resample

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "font.size": 11,
})

BASE = "/workspace/benchmarks_ai_research/routing/"

COLORS = {
    "CNN-only":        "#888780",
    "Entropy thresh":  "#378ADD",
    "Always escalate": "#D85A30",
    "Bandit":          "#BA7517",
    "Disagree thresh": "#1D9E75",
}

METHOD_ORDER = ["CNN-only", "Entropy thresh", "Always escalate", "Bandit", "Disagree thresh"]
K1 = 1.0
K2 = 5.0
eps = 1e-12


def bootstrap_ci(correct, n=1000, ci=95):
    scores = [np.mean(resample(correct)) for _ in range(n)]
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return lo, hi


def add_derived_features(df):
    df["disagree"]     = (df["y_cnn"] != df["y_clip"]).astype(int)
    df["clip_margin"]  = np.abs(df["clip_proba"] - 0.5)
    df["clip_entropy"] = -(
        df["clip_proba"] * np.log(df["clip_proba"] + eps) +
        (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + eps)
    )
    df["prob_gap"]      = np.abs(df["confidence"] - df["clip_proba"])
    df["cnn_correct"]   = (df["y_cnn"]  == df["y_true"]).astype(int)
    df["clip_correct"]  = (df["y_clip"] == df["y_true"]).astype(int)
    df["qwen_correct"]  = (df["y_qwen"] == df["y_true"]).astype(int)
    return df


def load_dataset(prefix, bandit_model_path, val_paths_path):
    df_cnn  = pd.read_csv(BASE + f"{prefix}bandit_dataset.csv")
    df_qwen = pd.read_csv(BASE + f"{prefix}qwen_preds.csv")
    df_clip = pd.read_csv(BASE + f"{prefix}clip_preds.csv")

    df = df_cnn.merge(df_qwen[["path", "y_qwen", "latency_qwen_ms"]], on="path")
    df = df.merge(df_clip[["path", "y_clip", "clip_proba", "latency_clip_ms"]], on="path")

    val_paths = np.load(val_paths_path, allow_pickle=True)
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    df = add_derived_features(df)

    bandit, scaler = joblib.load(bandit_model_path)
    return df, bandit, scaler


def run_all_methods(df, bandit, scaler):
    results = {}

    results["CNN-only"] = {
        "correct": df["cnn_correct"].tolist(),
        "cost": [K1] * len(df),
        "escalated": [0] * len(df)
    }

    results["Always escalate"] = {
        "correct": df["qwen_correct"].tolist(),
        "cost": [K1 + K2] * len(df),
        "escalated": [1] * len(df)
    }

    correct, costs, esc = [], [], []
    for _, row in df.iterrows():
        action = int(row["entropy1"] > 0.5)
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(K1 if action == 0 else K1 + K2)
        esc.append(action)
    results["Entropy thresh"] = {"correct": correct, "cost": costs, "escalated": esc}

    correct, costs, esc = [], [], []
    for _, row in df.iterrows():
        action = int(row["disagree"] == 1)
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(K1 if action == 0 else K1 + K2)
        esc.append(action)
    results["Disagree thresh"] = {"correct": correct, "cost": costs, "escalated": esc}

    correct, costs, esc = [], [], []
    for _, row in df.iterrows():
        ctx = scaler.transform([[
            row["confidence"], row["margin1"], row["entropy1"], row["logit"], row["y_cnn"],
            row["y_clip"], row["clip_proba"], row["clip_margin"], row["clip_entropy"],
            row["disagree"], row["prob_gap"]
        ]])
        action = bandit.predict(ctx)[0]
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(K1 if action == 0 else K1 + K2)
        esc.append(action)
    results["Bandit"] = {"correct": correct, "cost": costs, "escalated": esc}

    return results


def compute_summary(results):
    rows = []
    for method in METHOD_ORDER:
        r = results[method]
        acc = np.mean(r["correct"])
        lo, hi = bootstrap_ci(r["correct"])
        cost = np.mean(r["cost"])
        esc_rate = np.mean(r["escalated"])
        rows.append({
            "Method": method,
            "Accuracy": acc,
            "CI_lo": lo,
            "CI_hi": hi,
            "Avg Cost": cost,
            "Escalation Rate": esc_rate
        })
    return pd.DataFrame(rows)


def compute_agree_disagree(df):
    agree_df    = df[df["disagree"] == 0]
    disagree_df = df[df["disagree"] == 1]
    return {
        "n_agree":           len(agree_df),
        "n_disagree":        len(disagree_df),
        "agree_cnn_acc":     agree_df["cnn_correct"].mean(),
        "disagree_cnn_acc":  disagree_df["cnn_correct"].mean(),
        "agree_qwen_acc":    agree_df["qwen_correct"].mean(),
        "disagree_qwen_acc": disagree_df["qwen_correct"].mean(),
    }


def compute_escalation_breakdown(df, results):
    cases = {
        "CNN correct / CLIP agree":    df[(df["cnn_correct"]==1) & (df["disagree"]==0)],
        "CNN correct / CLIP disagree": df[(df["cnn_correct"]==1) & (df["disagree"]==1)],
        "CNN wrong / CLIP agree":      df[(df["cnn_correct"]==0) & (df["disagree"]==0)],
        "CNN wrong / CLIP disagree":   df[(df["cnn_correct"]==0) & (df["disagree"]==1)],
    }
    rows = []
    bandit_esc   = np.array(results["Bandit"]["escalated"])
    disagree_esc = np.array(results["Disagree thresh"]["escalated"])
    for label, subset in cases.items():
        if len(subset) == 0:
            continue
        idx = subset.index
        rows.append({
            "Case": label,
            "n": len(subset),
            "Bandit esc.": bandit_esc[idx].mean(),
            "Disagree esc.": disagree_esc[idx].mean(),
        })
    return pd.DataFrame(rows)


# ── Load datasets
print("Loading GenBuster...")
df_gb, bandit_gb, scaler_gb = load_dataset(
    "", "bandit_model.pkl",
    BASE + "bandit_val_paths.npy"
)
print("Loading SD14...")
df_sd, bandit_sd, scaler_sd = load_dataset(
   "sd14_", "bandit_model.pkl",
   BASE + "sd14_bandit_val_paths.npy"
)
print("Loading BigGAN...")
df_bg, bandit_bg, scaler_bg = load_dataset(
    "biggan_", BASE + "biggan_bandit_model.pkl",
    BASE + "biggan_bandit_val_paths.npy"
)

print("Running evaluations...")
res_gb = run_all_methods(df_gb, bandit_gb, scaler_gb)
res_sd = run_all_methods(df_sd, bandit_sd, scaler_sd)
res_bg = run_all_methods(df_bg, bandit_bg, scaler_bg)

summary_gb = compute_summary(res_gb)
summary_sd = compute_summary(res_sd)
summary_bg = compute_summary(res_bg)

ad_gb = compute_agree_disagree(df_gb)
ad_sd = compute_agree_disagree(df_sd)
ad_bg = compute_agree_disagree(df_bg)

esc_gb = compute_escalation_breakdown(df_gb, res_gb)
esc_sd = compute_escalation_breakdown(df_sd, res_sd)
esc_bg = compute_escalation_breakdown(df_bg, res_bg)

# ── Console output
for name, s in [("GenBuster", summary_gb), ("SD14", summary_sd), ("BigGAN", summary_bg)]:
    print(f"\n=== {name} ===")
    print(s.to_string(index=False))

print("\n=== Agreement Analysis ===")
for name, ad in [("GenBuster", ad_gb), ("SD14", ad_sd), ("BigGAN", ad_bg)]:
    print(f"{name}: agree n={ad['n_agree']} CNN={ad['agree_cnn_acc']:.3f} | "
          f"disagree n={ad['n_disagree']} CNN={ad['disagree_cnn_acc']:.3f}")


# ══════════════════════════════
# PLOT 1: Accuracy bars per dataset
# ══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, (title, summary) in zip(axes, [
    ("GenBuster-200K-mini", summary_gb),
    ("GenImage SD 1.4",     summary_sd),
    ("GenImage BigGAN",     summary_bg),
]):
    bars = ax.bar(
        range(len(METHOD_ORDER)),
        summary["Accuracy"],
        color=[COLORS[m] for m in METHOD_ORDER],
        width=0.6, alpha=0.9, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in METHOD_ORDER], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{row.Accuracy:.2f}", ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("Accuracy")
fig.suptitle("Detection accuracy across datasets and methods", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_accuracy_comparison.png")


# ══════════════════════════════
# PLOT 2: Cost-accuracy frontier
# ══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, summary) in zip(axes, [
    ("GenBuster-200K-mini", summary_gb),
    ("GenImage SD 1.4",     summary_sd),
    ("GenImage BigGAN",     summary_bg),
]):
    for _, row in summary.iterrows():
        m = row["Method"]
        ax.scatter(row["Avg Cost"], row["Accuracy"], color=COLORS[m], s=120, zorder=5)
        ax.annotate(m, (row["Avg Cost"], row["Accuracy"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color=COLORS[m])
    ax.set_xlabel("Average cost")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, 7)
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
axes[0].set_ylabel("Accuracy")
fig.suptitle("Cost-accuracy frontier", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_cost_accuracy_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_cost_accuracy_all.png")


# ══════════════════════════════
# PLOT 3: Agree vs disagree accuracy
# ══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, ad) in zip(axes, [
    ("GenBuster-200K-mini", ad_gb),
    ("GenImage SD 1.4",     ad_sd),
    ("GenImage BigGAN",     ad_bg),
]):
    x = np.arange(2)
    w = 0.3
    b1 = ax.bar(x - w/2, [ad["agree_cnn_acc"],  ad["disagree_cnn_acc"]],
                w, label="CNN",  color=COLORS["CNN-only"], alpha=0.9)
    b2 = ax.bar(x + w/2, [ad["agree_qwen_acc"], ad["disagree_qwen_acc"]],
                w, label="Qwen", color=COLORS["Always escalate"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Agree\n(n={ad['n_agree']})",
        f"Disagree\n(n={ad['n_disagree']})"
    ])
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("Accuracy")
fig.suptitle("CNN and Qwen accuracy by agreement status", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_agree_disagree_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_agree_disagree_all.png")


# ══════════════════════════════
# PLOT 4: Cross-dataset grouped bar
# ══════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
dataset_labels = ["GenBuster\n(video gen.)", "GenImage SD 1.4\n(diffusion)", "GenImage BigGAN\n(GAN)"]
summaries = [summary_gb, summary_sd, summary_bg]
x = np.arange(len(dataset_labels))
w = 0.14
offsets = np.linspace(-2*w, 2*w, len(METHOD_ORDER))
for i, method in enumerate(METHOD_ORDER):
    accs = [s[s["Method"] == method]["Accuracy"].values[0] for s in summaries]
    ax.bar(x + offsets[i], accs, w,
           label=method, color=COLORS[method], alpha=0.9,
           edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(dataset_labels, fontsize=10)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.3, 1.05)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.set_title("Method accuracy across all three datasets", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_cross_dataset.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_cross_dataset.png")


# ══════════════════════════════
# PLOT 5: Escalation behavior
# ══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, esc) in zip(axes, [
    ("GenBuster", esc_gb),
    ("GenImage SD 1.4", esc_sd),
    ("GenImage BigGAN", esc_bg),
]):
    x = np.arange(len(esc))
    w = 0.3
    ax.bar(x - w/2, esc["Bandit esc."],   w, label="Bandit",
           color=COLORS["Bandit"], alpha=0.9)
    ax.bar(x + w/2, esc["Disagree esc."], w, label="Disagree thresh",
           color=COLORS["Disagree thresh"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(esc["Case"], fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Escalation rate")
fig.suptitle("Escalation behavior by case type", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_escalation_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_escalation_all.png")


# ══════════════════════════════
# PLOT 6: Per-class accuracy
# ══════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods_to_show = ["CNN-only", "Disagree thresh", "Always escalate"]
for ax, (title, df, res) in zip(axes, [
    ("GenBuster-200K-mini", df_gb, res_gb),
    ("GenImage SD 1.4",     df_sd, res_sd),
    ("GenImage BigGAN",     df_bg, res_bg),
]):
    fake_accs, real_accs = [], []
    for method in methods_to_show:
        correct  = np.array(res[method]["correct"])
        fake_idx = (df["y_true"] == 1).values
        real_idx = (df["y_true"] == 0).values
        fake_accs.append(correct[fake_idx].mean())
        real_accs.append(correct[real_idx].mean())
    x = np.arange(len(methods_to_show))
    w = 0.3
    b1 = ax.bar(x - w/2, fake_accs, w, label="Fake (AI-gen.)",
                color="#D85A30", alpha=0.9)
    b2 = ax.bar(x + w/2, real_accs, w, label="Real",
                color="#1D9E75", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods_to_show, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("Per-class accuracy")
fig.suptitle("Per-class accuracy: fake vs real detection", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_per_class.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_per_class.png")


# ══════════════════════════════
# LATEX TABLES
# ══════════════════════════════
def make_latex_table(summary, dataset_name, label):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Detection results on {dataset_name}. CI denotes 95\\% bootstrap confidence interval.}}",
        f"\\label{{tab:{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Accuracy & 95\% CI & Avg Cost & Esc. Rate \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Method']} & {row['Accuracy']:.3f} & "
            f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] & "
            f"{row['Avg Cost']:.3f} & "
            f"{row['Escalation Rate']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

print("\n\n=== LATEX TABLES ===\n")
print(make_latex_table(summary_gb, "GenBuster-200K-mini", "genbuster"))
print()
print(make_latex_table(summary_sd, "GenImage SD~1.4", "sd14"))
print()
print(make_latex_table(summary_bg, "GenImage BigGAN", "biggan"))

print("\n% Cross-dataset summary")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Disagreement threshold accuracy across datasets and generator types.}")
print(r"\label{tab:crossdataset}")
print(r"\begin{tabular}{llcc}")
print(r"\toprule")
print(r"Dataset & Generator type & CNN-only & Disagree thresh \\")
print(r"\midrule")
for name, gtype, s in [
    ("GenBuster-200K-mini", "Video generators", summary_gb),
    ("GenImage SD~1.4",     "Diffusion images", summary_sd),
    ("GenImage BigGAN",     "GAN images",       summary_bg),
]:
    cnn = s[s["Method"]=="CNN-only"]["Accuracy"].values[0]
    dis = s[s["Method"]=="Disagree thresh"]["Accuracy"].values[0]
    print(f"{name} & {gtype} & {cnn:.3f} & {dis:.3f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\nDone. All plots saved to", BASE)
