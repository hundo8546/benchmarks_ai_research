import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    "Disagree ensemble": "#9B59B6"
}

METHOD_ORDER = ["CNN-only", "Entropy thresh", "Always escalate", "Bandit", "Disagree thresh","Disagree ensemble"]
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

        # Disagree-aware ensemble
    correct_ens, costs_ens = [], []
    for _, row in df.iterrows():
        pred = row["y_cnn"] if row["disagree"] == 0 else row["y_clip"]
        correct_ens.append(int(pred == row["y_true"]))
        costs_ens.append(K1)
    results["Disagree ensemble"] = {
        "correct": correct_ens,
        "cost": costs_ens,
        "escalated": [0] * len(df)
    }

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


# ══════════════════════════════════════════════
# ABLATION SETUP — use consistent val paths
# ══════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def run_ablation_bandit(df, features, val_paths):
    LAMBDA = 0.05
    K1 = 1.0
    K2 = 5.0

    df = df.copy()
    df["cost_stop"] = (df["latency_cnn_ms"] + df["latency_clip_ms"])
    df["cost_esc"]  = (df["latency_cnn_ms"] + df["latency_clip_ms"] + df["latency_qwen_ms"])
    max_cost = df["cost_esc"].max()
    df["cost_stop"] /= max_cost
    df["cost_esc"]  /= max_cost

    def compute_rewards(row):
        r_stop = (1 if row["y_cnn"] == row["y_true"] else -1) - LAMBDA * row["cost_stop"]
        r_esc  = (1 if row["y_qwen"] == row["y_true"] else -1) - LAMBDA * row["cost_esc"]
        return r_stop, r_esc

    rewards = df.apply(compute_rewards, axis=1)
    df["r_stop"] = [r[0] for r in rewards]
    df["r_esc"]  = [r[1] for r in rewards]
    df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)

    # Use consistent val paths — same split as main evaluation
    val_mask  = df["path"].isin(val_paths)
    train_mask = ~val_mask

    X = df[features].values
    y = df["target"].values

    train_idx = df[train_mask].index.values
    val_idx   = df[val_mask].index.values

    if len(np.unique(y[train_idx])) < 2:
        return None, None, None

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y[train_idx])

    correct, costs, esc_rates = [], [], []
    for i in val_idx:
        row = df.iloc[i]
        ctx    = scaler.transform([X[i]])
        action = model.predict(ctx)[0]
        pred   = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(K1 if action == 0 else K1 + K2)
        esc_rates.append(action)

    return np.mean(correct), np.mean(costs), np.mean(esc_rates)


def run_simple_ensemble(df, val_paths):
    # Majority vote: if clip_proba > 0.5 and y_cnn == 1 -> fake, else use clip
    K1 = 1.0
    correct, costs = [], []
    val_df = df[df["path"].isin(val_paths)]
    for _, row in val_df.iterrows():
        # soft vote: average CNN prob and clip_proba
        cnn_prob  = row["confidence"] if row["y_cnn"] == 1 else 1 - row["confidence"]
        clip_prob = row["clip_proba"]
        avg_prob  = (cnn_prob + clip_prob) / 2
        pred = 1 if avg_prob > 0.5 else 0
        correct.append(int(pred == row["y_true"]))
        costs.append(K1)  # no escalation — both always run
    return np.mean(correct), np.mean(costs)


# Load val paths for consistent splits
gb_val_paths = np.load(BASE + "bandit_val_paths.npy", allow_pickle=True)

# ══════════════════════════════════════════════
# ABLATION 1: Feature ablation
# ══════════════════════════════════════════════
feature_sets = {
    "CNN only":            ["confidence", "margin1", "entropy1", "logit", "y_cnn"],
    "CLIP only":           ["y_clip", "clip_proba", "clip_margin", "clip_entropy"],
    "CNN + disagree":      ["confidence", "margin1", "entropy1", "logit", "y_cnn", "disagree"],
    "CNN + CLIP (full)":   ["confidence", "margin1", "entropy1", "logit", "y_cnn",
                            "y_clip", "clip_proba", "clip_margin", "clip_entropy",
                            "disagree", "prob_gap"],
}

# Load full bandit dataset for ablation (not just val rows)
df_gb_full = pd.read_csv(BASE + "bandit_dataset.csv")
df_qwen_full = pd.read_csv(BASE + "qwen_preds.csv")
df_clip_full = pd.read_csv(BASE + "clip_preds.csv")
df_gb_abl = df_gb_full.merge(df_qwen_full[["path", "y_qwen", "latency_qwen_ms"]], on="path")
df_gb_abl = df_gb_abl.merge(df_clip_full[["path", "y_clip", "clip_proba", "latency_clip_ms"]], on="path")
df_gb_abl = add_derived_features(df_gb_abl)

print("\n=== ABLATION 1: Feature ablation (GenBuster) ===")
print(f"{'Feature set':<25} {'Accuracy':>10} {'Avg Cost':>10} {'Esc Rate':>10}")
print("-" * 58)
abl_rows = []

# Simple ensemble baseline first
ens_acc, ens_cost = run_simple_ensemble(df_gb_abl, gb_val_paths)
print(f"{'Simple ensemble':<25} {ens_acc:>10.3f} {ens_cost:>10.3f} {'N/A':>10}")
abl_rows.append({"Feature set": "Simple ensemble", "Accuracy": ens_acc, "Avg Cost": ens_cost, "Esc Rate": 0.0})

for name, feats in feature_sets.items():
    acc, cost, esc = run_ablation_bandit(df_gb_abl, feats, gb_val_paths)
    if acc is not None:
        print(f"{name:<25} {acc:>10.3f} {cost:>10.3f} {esc:>10.3f}")
        abl_rows.append({"Feature set": name, "Accuracy": acc, "Avg Cost": cost, "Esc Rate": esc})

# Also print main CNN-only for consistency check
cnn_only_acc = df_gb[df_gb["path"].isin(gb_val_paths)]["cnn_correct"].mean()
print(f"\nConsistency check — CNN-only (main eval): {cnn_only_acc:.3f}")

# ══════════════════════════════════════════════
# ABLATION 2: Entropy threshold sensitivity
# ══════════════════════════════════════════════
print("\n=== ABLATION 2: Entropy threshold sensitivity (GenBuster) ===")
print(f"{'Tau':<10} {'Accuracy':>10} {'Avg Cost':>10} {'Esc Rate':>10}")
print("-" * 43)
tau_rows = []
taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val_df = df_gb[df_gb["path"].isin(gb_val_paths)].reset_index(drop=True)

for tau in taus:
    correct, costs, esc = [], [], []
    for _, row in val_df.iterrows():
        action = int(row["entropy1"] > tau)
        pred   = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(1.0 if action == 0 else 6.0)
        esc.append(action)
    acc  = np.mean(correct)
    cost = np.mean(costs)
    esc_rate = np.mean(esc)
    print(f"{tau:<10} {acc:>10.3f} {cost:>10.3f} {esc_rate:>10.3f}")
    tau_rows.append({"Tau": tau, "Accuracy": acc, "Avg Cost": cost, "Esc Rate": esc_rate})

# ══════════════════════════════════════════════
# ABLATION PLOTS
# ══════════════════════════════════════════════

# Feature ablation bar chart
fig, ax = plt.subplots(figsize=(10, 4))
names = [r["Feature set"] for r in abl_rows]
accs  = [r["Accuracy"] for r in abl_rows]
abl_colors = ["#378ADD", "#888780", "#5DCAA5", "#BA7517", "#1D9E75"]
bars = ax.bar(range(len(names)), accs,
              color=abl_colors[:len(names)], alpha=0.9,
              edgecolor="white", linewidth=0.5, width=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.3, 0.85)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_title("Feature ablation: bandit routing accuracy (GenBuster)", fontsize=11, fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(BASE + "plot_ablation_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_ablation_features.png")

# Entropy threshold sensitivity plot
fig, ax1 = plt.subplots(figsize=(8, 4))
taus_plot  = [r["Tau"] for r in tau_rows]
accs_plot  = [r["Accuracy"] for r in tau_rows]
costs_plot = [r["Avg Cost"] for r in tau_rows]
escs_plot  = [r["Esc Rate"] for r in tau_rows]

ax2 = ax1.twinx()
ax1.plot(taus_plot, accs_plot,  color=COLORS["Disagree thresh"],
         marker="o", linewidth=2, label="Accuracy")
ax2.plot(taus_plot, escs_plot,  color=COLORS["Bandit"],
         marker="s", linewidth=2, linestyle="--", label="Escalation rate")
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax1.set_xlabel("Entropy threshold tau")
ax1.set_ylabel("Accuracy", color=COLORS["Disagree thresh"])
ax2.set_ylabel("Escalation rate", color=COLORS["Bandit"])
ax1.set_ylim(0.3, 0.85)
ax2.set_ylim(0.0, 1.0)
ax1.set_title("Entropy threshold sensitivity (GenBuster)", fontsize=11, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
plt.tight_layout()
plt.savefig(BASE + "plot_ablation_entropy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_ablation_entropy.png")

# ══════════════════════════════════════════════
# LATEX ABLATION TABLES
# ══════════════════════════════════════════════
print("\n% Feature ablation table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Feature ablation study for the bandit router on GenBuster-200K-mini. "
      r"Simple ensemble uses soft voting between CNN and CLIP without escalation.}")
print(r"\label{tab:ablation_features}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Feature set & Accuracy & Avg Cost & Esc. Rate \\")
print(r"\midrule")
for r in abl_rows:
    esc_str = f"{r['Esc Rate']:.2f}" if r["Esc Rate"] > 0 else "N/A"
    print(f"{r['Feature set']} & {r['Accuracy']:.3f} & {r['Avg Cost']:.3f} & {esc_str} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% Entropy threshold sensitivity table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Entropy threshold sensitivity on GenBuster-200K-mini. "
      r"$\tau = 0.5$ is used in all main experiments.}")
print(r"\label{tab:ablation_entropy}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"$\tau$ & Accuracy & Avg Cost & Esc. Rate \\")
print(r"\midrule")
for r in tau_rows:
    print(f"{r['Tau']} & {r['Accuracy']:.3f} & {r['Avg Cost']:.3f} & {r['Esc Rate']:.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# ══════════════════════════════════════════════
# ABLATION 3: Soft disagreement threshold sensitivity
# ══════════════════════════════════════════════
print("\n=== ABLATION 3: Disagreement threshold sensitivity (GenBuster) ===")
print(f"{'Threshold':<12} {'Accuracy':>10} {'Avg Cost':>10} {'Esc Rate':>10}")
print("-" * 45)

val_df = df_gb[df_gb["path"].isin(gb_val_paths)].reset_index(drop=True)
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
disagree_thresh_rows = []

for thresh in thresholds:
    correct, costs, esc = [], [], []
    for _, row in val_df.iterrows():
        # escalate if prob_gap exceeds threshold
        action = int(row["prob_gap"] > thresh)
        pred   = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(1.0 if action == 0 else 6.0)
        esc.append(action)
    acc      = np.mean(correct)
    cost     = np.mean(costs)
    esc_rate = np.mean(esc)
    print(f"{thresh:<12} {acc:>10.3f} {cost:>10.3f} {esc_rate:>10.3f}")
    disagree_thresh_rows.append({
        "Threshold": thresh,
        "Accuracy": acc,
        "Avg Cost": cost,
        "Esc Rate": esc_rate
    })

# Plot
fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
thresh_vals = [r["Threshold"] for r in disagree_thresh_rows]
accs_vals   = [r["Accuracy"]  for r in disagree_thresh_rows]
esc_vals    = [r["Esc Rate"]  for r in disagree_thresh_rows]

ax1.plot(thresh_vals, accs_vals, color=COLORS["Disagree thresh"],
         marker="o", linewidth=2, label="Accuracy")
ax2.plot(thresh_vals, esc_vals,  color=COLORS["Bandit"],
         marker="s", linewidth=2, linestyle="--", label="Escalation rate")
ax1.axvline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.3)
ax1.set_xlabel("Prob gap threshold")
ax1.set_ylabel("Accuracy", color=COLORS["Disagree thresh"])
ax2.set_ylabel("Escalation rate", color=COLORS["Bandit"])
ax1.set_ylim(0.3, 0.85)
ax2.set_ylim(0.0, 1.1)
ax1.set_title("Disagreement threshold sensitivity (GenBuster)",
              fontsize=11, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
plt.tight_layout()
plt.savefig(BASE + "plot_ablation_disagree_thresh.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_ablation_disagree_thresh.png")


# ══════════════════════════════════════════════
# ABLATION 4: Accuracy/cost efficiency metric
# ══════════════════════════════════════════════
print("\n=== ABLATION 4: Accuracy/Cost efficiency ===")
print(f"{'Dataset':<25} {'Method':<20} {'Accuracy':>10} {'Avg Cost':>10} {'Acc/Cost':>10}")
print("-" * 70)

efficiency_rows = []
for dname, summary in [("GenBuster", summary_gb), ("SD14", summary_sd), ("BigGAN", summary_bg)]:
    for _, row in summary.iterrows():
        eff = row["Accuracy"] / row["Avg Cost"]
        efficiency_rows.append({
            "Dataset": dname,
            "Method": row["Method"],
            "Accuracy": row["Accuracy"],
            "Avg Cost": row["Avg Cost"],
            "Acc/Cost": eff
        })
        print(f"{dname:<25} {row['Method']:<20} {row['Accuracy']:>10.3f} "
              f"{row['Avg Cost']:>10.3f} {eff:>10.3f}")

# Plot efficiency across datasets
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, dname in zip(axes, ["GenBuster", "SD14", "BigGAN"]):
    subset = [r for r in efficiency_rows if r["Dataset"] == dname]
    methods = [r["Method"] for r in subset]
    effs    = [r["Acc/Cost"] for r in subset]
    bars = ax.bar(range(len(methods)), effs,
                  color=[COLORS[m] for m in methods],
                  alpha=0.9, edgecolor="white", linewidth=0.5, width=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=8)
    ax.set_title(dname, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy / Cost")
    for bar, eff in zip(bars, effs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{eff:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Cost-normalized accuracy (Accuracy / Avg Cost)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(BASE + "plot_efficiency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_efficiency.png")


# ══════════════════════════════════════════════
# ABLATION 5: Cross-dataset bandit transfer
# ══════════════════════════════════════════════
print("\n=== ABLATION 5: Cross-dataset bandit transfer ===")

def train_bandit_on_dataset(df_train, val_paths_train):
    LAMBDA = 0.05
    K1, K2 = 1.0, 5.0

    df = df_train.copy()
    df["cost_stop"] = (df["latency_cnn_ms"] + df["latency_clip_ms"])
    df["cost_esc"]  = (df["latency_cnn_ms"] + df["latency_clip_ms"] + df["latency_qwen_ms"])
    max_cost = df["cost_esc"].max()
    df["cost_stop"] /= max_cost
    df["cost_esc"]  /= max_cost

    def compute_rewards(row):
        r_stop = (1 if row["y_cnn"] == row["y_true"] else -1) - LAMBDA * row["cost_stop"]
        r_esc  = (1 if row["y_qwen"] == row["y_true"] else -1) - LAMBDA * row["cost_esc"]
        return r_stop, r_esc

    rewards = df.apply(compute_rewards, axis=1)
    df["r_stop"] = [r[0] for r in rewards]
    df["r_esc"]  = [r[1] for r in rewards]
    df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)

    features = ["confidence", "margin1", "entropy1", "logit", "y_cnn",
                "y_clip", "clip_proba", "clip_margin", "clip_entropy",
                "disagree", "prob_gap"]

    # train on non-val rows
    train_mask = ~df["path"].isin(val_paths_train)
    train_idx  = df[train_mask].index.values

    X = df[features].values
    y = df["target"].values

    if len(np.unique(y[train_idx])) < 2:
        return None, None

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    model   = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y[train_idx])
    return model, scaler


def eval_transferred_bandit(model, scaler, df_test, val_paths_test):
    K1, K2 = 1.0, 5.0
    features = ["confidence", "margin1", "entropy1", "logit", "y_cnn",
                "y_clip", "clip_proba", "clip_margin", "clip_entropy",
                "disagree", "prob_gap"]

    val_df = df_test[df_test["path"].isin(val_paths_test)].reset_index(drop=True)
    correct, costs = [], []
    for _, row in val_df.iterrows():
        X_row  = scaler.transform([row[features].values])
        action = model.predict(X_row)[0]
        pred   = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(K1 if action == 0 else K1 + K2)
    return np.mean(correct), np.mean(costs)


# Load full datasets for transfer experiment
def load_full_dataset(prefix, val_paths_path):
    df_cnn  = pd.read_csv(BASE + f"{prefix}bandit_dataset.csv")
    df_qwen = pd.read_csv(BASE + f"{prefix}qwen_preds.csv")
    df_clip = pd.read_csv(BASE + f"{prefix}clip_preds.csv")
    df = df_cnn.merge(df_qwen[["path", "y_qwen", "latency_qwen_ms"]], on="path")
    df = df.merge(df_clip[["path", "y_clip", "clip_proba", "latency_clip_ms"]], on="path")
    df = add_derived_features(df)
    val_paths = np.load(val_paths_path, allow_pickle=True)
    return df, val_paths

df_gb_full2, gb_vp = load_full_dataset("", BASE + "bandit_val_paths.npy")
df_bg_full2, bg_vp = load_full_dataset("biggan_", BASE + "biggan_bandit_val_paths.npy")
df_sd_full2, sd_vp = load_full_dataset("sd14_", BASE + "sd14_bandit_val_paths.npy")

# Train on each dataset, eval on others
datasets = {
    "GenBuster": (df_gb_full2, gb_vp),
    "BigGAN":    (df_bg_full2, bg_vp),
    "SD14":      (df_sd_full2, sd_vp),
}

print(f"{'Train':<12} {'Test':<12} {'Accuracy':>10} {'Avg Cost':>10}")
print("-" * 47)
transfer_rows = []
for train_name, (df_train, vp_train) in datasets.items():
    model, scaler = train_bandit_on_dataset(df_train, vp_train)
    if model is None:
        continue
    for test_name, (df_test, vp_test) in datasets.items():
        acc, cost = eval_transferred_bandit(model, scaler, df_test, vp_test)
        marker = " (in-domain)" if train_name == test_name else ""
        print(f"{train_name:<12} {test_name:<12} {acc:>10.3f} {cost:>10.3f}{marker}")
        transfer_rows.append({
            "Train": train_name,
            "Test": test_name,
            "Accuracy": acc,
            "Avg Cost": cost,
            "In-domain": train_name == test_name
        })

# Cross-dataset transfer heatmap
import matplotlib.colors as mcolors
train_names = ["GenBuster", "BigGAN", "SD14"]
test_names  = ["GenBuster", "BigGAN", "SD14"]
matrix = np.zeros((3, 3))
for r in transfer_rows:
    i = train_names.index(r["Train"])
    j = test_names.index(r["Test"])
    matrix[i, j] = r["Accuracy"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(matrix, cmap="YlGn", vmin=0.4, vmax=1.0)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(test_names)
ax.set_yticklabels(train_names)
ax.set_xlabel("Test dataset")
ax.set_ylabel("Train dataset")
ax.set_title("Cross-dataset bandit transfer accuracy",
             fontsize=11, fontweight="bold")
for i in range(3):
    for j in range(3):
        label = f"{matrix[i,j]:.3f}"
        if i == j:
            label += "\n(in-domain)"
        ax.text(j, i, label, ha="center", va="center", fontsize=9,
                color="black")
plt.colorbar(im, ax=ax, label="Accuracy")
plt.tight_layout()
plt.savefig(BASE + "plot_transfer.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_transfer.png")

# LaTeX tables
print("\n% Disagreement threshold sensitivity table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Soft disagreement threshold sensitivity on GenBuster-200K-mini. "
      r"Threshold applied to $|p_{\text{CLIP}} - p_{\text{CNN}}|$. "
      r"Binary disagreement corresponds to threshold = 0.}")
print(r"\label{tab:ablation_disagree}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Threshold & Accuracy & Avg Cost & Esc. Rate \\")
print(r"\midrule")
for r in disagree_thresh_rows:
    marker = r" $\leftarrow$ selected" if r["Threshold"] == 0.0 else ""
    print(f"{r['Threshold']} & {r['Accuracy']:.3f} & "
          f"{r['Avg Cost']:.3f} & {r['Esc Rate']:.2f}{marker} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% Cross-dataset transfer table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Cross-dataset bandit transfer accuracy. "
      r"Diagonal entries are in-domain results.}")
print(r"\label{tab:transfer}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Train $\rightarrow$ Test & GenBuster & BigGAN & SD14 \\")
print(r"\midrule")
for train_name in train_names:
    row_vals = []
    for test_name in test_names:
        match = [r for r in transfer_rows
                 if r["Train"] == train_name and r["Test"] == test_name]
        val = f"{match[0]['Accuracy']:.3f}" if match else "N/A"
        if train_name == test_name:
            val = f"\\textbf{{{val}}}"
        row_vals.append(val)
    print(f"{train_name} & {' & '.join(row_vals)} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")


# ══════════════════════════════════════════════
# FIX 1: Binary vs soft disagreement — clearly separated
# ══════════════════════════════════════════════
print("\n=== Binary vs Soft Disagreement (GenBuster) ===")
print(f"{'Method':<35} {'Accuracy':>10} {'Avg Cost':>10} {'Esc Rate':>10}")
print("-" * 58)

val_df = df_gb[df_gb["path"].isin(gb_val_paths)].reset_index(drop=True)

# Binary disagreement (y_cnn != y_clip)
correct, costs, esc = [], [], []
for _, row in val_df.iterrows():
    action = int(row["y_cnn"] != row["y_clip"])
    pred   = row["y_cnn"] if action == 0 else row["y_qwen"]
    correct.append(int(pred == row["y_true"]))
    costs.append(1.0 if action == 0 else 6.0)
    esc.append(action)
bin_acc  = np.mean(correct)
bin_cost = np.mean(costs)
bin_esc  = np.mean(esc)
print(f"{'Binary disagree (y_cnn != y_clip)':<35} {bin_acc:>10.3f} {bin_cost:>10.3f} {bin_esc:>10.3f}")

# Soft threshold sweep on |clip_proba - cnn_prob|
soft_rows = []
for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    correct, costs, esc = [], [], []
    for _, row in val_df.iterrows():
        action = int(row["prob_gap"] > thresh)
        pred   = row["y_cnn"] if action == 0 else row["y_qwen"]
        correct.append(int(pred == row["y_true"]))
        costs.append(1.0 if action == 0 else 6.0)
        esc.append(action)
    acc  = np.mean(correct)
    cost = np.mean(costs)
    er   = np.mean(esc)
    marker = " <- selected" if thresh == 0.0 else ""
    print(f"{'Soft thresh |p_gap| > ' + str(thresh):<35} {acc:>10.3f} {cost:>10.3f} {er:>10.3f}{marker}")
    soft_rows.append({"Threshold": thresh, "Accuracy": acc,
                      "Avg Cost": cost, "Esc Rate": er})


# ══════════════════════════════════════════════
# FIX 2: Cross-dataset disagreement rule transfer
# ══════════════════════════════════════════════
print("\n=== Cross-dataset Disagreement Rule Transfer ===")
print("(No training — disagreement threshold has no parameters)")
print(f"{'Eval dataset':<20} {'Binary disagree':>18} {'Soft thresh=0.0':>18}")
print("-" * 58)

cross_disagree_rows = []
for dname, df_eval, vp_eval in [
    ("GenBuster", df_gb, gb_vp),
    ("SD14",      df_sd, sd_vp),
    ("BigGAN",    df_bg, bg_vp),
]:
    val_eval = df_eval[df_eval["path"].isin(vp_eval)].reset_index(drop=True)

    # Binary
    correct_bin, correct_soft = [], []
    for _, row in val_eval.iterrows():
        action_bin  = int(row["y_cnn"] != row["y_clip"])
        action_soft = int(row["prob_gap"] > 0.0)
        pred_bin    = row["y_cnn"] if action_bin  == 0 else row["y_qwen"]
        pred_soft   = row["y_cnn"] if action_soft == 0 else row["y_qwen"]
        correct_bin.append(int(pred_bin  == row["y_true"]))
        correct_soft.append(int(pred_soft == row["y_true"]))

    bin_a  = np.mean(correct_bin)
    soft_a = np.mean(correct_soft)
    print(f"{dname:<20} {bin_a:>18.3f} {soft_a:>18.3f}")
    cross_disagree_rows.append({
        "Dataset": dname,
        "Binary": bin_a,
        "Soft=0.0": soft_a
    })


# ══════════════════════════════════════════════
# FIX 3: CLIP-only in same evaluation setup
# ══════════════════════════════════════════════
print("\n=== CLIP-only in routing evaluation setup ===")
print(f"{'Dataset':<15} {'CLIP-only acc':>15} {'Avg Cost':>10}")
print("-" * 43)

clip_only_rows = []
for dname, df_eval, vp_eval in [
    ("GenBuster", df_gb, gb_vp),
    ("SD14",      df_sd, sd_vp),
    ("BigGAN",    df_bg, bg_vp),
]:
    val_eval = df_eval[df_eval["path"].isin(vp_eval)].reset_index(drop=True)
    correct  = (val_eval["y_clip"] == val_eval["y_true"]).astype(int).tolist()
    acc      = np.mean(correct)
    cost     = 1.0  # CLIP always runs, no escalation
    print(f"{dname:<15} {acc:>15.3f} {cost:>10.3f}")
    clip_only_rows.append({"Dataset": dname, "Accuracy": acc, "Avg Cost": cost})


# ══════════════════════════════════════════════
# UPDATED PLOTS
# ══════════════════════════════════════════════

# Plot 1: Binary vs soft disagreement
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
thresh_vals = [r["Threshold"] for r in soft_rows]
accs_soft   = [r["Accuracy"]  for r in soft_rows]
esc_soft    = [r["Esc Rate"]  for r in soft_rows]

ax1.plot(thresh_vals, accs_soft, color=COLORS["Disagree thresh"],
         marker="o", linewidth=2, label="Soft threshold accuracy")
ax2.plot(thresh_vals, esc_soft,  color=COLORS["Bandit"],
         marker="s", linewidth=2, linestyle="--", label="Escalation rate")
ax1.axhline(bin_acc, color=COLORS["Disagree thresh"], linewidth=1.5,
            linestyle=":", alpha=0.7, label=f"Binary disagree acc ({bin_acc:.3f})")
ax1.set_xlabel("Prob gap threshold")
ax1.set_ylabel("Accuracy", color=COLORS["Disagree thresh"])
ax2.set_ylabel("Escalation rate", color=COLORS["Bandit"])
ax1.set_ylim(0.3, 0.85)
ax2.set_ylim(0.0, 1.1)
ax1.set_title("Binary vs soft disagreement threshold (GenBuster)",
              fontsize=11, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(BASE + "plot_disagree_binary_vs_soft.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_disagree_binary_vs_soft.png")

# Plot 2: CLIP-only vs disagree threshold vs bandit across datasets
fig, ax = plt.subplots(figsize=(10, 5))
dataset_labels = ["GenBuster", "GenImage SD 1.4", "GenImage BigGAN"]
x = np.arange(3)
w = 0.2

clip_accs    = [r["Accuracy"] for r in clip_only_rows]
disagree_accs = [summary_gb[summary_gb["Method"]=="Disagree thresh"]["Accuracy"].values[0],
                 summary_sd[summary_sd["Method"]=="Disagree thresh"]["Accuracy"].values[0],
                 summary_bg[summary_bg["Method"]=="Disagree thresh"]["Accuracy"].values[0]]
bandit_accs  = [summary_gb[summary_gb["Method"]=="Bandit"]["Accuracy"].values[0],
                summary_sd[summary_sd["Method"]=="Bandit"]["Accuracy"].values[0],
                summary_bg[summary_bg["Method"]=="Bandit"]["Accuracy"].values[0]]
cnn_accs     = [summary_gb[summary_gb["Method"]=="CNN-only"]["Accuracy"].values[0],
                summary_sd[summary_sd["Method"]=="CNN-only"]["Accuracy"].values[0],
                summary_bg[summary_bg["Method"]=="CNN-only"]["Accuracy"].values[0]]

b1 = ax.bar(x - 1.5*w, cnn_accs,      w, label="CNN-only",
            color=COLORS["CNN-only"], alpha=0.9)
b2 = ax.bar(x - 0.5*w, clip_accs,     w, label="CLIP-only",
            color=COLORS["Entropy thresh"], alpha=0.9)
b3 = ax.bar(x + 0.5*w, bandit_accs,   w, label="Bandit",
            color=COLORS["Bandit"], alpha=0.9)
b4 = ax.bar(x + 1.5*w, disagree_accs, w, label="Disagree thresh",
            color=COLORS["Disagree thresh"], alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(dataset_labels)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.3, 1.05)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=9, ncol=2)
ax.set_title("CNN-only vs CLIP-only vs Bandit vs Disagree threshold",
             fontsize=11, fontweight="bold")
for bars in [b1, b2, b3, b4]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(BASE + "plot_clip_vs_disagree.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_clip_vs_disagree.png")


# ══════════════════════════════════════════════
# LATEX TABLES
# ══════════════════════════════════════════════
print("\n% Binary vs soft disagreement table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Binary versus soft disagreement threshold on "
      r"GenBuster-200K-mini. The soft threshold is applied to "
      r"$|p_{\text{CLIP}} - p_{\text{CNN}}|$.}")
print(r"\label{tab:disagree_binary_soft}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Method & Accuracy & Avg Cost & Esc. Rate \\")
print(r"\midrule")
print(f"Binary disagree ($y_{{\\text{{cnn}}}} \\neq y_{{\\text{{clip}}}}$) & "
      f"{bin_acc:.3f} & {bin_cost:.3f} & {bin_esc:.2f} \\\\")
print(r"\midrule")
for r in soft_rows:
    marker = r" $\leftarrow$ selected" if r["Threshold"] == 0.0 else ""
    print(f"Soft thresh $> {r['Threshold']}$ & {r['Accuracy']:.3f} & "
          f"{r['Avg Cost']:.3f} & {r['Esc Rate']:.2f}{marker} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% Cross-dataset disagreement transfer table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Cross-dataset disagreement rule accuracy. No retraining "
      r"is required as the disagreement threshold has no trainable parameters.}")
print(r"\label{tab:disagree_transfer}")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Dataset & Binary disagree & Soft thresh ($> 0.0$) \\")
print(r"\midrule")
for r in cross_disagree_rows:
    print(f"{r['Dataset']} & {r['Binary']:.3f} & {r['Soft=0.0']:.3f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% CLIP-only comparison table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{CLIP-only accuracy evaluated in the same routing setup "
      r"as all other methods. CLIP-only applies no escalation.}")
print(r"\label{tab:clip_only}")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Dataset & CLIP-only Accuracy & Avg Cost \\")
print(r"\midrule")
for r in clip_only_rows:
    print(f"{r['Dataset']} & {r['Accuracy']:.3f} & {r['Avg Cost']:.3f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\nAll fixes complete.")
#end
print("\nAll ablations complete.")
print("\nDone. All plots saved to", BASE)
