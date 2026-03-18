import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import resample

BASE = "/workspace/benchmarks_ai_research/routing/"
K1 = 1.0
K2 = 5.0

# ── Load data ──────────────────────────────────────────────────────────────
df_cnn  = pd.read_csv(BASE + "bandit_dataset.csv")
df_qwen = pd.read_csv(BASE + "qwen_preds.csv")
df_clip = pd.read_csv(BASE + "clip_preds.csv")

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
df = df.merge(df_clip[["path", "y_clip", "clip_proba"]], on="path")

val_paths = np.load(BASE + "bandit_val_paths.npy", allow_pickle=True)
df = df[df["path"].isin(val_paths)].reset_index(drop=True)

# Derived features
df["disagree"]    = (df["y_cnn"] != df["y_clip"]).astype(int)
df["clip_margin"] = np.abs(df["clip_proba"] - 0.5)
eps = 1e-12
df["clip_entropy"] = -(
    df["clip_proba"] * np.log(df["clip_proba"] + eps) +
    (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + eps)
)
df["prob_gap"] = np.abs(df["confidence"] - df["clip_proba"])

df["cnn_correct"]  = (df["y_cnn"]  == df["y_true"]).astype(int)
df["clip_correct"] = (df["y_clip"] == df["y_true"]).astype(int)
df["qwen_correct"] = (df["y_qwen"] == df["y_true"]).astype(int)

# ── Bootstrap CI helper ────────────────────────────────────────────────────
def bootstrap_ci(correct, n=1000, ci=95):
    scores = [np.mean(resample(correct)) for _ in range(n)]
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return lo, hi

# ── Routing methods ────────────────────────────────────────────────────────
bandit, scaler = joblib.load("bandit_model.pkl")

def run_bandit(df):
    correct, costs, escalated = [], [], []
    for _, row in df.iterrows():
        ctx = scaler.transform([[
            row["confidence"], row["margin1"], row["entropy1"], row["logit"], row["y_cnn"],
            row["y_clip"], row["clip_proba"], row["clip_margin"], row["clip_entropy"],
            row["disagree"], row["prob_gap"]
        ]])
        action = bandit.predict(ctx)[0]
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        cost = K1 if action == 0 else K1 + K2
        correct.append(int(pred == row["y_true"]))
        costs.append(cost)
        escalated.append(action)
    return correct, costs, escalated

def run_disagree_thresh(df):
    correct, costs, escalated = [], [], []
    for _, row in df.iterrows():
        action = int(row["disagree"] == 1)
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        cost = K1 if action == 0 else K1 + K2
        correct.append(int(pred == row["y_true"]))
        costs.append(cost)
        escalated.append(action)
    return correct, costs, escalated

def run_entropy_thresh(df, tau=0.5):
    correct, costs, escalated = [], [], []
    for _, row in df.iterrows():
        action = int(row["entropy1"] > tau)
        pred = row["y_cnn"] if action == 0 else row["y_qwen"]
        cost = K1 if action == 0 else K1 + K2
        correct.append(int(pred == row["y_true"]))
        costs.append(cost)
        escalated.append(action)
    return correct, costs, escalated

bandit_correct,  bandit_costs,  bandit_esc  = run_bandit(df)
disagree_correct, disagree_costs, disagree_esc = run_disagree_thresh(df)
entropy_correct, entropy_costs, entropy_esc  = run_entropy_thresh(df)

cnn_correct_list    = df["cnn_correct"].tolist()
always_correct_list = df["qwen_correct"].tolist()

df["bandit_escalate"]   = bandit_esc
df["disagree_escalate"] = disagree_esc

# ── Summary stats ──────────────────────────────────────────────────────────
methods = {
    "CNN-only":         (cnn_correct_list,    [K1]*len(df)),
    "Entropy thresh":   (entropy_correct,     entropy_costs),
    "Always escalate":  (always_correct_list, [K1+K2]*len(df)),
    "Bandit":           (bandit_correct,      bandit_costs),
    "Disagree thresh":  (disagree_correct,    disagree_costs),
}

print("\n=== Summary ===")
print(f"{'Method':<20} {'Acc':>6} {'CI_lo':>6} {'CI_hi':>6} {'Cost':>6}")
for name, (corr, costs) in methods.items():
    acc  = np.mean(corr)
    lo, hi = bootstrap_ci(corr)
    cost = np.mean(costs)
    print(f"{name:<20} {acc:>6.3f} {lo:>6.3f} {hi:>6.3f} {cost:>6.3f}")

# ── 1. Agree vs Disagree accuracy breakdown ────────────────────────────────
agree_df    = df[df["disagree"] == 0]
disagree_df = df[df["disagree"] == 1]

agree_cnn_acc    = agree_df["cnn_correct"].mean()
disagree_cnn_acc = disagree_df["cnn_correct"].mean()
agree_qwen_acc    = agree_df["qwen_correct"].mean()
disagree_qwen_acc = disagree_df["qwen_correct"].mean()

print("\n=== Agree vs Disagree Breakdown ===")
print(f"Agree cases:    {len(agree_df)} samples")
print(f"Disagree cases: {len(disagree_df)} samples")
print(f"CNN acc  — agree: {agree_cnn_acc:.3f}  disagree: {disagree_cnn_acc:.3f}")
print(f"Qwen acc — agree: {agree_qwen_acc:.3f}  disagree: {disagree_qwen_acc:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
w = 0.3
ax.bar(x - w/2, [agree_cnn_acc,  disagree_cnn_acc],  w, label="CNN",  color="#7F77DD")
ax.bar(x + w/2, [agree_qwen_acc, disagree_qwen_acc], w, label="Qwen", color="#1D9E75")
ax.set_xticks(x)
ax.set_xticklabels(["CNN & CLIP agree", "CNN & CLIP disagree"])
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by agreement status")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + "plot_agree_disagree.png", dpi=150)
plt.close()
print("Saved plot_agree_disagree.png")

# ── 2. Escalation behavior breakdown ──────────────────────────────────────
cases = {
    "CNN correct\nCLIP agree":    df[(df["cnn_correct"]==1) & (df["disagree"]==0)],
    "CNN correct\nCLIP disagree": df[(df["cnn_correct"]==1) & (df["disagree"]==1)],
    "CNN wrong\nCLIP agree":      df[(df["cnn_correct"]==0) & (df["disagree"]==0)],
    "CNN wrong\nCLIP disagree":   df[(df["cnn_correct"]==0) & (df["disagree"]==1)],
}

print("\n=== Escalation Behavior ===")
bandit_rates   = []
disagree_rates = []
labels = []
for label, subset in cases.items():
    if len(subset) == 0:
        continue
    b_rate = subset["bandit_escalate"].mean()
    d_rate = subset["disagree_escalate"].mean()
    bandit_rates.append(b_rate)
    disagree_rates.append(d_rate)
    labels.append(label.replace("\n", " / "))
    print(f"{label.replace(chr(10),' / '):<35} n={len(subset):>3}  bandit_esc={b_rate:.2f}  disagree_esc={d_rate:.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, bandit_rates,   w, label="Bandit",            color="#BA7517")
ax.bar(x + w/2, disagree_rates, w, label="Disagree threshold", color="#1D9E75")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Escalation rate")
ax.set_title("Escalation rate by case")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + "plot_escalation_behavior.png", dpi=150)
plt.close()
print("Saved plot_escalation_behavior.png")

# ── 3. Accuracy vs Cost frontier ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors = {
    "CNN-only":        "#888780",
    "Entropy thresh":  "#378ADD",
    "Always escalate": "#D85A30",
    "Bandit":          "#BA7517",
    "Disagree thresh": "#1D9E75",
}
for name, (corr, costs) in methods.items():
    acc  = np.mean(corr)
    lo, hi = bootstrap_ci(corr)
    cost = np.mean(costs)
    ax.errorbar(cost, acc, yerr=[[acc-lo],[hi-acc]],
                fmt="o", color=colors[name], capsize=4,
                markersize=8, label=name)
    ax.annotate(name, (cost, acc),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, color=colors[name])

ax.set_xlabel("Average cost")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs cost frontier")
ax.set_xlim(0, 7)
ax.set_ylim(0.4, 0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + "plot_cost_accuracy.png", dpi=150)
plt.close()
print("Saved plot_cost_accuracy.png")

# ── 4. Confidence interval bar plot ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names, accs, los, his = [], [], [], []
for name, (corr, _) in methods.items():
    acc = np.mean(corr)
    lo, hi = bootstrap_ci(corr)
    names.append(name)
    accs.append(acc)
    los.append(acc - lo)
    his.append(hi - acc)

bar_colors = [colors[n] for n in names]
x = np.arange(len(names))
ax.bar(x, accs, color=bar_colors, alpha=0.85,
       yerr=[los, his], capsize=5, error_kw={"elinewidth": 1.5})
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy with 95% bootstrap confidence intervals")
ax.set_ylim(0.3, 0.95)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(BASE + "plot_ci_bars.png", dpi=150)
plt.close()
print("Saved plot_ci_bars.png")

print("\nAll plots saved to", BASE)
