import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample

LAMBDA = 0.05
K1 = 1.0
K2 = 5.0
BASE = "/workspace/benchmarks_ai_research/routing/"
df_cnn  = pd.read_csv(BASE + "biggan_bandit_dataset.csv")
df_qwen = pd.read_csv(BASE + "biggan_qwen_preds.csv")
df_clip = pd.read_csv(BASE + "biggan_clip_preds.csv")

df = df_cnn.merge(df_qwen[["path", "y_qwen"]], on="path")
df = df.merge(df_clip[["path", "y_clip", "clip_proba"]], on="path")


val_paths = np.load(BASE + "biggan_bandit_val_paths.npy", allow_pickle=True)


df = df[df["path"].isin(val_paths)].reset_index(drop=True)

# Add these lines here
df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
df["clip_margin"] = np.abs(df["clip_proba"] - 0.5)
eps = 1e-12
df["clip_entropy"] = -(
    df["clip_proba"] * np.log(df["clip_proba"] + eps) +
    (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + eps)
)
df["prob_gap"] = np.abs(df["confidence"] - df["clip_proba"])


df = df[df["path"].isin(val_paths)].reset_index(drop=True)




print("CNN predicted fake rate:", np.mean(df["y_cnn"]))
# Load trained bandit
bandit, scaler = joblib.load(BASE+"biggan_bandit_model.pkl")
# and before predict:

# CNN-only
cnn_acc = np.mean(df["y_cnn"] == df["y_true"])
cnn_cost = K1

# Always escalate
always_acc = np.mean(df["y_qwen"] == df["y_true"])
always_cost = K1 + K2

# Bandit routing
correct = []
costs = []

# for _, row in df.iterrows():
#     context = np.array([[row["confidence"], row["margin1"], row["entropy1"],row["logit"],row["y_cnn"]]])
#     action = bandit.predict(context)[0]

#     if action == 0:
#         # stop
#         pred = row["y_cnn"]
#         cost = K1
#     else:
#         # escalate
#         pred = row["y_qwen"]
#         cost = K1 + K2

#     correct.append(pred == row["y_true"])
#     costs.append(cost)


results = []
for _, row in df.iterrows():
    context = np.array([[
    row["confidence"], row["margin1"], row["entropy1"], row["logit"], row["y_cnn"],
    row["y_clip"], row["clip_proba"], row["clip_margin"], row["clip_entropy"],
    row["disagree"], row["prob_gap"]
]])
    context = scaler.transform(context)
    action = bandit.predict(context)[0]
    if action == 0:
        pred = row["y_cnn"]
        cost = K1
    else:
        pred = row["y_qwen"]
        cost = K1 + K2
    correct.append(pred == row["y_true"])
    costs.append(cost)
    results.append({"path": row["path"], "escalate": action})

pd.DataFrame(results).to_csv(BASE + "biggan_bandit_decisions.csv", index=False)

bandit_acc = np.mean(correct)
bandit_cost = np.mean(costs)


# Threshold baseline: escalate if entropy > tau
TAU = 0.5
correct_thresh = []
costs_thresh = []
for _, row in df.iterrows():
    if row["entropy1"] > TAU:
        pred = row["y_qwen"]
        cost = K1 + K2
    else:
        pred = row["y_cnn"]
        cost = K1
    correct_thresh.append(pred == row["y_true"])
    costs_thresh.append(cost)

thresh_acc = np.mean(correct_thresh)
thresh_cost = np.mean(costs_thresh)

# Disagreement threshold baseline: escalate if CNN and CLIP disagree
correct_disagree = []
costs_disagree = []
for _, row in df.iterrows():
    if row["disagree"] == 1:
        pred = row["y_qwen"]
        cost = K1 + K2
    else:
        pred = row["y_cnn"]
        cost = K1
    correct_disagree.append(pred == row["y_true"])
    costs_disagree.append(cost)

disagree_acc = np.mean(correct_disagree)
disagree_cost = np.mean(costs_disagree)


def bootstrap_ci(correct, n_bootstrap=1000, ci=95):
    scores = []
    for _ in range(n_bootstrap):
        sample = resample(correct, random_state=None)
        scores.append(np.mean(sample))
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper



# Framework-CNN versions (CNN prediction on stop branch)
correct_cnn_stop, costs_cnn_stop = [], []
for _, row in df.iterrows():
    context = np.array([[
        row["confidence"], row["margin1"], row["entropy1"], row["logit"], row["y_cnn"],
        row["y_clip"], row["clip_proba"], row["clip_margin"], row["clip_entropy"],
        row["disagree"], row["prob_gap"]
    ]])
    context = scaler.transform(context)
    action = bandit.predict(context)[0]
    pred = row["y_cnn"] if action == 0 else row["y_qwen"]
    correct_cnn_stop.append(pred == row["y_true"])
    costs_cnn_stop.append(K1 if action == 0 else K1 + K2)

correct_disagree_cnn, costs_disagree_cnn = [], []
for _, row in df.iterrows():
    if row["disagree"] == 1:
        pred = row["y_qwen"]
        cost = K1 + K2
    else:
        pred = row["y_cnn"]
        cost = K1
    correct_disagree_cnn.append(pred == row["y_true"])
    costs_disagree_cnn.append(cost)

# CLIP-only baseline
clip_only_correct = (df["y_clip"] == df["y_true"]).astype(int).tolist()

bandit_cnn_lo, bandit_cnn_hi = bootstrap_ci(correct_cnn_stop)
disagree_cnn_lo, disagree_cnn_hi = bootstrap_ci(correct_disagree_cnn)
clip_lo, clip_hi = bootstrap_ci(clip_only_correct)

print("CNN-only accuracy:", cnn_acc, "cost:", cnn_cost)
print("Always escalate accuracy:", always_acc, "cost:", always_cost)
print("Bandit accuracy:", bandit_acc, "avg cost:", bandit_cost)
print("Qwen fake accuracy:",
      np.mean(df[df["y_true"]==1]["y_qwen"] ==
              df[df["y_true"]==1]["y_true"]))
print("Qwen real accuracy:",
      np.mean(df[df["y_true"]==0]["y_qwen"] ==
              df[df["y_true"]==0]["y_true"]))
print("Qwen predicted fake rate:", np.mean(df["y_qwen"]==1))
print("Entropy threshold accuracy:", thresh_acc, "cost:", thresh_cost)
print("Disagreement threshold accuracy:", disagree_acc, "cost:", disagree_cost)

# Disagreement-aware CNN/CLIP ensemble (no Qwen needed)
correct_ensemble = []
costs_ensemble = []
for _, row in df.iterrows():
    if row["disagree"] == 0:
        pred = row["y_cnn"]  # models agree, trust CNN
        cost = K1
    else:
        pred = row["y_clip"]  # models disagree, trust CLIP
        cost = K1  # CLIP already ran, no extra cost
    correct_ensemble.append(int(pred == row["y_true"]))
    costs_ensemble.append(cost)

ensemble_acc = np.mean(correct_ensemble)
ensemble_cost = np.mean(costs_ensemble)
ensemble_lo, ensemble_hi = bootstrap_ci(correct_ensemble)
print(f"Disagree-aware ensemble: {ensemble_acc:.3f} [{ensemble_lo:.3f}, {ensemble_hi:.3f}] cost: {ensemble_cost:.3f}")





# After computing all accuracies, add:
cnn_correct = (df["y_cnn"] == df["y_true"]).astype(int).tolist()
always_correct = (df["y_qwen"] == df["y_true"]).astype(int).tolist()

cnn_lo, cnn_hi = bootstrap_ci(cnn_correct)
always_lo, always_hi = bootstrap_ci(always_correct)
bandit_lo, bandit_hi = bootstrap_ci(correct)
thresh_lo, thresh_hi = bootstrap_ci(correct_thresh)
disagree_lo, disagree_hi = bootstrap_ci(correct_disagree)

print(f"CNN-only:     {cnn_acc:.3f} [{cnn_lo:.3f}, {cnn_hi:.3f}]")
print(f"Always esc:   {always_acc:.3f} [{always_lo:.3f}, {always_hi:.3f}]")
print(f"Bandit:       {bandit_acc:.3f} [{bandit_lo:.3f}, {bandit_hi:.3f}]")
print(f"Entropy thresh: {thresh_acc:.3f} [{thresh_lo:.3f}, {thresh_hi:.3f}]")
print(f"Disagree thresh: {disagree_acc:.3f} [{disagree_lo:.3f}, {disagree_hi:.3f}]")


print("\n=== Framework comparison ===")
print(f"CLIP-only:              {np.mean(clip_only_correct):.3f} [{clip_lo:.3f}, {clip_hi:.3f}] cost: 1.0")
print(f"Framework-CNN bandit:   {np.mean(correct_cnn_stop):.3f} [{bandit_cnn_lo:.3f}, {bandit_cnn_hi:.3f}] cost: {np.mean(costs_cnn_stop):.3f}")
print(f"Framework-CLIP bandit:  {bandit_acc:.3f} [{bandit_lo:.3f}, {bandit_hi:.3f}] cost: {bandit_cost:.3f}")
print(f"Framework-CNN disagree: {np.mean(correct_disagree_cnn):.3f} [{disagree_cnn_lo:.3f}, {disagree_cnn_hi:.3f}] cost: {np.mean(costs_disagree_cnn):.3f}")
print(f"Framework-CLIP disagree:{disagree_acc:.3f} [{disagree_lo:.3f}, {disagree_hi:.3f}] cost: {disagree_cost:.3f}")


print(df["y_true"].value_counts())
print(df["y_cnn"].value_counts())
print(df["y_clip"].value_counts())