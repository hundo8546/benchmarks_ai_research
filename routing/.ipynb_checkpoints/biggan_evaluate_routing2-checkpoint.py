import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample

# ============================================
# COST MODEL
# ============================================
# Latency for all detectors; tokens additionally for VLM.
# CNN and CLIP run sequentially in our implementation (independent, order-invariant).
# Measured on NVIDIA RTX A4500.
LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_CNN_MS + LATENCY_CLIP_MS  # 56.2ms

# ============================================
# LOAD DATA
# ============================================
BASE = "/workspace/benchmarks_ai_research/routing/"

df_cnn = pd.read_csv(BASE + "biggan_bandit_dataset.csv")
df_qwen = pd.read_csv(BASE + "biggan_qwen_preds.csv")
df_clip = pd.read_csv(BASE + "biggan_clip_preds.csv")

df = df_cnn.merge(
    df_qwen[["path", "y_qwen", "y_qwen_text", "p_fake_qwen", "qwen_margin",
             "qwen_input_tokens", "qwen_output_tokens", "latency_qwen_ms"]],
    on="path"
)
df = df.merge(df_clip[["path", "y_clip", "clip_proba"]], on="path")

val_paths = np.load(BASE + "biggan_bandit_val_paths.npy", allow_pickle=True)
df = df[df["path"].isin(val_paths)].reset_index(drop=True)

# Derived features
df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
df["clip_margin"] = np.abs(df["clip_proba"] - 0.5)
eps = 1e-12
df["clip_entropy"] = -(
    df["clip_proba"] * np.log(df["clip_proba"] + eps) +
    (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + eps)
)
df["prob_gap"] = np.abs(df["confidence"] - df["clip_proba"])

# Fill NaN for safety
df["qwen_input_tokens"] = df["qwen_input_tokens"].fillna(0)
df["qwen_output_tokens"] = df["qwen_output_tokens"].fillna(0)
df["latency_qwen_ms"] = df["latency_qwen_ms"].fillna(0)
df["p_fake_qwen"] = df["p_fake_qwen"].fillna(0.5)

print(f"Evaluating on {len(df)} samples")
print(f"CNN predicted fake rate: {df['y_cnn'].mean():.3f}")
print(f"CLIP predicted fake rate: {df['y_clip'].mean():.3f}")
print(f"Qwen predicted fake rate: {df['y_qwen'].mean():.3f}")

# ============================================
# HELPER: EVALUATE A ROUTING POLICY
# ============================================
def evaluate_policy(df, get_action):
    """
    get_action(row) -> (pred, invoke_vlm: bool)
    Returns: dict with accuracy, latency, tokens, etc.
    """
    correct = []
    latencies = []
    in_tokens = []
    out_tokens = []

    for _, row in df.iterrows():
        pred, invoke_vlm = get_action(row)
        if invoke_vlm:
            latency = LATENCY_CHEAP_MS + row["latency_qwen_ms"]
            in_tok = row["qwen_input_tokens"]
            out_tok = row["qwen_output_tokens"]
        else:
            latency = LATENCY_CHEAP_MS
            in_tok = 0
            out_tok = 0
        correct.append(int(pred == row["y_true"]))
        latencies.append(latency)
        in_tokens.append(in_tok)
        out_tokens.append(out_tok)

    return {
        "correct": correct,
        "accuracy": np.mean(correct),
        "mean_latency_ms": np.mean(latencies),
        "mean_input_tokens": np.mean(in_tokens),
        "mean_output_tokens": np.mean(out_tokens),
        "total_output_tokens": np.sum(out_tokens),
        "escalation_rate": np.mean([l > LATENCY_CHEAP_MS for l in latencies]),
    }

def bootstrap_ci(correct, n_bootstrap=1000, ci=95):
    scores = []
    for _ in range(n_bootstrap):
        sample = resample(correct, random_state=None)
        scores.append(np.mean(sample))
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper

# ============================================
# METHOD DEFINITIONS
# ============================================

# CNN-only: no CLIP, no Qwen. Latency is just CNN.
cnn_only_correct = (df["y_cnn"] == df["y_true"]).astype(int).tolist()
cnn_only_acc = np.mean(cnn_only_correct)
cnn_only_latency = LATENCY_CNN_MS

# CLIP-only: no CNN, no Qwen. Latency is just CLIP.
clip_only_correct = (df["y_clip"] == df["y_true"]).astype(int).tolist()
clip_only_acc = np.mean(clip_only_correct)
clip_only_latency = LATENCY_CLIP_MS

# Always escalate to Qwen
always_esc = evaluate_policy(df, lambda r: (r["y_qwen"], True))

# Entropy threshold (tau = 0.5)
TAU = 0.5
entropy_result = evaluate_policy(
    df,
    lambda r: (r["y_qwen"] if r["entropy1"] > TAU else r["y_cnn"],
               r["entropy1"] > TAU)
)

# Disagreement threshold -> Qwen
disagree_thresh_result = evaluate_policy(
    df,
    lambda r: (r["y_qwen"] if r["disagree"] == 1 else r["y_cnn"],
               r["disagree"] == 1)
)

# Disagreement-aware ensemble: CLIP on disagreement, no VLM
ensemble_correct = []
for _, row in df.iterrows():
    pred = row["y_clip"] if row["disagree"] == 1 else row["y_cnn"]
    ensemble_correct.append(int(pred == row["y_true"]))
ensemble_acc = np.mean(ensemble_correct)
ensemble_latency = LATENCY_CHEAP_MS  # no VLM ever invoked

# Bandit router
bandit, scaler = joblib.load("bandit_model.pkl")

def bandit_action(row):
    context = np.array([[
        row["confidence"], row["margin1"], row["entropy1"], row["logit"], row["y_cnn"],
        row["y_clip"], row["clip_proba"], row["clip_margin"], row["clip_entropy"],
        row["disagree"], row["prob_gap"]
    ]])
    context = scaler.transform(context)
    action = bandit.predict(context)[0]
    pred = row["y_qwen"] if action == 1 else row["y_cnn"]
    return pred, bool(action == 1)

bandit_result = evaluate_policy(df, bandit_action)

# ============================================
# MATCHED-RATE ENTROPY (NEW - for fair comparison)
# ============================================
disagree_rate = df["disagree"].mean()
tau_matched = np.quantile(df["entropy1"], 1 - disagree_rate)

entropy_matched_result = evaluate_policy(
    df,
    lambda r: (r["y_qwen"] if r["entropy1"] > tau_matched else r["y_cnn"],
               r["entropy1"] > tau_matched)
)

# ============================================
# RESULTS TABLE
# ============================================
print("\n" + "="*100)
print(f"{'Method':<30s}{'Acc':>8s}{'95% CI':>20s}{'Lat(ms)':>12s}{'In tok':>12s}{'Out tok':>12s}{'Esc':>8s}")
print("="*100)

def print_row(name, correct, mean_latency, mean_in=0, mean_out=0, esc_rate=0.0):
    acc = np.mean(correct)
    lo, hi = bootstrap_ci(correct)
    print(f"{name:<30s}{acc:>8.3f}  [{lo:.3f}, {hi:.3f}]   "
          f"{mean_latency:>10.1f}  {mean_in:>10.1f}  {mean_out:>10.1f}  {esc_rate:>6.2f}")

print_row("CNN-only", cnn_only_correct, cnn_only_latency)
print_row("CLIP-only", clip_only_correct, clip_only_latency)
print_row("Disagree ensemble", ensemble_correct, ensemble_latency)
print_row("Entropy thresh (tau=0.5)", entropy_result["correct"],
          entropy_result["mean_latency_ms"], entropy_result["mean_input_tokens"],
          entropy_result["mean_output_tokens"], entropy_result["escalation_rate"])
print_row(f"Entropy matched rate ({disagree_rate:.2f})", entropy_matched_result["correct"],
          entropy_matched_result["mean_latency_ms"], entropy_matched_result["mean_input_tokens"],
          entropy_matched_result["mean_output_tokens"], entropy_matched_result["escalation_rate"])
print_row("Bandit", bandit_result["correct"],
          bandit_result["mean_latency_ms"], bandit_result["mean_input_tokens"],
          bandit_result["mean_output_tokens"], bandit_result["escalation_rate"])
print_row("Disagree thresh", disagree_thresh_result["correct"],
          disagree_thresh_result["mean_latency_ms"], disagree_thresh_result["mean_input_tokens"],
          disagree_thresh_result["mean_output_tokens"], disagree_thresh_result["escalation_rate"])
print_row("Always escalate", always_esc["correct"],
          always_esc["mean_latency_ms"], always_esc["mean_input_tokens"],
          always_esc["mean_output_tokens"], always_esc["escalation_rate"])

print("="*100)

# ============================================
# SANITY CHECKS
# ============================================
print("\n=== Sanity checks ===")
print(f"Qwen fake recall: {np.mean(df[df['y_true']==1]['y_qwen'] == 1):.3f}")
print(f"Qwen real recall: {np.mean(df[df['y_true']==0]['y_qwen'] == 0):.3f}")
print(f"Mean Qwen latency: {df['latency_qwen_ms'].mean():.1f} ms")
print(f"Mean Qwen input tokens: {df['qwen_input_tokens'].mean():.1f}")
print(f"Mean Qwen output tokens: {df['qwen_output_tokens'].mean():.1f}")
print(f"Disagreement rate: {disagree_rate:.3f}")
print(f"Matched-rate entropy tau: {tau_matched:.4f}")
print(f"Ensemble vs CLIP differ on {sum(np.array(ensemble_correct) != np.array(clip_only_correct))} samples")