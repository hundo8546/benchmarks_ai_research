import pandas as pd
import numpy as np

BASE = "/workspace/benchmarks_ai_research/routing/"

# Per-dataset target escalation rates (from the disagreement rule)
DATASETS = {
    "GenBuster": {
        "file": BASE + "merged_dataset.csv",
        "target_esc_rate": 0.44,
    },
    "SD14": {
        "file": BASE + "sd14_merged_dataset.csv",       # adjust to your actual path
        "target_esc_rate": 0.49,
    },
    "BigGAN": {
        "file": BASE + "biggan_merged_dataset.csv",     # adjust to your actual path
        "target_esc_rate": 0.28,
    },
}

K1 = 1.0
K2 = 5.0

for name, cfg in DATASETS.items():
    df = pd.read_csv(cfg["file"])
    target_rate = cfg["target_esc_rate"]

    # Pick the entropy threshold that escalates exactly target_rate of samples
    # i.e. the (1 - target_rate) quantile of entropy values
    tau_matched = np.quantile(df["entropy1"], 1 - target_rate)

    # Apply the matched-rate entropy rule
    escalate_mask = df["entropy1"] > tau_matched
    preds = np.where(escalate_mask, df["y_qwen"], df["y_cnn"])
    acc = np.mean(preds == df["y_true"])
    esc_rate = np.mean(escalate_mask)
    cost = np.mean(np.where(escalate_mask, K1 + K2, K1))

    # For comparison: disagreement rule at its natural rate
    dis_preds = np.where(df["disagree"] == 1, df["y_qwen"], df["y_cnn"])
    dis_acc = np.mean(dis_preds == df["y_true"])
    dis_rate = np.mean(df["disagree"] == 1)
    dis_cost = np.mean(np.where(df["disagree"] == 1, K1 + K2, K1))

    print(f"\n=== {name} ===")
    print(f"Target escalation rate: {target_rate:.2f}")
    print(f"Matched-rate entropy threshold tau: {tau_matched:.4f}")
    print(f"Entropy @ matched rate:  acc={acc:.3f}  esc={esc_rate:.3f}  cost={cost:.3f}")
    print(f"Disagreement (reference): acc={dis_acc:.3f}  esc={dis_rate:.3f}  cost={dis_cost:.3f}")
    print(f"Gap (disagree - entropy): {dis_acc - acc:+.3f}")