"""
Retrain bandit routers on new Qwen v2 predictions.

The original bandits were trained on old Qwen outputs (broken text decoding,
near-zero fake recall on static images). After updating to score-decoded Qwen
with the neutral prompt, those bandit policies are mismatched against the new
verifier. This script retrains the bandit on each dataset using the same
feature vector and reward formulation, then saves new pickle files that
analysis2.py will load.

Run this before analysis2.py. Takes ~10 seconds total.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
BASE = "/workspace/benchmarks_ai_research/routing/"
EPS = 1e-12
BANDIT_LAMBDA = 0.05

LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_CNN_MS + LATENCY_CLIP_MS

FULL_BANDIT_FEATURES = [
    "confidence",
    "margin1",
    "entropy1",
    "logit",
    "y_cnn",
    "y_clip",
    "clip_proba",
    "clip_margin",
    "clip_entropy",
    "disagree",
    "prob_gap",
]

# (prefix, qwen_filename, val_paths_file, output_pickle)
DATASETS = [
    ("",        "qwen_preds.csv",        "bandit_val_paths.npy",        "bandit_model.pkl"),
    ("sd14_",   "sd14_qwen_preds.csv",   "sd14_bandit_val_paths.npy",   "sd14_bandit_model.pkl"),
    ("biggan_", "biggan_qwen_preds.csv", "biggan_bandit_val_paths.npy", "biggan_bandit_model.pkl"),
]


# ============================================================
# FEATURE ENGINEERING (mirrors analysis2.py)
# ============================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)

    df["clip_margin"] = np.abs(df["clip_proba"] - 0.5)
    df["clip_entropy"] = -(
        df["clip_proba"] * np.log(df["clip_proba"] + EPS)
        + (1 - df["clip_proba"]) * np.log(1 - df["clip_proba"] + EPS)
    )

    df["p_fake_cnn"] = np.where(df["y_cnn"] == 1, df["confidence"], 1 - df["confidence"])
    df["cnn_margin"] = np.abs(df["p_fake_cnn"] - 0.5)
    df["prob_gap"] = np.abs(df["p_fake_cnn"] - df["clip_proba"])

    # Force fixed cheap-stage latency
    df["latency_cnn_ms"] = LATENCY_CNN_MS
    df["latency_clip_ms"] = LATENCY_CLIP_MS
    df["latency_stop_ms"] = LATENCY_CHEAP_MS
    df["latency_escalate_ms"] = LATENCY_CHEAP_MS + df["latency_qwen_ms"]

    return df


# ============================================================
# LOAD + MERGE
# ============================================================
def load_dataset(prefix: str, qwen_filename: str, val_paths_file: str):
    df_cnn = pd.read_csv(os.path.join(BASE, f"{prefix}bandit_dataset.csv"))
    df_clip = pd.read_csv(os.path.join(BASE, f"{prefix}clip_preds.csv"))
    df_qwen = pd.read_csv(os.path.join(BASE, qwen_filename))

    # Verify we're loading the v2 Qwen file
    if "p_fake_qwen" not in df_qwen.columns:
        raise RuntimeError(
            f"{qwen_filename} does not have p_fake_qwen column. "
            f"You're loading the old Qwen file, not the v2 output. "
            f"Check the filename."
        )

    clip_cols = [c for c in ["path", "y_clip", "clip_proba", "latency_clip_ms"] if c in df_clip.columns]
    qwen_cols = [c for c in ["path", "y_qwen", "latency_qwen_ms"] if c in df_qwen.columns]

    df = df_cnn.merge(df_qwen[qwen_cols], on="path", how="inner")
    df = df.merge(df_clip[clip_cols], on="path", how="inner")

    df = add_derived_features(df)

    val_paths = np.load(os.path.join(BASE, val_paths_file), allow_pickle=True)
    return df, val_paths


# ============================================================
# BANDIT TRAINING
# ============================================================
def compute_bandit_targets(df: pd.DataFrame, lam: float = BANDIT_LAMBDA) -> pd.DataFrame:
    """
    Compute per-sample targets for the bandit by simulating the reward of
    stopping at CNN vs escalating to Qwen, including latency cost.
    Target = 1 if escalate reward > stop reward, else 0.
    """
    df = df.copy()

    max_cost = float(df["latency_escalate_ms"].max())
    cost_stop = df["latency_stop_ms"] / max_cost
    cost_esc = df["latency_escalate_ms"] / max_cost

    df["r_stop"] = np.where(df["y_cnn"] == df["y_true"], 1.0, -1.0) - lam * cost_stop
    df["r_esc"] = np.where(df["y_qwen"] == df["y_true"], 1.0, -1.0) - lam * cost_esc
    df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)

    return df


def train_bandit(df: pd.DataFrame, val_paths: np.ndarray):
    """
    Train a logistic regression bandit on the non-val portion of the dataset.
    Returns (model, scaler) ready to be pickled.
    """
    df = compute_bandit_targets(df)

    train_mask = ~df["path"].isin(val_paths)
    train_idx = df[train_mask].index.values

    X = df[FULL_BANDIT_FEATURES].values
    y = df["target"].values

    X_train = X[train_idx]
    y_train = y[train_idx]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            f"Only one class in training targets (all {np.unique(y_train)}). "
            f"Cannot train bandit — try adjusting LAMBDA or check your data."
        )

    # Class balance info
    pos_rate = np.mean(y_train)
    print(f"  Training samples: {len(y_train)}, positive (escalate) rate: {pos_rate:.3f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    # Quick in-sample sanity
    train_acc = model.score(X_train_scaled, y_train)
    print(f"  In-sample target prediction accuracy: {train_acc:.3f}")

    return model, scaler


def evaluate_bandit_on_val(df: pd.DataFrame, val_paths: np.ndarray, model, scaler):
    """
    Quick sanity-check eval of the trained bandit on the val split.
    Reports routed accuracy and escalation rate.
    """
    val_df = df[df["path"].isin(val_paths)].reset_index(drop=True)

    X_val = val_df[FULL_BANDIT_FEATURES].values
    X_val_scaled = scaler.transform(X_val)
    actions = model.predict(X_val_scaled).astype(int)

    preds = np.where(actions == 1, val_df["y_qwen"].values, val_df["y_cnn"].values)
    correct = (preds == val_df["y_true"].values).astype(int)

    acc = float(np.mean(correct))
    esc_rate = float(np.mean(actions))

    return acc, esc_rate


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Retraining bandits on new Qwen v2 predictions")
    print("=" * 60)

    for prefix, qwen_filename, val_paths_file, output_pickle in DATASETS:
        dataset_name = prefix.rstrip("_") if prefix else "GenBuster"
        print(f"\n--- {dataset_name} ---")

        try:
            df, val_paths = load_dataset(prefix, qwen_filename, val_paths_file)
        except FileNotFoundError as e:
            print(f"  SKIPPED: file not found: {e}")
            continue
        except RuntimeError as e:
            print(f"  SKIPPED: {e}")
            continue

        print(f"  Loaded {len(df)} total samples, {len(val_paths)} val samples")
        print(f"  Qwen fake rate: {df['y_qwen'].mean():.3f}")

        # Optionally back up the old pickle
        output_path = os.path.join(BASE, output_pickle)
        if os.path.exists(output_path):
            backup_path = output_path + ".old_qwen_backup"
            if not os.path.exists(backup_path):
                os.rename(output_path, backup_path)
                print(f"  Backed up old pickle to {os.path.basename(backup_path)}")
            else:
                print(f"  Backup already exists at {os.path.basename(backup_path)}, overwriting current pickle only")

        # Train
        model, scaler = train_bandit(df, val_paths)

        # Eval on val split
        val_acc, val_esc = evaluate_bandit_on_val(df, val_paths, model, scaler)
        print(f"  Val set: accuracy={val_acc:.3f}  escalation_rate={val_esc:.3f}")

        # Save
        joblib.dump((model, scaler), output_path)
        print(f"  Saved {output_pickle}")

    print("\n" + "=" * 60)
    print("Done. Run analysis2.py next to regenerate all tables.")
    print("=" * 60)


if __name__ == "__main__":
    main()