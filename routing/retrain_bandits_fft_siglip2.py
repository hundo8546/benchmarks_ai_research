"""
Retrain bandit routers on FFT + SigLIP2 features (fixing the Table 15 vs 16
unfair-comparison issue raised in review): the original bandit
(retrain_bandits.py) was trained on CNNSpot + CLIP features while the
disagreement rule it's compared against uses FFT + SigLIP2 + Qwen. This
script retrains the bandit with the same FFT + SigLIP2 backbone so both
routing policies are compared on equal footing.

Mirrors retrain_bandits.py's feature engineering and reward formulation,
substituting:
  y_cnn, confidence, margin1, entropy1, logit   -> FFT probe outputs
  y_clip, clip_proba                            -> SigLIP2 probe outputs

Produces {prefix}bandit_model_fft_siglip2.pkl per dataset.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extractors"))
from extractors.common import DATASETS as DS_CFG, ROUTING_BASE, load_val_paths

BASE = "/workspace/benchmarks_ai_research/routing/"
EPS = 1e-12
BANDIT_LAMBDA = 0.05

LATENCY_FFT_MS = 2.5
LATENCY_SIGLIP2_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_FFT_MS + LATENCY_SIGLIP2_MS

FULL_BANDIT_FEATURES = [
    "confidence",
    "margin1",
    "entropy1",
    "logit",
    "y_fft",
    "y_siglip2",
    "siglip2_proba",
    "siglip2_margin",
    "siglip2_entropy",
    "disagree",
    "prob_gap",
]

# (dataset_key, prefix, qwen_filename, val_paths_file, output_pickle)
DATASETS = [
    ("GenBuster", "",        "qwen_preds.csv",        "bandit_val_paths.npy",        "bandit_model_fft_siglip2.pkl"),
    ("SD14",      "sd14_",   "sd14_qwen_preds.csv",   "sd14_bandit_val_paths.npy",   "sd14_bandit_model_fft_siglip2.pkl"),
    ("BigGAN",    "biggan_", "biggan_qwen_preds.csv", "biggan_bandit_val_paths.npy", "biggan_bandit_model_fft_siglip2.pkl"),
]


def _binary_entropy(p):
    return -(p * np.log(p + EPS) + (1 - p) * np.log(1 - p + EPS))


def get_fft_frame(ds_name, val_paths):
    """Apply the cached in-domain FFT probe to every sample (train+val)."""
    cfg = DS_CFG[ds_name]
    prefix = cfg["prefix"]
    feat_file = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")
    probe_pkl = os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl")

    df = pd.read_csv(feat_file)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)

    clf, scaler = joblib.load(probe_pkl)
    Xs = scaler.transform(X)
    y_fft = clf.predict(Xs).astype(int)
    p_fake = clf.predict_proba(Xs)[:, 1]
    logit = clf.decision_function(Xs)

    out = pd.DataFrame({
        "path": df["path"],
        "y_fft": y_fft,
        "confidence": np.maximum(p_fake, 1 - p_fake),
        "margin1": np.abs(p_fake - 0.5),
        "entropy1": _binary_entropy(p_fake),
        "logit": logit,
    })
    return out


def get_siglip2_frame(ds_name):
    """Apply the cached in-domain SigLIP2 probe to every sample (train+val)."""
    cfg = DS_CFG[ds_name]
    prefix = cfg["prefix"]
    feat_file = os.path.join(ROUTING_BASE, f"{prefix}siglip2_features.csv")
    probe_pkl = os.path.join(ROUTING_BASE, f"{prefix}siglip2_probe.pkl")

    df = pd.read_csv(feat_file)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)

    clf, scaler = joblib.load(probe_pkl)
    Xs = scaler.transform(X)
    y_siglip2 = clf.predict(Xs).astype(int)
    p_fake = clf.predict_proba(Xs)[:, 1]

    out = pd.DataFrame({
        "path": df["path"],
        "y_true": df["y_true"].astype(int),
        "y_siglip2": y_siglip2,
        "siglip2_proba": p_fake,
        "siglip2_margin": np.abs(p_fake - 0.5),
        "siglip2_entropy": _binary_entropy(p_fake),
    })
    return out


def load_dataset(ds_name, qwen_filename, val_paths_file):
    cfg = DS_CFG[ds_name]
    val_paths = np.load(os.path.join(BASE, val_paths_file), allow_pickle=True)

    df_fft = get_fft_frame(ds_name, val_paths)
    df_sig = get_siglip2_frame(ds_name)
    df_qwen = pd.read_csv(os.path.join(BASE, qwen_filename))

    if "y_qwen" not in df_qwen.columns:
        raise RuntimeError(f"{qwen_filename} does not have a y_qwen column.")
    qwen_cols = [c for c in ["path", "y_qwen", "latency_qwen_ms"] if c in df_qwen.columns]

    df = df_sig.merge(df_fft, on="path", how="inner")
    df = df.merge(df_qwen[qwen_cols], on="path", how="inner")

    df["disagree"] = (df["y_fft"] != df["y_siglip2"]).astype(int)
    df["prob_gap"] = np.abs(df["confidence"].where(df["y_fft"] == 1, 1 - df["confidence"])
                             - df["siglip2_proba"])

    df["latency_fft_ms"] = LATENCY_FFT_MS
    df["latency_siglip2_ms"] = LATENCY_SIGLIP2_MS
    df["latency_stop_ms"] = LATENCY_CHEAP_MS
    df["latency_escalate_ms"] = LATENCY_CHEAP_MS + df["latency_qwen_ms"]

    return df, val_paths


def compute_bandit_targets(df: pd.DataFrame, lam: float = BANDIT_LAMBDA) -> pd.DataFrame:
    df = df.copy()
    max_cost = float(df["latency_escalate_ms"].max())
    cost_stop = df["latency_stop_ms"] / max_cost
    cost_esc = df["latency_escalate_ms"] / max_cost

    df["r_stop"] = np.where(df["y_fft"] == df["y_true"], 1.0, -1.0) - lam * cost_stop
    df["r_esc"] = np.where(df["y_qwen"] == df["y_true"], 1.0, -1.0) - lam * cost_esc
    df["target"] = (df["r_esc"] > df["r_stop"]).astype(int)
    return df


def train_bandit(df: pd.DataFrame, val_paths: np.ndarray):
    df = compute_bandit_targets(df)
    train_mask = ~df["path"].isin(val_paths)
    train_idx = df[train_mask].index.values

    internal_split = False
    if len(train_idx) == 0:
        # Qwen predictions don't cover the training-split portion of this
        # dataset (only the held-out val set). Fall back to an internal
        # 70/30 split of the available (val-covered) rows so the bandit can
        # still be fit; this deviates from the standard train/val protocol
        # and should be caveated wherever these numbers are reported.
        internal_split = True
        rng = np.random.default_rng(42)
        all_idx = df.index.values
        n_train = int(len(all_idx) * 0.7)
        perm = rng.permutation(all_idx)
        train_idx = perm[:n_train]
        print(f"  WARNING: no Qwen coverage outside the val split; using an "
              f"internal 70/30 split of the {len(all_idx)} available rows "
              f"instead ({n_train} train / {len(all_idx) - n_train} held out).")

    X = df[FULL_BANDIT_FEATURES].values
    y = df["target"].values

    X_train = X[train_idx]
    y_train = y[train_idx]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            f"Only one class in training targets (all {np.unique(y_train)}). "
            f"Cannot train bandit."
        )

    pos_rate = np.mean(y_train)
    print(f"  Training samples: {len(y_train)}, positive (escalate) rate: {pos_rate:.3f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    print(f"  In-sample target prediction accuracy: {train_acc:.3f}")

    eval_idx = None
    if internal_split:
        eval_idx = np.setdiff1d(df.index.values, train_idx)

    return model, scaler, internal_split, eval_idx


def evaluate_bandit(df: pd.DataFrame, eval_idx, model, scaler):
    val_df = df.loc[eval_idx].reset_index(drop=True)
    X_val = val_df[FULL_BANDIT_FEATURES].values
    X_val_scaled = scaler.transform(X_val)
    actions = model.predict(X_val_scaled).astype(int)

    preds = np.where(actions == 1, val_df["y_qwen"].values, val_df["y_fft"].values)
    correct = (preds == val_df["y_true"].values).astype(int)

    acc = float(np.mean(correct))
    esc_rate = float(np.mean(actions))
    return acc, esc_rate, val_df


def main():
    print("=" * 60)
    print("Retraining bandits on FFT + SigLIP2 + Qwen features")
    print("=" * 60)

    for ds_name, prefix, qwen_filename, val_paths_file, output_pickle in DATASETS:
        print(f"\n--- {ds_name} ---")
        try:
            df, val_paths = load_dataset(ds_name, qwen_filename, val_paths_file)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  SKIPPED: {e}")
            continue

        print(f"  Loaded {len(df)} total samples, {len(val_paths)} val samples")
        print(f"  Qwen fake rate: {df['y_qwen'].mean():.3f}")
        print(f"  Disagree rate (FFT vs SigLIP2): {df['disagree'].mean():.3f}")

        model, scaler, internal_split, eval_idx = train_bandit(df, val_paths)
        if internal_split:
            eval_label = "internal held-out split (NOT the official val set)"
        else:
            eval_idx = df[df["path"].isin(val_paths)].index.values
            eval_label = "official val split"

        val_acc, val_esc, eval_df = evaluate_bandit(df, eval_idx, model, scaler)
        print(f"  {eval_label}: bandit accuracy={val_acc:.3f}  escalation_rate={val_esc:.3f}")

        # Disagreement-rule accuracy on the same evaluation rows, for direct comparison
        dis_pred = np.where(eval_df["disagree"] == 1, eval_df["y_qwen"], eval_df["y_fft"])
        dis_acc = float(np.mean(dis_pred == eval_df["y_true"]))
        dis_esc = float(eval_df["disagree"].mean())
        print(f"  {eval_label}: disagreement-rule accuracy={dis_acc:.3f}  escalation_rate={dis_esc:.3f}")
        if internal_split:
            print(f"  CAVEAT: {ds_name} Qwen predictions only cover the official val "
                  f"split, so this bandit was fit/evaluated on an internal split of "
                  f"those same 400 samples rather than the training-split protocol "
                  f"used for SD14/BigGAN. Not directly comparable to the other two "
                  f"datasets' numbers without a Qwen rerun on GenBuster's training split.")

        output_path = os.path.join(BASE, output_pickle)
        joblib.dump((model, scaler), output_path)
        print(f"  Saved {output_pickle}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
