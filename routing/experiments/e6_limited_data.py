"""
E6 — Limited-Data Probe Study.
Tests the deployment regime where disagreement routing becomes useful:
when the semantic detector is trained with only a fraction of available labels.

Hypothesis: at low-data fractions, routing improves over semantic-only because
the semantic probe is weaker and the cheap-path accuracy drops.

Evaluates: Semantic-only vs Disagreement-cascade (semantic + CNNSpot + Qwen)
at train fractions: 1%, 5%, 10%, 25%, 50%, 100%.

Produces: results/e6_limited_data.csv
Success: routing outperforms semantic-only at low-data fractions.

Usage:
    python e6_limited_data.py [--datasets GenBuster SD14 BigGAN]
                              [--backbones CLIP SigLIP2 DINOv2]
"""
import argparse
import gc
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import (
    DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir, bootstrap_ci,
    LATENCY_CHEAP_MS, load_val_paths,
)

LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

BACKBONE_MAP = {
    "CLIP": {"feat_prefix": "clip_features", "latency_col": "latency_clip_ms"},
    "SigLIP2": {"feat_prefix": "siglip2_features", "latency_col": "latency_siglip2_ms"},
    "DINOv2": {"feat_prefix": "dinov2_features", "latency_col": "latency_dinov2_ms"},
}


def load_and_split_features(feat_file, val_paths_set):
    pq_file = feat_file.replace(".csv", ".parquet")
    if os.path.exists(pq_file):
        df = pd.read_parquet(pq_file)
    else:
        df = pd.read_csv(feat_file)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    val_mask = df["path"].isin(val_paths_set)
    train_mask = ~val_mask
    return df, feat_cols, val_mask, train_mask


def train_probe_fraction(df, feat_cols, train_mask, val_mask, frac, c=0.1):
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # Sub-sample training data
    n_train = max(4, int(len(train_idx) * frac))
    y_tr = y[train_idx]
    pos = train_idx[y_tr == 1]
    neg = train_idx[y_tr == 0]
    rng = np.random.default_rng(42)
    n_pos = max(2, n_train // 2)
    n_neg = max(2, n_train - n_pos)
    sampled = np.concatenate([
        rng.choice(pos, min(n_pos, len(pos)), replace=False),
        rng.choice(neg, min(n_neg, len(neg)), replace=False),
    ])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[sampled])
    X_val = scaler.transform(X[val_idx])
    y_tr_s = y[sampled]
    y_val = y[val_idx]

    clf = LogisticRegression(max_iter=2000, C=c, random_state=42)
    clf.fit(X_tr, y_tr_s)

    y_pred = clf.predict(X_val)
    return clf, scaler, y_pred, y_val, df.iloc[val_idx]["path"].values, len(sampled)


def evaluate_routing(df_val_feat, y_clip_pred, df_cnn_val, df_qwen_val):
    """Evaluate semantic-only vs disagreement cascade on the val set."""
    paths_feat = df_val_feat["path"].values
    y_true = df_val_feat["y_true"].values

    # Merge CNN predictions
    cnn_lookup = dict(zip(df_cnn_val["path"], df_cnn_val["y_cnn"]))
    qwen_lookup = dict(zip(df_qwen_val["path"], df_qwen_val["y_qwen"]))
    lat_lookup = {p: v for p, v in zip(df_qwen_val["path"],
                                        df_qwen_val.get("latency_qwen_ms",
                                                         pd.Series([0]*len(df_qwen_val))))}

    y_cnn = np.array([cnn_lookup.get(p, -1) for p in paths_feat])
    y_qwen = np.array([qwen_lookup.get(p, -1) for p in paths_feat])
    lat_qwen = np.array([lat_lookup.get(p, 0) for p in paths_feat])

    valid = (y_cnn != -1) & (y_qwen != -1)
    if valid.sum() < 10:
        return None

    y_true_v = y_true[valid]
    y_clip_v = y_clip_pred[valid]
    y_cnn_v = y_cnn[valid]
    y_qwen_v = y_qwen[valid]
    lat_qwen_v = lat_qwen[valid]

    disagree = (y_cnn_v != y_clip_v).astype(bool)

    # Semantic-only
    sem_acc = float(np.mean(y_clip_v == y_true_v))

    # Disagreement cascade
    casc_pred = np.where(disagree, y_qwen_v, y_clip_v)
    casc_acc = float(np.mean(casc_pred == y_true_v))
    casc_lats = np.where(disagree,
                         LATENCY_CHEAP_MS + lat_qwen_v,
                         LATENCY_CLIP_MS)

    agree_acc = float(np.mean(y_clip_v[~disagree] == y_true_v[~disagree])) if (~disagree).sum() > 0 else float("nan")
    dis_acc = float(np.mean(y_qwen_v[disagree] == y_true_v[disagree])) if disagree.sum() > 0 else float("nan")

    return {
        "n_valid": int(valid.sum()),
        "disagree_rate": float(np.mean(disagree)),
        "semantic_only_acc": sem_acc,
        "cascade_acc": casc_acc,
        "routing_gain": casc_acc - sem_acc,
        "cascade_mean_latency_ms": float(np.mean(casc_lats)),
        "agree_acc": agree_acc,
        "disagree_qwen_acc": dis_acc,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--backbones", nargs="+", default=["CLIP", "SigLIP2", "DINOv2"])
    p.add_argument("--append", action="store_true",
                   help="Append to existing results CSV instead of overwriting")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    all_rows = []

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]

        # Load CNN and Qwen predictions (needed for routing evaluation)
        if not os.path.exists(cfg["cnnspot_csv"]) or not os.path.exists(cfg["qwen_preds"]):
            print(f"Skipping {ds_name}: missing CNN or Qwen predictions")
            continue

        df_cnn = pd.read_csv(cfg["cnnspot_csv"])
        df_qwen = pd.read_csv(cfg["qwen_preds"])
        # Use load_val_paths to get consistent split
        _val_paths = load_val_paths(cfg, df_cnn["path"].tolist())
        df_cnn_val = df_cnn[df_cnn["path"].isin(_val_paths)].reset_index(drop=True)
        df_qwen_val = df_qwen[df_qwen["path"].isin(_val_paths)].reset_index(drop=True)

        print(f"\n=== {ds_name} ===")

        for bb_name in args.backbones:
            if bb_name not in BACKBONE_MAP:
                continue
            bb_cfg = BACKBONE_MAP[bb_name]
            feat_file = os.path.join(ROUTING_BASE,
                                     f"{cfg['prefix']}{bb_cfg['feat_prefix']}.csv")
            if not os.path.exists(feat_file):
                print(f"  {bb_name}: missing features, skipping")
                continue

            print(f"\n  Backbone: {bb_name}")
            df_feat, feat_cols, val_mask, train_mask = load_and_split_features(
                feat_file, _val_paths
            )
            df_feat_val = df_feat[val_mask].reset_index(drop=True)

            print(f"  {'Frac':>6} {'NTrain':>8} {'SemOnly':>9} {'Cascade':>9} {'Gain':>8} {'DisRate':>9}")
            print("  " + "-" * 55)

            for frac in FRACTIONS:
                clf, scaler, y_pred, y_val, val_paths_arr, n_train = train_probe_fraction(
                    df_feat, feat_cols, train_mask, val_mask, frac
                )

                rt = evaluate_routing(df_feat_val, y_pred, df_cnn_val, df_qwen_val)
                if rt is None:
                    continue

                sem_acc = float(accuracy_score(y_val, y_pred))
                gain = rt["routing_gain"]
                dis_rate = rt["disagree_rate"]

                mark = " *" if gain > 0.005 else ""
                print(f"  {frac:>6.0%} {n_train:>8d} {sem_acc:>9.3f} "
                      f"{rt['cascade_acc']:>9.3f} {gain:>+8.3f} {dis_rate:>9.3f}{mark}")

                all_rows.append({
                    "dataset": ds_name,
                    "backbone": bb_name,
                    "train_fraction": frac,
                    "n_train": n_train,
                    "semantic_only_acc": sem_acc,
                    **rt,
                })

            # Explicitly release the large feature DataFrame before next backbone
            del df_feat, df_feat_val
            gc.collect()

    df_out = pd.DataFrame(all_rows)
    out_file = os.path.join(RESULTS_DIR, "e6_limited_data.csv")
    if args.append and os.path.exists(out_file):
        df_out.to_csv(out_file, mode="a", index=False, header=False)
        print(f"\nAppended {len(df_out)} rows to {out_file}")
    else:
        df_out.to_csv(out_file, index=False)
        print(f"\nSaved {out_file}")

    # Success check: routing outperforms semantic-only at low-data fractions
    low_data = df_out[df_out["train_fraction"] <= 0.10]
    if len(low_data) > 0:
        routing_wins = int((low_data["routing_gain"] > 0.005).sum())
        print(f"\nAt ≤10% data: routing outperforms semantic-only in "
              f"{routing_wins}/{len(low_data)} cases")

    print("\nE6 complete.")


if __name__ == "__main__":
    main()
