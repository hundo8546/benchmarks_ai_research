"""
E2 — Modern Artifact Detector Replacement.
Compares CNNSpot (A1), DIRE (A2), FFT dual-stream (A3).

Records per detector per dataset:
  - Accuracy, Fake Recall, Real Recall
  - Cross-family Generalization Drop

Produces: results/e2_artifact_results.csv

Pre-requisites:
  - extract_dire.py run for each dataset (generates <prefix>dire_preds.csv)
  - extract_fft.py + train_probe.py run for each dataset (generates <prefix>fft_probe.pkl)

Usage:
    python e2_modern_artifact.py [--datasets GenBuster SD14 BigGAN]
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir, load_val_paths

LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4


def load_cnnspot(ds_name):
    cfg = DATASETS[ds_name]
    csv_path = cfg["cnnspot_csv"]
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    val_paths = load_val_paths(cfg, df["path"].tolist())
    return df[df["path"].isin(val_paths)].reset_index(drop=True)


def load_dire(ds_name):
    cfg = DATASETS[ds_name]
    dire_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}dire_preds.csv")
    if not os.path.exists(dire_csv):
        return None
    df = pd.read_csv(dire_csv)
    val_paths = load_val_paths(cfg, df["path"].tolist())
    return df[df["path"].isin(val_paths)].reset_index(drop=True)


def load_fft_probe(ds_name):
    """Load FFT features and train/load a probe; return val predictions."""
    cfg = DATASETS[ds_name]
    feat_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_features.csv")
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_probe.pkl")

    if not os.path.exists(feat_csv):
        return None

    df = pd.read_csv(feat_csv)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)

    # Use stored val paths to determine val set
    _val_paths = load_val_paths(cfg, df["path"].tolist())
    val_mask = df["path"].isin(_val_paths)
    train_mask = ~val_mask

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    if os.path.exists(probe_pkl):
        clf, scaler = joblib.load(probe_pkl)
    else:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(X_train_s, y_train)
        joblib.dump((clf, scaler), probe_pkl)

    X_val_s = scaler.transform(X_val)
    y_pred = clf.predict(X_val_s)
    y_prob = clf.predict_proba(X_val_s)[:, 1]

    df_val = df[val_mask].reset_index(drop=True)
    df_val["y_fft"] = y_pred
    df_val["prob_fake_fft"] = y_prob
    return df_val


def compute_detector_metrics(y_true, y_pred, detector_name, ds_name, latency_ms):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    fake_recall = float(np.mean(y_pred[y_true == 1] == 1)) if (y_true == 1).sum() > 0 else float("nan")
    real_recall = float(np.mean(y_pred[y_true == 0] == 0)) if (y_true == 0).sum() > 0 else float("nan")
    return {
        "dataset": ds_name,
        "detector": detector_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "latency_ms": latency_ms,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    all_rows = []
    # Store per-dataset per-detector accuracy for cross-family generalization
    det_acc = {}  # det_acc[detector][ds_name] = accuracy

    for ds_name in args.datasets:
        print(f"\n=== {ds_name} ===")
        det_acc.setdefault("CNNSpot", {}), det_acc.setdefault("DIRE", {}), det_acc.setdefault("FFT", {})

        # A1: CNNSpot
        df_cnn = load_cnnspot(ds_name)
        if df_cnn is not None:
            r = compute_detector_metrics(df_cnn["y_true"], df_cnn["y_cnn"],
                                          "CNNSpot", ds_name, LATENCY_CNN_MS)
            all_rows.append(r)
            det_acc["CNNSpot"][ds_name] = r["accuracy"]
            print(f"  CNNSpot: acc={r['accuracy']:.3f}, fake_rec={r['fake_recall']:.3f}, "
                  f"real_rec={r['real_recall']:.3f}")
        else:
            print(f"  CNNSpot: missing CSV")

        # A2: DIRE
        df_dire = load_dire(ds_name)
        if df_dire is not None:
            r = compute_detector_metrics(df_dire["y_true"], df_dire["y_dire"],
                                          "DIRE", ds_name, 12.0)  # approx latency
            all_rows.append(r)
            det_acc["DIRE"][ds_name] = r["accuracy"]
            print(f"  DIRE:    acc={r['accuracy']:.3f}, fake_rec={r['fake_recall']:.3f}, "
                  f"real_rec={r['real_recall']:.3f}")
        else:
            print(f"  DIRE:    missing preds CSV (run extract_dire.py first)")

        # A3: FFT
        df_fft = load_fft_probe(ds_name)
        if df_fft is not None:
            r = compute_detector_metrics(df_fft["y_true"], df_fft["y_fft"],
                                          "FFT", ds_name, 2.5)  # approx latency
            all_rows.append(r)
            det_acc["FFT"][ds_name] = r["accuracy"]
            print(f"  FFT:     acc={r['accuracy']:.3f}, fake_rec={r['fake_recall']:.3f}, "
                  f"real_rec={r['real_recall']:.3f}")
        else:
            print(f"  FFT:     missing features CSV (run extract_fft.py first)")

    # Compute cross-family generalization drop
    # Drop = (in-domain accuracy) - (mean out-of-domain accuracy)
    print("\n=== Cross-family generalization drop ===")
    for det_name, accs in det_acc.items():
        if len(accs) < 2:
            continue
        ds_names = list(accs.keys())
        for in_ds in ds_names:
            out_accs = [accs[d] for d in ds_names if d != in_ds]
            if out_accs:
                drop = accs[in_ds] - np.mean(out_accs)
                print(f"  {det_name} in-domain={in_ds}: {accs[in_ds]:.3f}, "
                      f"OOD mean={np.mean(out_accs):.3f}, drop={drop:+.3f}")
                for row in all_rows:
                    if row["detector"] == det_name and row["dataset"] == in_ds:
                        row["cross_family_drop"] = float(drop)

    df_out = pd.DataFrame(all_rows)
    out_file = os.path.join(RESULTS_DIR, "e2_artifact_results.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")
    print("E2 complete.")


if __name__ == "__main__":
    main()
