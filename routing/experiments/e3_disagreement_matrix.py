"""
E3 — Cross-Detector Disagreement Matrix.
Core revision experiment: measures disagreement behavior across all 9 detector pairings.

Artifact detectors (A): CNNSpot, DIRE, FFT
Semantic detectors (S): CLIP, SigLIP2, DINOv2

For each (A, S) pairing and dataset, computes:
  - Agreement Rate: P(agree)
  - Cheap Path Accuracy: P(correct | agree)
  - Hard Case Difficulty: P(correct | disagree)
  - Escalation Gain Potential: Δ

Produces: results/e3_disagreement_matrix.csv
Success: at least one pairing achieves cheap-path accuracy > 90% and
         substantially harder disagreement subset.

Usage:
    python e3_disagreement_matrix.py [--datasets GenBuster SD14 BigGAN]
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir, LATENCY_CHEAP_MS, load_val_paths

LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4

ARTIFACT_DETECTORS = {
    "CNNSpot": {"pred_col": "y_cnn", "prob_col": "confidence"},
    "DIRE": {"pred_col": "y_dire", "prob_col": "prob_fake_dire"},
    "FFT": {"pred_col": "y_fft", "prob_col": "prob_fake_fft"},
}

SEMANTIC_DETECTORS = {
    "CLIP": {"feat_prefix": "clip_features", "probe_prefix": "clip_classifier"},
    "SigLIP2": {"feat_prefix": "siglip2_features", "probe_prefix": "siglip2_probe"},
    "DINOv2": {"feat_prefix": "dinov2_features", "probe_prefix": "dinov2_probe"},
}


def load_artifact_preds(ds_name, det_name):
    """Load artifact detector predictions for the val split."""
    cfg = DATASETS[ds_name]
    det_cfg = ARTIFACT_DETECTORS[det_name]

    if det_name == "CNNSpot":
        csv_path = cfg["cnnspot_csv"]
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        val_paths = load_val_paths(cfg, df["path"].tolist())
        df = df[df["path"].isin(val_paths)].reset_index(drop=True)
        return df.rename(columns={det_cfg["pred_col"]: "y_artifact",
                                   det_cfg["prob_col"]: "prob_artifact"})

    elif det_name == "DIRE":
        csv_path = os.path.join(ROUTING_BASE, f"{cfg['prefix']}dire_preds.csv")
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        val_paths = load_val_paths(cfg, df["path"].tolist())
        df = df[df["path"].isin(val_paths)].reset_index(drop=True)
        return df.rename(columns={"y_dire": "y_artifact", "prob_fake_dire": "prob_artifact"})

    elif det_name == "FFT":
        feat_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_features.csv")
        probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_probe.pkl")
        if not os.path.exists(feat_csv):
            return None
        df = pd.read_csv(feat_csv)
        feat_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feat_cols].values.astype(np.float32)
        y = df["y_true"].values.astype(int)
        val_paths = load_val_paths(cfg, df["path"].tolist())
        val_mask = df["path"].isin(val_paths)
        train_mask = ~val_mask
        if os.path.exists(probe_pkl):
            clf, scaler = joblib.load(probe_pkl)
        else:
            scaler = StandardScaler()
            clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
            clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
            joblib.dump((clf, scaler), probe_pkl)
        X_val = scaler.transform(X[val_mask])
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        df_val = df[val_mask].reset_index(drop=True)
        df_val["y_artifact"] = y_pred
        df_val["prob_artifact"] = y_prob
        return df_val[["path", "y_true", "y_artifact", "prob_artifact"]]

    return None


def load_semantic_preds(ds_name, sem_name):
    """Load or compute semantic detector predictions for the val split."""
    cfg = DATASETS[ds_name]
    sem_cfg = SEMANTIC_DETECTORS[sem_name]

    feat_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{sem_cfg['feat_prefix']}.csv")
    if not os.path.exists(feat_csv):
        return None

    df = pd.read_csv(feat_csv)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)

    val_paths = load_val_paths(cfg, df["path"].tolist())
    val_mask = df["path"].isin(val_paths)
    train_mask = ~val_mask

    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{sem_cfg['probe_prefix']}.pkl")

    if os.path.exists(probe_pkl):
        clf, scaler = joblib.load(probe_pkl)
    else:
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
        joblib.dump((clf, scaler), probe_pkl)

    X_val = scaler.transform(X[val_mask])
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    df_val = df[val_mask].reset_index(drop=True)
    df_val["y_semantic"] = y_pred
    df_val["prob_semantic"] = y_prob
    return df_val[["path", "y_true", "y_semantic", "prob_semantic"]]


def compute_pairing_metrics(df_art, df_sem, ds_name, art_name, sem_name):
    """Merge artifact and semantic predictions and compute disagreement metrics."""
    df = df_art.merge(df_sem, on=["path", "y_true"], how="inner")
    if len(df) == 0:
        return None

    y_true = df["y_true"].values
    y_art = df["y_artifact"].values
    y_sem = df["y_semantic"].values

    disagree = (y_art != y_sem).astype(bool)
    agree = ~disagree

    n_total = len(df)
    n_agree = int(agree.sum())
    n_disagree = int(disagree.sum())

    cheap_acc = float(np.mean(y_sem[agree] == y_true[agree])) if n_agree > 0 else float("nan")
    hard_acc = float(np.mean(y_sem[disagree] == y_true[disagree])) if n_disagree > 0 else float("nan")
    gain = (cheap_acc - hard_acc) if not (np.isnan(cheap_acc) or np.isnan(hard_acc)) else float("nan")

    # On agreement subset, use semantic (which = artifact here)
    agree_art_acc = float(np.mean(y_art[agree] == y_true[agree])) if n_agree > 0 else float("nan")

    return {
        "dataset": ds_name,
        "artifact": art_name,
        "semantic": sem_name,
        "pairing": f"{art_name}+{sem_name}",
        "n_total": n_total,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_rate": float(np.mean(agree)),
        "cheap_path_accuracy": cheap_acc,
        "hard_case_accuracy": hard_acc,
        "escalation_gain_potential": gain,
        "artifact_only_acc": float(np.mean(y_art == y_true)),
        "semantic_only_acc": float(np.mean(y_sem == y_true)),
        "meets_90pct_criterion": cheap_acc > 0.90 if not np.isnan(cheap_acc) else False,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    all_rows = []

    for ds_name in args.datasets:
        print(f"\n=== {ds_name} ===")
        print(f"{'Artifact':<12} {'Semantic':<12} {'AgreeRate':>10} "
              f"{'CheapAcc':>10} {'HardAcc':>10} {'Gain':>8} {'90%?':>6}")
        print("-" * 70)

        for art_name in ARTIFACT_DETECTORS:
            df_art = load_artifact_preds(ds_name, art_name)
            if df_art is None:
                print(f"  {art_name}: missing, skipping")
                continue

            for sem_name in SEMANTIC_DETECTORS:
                df_sem = load_semantic_preds(ds_name, sem_name)
                if df_sem is None:
                    print(f"  {art_name}+{sem_name}: missing semantic features")
                    continue

                metrics = compute_pairing_metrics(df_art, df_sem, ds_name, art_name, sem_name)
                if metrics is None:
                    print(f"  {art_name}+{sem_name}: no overlapping data")
                    continue

                all_rows.append(metrics)
                mark = " *" if metrics["meets_90pct_criterion"] else ""
                print(f"  {art_name:<12} {sem_name:<12} "
                      f"{metrics['agreement_rate']:>10.3f} "
                      f"{metrics['cheap_path_accuracy']:>10.3f} "
                      f"{metrics['hard_case_accuracy']:>10.3f} "
                      f"{metrics['escalation_gain_potential']:>8.3f}{mark}")

    df_out = pd.DataFrame(all_rows)
    out_file = os.path.join(RESULTS_DIR, "e3_disagreement_matrix.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")

    # Check success criterion
    if len(df_out) > 0:
        best = df_out.loc[df_out["cheap_path_accuracy"].idxmax()]
        print(f"\nBest pairing: {best['pairing']} on {best['dataset']}")
        print(f"  Cheap-path acc = {best['cheap_path_accuracy']:.3f}")
        print(f"  Hard-case acc  = {best['hard_case_accuracy']:.3f}")
        print(f"  Gain potential = {best['escalation_gain_potential']:.3f}")
        n_pass = int(df_out["meets_90pct_criterion"].sum())
        print(f"\nPairings meeting >90% cheap-path criterion: {n_pass}/{len(df_out)}")

    print("\nE3 complete.")


if __name__ == "__main__":
    main()
