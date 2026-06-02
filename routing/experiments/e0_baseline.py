"""
E0 — Baseline Reproduction.
Reproduces Tables 2, 3, 4 from the original paper using:
  Semantic: CLIP ViT-B/32
  Artifact: CNNSpot
  Verifier: Qwen2.5-VL-7B

Produces: results/e0_table2.csv, e0_table3.csv, e0_table4.csv, e0_disagreement_stats.csv
Pass criterion: < 1.5% deviation from published numbers.

Usage:
    python e0_baseline.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.utils import resample

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import DATASETS, RESULTS_DIR, ensure_results_dir, bootstrap_ci, load_val_paths

ROUTING_BASE = "/workspace/benchmarks_ai_research/routing"
LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_CNN_MS + LATENCY_CLIP_MS


def load_dataset(ds_name):
    cfg = DATASETS[ds_name]
    df_cnn = pd.read_csv(cfg["cnnspot_csv"])
    df_qwen = pd.read_csv(cfg["qwen_preds"])
    df_clip = pd.read_csv(cfg["clip_preds"])

    qwen_cols = ["path", "y_qwen"]
    for col in ["y_qwen_text", "p_fake_qwen", "qwen_margin",
                "qwen_input_tokens", "qwen_output_tokens", "latency_qwen_ms"]:
        if col in df_qwen.columns:
            qwen_cols.append(col)

    df = df_cnn.merge(df_qwen[qwen_cols], on="path", how="inner")
    df = df.merge(df_clip[["path", "y_clip", "clip_proba"]], on="path", how="inner")

    val_paths = load_val_paths(cfg, df["path"].tolist())
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)

    # Fill Qwen columns if missing
    for col in ["qwen_input_tokens", "qwen_output_tokens", "latency_qwen_ms", "p_fake_qwen"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
    return df


def evaluate_methods(df, dataset_name):
    y_true = df["y_true"].values
    y_cnn = df["y_cnn"].values
    y_clip = df["y_clip"].values
    y_qwen = df["y_qwen"].values
    disagree = df["disagree"].values

    rows = []

    def row(name, y_pred, latency_ms, esc_rate=0.0,
            mean_in=0.0, mean_out=0.0):
        correct = (y_pred == y_true).astype(int).tolist()
        acc = np.mean(correct)
        lo, hi = bootstrap_ci(correct)
        rows.append({
            "dataset": dataset_name,
            "method": name,
            "accuracy": acc,
            "ci_lo": lo,
            "ci_hi": hi,
            "mean_latency_ms": latency_ms,
            "escalation_rate": esc_rate,
            "mean_input_tokens": mean_in,
            "mean_output_tokens": mean_out,
        })

    # CNN-only
    row("CNN-only", y_cnn, LATENCY_CNN_MS)

    # CLIP-only
    row("CLIP-only", y_clip, LATENCY_CLIP_MS)

    # Disagree ensemble (= CLIP-only, no VLM)
    ensemble_pred = np.where(disagree == 1, y_clip, y_cnn)
    row("Disagree ensemble", ensemble_pred, LATENCY_CHEAP_MS)

    # Disagree threshold -> Qwen
    dis_pred = np.where(disagree == 1, y_qwen, y_cnn)
    lats = np.where(disagree == 1,
                    LATENCY_CHEAP_MS + df["latency_qwen_ms"].values,
                    LATENCY_CHEAP_MS)
    in_toks = np.where(disagree == 1, df["qwen_input_tokens"].values, 0.0)
    out_toks = np.where(disagree == 1, df["qwen_output_tokens"].values, 0.0)
    row("Disagree thresh", dis_pred,
        float(np.mean(lats)),
        float(np.mean(disagree)),
        float(np.mean(in_toks)),
        float(np.mean(out_toks)))

    # Always escalate
    always_lats = LATENCY_CHEAP_MS + df["latency_qwen_ms"].values
    row("Always escalate", y_qwen,
        float(np.mean(always_lats)), 1.0,
        float(df["qwen_input_tokens"].mean()),
        float(df["qwen_output_tokens"].mean()))

    # Bandit (if available)
    prefix = DATASETS[dataset_name]["prefix"]
    bandit_pkl = os.path.join(ROUTING_BASE, f"{prefix}bandit_model.pkl")
    if not os.path.exists(bandit_pkl):
        # Fallback to root-level bandit_model.pkl
        bandit_pkl = os.path.join(os.path.dirname(ROUTING_BASE), "bandit_model.pkl")
    if os.path.exists(bandit_pkl):
        try:
            bandit, scaler = joblib.load(bandit_pkl)
            feat_cols = ["confidence", "margin1", "entropy1", "logit", "y_cnn",
                         "y_clip", "clip_proba"]
            feat_cols = [c for c in feat_cols if c in df.columns]
            clip_margin = np.abs(df["clip_proba"].values - 0.5)
            eps = 1e-12
            clip_entropy = -(df["clip_proba"].values * np.log(df["clip_proba"].values + eps) +
                             (1 - df["clip_proba"].values) * np.log(1 - df["clip_proba"].values + eps))
            prob_gap = np.abs(df["confidence"].values - df["clip_proba"].values)
            X = np.column_stack([df["confidence"].values, df["margin1"].values,
                                  df["entropy1"].values, df["logit"].values,
                                  df["y_cnn"].values, df["y_clip"].values,
                                  df["clip_proba"].values, clip_margin,
                                  clip_entropy, disagree, prob_gap])
            X_s = scaler.transform(X)
            actions = bandit.predict(X_s)
            bandit_pred = np.where(actions == 1, y_qwen, y_cnn)
            b_lats = np.where(actions == 1,
                              LATENCY_CHEAP_MS + df["latency_qwen_ms"].values,
                              LATENCY_CHEAP_MS)
            row("Bandit", bandit_pred, float(np.mean(b_lats)),
                float(np.mean(actions)),
                float(np.mean(np.where(actions == 1, df["qwen_input_tokens"].values, 0))),
                float(np.mean(np.where(actions == 1, df["qwen_output_tokens"].values, 0))))
        except Exception as e:
            print(f"  Bandit failed for {dataset_name}: {e}")

    return pd.DataFrame(rows)


def disagreement_stats(df, dataset_name):
    disagree_mask = df["disagree"] == 1
    agree_mask = df["disagree"] == 0

    stats = {
        "dataset": dataset_name,
        "n_total": len(df),
        "n_disagree": int(disagree_mask.sum()),
        "disagree_rate": float(df["disagree"].mean()),
        "agree_clip_acc": float(np.mean(df.loc[agree_mask, "y_clip"] == df.loc[agree_mask, "y_true"])),
        "disagree_clip_acc": float(np.mean(df.loc[disagree_mask, "y_clip"] == df.loc[disagree_mask, "y_true"])),
        "disagree_cnn_acc": float(np.mean(df.loc[disagree_mask, "y_cnn"] == df.loc[disagree_mask, "y_true"])),
        "disagree_qwen_acc": float(np.mean(df.loc[disagree_mask, "y_qwen"] == df.loc[disagree_mask, "y_true"])),
        "clip_overall_acc": float(np.mean(df["y_clip"] == df["y_true"])),
        "cnn_overall_acc": float(np.mean(df["y_cnn"] == df["y_true"])),
    }
    return stats


def main():
    ensure_results_dir()
    all_results = []
    all_dis_stats = []

    datasets_available = []
    for ds_name, cfg in DATASETS.items():
        missing = [f for f in [cfg["cnnspot_csv"], cfg["clip_preds"], cfg["qwen_preds"]]
                   if not os.path.exists(f)]
        if missing:
            print(f"Skipping {ds_name}: missing {missing}")
        else:
            datasets_available.append(ds_name)

    for ds_name in datasets_available:
        print(f"\n=== {ds_name} ===")
        df = load_dataset(ds_name)
        print(f"  {len(df)} validation samples")

        results = evaluate_methods(df, ds_name)
        all_results.append(results)

        dis_stats = disagreement_stats(df, ds_name)
        all_dis_stats.append(dis_stats)

        # Print table
        print(f"\n  {'Method':<28} {'Acc':>7} {'95% CI':>16} {'Lat(ms)':>10} {'Esc':>7}")
        print("  " + "-" * 75)
        for _, r in results.iterrows():
            ci = f"[{r.ci_lo:.3f},{r.ci_hi:.3f}]"
            print(f"  {r.method:<28} {r.accuracy:>7.3f} {ci:>16} "
                  f"{r.mean_latency_ms:>10.1f} {r.escalation_rate:>7.2f}")

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        # Split into per-dataset tables (Table 2 = GenBuster, 3 = SD14, 4 = BigGAN)
        table_map = {"GenBuster": "e0_table2.csv", "SD14": "e0_table3.csv", "BigGAN": "e0_table4.csv"}
        for ds_name, fname in table_map.items():
            sub = df_all[df_all["dataset"] == ds_name]
            if len(sub) > 0:
                sub.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
                print(f"\nSaved {fname}")

        df_dis = pd.DataFrame(all_dis_stats)
        df_dis.to_csv(os.path.join(RESULTS_DIR, "e0_disagreement_stats.csv"), index=False)
        print("Saved e0_disagreement_stats.csv")

    print("\nE0 complete.")


if __name__ == "__main__":
    main()
