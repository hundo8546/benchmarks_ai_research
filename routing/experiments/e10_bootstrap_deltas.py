"""
E10 -- Bootstrap confidence intervals on the delta columns reported in
tab:cross_family_deployment (Table 14) and tab:e8_results (Table 4).

Reviewer 2 asked for uncertainty quantification on the reported deltas, not
just point estimates, since some are small relative to the evaluation set
sizes (n=400-480 for cross-family, n=1400-1660 disagreement-subset for
SDXL/FLUX).

For each delta this uses a *paired* bootstrap: each resample redraws sample
indices once and recomputes both accuracies (e.g. cascade and stale-semantic)
on that same resample, so correlation between the two arms (same samples,
same errors) is preserved. This gives a narrower, more correct CI on the
difference than combining independently bootstrapped CIs for each arm would.

Produces: results/e10_cross_family_delta_cis.csv, results/e10_e8_delta_cis.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cross_family_deployment import DATASETS, load_dataset, apply_probe_to_val

BASE = "/workspace/benchmarks_ai_research/routing/"
N_BOOT = 10000
RNG = np.random.default_rng(42)


def paired_bootstrap_delta(y_true, pred_a, pred_b, n_boot=N_BOOT):
    """Bootstrap CI on mean(pred_a == y_true) - mean(pred_b == y_true)."""
    n = len(y_true)
    correct_a = (pred_a == y_true).astype(np.float64)
    correct_b = (pred_b == y_true).astype(np.float64)
    point = correct_a.mean() - correct_b.mean()
    idx = RNG.integers(0, n, size=(n_boot, n))
    deltas = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, lo, hi


def cross_family_deltas():
    print("Loading cross-family datasets...")
    data = {}
    for name in DATASETS:
        df, vp, sem_lookup, model, scaler, feat_cols = load_dataset(name)
        data[name] = {"df": df, "val_paths": vp, "sem_lookup": sem_lookup,
                      "model": model, "scaler": scaler, "feat_cols": feat_cols}

    rows = []
    for train_name in DATASETS:
        for test_name in DATASETS:
            if train_name == test_name:
                continue
            train_d, test_d = data[train_name], data[test_name]
            val_df, y_stale = apply_probe_to_val(
                train_d["model"], train_d["scaler"],
                test_d["df"], test_d["val_paths"],
                test_d["sem_lookup"], test_d["feat_cols"])

            y_true = val_df["y_true"].values
            y_fft = val_df["y_fft"].values
            y_qwen = val_df["y_qwen"].values
            disagree = (y_fft != y_stale)
            pred_cascade = np.where(disagree, y_qwen, y_fft)

            d_stale, lo_stale, hi_stale = paired_bootstrap_delta(y_true, pred_cascade, y_stale)
            d_fft, lo_fft, hi_fft = paired_bootstrap_delta(y_true, pred_cascade, y_fft)

            sig_stale = "yes" if (lo_stale > 0 or hi_stale < 0) else "no"
            sig_fft = "yes" if (lo_fft > 0 or hi_fft < 0) else "no"

            print(f"{train_name:>10} -> {test_name:<10}  n={len(val_df):4d}  "
                  f"delta_stale={d_stale:+.3f} [{lo_stale:+.3f},{hi_stale:+.3f}] sig={sig_stale}  "
                  f"delta_fft={d_fft:+.3f} [{lo_fft:+.3f},{hi_fft:+.3f}] sig={sig_fft}")

            rows.append({
                "train": train_name, "test": test_name, "n": len(val_df),
                "delta_stale": d_stale, "delta_stale_ci_lo": lo_stale, "delta_stale_ci_hi": hi_stale,
                "delta_stale_sig_at_95": sig_stale,
                "delta_fft": d_fft, "delta_fft_ci_lo": lo_fft, "delta_fft_ci_hi": hi_fft,
                "delta_fft_sig_at_95": sig_fft,
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "results", "e10_cross_family_delta_cis.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    return out


def e8_deltas():
    print("\nLoading SDXL/FLUX e8 per-sample data...")
    import joblib

    def get_predictions_stale(feat_file, probe_pkl):
        clf, scaler = joblib.load(os.path.join(BASE, probe_pkl))
        df = pd.read_csv(os.path.join(BASE, feat_file))
        feat_cols = [c for c in df.columns if c.startswith("f")]
        X = df[feat_cols].values.astype(np.float32)
        Xs = scaler.transform(X)
        df = df[["path", "y_true"]].copy()
        df["y_pred"] = clf.predict(Xs)
        return df

    # CLIP omitted: no SD14-trained CLIP probe/features exist in this
    # environment (see cross_family_deployment.py docstring), so its e8 row
    # cannot be reconstructed from existing artifacts without new GPU feature
    # extraction. SigLIP2 and DINOv2 stale probes/features are both present.
    SEM_CONFIGS = ["siglip2", "dinov2"]

    rows = []
    for ds in ["sdxl", "flux"]:
        df_fft = get_predictions_stale(f"{ds}_fft_features.csv", "sd14_fft_probe.pkl")
        df_fft = df_fft.rename(columns={"y_pred": "y_fft"})

        for sem in SEM_CONFIGS:
            df_sem = get_predictions_stale(f"{ds}_{sem}_features.csv", f"sd14_{sem}_probe.pkl")
            df_sem = df_sem.rename(columns={"y_pred": "y_sem"})
            df = df_fft.merge(df_sem[["path", "y_sem"]], on="path")

            gpt = pd.read_csv(os.path.join(BASE, f"{ds}_gpt55_{sem}_disagree_preds.csv"))
            qwen = pd.read_csv(os.path.join(BASE, f"{ds}_qwen_{sem}_disagree_preds.csv"))
            df = df.merge(gpt[["path", "y_gpt55"]], on="path", how="left")
            df = df.merge(qwen[["path", "y_qwen"]], on="path", how="left")

            disagree = df["y_fft"] != df["y_sem"]
            y_true = df["y_true"].values
            cascade_gpt = np.where(disagree, df["y_gpt55"], df["y_sem"])
            cascade_qwen = np.where(disagree, df["y_qwen"], df["y_sem"])

            d_gpt, lo_gpt, hi_gpt = paired_bootstrap_delta(y_true, cascade_gpt, df["y_sem"].values)
            d_qwen, lo_qwen, hi_qwen = paired_bootstrap_delta(y_true, cascade_qwen, df["y_sem"].values)

            sig_gpt = "yes" if (lo_gpt > 0 or hi_gpt < 0) else "no"
            sig_qwen = "yes" if (lo_qwen > 0 or hi_qwen < 0) else "no"

            print(f"{ds.upper():>6} + {sem:<8} n={len(df):4d}  "
                  f"gain_gpt55={d_gpt:+.3f} [{lo_gpt:+.3f},{hi_gpt:+.3f}] sig={sig_gpt}  "
                  f"gain_qwen={d_qwen:+.3f} [{lo_qwen:+.3f},{hi_qwen:+.3f}] sig={sig_qwen}")

            rows.append({
                "dataset": ds.upper(), "semantic": sem.capitalize() if sem != "siglip2" else "SigLIP2",
                "n": len(df),
                "gain_gpt55": d_gpt, "gain_gpt55_ci_lo": lo_gpt, "gain_gpt55_ci_hi": hi_gpt,
                "gain_gpt55_sig_at_95": sig_gpt,
                "gain_qwen": d_qwen, "gain_qwen_ci_lo": lo_qwen, "gain_qwen_ci_hi": hi_qwen,
                "gain_qwen_sig_at_95": sig_qwen,
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "results", "e10_e8_delta_cis.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    return out


if __name__ == "__main__":
    cross_family_deltas()
    e8_deltas()
