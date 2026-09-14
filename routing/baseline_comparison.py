"""
Reviewer 1 Comment 3 — random / majority-class / best-cheap-detector baselines
on the verifier-on-disagreement-subset cells across Tables:
  verifier_comparison, e8_results, cross_family_deployment, limited_data.

biggan_shift is NOT included here: the distant-shift (SD14-stale-probe ->
GenImage BigGAN, n=1500) per-sample predictions are not present in this repo
checkout, and GenImage BigGAN images are not currently downloaded (blocked
on rclone GenImage transfer). See claude_context_notes/resubmissionplan.md
Section 3.3/3.1.

Usage:
    python baseline_comparison.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROUTING_BASE)
sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))

from extractors.common import DATASETS, RESULTS_DIR, ensure_results_dir, load_val_paths
from baseline_utils import compare_verifier_to_baselines

import cross_family_deployment as cfd
import e8_modern_diffusion as e8
import e6_limited_data as e6

ROWS = []
SUBSETS = []  # raw per-cell arrays, for gate_analysis.py to reuse


def add_row(table, config, verifier, y_true, y_verifier, y_cheap1, y_cheap2,
            cheap1_name, cheap2_name):
    y_true = np.asarray(y_true)
    y_verifier = np.asarray(y_verifier)
    y_cheap1 = np.asarray(y_cheap1)
    y_cheap2 = np.asarray(y_cheap2)

    res = compare_verifier_to_baselines(
        y_true, y_verifier, y_cheap1, y_cheap2, cheap1_name, cheap2_name)
    res.update({"table": table, "config": config, "verifier": verifier})
    ROWS.append(res)

    SUBSETS.append({
        "table": table, "config": config, "verifier": verifier,
        "y_true": y_true, "y_cheap1": y_cheap1, "y_cheap2": y_cheap2, "y_verifier": y_verifier,
        "cheap1_name": cheap1_name, "cheap2_name": cheap2_name,
    })


# ─────────────────────────────────────────────────────────────────────────
# 1. verifier_comparison: 3 in-domain datasets x {Qwen2.5-VL-7B, GPT-5.5}
#    on the FFT+SigLIP2 disagreement subset.
# ─────────────────────────────────────────────────────────────────────────
def get_fft_siglip2_disagree_val(ds_name):
    cfg = DATASETS[ds_name]
    prefix = cfg["prefix"]

    df_fft = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"))
    fft_feat_cols = [c for c in df_fft.columns if c.startswith("f")]
    clf_fft, scaler_fft = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl"))

    df_sig = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}siglip2_features.csv"))
    sig_feat_cols = [c for c in df_sig.columns if c.startswith("f")]
    clf_sig, scaler_sig = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}siglip2_probe.pkl"))

    val_paths = load_val_paths(cfg, df_fft["path"].tolist())

    df_fft_val = df_fft[df_fft["path"].isin(val_paths)].copy()
    df_fft_val["y_fft"] = clf_fft.predict(
        scaler_fft.transform(df_fft_val[fft_feat_cols].values.astype(np.float32)))

    df_sig_val = df_sig[df_sig["path"].isin(val_paths)].copy()
    df_sig_val["y_siglip2"] = clf_sig.predict(
        scaler_sig.transform(df_sig_val[sig_feat_cols].values.astype(np.float32)))

    df = df_fft_val[["path", "y_true", "y_fft"]].merge(
        df_sig_val[["path", "y_siglip2"]], on="path", how="inner")
    df["disagree"] = (df["y_fft"] != df["y_siglip2"]).astype(bool)
    return df[df["disagree"]].reset_index(drop=True)


def run_verifier_comparison():
    ds_cfg = [
        ("GenBuster", "qwen_preds.csv", "gpt55_disagree_preds.csv"),
        ("SD14", "sd14_qwen_preds.csv", "sd14_gpt55_disagree_preds.csv"),
        ("BigGAN", "biggan_qwen_preds.csv", "biggan_gpt55_disagree_preds.csv"),
    ]
    for ds_name, qwen_file, gpt_file in ds_cfg:
        df_dis = get_fft_siglip2_disagree_val(ds_name)

        df_qwen = pd.read_csv(os.path.join(ROUTING_BASE, qwen_file))
        qwen_lookup = dict(zip(df_qwen["path"], df_qwen["y_qwen"]))
        df_dis["y_qwen"] = df_dis["path"].map(qwen_lookup)
        d_qwen = df_dis.dropna(subset=["y_qwen"])
        if len(d_qwen) > 0:
            add_row("verifier_comparison", ds_name, "Qwen2.5-VL-7B",
                    d_qwen["y_true"], d_qwen["y_qwen"], d_qwen["y_fft"], d_qwen["y_siglip2"],
                    "FFT", "SigLIP2")

        gpt_path = os.path.join(ROUTING_BASE, gpt_file)
        if os.path.exists(gpt_path):
            df_gpt = pd.read_csv(gpt_path)
            verifier_col = "y_verifier" if "y_verifier" in df_gpt.columns else "y_gpt55"
            gpt_lookup = dict(zip(df_gpt["path"], df_gpt[verifier_col]))
            df_dis["y_gpt55"] = df_dis["path"].map(gpt_lookup)
            d_gpt = df_dis.dropna(subset=["y_gpt55"])
            if len(d_gpt) > 0:
                add_row("verifier_comparison", ds_name, "GPT-5.5",
                        d_gpt["y_true"], d_gpt["y_gpt55"], d_gpt["y_fft"], d_gpt["y_siglip2"],
                        "FFT", "SigLIP2")


# ─────────────────────────────────────────────────────────────────────────
# 2. e8_results: {SDXL, FLUX} x {SigLIP2, DINOv2} x {Qwen, GPT-5.5}.
#    CLIP skipped: SD14 lacks raw CLIP features in this checkout (documented
#    in cross_family_deployment.py), so the CLIP stale probe cannot be
#    reconstructed here.
# ─────────────────────────────────────────────────────────────────────────
def run_e8_results():
    for ds_name, cfg in e8.E8_DATASETS.items():
        prefix = cfg["prefix"]
        for sem_name in ["SigLIP2", "DINOv2"]:
            feat_key = e8.SEMANTICS[sem_name]
            feat_file = os.path.join(ROUTING_BASE, f"{prefix}{feat_key}.csv")
            fft_file = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")
            probe_key = {"SigLIP2": "stale_probe_siglip2", "DINOv2": "stale_probe_dinov2"}[sem_name]

            if not (os.path.exists(feat_file) and os.path.exists(fft_file)):
                print(f"  [e8] {ds_name}/{sem_name}: missing features, skip")
                continue

            df_sem = e8.get_predictions_stale(feat_file, cfg[probe_key], label=f"stale {sem_name}")
            df_fft = e8.get_predictions_stale(fft_file, cfg["stale_fft_probe"], label="stale FFT")
            if df_sem is None or df_fft is None:
                print(f"  [e8] {ds_name}/{sem_name}: probe missing, skip")
                continue

            df = df_fft.rename(columns={"y_pred": "y_fft"}).merge(
                df_sem.rename(columns={"y_pred": "y_sem"})[["path", "y_sem"]],
                on="path", how="inner")
            df["disagree"] = (df["y_fft"] != df["y_sem"]).astype(bool)
            df_dis = df[df["disagree"]].reset_index(drop=True)

            for verifier_tag, file_tag, col in [
                ("Qwen2.5-VL-7B", "qwen", "y_qwen"),
                ("GPT-5.5", "gpt55", None),
            ]:
                preds_csv = os.path.join(
                    ROUTING_BASE, f"{prefix}{file_tag}_{sem_name.lower()}_disagree_preds.csv")
                if not os.path.exists(preds_csv):
                    print(f"  [e8] {ds_name}/{sem_name}/{verifier_tag}: no preds file, skip")
                    continue
                df_v = pd.read_csv(preds_csv)
                verifier_col = col or ("y_gpt55" if "y_gpt55" in df_v.columns else "y_verifier")
                merged = df_dis.merge(df_v[["path", verifier_col]], on="path", how="inner")
                if len(merged) == 0:
                    continue
                add_row("e8_results", f"{ds_name}/{sem_name}", verifier_tag,
                        merged["y_true"], merged[verifier_col], merged["y_fft"], merged["y_sem"],
                        "StaleFFT", f"Stale{sem_name}")


# ─────────────────────────────────────────────────────────────────────────
# 3. cross_family_deployment: 6 train->test pairings, Qwen verifier.
# ─────────────────────────────────────────────────────────────────────────
def run_cross_family_deployment():
    data = {}
    for name in cfd.DATASETS:
        df, vp, sem_lookup, model, scaler, feat_cols = cfd.load_dataset(name)
        data[name] = {"df": df, "val_paths": vp, "sem_lookup": sem_lookup,
                      "model": model, "scaler": scaler, "feat_cols": feat_cols}

    for train_name in cfd.DATASETS:
        for test_name in cfd.DATASETS:
            if train_name == test_name:
                continue
            train_d, test_d = data[train_name], data[test_name]
            val_df, y_stale = cfd.apply_probe_to_val(
                train_d["model"], train_d["scaler"],
                test_d["df"], test_d["val_paths"], test_d["sem_lookup"], test_d["feat_cols"])

            y_true = val_df["y_true"].values
            y_fft = val_df["y_fft"].values
            y_qwen = val_df["y_qwen"].values
            disagree = (y_fft != y_stale).astype(bool)

            if disagree.sum() == 0:
                continue
            add_row("cross_family_deployment", f"{train_name}->{test_name}", "Qwen2.5-VL-7B",
                    y_true[disagree], y_qwen[disagree], y_fft[disagree], y_stale[disagree],
                    "FFT", "StaleSigLIP2")


# ─────────────────────────────────────────────────────────────────────────
# 5. biggan_shift: SD14-stale FFT+SigLIP2 disagreement subset on GenImage
#    BigGAN (n=1500), GPT-5.5 and Qwen2.5-VL-7B verifiers. Qwen2.5-VL-72B
#    omitted -- doesn't fit on this machine's 20GB GPU.
# ─────────────────────────────────────────────────────────────────────────
def run_biggan_shift():
    pool = pd.read_csv(os.path.join(ROUTING_BASE, "biggan_distant_shift_1500.csv"))

    gpt_path = os.path.join(ROUTING_BASE, "biggan_distant_shift_gpt55_preds.csv")
    if os.path.exists(gpt_path):
        df_gpt = pd.read_csv(gpt_path)
        merged = pool.merge(df_gpt[["path", "y_verifier"]], on="path", how="inner").dropna(subset=["y_verifier"])
        if len(merged) > 0:
            add_row("biggan_shift", "BigGAN-distant-shift", "GPT-5.5",
                    merged["y_true"], merged["y_verifier"], merged["y_stale_fft"], merged["y_stale_sig"],
                    "StaleFFT", "StaleSigLIP2")

    qwen_path = os.path.join(ROUTING_BASE, "biggan_distant_shift_qwen7b_preds.csv")
    if os.path.exists(qwen_path):
        df_qwen = pd.read_csv(qwen_path)
        merged = pool.merge(df_qwen[["path", "y_qwen"]], on="path", how="inner").dropna(subset=["y_qwen"])
        if len(merged) > 0:
            add_row("biggan_shift", "BigGAN-distant-shift", "Qwen2.5-VL-7B",
                    merged["y_true"], merged["y_qwen"], merged["y_stale_fft"], merged["y_stale_sig"],
                    "StaleFFT", "StaleSigLIP2")

    qwen72b_path = os.path.join(ROUTING_BASE, "biggan_distant_shift_qwen72b_preds.csv")
    if os.path.exists(qwen72b_path):
        df_qwen72b = pd.read_csv(qwen72b_path)
        merged = pool.merge(df_qwen72b[["path", "y_qwen"]], on="path", how="inner").dropna(subset=["y_qwen"])
        if len(merged) > 0:
            add_row("biggan_shift", "BigGAN-distant-shift", "Qwen2.5-VL-72B",
                    merged["y_true"], merged["y_qwen"], merged["y_stale_fft"], merged["y_stale_sig"],
                    "StaleFFT", "StaleSigLIP2")


# ─────────────────────────────────────────────────────────────────────────
# 4. limited_data: SD14 x {SigLIP2, DINOv2} x 6 train fractions, Qwen verifier.
# ─────────────────────────────────────────────────────────────────────────
def run_limited_data():
    ds_name = "SD14"
    cfg = e6.DATASETS[ds_name]

    df_fft_val = e6.get_fft_val_preds(cfg)
    df_qwen = pd.read_csv(cfg["qwen_preds"])
    val_paths = load_val_paths(cfg, df_fft_val["path"].tolist())
    df_qwen_val = df_qwen[df_qwen["path"].isin(val_paths)].reset_index(drop=True)
    fft_lookup = dict(zip(df_fft_val["path"], df_fft_val["y_fft"]))
    qwen_lookup = dict(zip(df_qwen_val["path"], df_qwen_val["y_qwen"]))

    for bb_name in ["SigLIP2", "DINOv2"]:
        bb_cfg = e6.BACKBONE_MAP[bb_name]
        feat_file = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{bb_cfg['feat_prefix']}.csv")
        if not os.path.exists(feat_file):
            continue
        df_feat, feat_cols, val_mask, train_mask = e6.load_and_split_features(feat_file, val_paths)
        df_feat_val = df_feat[val_mask].reset_index(drop=True)

        for frac in e6.FRACTIONS:
            clf, scaler, y_pred, y_val, val_paths_arr, n_train = e6.train_probe_fraction(
                df_feat, feat_cols, train_mask, val_mask, frac)

            paths = df_feat_val["path"].values
            y_true_all = df_feat_val["y_true"].values
            y_fft_all = np.array([fft_lookup.get(p, -1) for p in paths])
            y_qwen_all = np.array([qwen_lookup.get(p, -1) for p in paths])
            valid = (y_fft_all != -1) & (y_qwen_all != -1)
            disagree = valid & (y_fft_all != y_pred)

            if disagree.sum() == 0:
                continue
            add_row("limited_data", f"{ds_name}/{bb_name}/frac={frac:.0%}", "Qwen2.5-VL-7B",
                    y_true_all[disagree], y_qwen_all[disagree], y_fft_all[disagree],
                    y_pred[disagree], "FFT", bb_name)


def main():
    ensure_results_dir()
    print("Running verifier_comparison...")
    run_verifier_comparison()
    print("Running e8_results...")
    run_e8_results()
    print("Running cross_family_deployment...")
    run_cross_family_deployment()
    print("Running limited_data...")
    run_limited_data()
    print("Running biggan_shift...")
    run_biggan_shift()

    df = pd.DataFrame(ROWS)
    cols = ["table", "config", "verifier", "n", "verifier_acc",
            "random_mean", "random_ci_lo", "random_ci_hi", "clears_random",
            "majority_acc", "verifier_minus_majority_ci_lo", "verifier_minus_majority_ci_hi", "clears_majority",
            "best_cheap_acc", "best_cheap_source",
            "verifier_minus_best_cheap_ci_lo", "verifier_minus_best_cheap_ci_hi", "clears_best_cheap"]
    df = df[cols]
    out_path = os.path.join(RESULTS_DIR, "baseline_comparison_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(df)} rows)\n")

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nTotal cells: {len(df)}")
    print(f"Clears random:      {df['clears_random'].sum()}/{len(df)}")
    print(f"Clears majority:    {df['clears_majority'].sum()}/{len(df)}")
    print(f"Clears best-cheap:  {df['clears_best_cheap'].sum()}/{len(df)}")
    print("\nCells where verifier does NOT clear best-cheap-detector baseline:")
    print(df[~df["clears_best_cheap"]][["table", "config", "verifier", "n", "verifier_acc", "best_cheap_acc", "best_cheap_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
