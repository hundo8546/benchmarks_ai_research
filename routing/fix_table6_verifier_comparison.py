"""
Fix Table 6 (verifier_comparison): the existing gpt55_disagree_preds.csv /
Qwen accuracy numbers reported there were computed on the CNNSpot+CLIP
disagreement subset (see experiments/e4_gpt55.py get_disagree_subset), not
the FFT+SigLIP2 disagreement subset the rest of the paper's cascade uses
(Tables 3-5, 9, 10, 14). This recomputes the correct FFT+SigLIP2
disagreement subset per dataset, evaluates Qwen accuracy on it (Qwen
predictions already cover the full val set, so no new calls needed), and
reports how much of that subset already has a GPT-5.5 prediction from the
old (CNNSpot+CLIP-based) run vs. how many new GPT-5.5 calls would be needed
to fully cover it.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extractors"))
from extractors.common import DATASETS, ROUTING_BASE, load_val_paths

DATASET_CFG = [
    ("GenBuster", "", "qwen_preds.csv", "gpt55_disagree_preds.csv"),
    ("SD14", "sd14_", "sd14_qwen_preds.csv", "sd14_gpt55_disagree_preds.csv"),
    ("BigGAN", "biggan_", "biggan_qwen_preds.csv", "biggan_gpt55_disagree_preds.csv"),
]


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
    return df


def main():
    rows = []
    for ds_name, prefix, qwen_file, gpt_file in DATASET_CFG:
        print(f"\n=== {ds_name} ===")
        df = get_fft_siglip2_disagree_val(ds_name)
        df_dis = df[df["disagree"]].reset_index(drop=True)
        n_dis = len(df_dis)
        print(f"  FFT+SigLIP2 disagreement subset: n={n_dis} "
              f"(vs old CNNSpot+CLIP subset used in current Table 6)")

        # Qwen accuracy on the CORRECT disagreement subset (full val coverage exists)
        df_qwen = pd.read_csv(os.path.join(ROUTING_BASE, qwen_file))
        qwen_lookup = dict(zip(df_qwen["path"], df_qwen["y_qwen"]))
        y_qwen = df_dis["path"].map(qwen_lookup)
        valid_qwen = y_qwen.notna()
        qwen_acc = float(np.mean(y_qwen[valid_qwen] == df_dis.loc[valid_qwen, "y_true"]))
        print(f"  Qwen accuracy on correct disagreement subset: {qwen_acc:.3f} "
              f"(n={int(valid_qwen.sum())}/{n_dis} covered)")

        # GPT-5.5: how much of the correct subset already has a prediction
        # from the old CNNSpot+CLIP-based disagree_preds file?
        gpt_path = os.path.join(ROUTING_BASE, gpt_file)
        covered_acc, n_covered, n_missing = float("nan"), 0, n_dis
        if os.path.exists(gpt_path):
            df_gpt_old = pd.read_csv(gpt_path)
            gpt_lookup = dict(zip(df_gpt_old["path"], df_gpt_old["y_verifier"]))
            y_gpt = df_dis["path"].map(gpt_lookup)
            covered = y_gpt.notna()
            n_covered = int(covered.sum())
            n_missing = n_dis - n_covered
            if n_covered > 0:
                covered_acc = float(np.mean(y_gpt[covered] == df_dis.loc[covered, "y_true"]))
            print(f"  GPT-5.5 coverage of correct subset: {n_covered}/{n_dis} already have "
                  f"a prediction from the old run, {n_missing} would need new calls")
            print(f"  GPT-5.5 accuracy on the {n_covered} already-covered correct-subset images: "
                  f"{covered_acc:.3f}")

        rows.append({
            "dataset": ds_name,
            "n_disagree_fft_siglip2": n_dis,
            "qwen_acc_correct_subset": qwen_acc,
            "n_qwen_covered": int(valid_qwen.sum()),
            "gpt55_n_covered_of_correct_subset": n_covered,
            "gpt55_n_missing_needs_new_calls": n_missing,
            "gpt55_acc_on_covered_correct_subset": covered_acc,
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(ROUTING_BASE, "results", "table6_fft_siglip2_fix.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
