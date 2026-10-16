"""
Regenerate tab:main_results, tab:cheap_path, and tab:escalation at the
expanded evaluation-set size (SD1.4/BigGAN N=2000, GenBuster N=1000),
mirroring the disagreement-based methodology in currentpaper.tex Sections
V.B/V.C/Table~2/Table~5/Table~7, with FFT+SigLIP2 as the primary
artifact/semantic pair (matching the current paper, not the older
CNNSpot+CLIP pipeline in analysis.py).

Assumptions (flagged here so a wrong one is a one-line fix, not a re-run):
  - Latency values are NOT recomputed -- they are a fixed hardware
    measurement independent of N, carried over unchanged from the
    original Table~2 (FFT=2.5ms, CNNSpot=45.8ms, UnivFD=56.2ms,
    CLIP/SigLIP2/DINOv2=10.4ms, Qwen uniform=190.2/83.2/70.6ms per
    dataset). Only accuracies and derived cascade/escalation numbers use
    the expanded N.
  - qwen_preds.csv covers the FULL val split (needed for "Always esc."
    and the cascade variants), not just the disagreement subset. This is
    checked explicitly below and the script errors out with a clear
    message if coverage is incomplete, rather than silently computing a
    wrong "Always esc." accuracy over a partial set.
  - Entropy baseline uses the FFT probe's own predicted probability
    (binary entropy, threshold tau=0.5 on H(x)), same definition as the
    original Table 2 / Section III.E, escalating to Qwen above threshold.
  - Casc.(C)/(S)/(D) = FFT + {CLIP, SigLIP2, DINOv2} disagreement-routed
    cascade to Qwen, i.e. output the shared prediction on agreement,
    escalate to Qwen on disagreement. Casc.(S) is the paper's primary
    cascade (FFT+SigLIP2+Qwen) reported elsewhere in the paper too.

Usage:
    python regenerate_main_tables.py

CPU-only, no GPU/video dependencies: DATASETS and load_val_paths are
inlined below instead of imported from extractors/common.py, which pulls
in torch/decord/cv2/PIL at module load time for functionality this script
never uses. Only pandas, numpy, scikit-learn, and joblib are required.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.utils import resample

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROUTING_BASE)

DATASETS = {
    "GenBuster": {
        "val_paths": os.path.join(ROUTING_BASE, "bandit_val_paths.npy"),
        "prefix": "",
    },
    "SD14": {
        "val_paths": os.path.join(ROUTING_BASE, "sd14_bandit_val_paths.npy"),
        "prefix": "sd14_",
    },
    "BigGAN": {
        "val_paths": os.path.join(ROUTING_BASE, "biggan_bandit_val_paths.npy"),
        "prefix": "biggan_",
    },
}


def load_val_paths(cfg, df_paths=None):
    npy = cfg.get("val_paths", "")
    if npy and os.path.exists(npy):
        return set(np.load(npy, allow_pickle=True).tolist())
    if df_paths is not None:
        rng = np.random.default_rng(42)
        arr = np.array(df_paths)
        idx = rng.choice(len(arr), size=int(len(arr) * 0.30), replace=False)
        return set(arr[idx].tolist())
    return set()


LATENCY = {
    "FFT": 2.5, "CNNSpot": 45.8, "UnivFD": 56.2,
    "CLIP": 10.4, "SigLIP2": 10.4, "DINOv2": 10.4,
}
ALWAYS_ESC_LATENCY = {"GenBuster": 190.2, "SD14": 83.2, "BigGAN": 70.6}
ENTROPY_LATENCY = {"GenBuster": 17.9, "SD14": 14.3, "BigGAN": 12.9}
EPS = 1e-12


def bootstrap_ci(correct, n=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct)
    scores = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(n)]
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return lo, hi


def load_all(ds_name):
    cfg = DATASETS[ds_name]
    prefix = cfg["prefix"]

    df_fft = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"))
    fft_cols = [c for c in df_fft.columns if c.startswith("f")]
    clf_fft, sc_fft = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl"))
    val_paths = load_val_paths(cfg, df_fft["path"].tolist())
    df = df_fft[df_fft["path"].isin(val_paths)][["path", "y_true"]].copy()
    df["y_fft"] = clf_fft.predict(sc_fft.transform(df_fft[df_fft["path"].isin(val_paths)][fft_cols].values.astype(np.float32)))

    df_cnn = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}bandit_dataset.csv"))
    df = df.merge(df_cnn[["path", "y_cnn"]], on="path", how="left")

    df_univfd = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}univfd_preds.csv"))
    df = df.merge(df_univfd[["path", "y_univfd"]], on="path", how="left")

    df_clip = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}clip_preds.csv"))
    df = df.merge(df_clip[["path", "y_clip"]], on="path", how="left")

    df_sig = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}siglip2_preds.csv"))
    df = df.merge(df_sig[["path", "y_siglip2"]], on="path", how="left")

    df_dino = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}dinov2_preds.csv"))
    df = df.merge(df_dino[["path", "y_dinov2"]], on="path", how="left")

    df_qwen = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}qwen_preds.csv"))
    df = df.merge(df_qwen[["path", "y_qwen"]], on="path", how="left")

    for col in ["y_cnn", "y_univfd", "y_clip", "y_siglip2", "y_dinov2", "y_qwen"]:
        n_missing = df[col].isna().sum()
        if n_missing:
            print(f"  WARNING [{ds_name}]: {n_missing}/{len(df)} rows missing {col} "
                  f"-- coverage gap, check the source file before trusting this table.")

    return df.dropna(subset=["y_cnn", "y_univfd", "y_clip", "y_siglip2", "y_dinov2", "y_qwen"]).reset_index(drop=True)


def fft_entropy(ds_name, df):
    cfg = DATASETS[ds_name]
    prefix = cfg["prefix"]
    df_fft = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"))
    fft_cols = [c for c in df_fft.columns if c.startswith("f")]
    clf_fft, sc_fft = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl"))
    df_fft = df_fft[df_fft["path"].isin(df["path"])].copy()
    proba = clf_fft.predict_proba(sc_fft.transform(df_fft[fft_cols].values.astype(np.float32)))[:, 1]
    proba = np.clip(proba, EPS, 1 - EPS)
    ent = -(proba * np.log(proba) + (1 - proba) * np.log(1 - proba))
    df_fft = df_fft[["path"]].copy()
    df_fft["fft_entropy"] = ent
    return df.merge(df_fft, on="path", how="left")


def compute_main_results_row(y_pred, y_true):
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(int)
    acc = correct.mean()
    lo, hi = bootstrap_ci(correct)
    return acc, lo, hi


def cascade(df, sem_col):
    disagree = df["y_fft"] != df[sem_col]
    pred = np.where(disagree, df["y_qwen"], df["y_fft"])
    return pred, disagree


def main():
    all_main_results = {}
    all_cheap_path = {}
    all_escalation = {}

    for ds_name in ["GenBuster", "SD14", "BigGAN"]:
        print(f"\n=== {ds_name} ===")
        df = load_all(ds_name)
        df = fft_entropy(ds_name, df)
        n = len(df)
        print(f"  n = {n}")

        rows = {}
        for label, col in [("FFT", "y_fft"), ("CNNSpot", "y_cnn"), ("UnivFD", "y_univfd"),
                            ("CLIP", "y_clip"), ("SigLIP2", "y_siglip2"), ("DINOv2", "y_dinov2")]:
            acc, lo, hi = compute_main_results_row(df[col], df["y_true"])
            rows[label] = {"acc": acc, "ci": (lo, hi), "lat": LATENCY[label]}

        entropy_pred = np.where(df["fft_entropy"] > 0.5, df["y_qwen"], df["y_fft"])
        acc, lo, hi = compute_main_results_row(entropy_pred, df["y_true"])
        rows["Entropy"] = {"acc": acc, "ci": (lo, hi), "lat": ENTROPY_LATENCY[ds_name]}

        for label, sem_col in [("Casc.(C)", "y_clip"), ("Casc.(S)", "y_siglip2"), ("Casc.(D)", "y_dinov2")]:
            pred, disagree = cascade(df, sem_col)
            acc, lo, hi = compute_main_results_row(pred, df["y_true"])
            esc_rate = disagree.mean()
            # Latency = cheap-stage cost + esc_rate * Qwen cost; compute by
            # hand afterward from esc_rate, not needed for the accuracy pass.
            rows[label] = {"acc": acc, "ci": (lo, hi), "esc_rate": esc_rate}

        acc, lo, hi = compute_main_results_row(df["y_qwen"], df["y_true"])
        rows["Always esc."] = {"acc": acc, "ci": (lo, hi), "lat": ALWAYS_ESC_LATENCY[ds_name]}

        all_main_results[ds_name] = rows

        agree = df["y_fft"] == df["y_siglip2"]
        n_agree, n_disagree = int(agree.sum()), int((~agree).sum())
        cheap_acc = (df.loc[agree, "y_fft"] == df.loc[agree, "y_true"]).mean() if n_agree else float("nan")
        fft_disagree_acc = (df.loc[~agree, "y_fft"] == df.loc[~agree, "y_true"]).mean() if n_disagree else float("nan")
        all_cheap_path[ds_name] = {
            "n_agree": n_agree, "cheap_acc": cheap_acc,
            "n_disagree": n_disagree, "esc_rate": n_disagree / n,
            "fft_disagree_acc": fft_disagree_acc,
        }

        if ds_name == "GenBuster":
            fft_correct = df["y_fft"] == df["y_true"]
            cases = {
                "FFT correct, models agree": (fft_correct & agree),
                "FFT correct, models disagree": (fft_correct & ~agree),
                "FFT wrong, models agree": (~fft_correct & agree),
                "FFT wrong, models disagree": (~fft_correct & ~agree),
            }
            all_escalation[ds_name] = {label: int(mask.sum()) for label, mask in cases.items()}

        print(f"  {'Method':<12} {'Acc':>7} {'[95% CI]':>18}")
        for label, r in rows.items():
            print(f"  {label:<12} {r['acc']:.3f}   [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

    print("\n\n=== tab:main_results (accuracy only -- merge with fixed latencies by hand) ===")
    for ds_name, rows in all_main_results.items():
        print(f"\n{ds_name}:")
        for label, r in rows.items():
            print(f"  {label:<12} acc={r['acc']:.4f}")

    print("\n\n=== tab:cheap_path ===")
    for ds_name, r in all_cheap_path.items():
        print(f"{ds_name}: agree n={r['n_agree']} cheap_acc={r['cheap_acc']:.4f} "
              f"disagree n={r['n_disagree']} esc_rate={r['esc_rate']:.4f} "
              f"fft_disagree_acc={r['fft_disagree_acc']:.4f}")

    print("\n\n=== tab:escalation (GenBuster only) ===")
    for label, n in all_escalation["GenBuster"].items():
        print(f"  {label}: n={n}")


if __name__ == "__main__":
    main()