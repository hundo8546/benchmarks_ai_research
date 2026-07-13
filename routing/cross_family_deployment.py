"""
Cross-family deployment experiment.

For each (train, test) dataset pair where train != test:
- Take the SigLIP2 probe trained on the train dataset (stale probe).
- Apply it to the test dataset.
- Build the disagreement-routed cascade using the FFT artifact detector
  (trained in-domain on the test dataset, matching Tables 3-5) + stale SigLIP2 + Qwen.
- Compare against (a) stale SigLIP2 only, (b) in-domain SigLIP2 only, (c) FFT-only.

FFT is trained in-domain per test dataset rather than transferred cross-family,
since the rest of the paper reports FFT accuracy as an in-domain quantity
(Tables 3-5) and FFT's spectral features are not family-specific in the way the
semantic probe is (Section VI.D). This matches reviewer request to add an
FFT-only reference column to the cross-family deployment table so cascade gain
can be checked against the stronger of the two cheap detectors, not just CNNSpot.

Uses SigLIP2 rather than CLIP as the semantic detector: this is the backbone
used throughout the paper's primary results (Tables 3-5, 9, 10, 14), whereas
this script previously used CLIP, a secondary comparison backbone whose raw
feature vectors are not available for SD14/BigGAN in this environment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = "/workspace/benchmarks_ai_research/routing/"

DATASETS = {
    "GenBuster": {
        "prefix": "",
        "qwen_file": "qwen_preds.csv",
        "val_paths": "bandit_val_paths.npy",
        "siglip2_features": "siglip2_features.csv",
        "siglip2_probe": "siglip2_probe.pkl",
        "fft_features": "fft_features.csv",
        "fft_probe": "fft_probe.pkl",
    },
    "SD14": {
        "prefix": "sd14_",
        "qwen_file": "sd14_qwen_preds.csv",
        "val_paths": "sd14_bandit_val_paths.npy",
        "siglip2_features": "sd14_siglip2_features.csv",
        "siglip2_probe": "sd14_siglip2_probe.pkl",
        "fft_features": "sd14_fft_features.csv",
        "fft_probe": "sd14_fft_probe.pkl",
    },
    "BigGAN": {
        "prefix": "biggan_",
        "qwen_file": "biggan_qwen_preds.csv",
        "val_paths": "biggan_bandit_val_paths.npy",
        "siglip2_features": "biggan_siglip2_features.csv",
        "siglip2_probe": "biggan_siglip2_probe.pkl",
        "fft_features": "biggan_fft_features.csv",
        "fft_probe": "biggan_fft_probe.pkl",
    },
}


def get_fft_preds(name, val_paths):
    """Train (or load cached) in-domain FFT probe for `name`, predict on its val split."""
    cfg = DATASETS[name]
    df = pd.read_csv(os.path.join(BASE, cfg["fft_features"]))
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)

    val_set = set(val_paths.tolist())
    val_mask = df["path"].isin(val_set)
    train_mask = ~val_mask

    probe_pkl = os.path.join(BASE, cfg["fft_probe"])
    if os.path.exists(probe_pkl):
        clf, scaler = joblib.load(probe_pkl)
    else:
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
        joblib.dump((clf, scaler), probe_pkl)

    X_val = scaler.transform(X[val_mask])
    y_fft = clf.predict(X_val).astype(int)
    fft_lookup = dict(zip(df.loc[val_mask, "path"], y_fft))
    return fft_lookup


def load_dataset(name):
    cfg = DATASETS[name]

    df_cnn = pd.read_csv(os.path.join(BASE, f"{cfg['prefix']}bandit_dataset.csv"))
    df_qwen = pd.read_csv(os.path.join(BASE, cfg["qwen_file"]))

    qwen_cols = [c for c in ["path", "y_qwen"] if c in df_qwen.columns]
    df = df_cnn.merge(df_qwen[qwen_cols], on="path", how="inner")

    val_paths = np.load(os.path.join(BASE, cfg["val_paths"]), allow_pickle=True)

    df_sem = pd.read_csv(os.path.join(BASE, cfg["siglip2_features"]))
    feat_cols = [c for c in df_sem.columns if c.startswith("f")]
    sem_lookup = {row["path"]: row[feat_cols].values.astype(np.float32)
                  for _, row in df_sem.iterrows()}

    model, scaler = joblib.load(os.path.join(BASE, cfg["siglip2_probe"]))

    fft_lookup = get_fft_preds(name, val_paths)
    df["y_fft"] = df["path"].map(fft_lookup)
    df = df.dropna(subset=["y_fft"]).reset_index(drop=True)
    df["y_fft"] = df["y_fft"].astype(int)

    return df, val_paths, sem_lookup, model, scaler, feat_cols


def apply_probe_to_val(model, scaler, df, val_paths, sem_lookup, feat_cols):
    val_df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    missing = [p for p in val_df["path"] if p not in sem_lookup]
    if missing:
        print(f"  WARNING: {len(missing)} val paths missing from siglip2 features, dropping")
        val_df = val_df[val_df["path"].isin(sem_lookup)].reset_index(drop=True)

    X = np.stack([sem_lookup[p] for p in val_df["path"].values])
    Xs = scaler.transform(X)
    y_sem = model.predict(Xs).astype(int)
    return val_df, y_sem


def evaluate(val_df, y_sem):
    y_true = val_df["y_true"].values
    y_fft = val_df["y_fft"].values
    y_qwen = val_df["y_qwen"].values

    disagree = (y_fft != y_sem).astype(int)
    pred_cascade = np.where(disagree == 1, y_qwen, y_fft)

    n_dis = int(disagree.sum())
    qwen_on_dis = float(np.mean(y_qwen[disagree == 1] == y_true[disagree == 1])) if n_dis > 0 else float("nan")

    return {
        "siglip2_acc": float(np.mean(y_sem == y_true)),
        "fft_acc": float(np.mean(y_fft == y_true)),
        "cascade_acc": float(np.mean(pred_cascade == y_true)),
        "disagree_rate": float(np.mean(disagree)),
        "n_disagree": n_dis,
        "qwen_on_disagree": qwen_on_dis,
    }


def main():
    print("Loading datasets and probes...")
    data = {}
    for name in DATASETS:
        df, vp, sem_lookup, model, scaler, feat_cols = load_dataset(name)
        data[name] = {
            "df": df, "val_paths": vp, "sem_lookup": sem_lookup,
            "model": model, "scaler": scaler, "feat_cols": feat_cols,
        }
        print(f"  {name}: {len(df)} samples, {len(vp)} val, {len(sem_lookup)} siglip2 features")

    print("\n" + "=" * 108)
    print("Cross-family deployment experiment (FFT + SigLIP2)")
    print("=" * 108)
    print(f"{'Train':<10} {'Test':<10} {'FFT':>8} {'StaleSigLIP2':>13} {'Cascade':>10} {'Delta':>8} {'InDomSig2':>11} {'QwenOnDis':>11} {'DisRate':>9}")
    print("-" * 108)

    rows = []
    for train_name in DATASETS:
        for test_name in DATASETS:
            train_d = data[train_name]
            test_d = data[test_name]

            # Stale: apply train_name's SigLIP2 probe to test_name's data
            val_df, y_stale = apply_probe_to_val(
                train_d["model"], train_d["scaler"],
                test_d["df"], test_d["val_paths"],
                test_d["sem_lookup"], test_d["feat_cols"],
            )
            stale = evaluate(val_df, y_stale)

            # In-domain: apply test_name's own SigLIP2 probe
            _, y_indom = apply_probe_to_val(
                test_d["model"], test_d["scaler"],
                test_d["df"], test_d["val_paths"],
                test_d["sem_lookup"], test_d["feat_cols"],
            )
            indom = evaluate(val_df, y_indom)

            delta = stale["cascade_acc"] - stale["siglip2_acc"]
            delta_vs_fft = stale["cascade_acc"] - stale["fft_acc"]
            tag = " *" if train_name == test_name else ""

            print(
                f"{train_name:<10} {test_name:<10} "
                f"{stale['fft_acc']:>8.3f} "
                f"{stale['siglip2_acc']:>13.3f} "
                f"{stale['cascade_acc']:>10.3f} "
                f"{delta:>+8.3f} "
                f"{indom['siglip2_acc']:>11.3f} "
                f"{stale['qwen_on_disagree']:>11.3f} "
                f"{stale['disagree_rate']:>9.3f}{tag}"
            )

            rows.append({
                "train": train_name, "test": test_name,
                "fft_acc": stale["fft_acc"],
                "stale_siglip2_acc": stale["siglip2_acc"],
                "cascade_acc": stale["cascade_acc"],
                "delta": delta,
                "delta_vs_fft": delta_vs_fft,
                "indomain_siglip2_acc": indom["siglip2_acc"],
                "qwen_on_disagree": stale["qwen_on_disagree"],
                "disagree_rate": stale["disagree_rate"],
                "n_disagree": stale["n_disagree"],
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(BASE, "cross_family_deployment.csv"), index=False)
    print(f"\nSaved cross_family_deployment.csv  (* = in-domain row)")

    print("\n" + "=" * 108)
    print("Off-diagonal summary: does the cascade recover accuracy lost by stale SigLIP2, and beat FFT-only?")
    print("=" * 108)
    off = df_out[df_out["train"] != df_out["test"]]
    for _, r in off.iterrows():
        sign = "+" if r["delta"] >= 0 else ""
        sign_fft = "+" if r["delta_vs_fft"] >= 0 else ""
        print(f"  {r['train']:>10} -> {r['test']:<10}  "
              f"FFT-only {r['fft_acc']:.3f}  "
              f"stale SigLIP2 {r['stale_siglip2_acc']:.3f}  "
              f"cascade {r['cascade_acc']:.3f}  "
              f"delta-vs-stale {sign}{r['delta']:.3f}  "
              f"delta-vs-FFT {sign_fft}{r['delta_vs_fft']:.3f}  "
              f"(Qwen on disagree {r['qwen_on_disagree']:.3f})")

    print(f"\nMean cascade-vs-stale delta across off-diagonal: {off['delta'].mean():+.3f}")
    print(f"Mean cascade-vs-FFT-only delta across off-diagonal: {off['delta_vs_fft'].mean():+.3f}")
    print(f"Cells where cascade beats stale SigLIP2: {(off['delta'] > 0).sum()}/{len(off)}")
    print(f"Cells where cascade beats FFT-only:      {(off['delta_vs_fft'] > 0).sum()}/{len(off)}")


if __name__ == "__main__":
    main()
