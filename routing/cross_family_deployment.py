"""
Cross-family deployment experiment.

For each (train, test) dataset pair where train != test:
- Take the CLIP probe trained on the train dataset (stale probe).
- Apply it to the test dataset.
- Build the disagreement-routed cascade using stale CLIP + CNN + Qwen.
- Compare against (a) stale CLIP only, (b) in-domain CLIP only.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE = "/workspace/benchmarks_ai_research/routing/"

DATASETS = {
    "GenBuster": {
        "prefix": "",
        "qwen_file": "qwen_preds.csv",
        "val_paths": "bandit_val_paths.npy",
        "clip_features": "clip_features.csv",
        "clip_classifier": "clip_classifier.pkl",
    },
    "SD14": {
        "prefix": "sd14_",
        "qwen_file": "sd14_qwen_preds.csv",
        "val_paths": "sd14_bandit_val_paths.npy",
        "clip_features": "sd14_clip_features.csv",
        "clip_classifier": "sd14_clip_classifier.pkl",
    },
    "BigGAN": {
        "prefix": "biggan_",
        "qwen_file": "biggan_qwen_preds.csv",
        "val_paths": "biggan_bandit_val_paths.npy",
        "clip_features": "biggan_clip_features.csv",
        "clip_classifier": "biggan_clip_classifier.pkl",
    },
}


def load_dataset(name):
    cfg = DATASETS[name]

    df_cnn = pd.read_csv(os.path.join(BASE, f"{cfg['prefix']}bandit_dataset.csv"))
    df_qwen = pd.read_csv(os.path.join(BASE, cfg["qwen_file"]))

    qwen_cols = [c for c in ["path", "y_qwen"] if c in df_qwen.columns]
    df = df_cnn.merge(df_qwen[qwen_cols], on="path", how="inner")

    val_paths = np.load(os.path.join(BASE, cfg["val_paths"]), allow_pickle=True)

    df_clip = pd.read_csv(os.path.join(BASE, cfg["clip_features"]))
    feat_cols = [c for c in df_clip.columns if c.startswith("f")]
    clip_lookup = {row["path"]: row[feat_cols].values.astype(np.float32)
                   for _, row in df_clip.iterrows()}

    model, scaler = joblib.load(os.path.join(BASE, cfg["clip_classifier"]))

    return df, val_paths, clip_lookup, model, scaler, feat_cols


def apply_probe_to_val(model, scaler, df, val_paths, clip_lookup, feat_cols):
    val_df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    missing = [p for p in val_df["path"] if p not in clip_lookup]
    if missing:
        print(f"  WARNING: {len(missing)} val paths missing from clip features, dropping")
        val_df = val_df[val_df["path"].isin(clip_lookup)].reset_index(drop=True)

    X = np.stack([clip_lookup[p] for p in val_df["path"].values])
    Xs = scaler.transform(X)
    y_clip = model.predict(Xs).astype(int)
    return val_df, y_clip


def evaluate(val_df, y_clip):
    y_true = val_df["y_true"].values
    y_cnn = val_df["y_cnn"].values
    y_qwen = val_df["y_qwen"].values

    disagree = (y_cnn != y_clip).astype(int)
    pred_cascade = np.where(disagree == 1, y_qwen, y_cnn)

    n_dis = int(disagree.sum())
    qwen_on_dis = float(np.mean(y_qwen[disagree == 1] == y_true[disagree == 1])) if n_dis > 0 else float("nan")

    return {
        "clip_acc": float(np.mean(y_clip == y_true)),
        "cnn_acc": float(np.mean(y_cnn == y_true)),
        "cascade_acc": float(np.mean(pred_cascade == y_true)),
        "disagree_rate": float(np.mean(disagree)),
        "n_disagree": n_dis,
        "qwen_on_disagree": qwen_on_dis,
    }


def main():
    print("Loading datasets and probes...")
    data = {}
    for name in DATASETS:
        df, vp, clip_lookup, model, scaler, feat_cols = load_dataset(name)
        data[name] = {
            "df": df, "val_paths": vp, "clip_lookup": clip_lookup,
            "model": model, "scaler": scaler, "feat_cols": feat_cols,
        }
        print(f"  {name}: {len(df)} samples, {len(vp)} val, {len(clip_lookup)} clip features")

    print("\n" + "=" * 95)
    print("Cross-family deployment experiment")
    print("=" * 95)
    print(f"{'Train':<10} {'Test':<10} {'StaleCLIP':>10} {'Cascade':>10} {'Delta':>8} {'InDomCLIP':>11} {'QwenOnDis':>11} {'DisRate':>9}")
    print("-" * 95)

    rows = []
    for train_name in DATASETS:
        for test_name in DATASETS:
            train_d = data[train_name]
            test_d = data[test_name]

            # Stale: apply train_name's CLIP probe to test_name's data
            val_df, y_stale = apply_probe_to_val(
                train_d["model"], train_d["scaler"],
                test_d["df"], test_d["val_paths"],
                test_d["clip_lookup"], test_d["feat_cols"],
            )
            stale = evaluate(val_df, y_stale)

            # In-domain: apply test_name's own CLIP probe
            _, y_indom = apply_probe_to_val(
                test_d["model"], test_d["scaler"],
                test_d["df"], test_d["val_paths"],
                test_d["clip_lookup"], test_d["feat_cols"],
            )
            indom = evaluate(val_df, y_indom)

            delta = stale["cascade_acc"] - stale["clip_acc"]
            tag = " *" if train_name == test_name else ""

            print(
                f"{train_name:<10} {test_name:<10} "
                f"{stale['clip_acc']:>10.3f} "
                f"{stale['cascade_acc']:>10.3f} "
                f"{delta:>+8.3f} "
                f"{indom['clip_acc']:>11.3f} "
                f"{stale['qwen_on_disagree']:>11.3f} "
                f"{stale['disagree_rate']:>9.3f}{tag}"
            )

            rows.append({
                "train": train_name, "test": test_name,
                "stale_clip_acc": stale["clip_acc"],
                "cascade_acc": stale["cascade_acc"],
                "delta": delta,
                "indomain_clip_acc": indom["clip_acc"],
                "qwen_on_disagree": stale["qwen_on_disagree"],
                "disagree_rate": stale["disagree_rate"],
                "n_disagree": stale["n_disagree"],
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(BASE, "cross_family_deployment.csv"), index=False)
    print(f"\nSaved cross_family_deployment.csv  (* = in-domain row)")

    print("\n" + "=" * 95)
    print("Off-diagonal summary: does the cascade recover accuracy lost by stale CLIP?")
    print("=" * 95)
    off = df_out[df_out["train"] != df_out["test"]]
    for _, r in off.iterrows():
        sign = "+" if r["delta"] >= 0 else ""
        print(f"  {r['train']:>10} -> {r['test']:<10}  "
              f"stale CLIP {r['stale_clip_acc']:.3f}  "
              f"cascade {r['cascade_acc']:.3f}  "
              f"delta {sign}{r['delta']:.3f}  "
              f"(Qwen on disagree {r['qwen_on_disagree']:.3f})")

    print(f"\nMean cascade-vs-stale delta across off-diagonal: {off['delta'].mean():+.3f}")
    print(f"Cells where cascade beats stale CLIP: {(off['delta'] > 0).sum()}/{len(off)}")


if __name__ == "__main__":
    main()