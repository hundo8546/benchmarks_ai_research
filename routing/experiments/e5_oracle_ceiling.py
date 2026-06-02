"""
E5 — Oracle Ceiling Experiment.
Simulates a perfect verifier at controlled accuracy levels (55%–95%) on the
disagreement subset to find the break-even point where the cascade beats
the semantic-only baseline.

No GPU required — runs on pre-computed disagreement subsets.

Produces: results/e5_oracle_results.csv
Required output: break-even verifier accuracy per dataset.

Usage:
    python e5_oracle_ceiling.py [--datasets GenBuster SD14 BigGAN]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir, load_val_paths

SIM_ACCURACIES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
N_TRIALS = 200  # Monte-Carlo trials per accuracy level
RNG_SEED = 42


def load_val_df(ds_name):
    """Load merged val-set dataframe with CNN, CLIP predictions."""
    cfg = DATASETS[ds_name]

    if not os.path.exists(cfg["cnnspot_csv"]) or not os.path.exists(cfg["clip_preds"]):
        return None

    df_cnn = pd.read_csv(cfg["cnnspot_csv"])
    df_clip = pd.read_csv(cfg["clip_preds"])
    df = df_cnn.merge(df_clip[["path", "y_clip", "clip_proba"]], on="path", how="inner")
    vp = load_val_paths(cfg, df["path"].tolist())
    df = df[df["path"].isin(vp)].reset_index(drop=True)
    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
    return df


def simulate_oracle_cascade(df, verifier_acc, n_trials, rng):
    """
    Simulate a verifier with a given accuracy on the disagreement subset.
    On agreement: use cheap-path (semantic/CNN, whichever is correct).
    On disagree: use oracle verifier that is correct with prob = verifier_acc.

    Returns mean cascade accuracy over n_trials.
    """
    y_true = df["y_true"].values
    y_clip = df["y_clip"].values
    y_cnn = df["y_cnn"].values
    disagree = df["disagree"].values.astype(bool)

    n = len(df)
    accs = []
    for _ in range(n_trials):
        # Agreement cases: use y_clip (same as disagree-ensemble baseline)
        pred = y_clip.copy()

        # Disagreement cases: oracle is correct with prob verifier_acc
        dis_idx = np.where(disagree)[0]
        if len(dis_idx) > 0:
            y_true_dis = y_true[dis_idx]
            # Oracle: flip correct answer with prob (1 - verifier_acc)
            correct_mask = rng.random(len(dis_idx)) < verifier_acc
            oracle_pred = y_true_dis.copy()
            # When wrong, flip label
            oracle_pred[~correct_mask] = 1 - oracle_pred[~correct_mask]
            pred[dis_idx] = oracle_pred

        accs.append(float(np.mean(pred == y_true)))
    return float(np.mean(accs)), float(np.std(accs))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    rng = np.random.default_rng(RNG_SEED)
    all_rows = []

    for ds_name in args.datasets:
        df = load_val_df(ds_name)
        if df is None:
            print(f"Skipping {ds_name}: missing data")
            continue

        # Compute baselines
        clip_acc = float(np.mean(df["y_clip"] == df["y_true"]))
        cnn_acc = float(np.mean(df["y_cnn"] == df["y_true"]))
        dis_rate = float(df["disagree"].mean())
        n_dis = int(df["disagree"].sum())

        print(f"\n=== {ds_name} ===")
        print(f"  CLIP-only accuracy: {clip_acc:.3f} (this is the target to beat)")
        print(f"  CNN-only accuracy:  {cnn_acc:.3f}")
        print(f"  Disagreement rate:  {dis_rate:.3f} ({n_dis}/{len(df)} samples)")
        print(f"\n  {'VerifierAcc':>12} {'CascadeAcc':>12} {'±Std':>8} {'Delta':>8} {'Beats?':>8}")
        print("  " + "-" * 55)

        break_even = None
        for v_acc in SIM_ACCURACIES:
            casc_acc, casc_std = simulate_oracle_cascade(df, v_acc, N_TRIALS, rng)
            delta = casc_acc - clip_acc
            beats = casc_acc > clip_acc
            if beats and break_even is None:
                break_even = v_acc
            mark = " *" if beats else ""
            print(f"  {v_acc:>12.0%} {casc_acc:>12.3f} {casc_std:>8.3f} "
                  f"{delta:>+8.3f}{mark}")

            all_rows.append({
                "dataset": ds_name,
                "verifier_accuracy": v_acc,
                "cascade_accuracy": casc_acc,
                "cascade_std": casc_std,
                "delta_vs_clip": delta,
                "beats_clip_baseline": beats,
                "clip_baseline": clip_acc,
                "cnn_baseline": cnn_acc,
                "disagree_rate": dis_rate,
            })

        if break_even is not None:
            print(f"\n  Break-even verifier accuracy: {break_even:.0%}")
        else:
            print(f"\n  No break-even found in [55%, 100%] range")
            print(f"  NOTE: Cascade cannot beat CLIP-only even with a perfect verifier")
            print(f"  This is consistent with disagree subset being harder for VLMs")
            # Compute theoretical maximum (perfect oracle on disagree)
            casc_perfect, _ = simulate_oracle_cascade(df, 1.0, 1, rng)
            print(f"  Perfect oracle cascade: {casc_perfect:.3f} vs CLIP: {clip_acc:.3f}")

    df_out = pd.DataFrame(all_rows)
    out_file = os.path.join(RESULTS_DIR, "e5_oracle_results.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")

    # Summary: break-even per dataset
    print("\n=== Summary: Break-Even Points ===")
    for ds_name in args.datasets:
        sub = df_out[df_out["dataset"] == ds_name]
        if len(sub) == 0:
            continue
        clip_base = sub["clip_baseline"].iloc[0]
        winners = sub[sub["beats_clip_baseline"]]
        if len(winners) > 0:
            be = winners["verifier_accuracy"].min()
            print(f"  {ds_name}: break-even at {be:.0%} verifier accuracy "
                  f"(vs CLIP baseline {clip_base:.3f})")
        else:
            print(f"  {ds_name}: no break-even (CLIP baseline={clip_base:.3f} too strong)")

    print("\nE5 complete.")


if __name__ == "__main__":
    main()
