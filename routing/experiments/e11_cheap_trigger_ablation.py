"""
E11 -- Cheap-trigger ablation: does escalation need FFT's spectral signal, or
would any first-stage trigger of the same size do?

Reviewer 2's concern: FFT contributes nothing on agreement cases by
construction (the cascade falls back to the semantic probe's own prediction
there), so the disagreement rule's apparent value could in principle come
just from escalating *some* fixed-size subset of inputs to a stronger
verifier, regardless of whether FFT's spectral features pick out genuinely
hard cases. This isolates that by comparing the real FFT-vs-semantic
disagreement-routed cascade against a random-trigger cascade that escalates
an equally-sized random subset of inputs to the same verifier (Qwen),
holding escalation rate (and therefore latency/cost) fixed.

If disagreement-routing beats random-routing at matched escalation rate,
the FFT detector's actual spectral signal is doing the selection work (it
identifies cases where the semantic probe is likely wrong, not just an
arbitrary subset). If they are statistically indistinguishable, FFT is not
contributing more than "having some first-stage filter."

In-domain GenBuster / SD~1.4 / BigGAN, FFT + SigLIP2 + Qwen, matching
Table~2 (tab:main_results) escalation rates.

Produces: results/e11_cheap_trigger_ablation.csv
"""
import os
import joblib
import numpy as np
import pandas as pd

BASE = "/workspace/benchmarks_ai_research/routing/"
N_TRIALS = 2000
RNG = np.random.default_rng(42)

DATASETS = {
    "GenBuster": {"prefix": ""},
    "SD14": {"prefix": "sd14_"},
    "BigGAN": {"prefix": "biggan_"},
}


def predict_indomain(prefix, name):
    """Train (or load cached) in-domain probe, predict on its own val split."""
    fft_feat = pd.read_csv(os.path.join(BASE, f"{prefix}fft_features.csv"))
    sem_feat = pd.read_csv(os.path.join(BASE, f"{prefix}siglip2_features.csv"))
    val_paths = set(np.load(os.path.join(BASE, f"{prefix}bandit_val_paths.npy"), allow_pickle=True).tolist())

    fft_clf, fft_scaler = joblib.load(os.path.join(BASE, f"{prefix}fft_probe.pkl"))
    sem_clf, sem_scaler = joblib.load(os.path.join(BASE, f"{prefix}siglip2_probe.pkl"))

    fft_val = fft_feat[fft_feat["path"].isin(val_paths)].reset_index(drop=True)
    sem_val = sem_feat[sem_feat["path"].isin(val_paths)].reset_index(drop=True)

    fft_cols = [c for c in fft_val.columns if c.startswith("f")]
    sem_cols = [c for c in sem_val.columns if c.startswith("f")]

    fft_val = fft_val.copy()
    fft_val["y_fft"] = fft_clf.predict(fft_scaler.transform(fft_val[fft_cols].values.astype(np.float32)))
    sem_val = sem_val.copy()
    sem_val["y_sem"] = sem_clf.predict(sem_scaler.transform(sem_val[sem_cols].values.astype(np.float32)))

    df = fft_val[["path", "y_true", "y_fft"]].merge(sem_val[["path", "y_sem"]], on="path")

    qwen = pd.read_csv(os.path.join(BASE, f"{prefix}qwen_preds.csv"))[["path", "y_qwen"]]
    df = df.merge(qwen, on="path", how="inner")
    return df


def main():
    rows = []
    for name, cfg in DATASETS.items():
        df = predict_indomain(cfg["prefix"], name)
        y_true = df["y_true"].values
        y_fft = df["y_fft"].values
        y_sem = df["y_sem"].values
        y_qwen = df["y_qwen"].values
        n = len(df)

        disagree = (y_fft != y_sem)
        n_dis = int(disagree.sum())
        esc_rate = n_dis / n

        cascade_pred = np.where(disagree, y_qwen, y_sem)
        real_acc = float(np.mean(cascade_pred == y_true))

        # Random-trigger baseline: escalate a random subset of the same size,
        # averaged over N_TRIALS draws.
        random_accs = np.empty(N_TRIALS)
        for t in range(N_TRIALS):
            sel = np.zeros(n, dtype=bool)
            idx = RNG.choice(n, size=n_dis, replace=False)
            sel[idx] = True
            pred = np.where(sel, y_qwen, y_sem)
            random_accs[t] = np.mean(pred == y_true)

        rand_mean = float(random_accs.mean())
        rand_lo, rand_hi = np.percentile(random_accs, [2.5, 97.5])
        sig = "yes" if (real_acc < rand_lo or real_acc > rand_hi) else "no"

        print(f"{name:>10}: n={n:4d} esc_rate={esc_rate:.3f}  "
              f"disagreement-cascade={real_acc:.3f}  "
              f"random-trigger={rand_mean:.3f} [{rand_lo:.3f},{rand_hi:.3f}]  "
              f"disagreement-outside-random-CI={sig}")

        rows.append({
            "dataset": name, "n": n, "escalation_rate": esc_rate,
            "disagreement_cascade_acc": real_acc,
            "random_trigger_mean_acc": rand_mean,
            "random_trigger_ci_lo": rand_lo, "random_trigger_ci_hi": rand_hi,
            "disagreement_beats_random_at_95": sig,
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "results", "e11_cheap_trigger_ablation.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
