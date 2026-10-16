"""
routing.pipeline.tables.main_results — tab_main_results, tab_cheap_path,
tab_escalation.

Ports the computation in routing/regenerate_main_tables.py (which only
printed) into importable, CSV-writing functions, with ONE change: eval-set
membership now comes from pipeline.splits.get_canonical_eval_paths() instead
of this module's own inlined load_val_paths() fallback-to-30%-split logic.

Why this matters (do not revert without reading routing/README.md /
SPLITS.md): extractors/common.py's load_val_paths() silently falls back to a
reproducible-but-WRONG 30%-of-pool split if the expected
`{prefix}bandit_val_paths.npy` file is missing. regenerate_main_tables.py
carried an inlined copy of that same fallback. experiments/e0_fft_variant.py's
GenBuster run was quietly affected by this at some point, producing a
GenBuster FFT accuracy of 0.7413 in the old results/e0_fft_table.csv instead
of the paper's correct 0.727. pipeline.splits.get_canonical_eval_paths() has
NO such fallback -- it raises FileNotFoundError instead of silently
mis-splitting, so this class of bug cannot recur silently here.

CPU-only: no torch/cv2/decord import at module load time. pipeline.splits
(and therefore extractors.common, which imports those) is only imported
lazily inside get_eval_paths(), i.e. paid for at actual computation time, not
at `import` time.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.utils import resample

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["GenBuster", "SD14", "BigGAN"]
PREFIX = {"GenBuster": "", "SD14": "sd14_", "BigGAN": "biggan_"}

LATENCY = {
    "FFT": 2.5, "CNNSpot": 45.8, "UnivFD": 56.2,
    "CLIP": 10.4, "SigLIP2": 10.4, "DINOv2": 10.4,
}
ALWAYS_ESC_LATENCY = {"GenBuster": 190.2, "SD14": 83.2, "BigGAN": 70.6}
ENTROPY_LATENCY = {"GenBuster": 17.9, "SD14": 14.3, "BigGAN": 12.9}
EPS = 1e-12


def get_eval_paths(ds_name):
    """Canonical eval-set paths for `ds_name`, via pipeline.splits (lazy
    import -- see module docstring for why this is deferred, not top-level)."""
    import sys
    sys.path.insert(0, ROUTING_BASE)
    from pipeline.splits import get_canonical_eval_paths
    return get_canonical_eval_paths(ds_name)


def bootstrap_ci(correct, n=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct)
    scores = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(n)]
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return lo, hi


def load_all(ds_name):
    """Build the per-sample eval-set dataframe (y_true + every cheap/verifier
    prediction) for one dataset, restricted to the canonical eval set."""
    prefix = PREFIX[ds_name]
    eval_paths = get_eval_paths(ds_name)

    df_fft = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"))
    fft_cols = [c for c in df_fft.columns if c.startswith("f")]
    clf_fft, sc_fft = joblib.load(os.path.join(ROUTING_BASE, f"{prefix}fft_probe.pkl"))
    eval_mask = df_fft["path"].isin(eval_paths)
    df = df_fft[eval_mask][["path", "y_true"]].copy()
    df["y_fft"] = clf_fft.predict(sc_fft.transform(df_fft.loc[eval_mask, fft_cols].values.astype(np.float32)))

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

    coverage_warnings = []
    for col in ["y_cnn", "y_univfd", "y_clip", "y_siglip2", "y_dinov2", "y_qwen"]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            coverage_warnings.append(
                f"{ds_name}: {n_missing}/{len(df)} rows missing {col} -- coverage gap")

    df = df.dropna(subset=["y_cnn", "y_univfd", "y_clip", "y_siglip2", "y_dinov2", "y_qwen"]).reset_index(drop=True)
    return df, coverage_warnings


def fft_entropy(ds_name, df):
    prefix = PREFIX[ds_name]
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


def _acc_ci(y_pred, y_true):
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(int)
    acc = correct.mean()
    lo, hi = bootstrap_ci(correct)
    return acc, lo, hi


def _cascade(df, sem_col):
    disagree = df["y_fft"] != df[sem_col]
    pred = np.where(disagree, df["y_qwen"], df["y_fft"])
    return pred, disagree


def _compute_all(datasets=DATASETS):
    """Shared pass over all datasets; returns (main_results_rows,
    cheap_path_rows, escalation_rows, warnings) exactly mirroring
    regenerate_main_tables.py's main(), just captured into data structures
    instead of printed."""
    main_rows, cheap_rows, esc_rows, warnings = [], [], [], []

    for ds_name in datasets:
        df, cov_warn = load_all(ds_name)
        warnings.extend(cov_warn)
        df = fft_entropy(ds_name, df)
        n = len(df)

        for label, col in [("FFT", "y_fft"), ("CNNSpot", "y_cnn"), ("UnivFD", "y_univfd"),
                            ("CLIP", "y_clip"), ("SigLIP2", "y_siglip2"), ("DINOv2", "y_dinov2")]:
            acc, lo, hi = _acc_ci(df[col], df["y_true"])
            main_rows.append({"dataset": ds_name, "method": label, "n": n,
                               "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                               "latency_ms": LATENCY[label], "escalation_rate": 0.0})

        entropy_pred = np.where(df["fft_entropy"] > 0.5, df["y_qwen"], df["y_fft"])
        acc, lo, hi = _acc_ci(entropy_pred, df["y_true"])
        main_rows.append({"dataset": ds_name, "method": "Entropy", "n": n,
                           "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                           "latency_ms": ENTROPY_LATENCY[ds_name],
                           "escalation_rate": float((df["fft_entropy"] > 0.5).mean())})

        for label, sem_col in [("Casc.(C)", "y_clip"), ("Casc.(S)", "y_siglip2"), ("Casc.(D)", "y_dinov2")]:
            pred, disagree = _cascade(df, sem_col)
            acc, lo, hi = _acc_ci(pred, df["y_true"])
            esc_rate = float(disagree.mean())
            lat = LATENCY["FFT"] + LATENCY["SigLIP2"] + esc_rate * ALWAYS_ESC_LATENCY[ds_name]
            main_rows.append({"dataset": ds_name, "method": label, "n": n,
                               "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                               "latency_ms": lat, "escalation_rate": esc_rate})

        acc, lo, hi = _acc_ci(df["y_qwen"], df["y_true"])
        main_rows.append({"dataset": ds_name, "method": "Always esc.", "n": n,
                           "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                           "latency_ms": ALWAYS_ESC_LATENCY[ds_name], "escalation_rate": 1.0})

        agree = df["y_fft"] == df["y_siglip2"]
        n_agree, n_disagree = int(agree.sum()), int((~agree).sum())
        cheap_acc = (df.loc[agree, "y_fft"] == df.loc[agree, "y_true"]).mean() if n_agree else float("nan")
        fft_disagree_acc = (df.loc[~agree, "y_fft"] == df.loc[~agree, "y_true"]).mean() if n_disagree else float("nan")
        cheap_rows.append({
            "dataset": ds_name, "n_agree": n_agree, "cheap_acc": cheap_acc,
            "n_disagree": n_disagree, "esc_rate": n_disagree / n,
            "fft_disagree_acc": fft_disagree_acc,
        })

        if ds_name == "GenBuster":
            fft_correct = df["y_fft"] == df["y_true"]
            cases = {
                "FFT correct, models agree": (fft_correct & agree),
                "FFT correct, models disagree": (fft_correct & ~agree),
                "FFT wrong, models agree": (~fft_correct & agree),
                "FFT wrong, models disagree": (~fft_correct & ~agree),
            }
            for label, mask in cases.items():
                esc_rows.append({"case": label, "n": int(mask.sum()),
                                  "disagree_esc": float((~agree)[mask].mean()) if mask.sum() else float("nan")})

    return main_rows, cheap_rows, esc_rows, warnings


def compute_main_results(datasets=DATASETS):
    """tab_main_results: dataset x method accuracy/CI/latency/esc_rate."""
    main_rows, _, _, _ = _compute_all(datasets)
    return pd.DataFrame(main_rows)


def compute_cheap_path(datasets=DATASETS):
    """tab_cheap_path: agreement-path accuracy vs disagreement-escalation rate."""
    _, cheap_rows, _, _ = _compute_all(datasets)
    return pd.DataFrame(cheap_rows)


def compute_escalation(datasets=("GenBuster",)):
    """tab_escalation: GenBuster-only 2x2 case breakdown (FFT correct/wrong x
    models agree/disagree), matching currentpaper.tex Table~\\ref{tab:escalation}."""
    _, _, esc_rows, _ = _compute_all(datasets)
    return pd.DataFrame(esc_rows)


def compute_all_with_warnings(datasets=DATASETS):
    """Convenience for generate_paper_results.py's audit log: returns the
    three DataFrames plus the list of coverage warnings in one pass (so the
    audit log doesn't require calling _compute_all three separate times)."""
    main_rows, cheap_rows, esc_rows, warnings = _compute_all(datasets)
    return pd.DataFrame(main_rows), pd.DataFrame(cheap_rows), pd.DataFrame(esc_rows), warnings
