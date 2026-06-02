"""
E0-FFT — Rerun E0 baseline substituting FFT probe for CNNSpot.
Generates FFT predictions on the val split (trains/loads probe via E3 convention),
then evaluates the same methods as E0: FFT-only, Semantic-only, Disagree-ensemble,
Disagree-thresh (Qwen escalation), Always-escalate.

Produces: results/e0_fft_table.csv, results/e0_fft_disagreement_stats.csv
"""
import os, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import (
    DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir,
    bootstrap_ci, load_val_paths,
)

LATENCY_FFT_MS   = 2.5
LATENCY_SEM_MS   = 10.4   # CLIP
LATENCY_CHEAP_MS = LATENCY_FFT_MS + LATENCY_SEM_MS


SEMANTICS = {
    "CLIP":    "clip_features",
    "SigLIP2": "siglip2_features",
    "DINOv2":  "dinov2_features",
}
SEM_PREDS_FILES = {
    "CLIP":    "clip_preds",      # key in DATASETS cfg
}


def get_fft_preds(ds_name):
    cfg = DATASETS[ds_name]
    feat_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_features.csv")
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}fft_probe.pkl")
    df = pd.read_csv(feat_csv)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)
    val_paths = load_val_paths(cfg, df["path"].tolist())
    val_mask = df["path"].isin(val_paths)
    train_mask = ~val_mask
    if os.path.exists(probe_pkl):
        clf, scaler = joblib.load(probe_pkl)
        print(f"  Loaded FFT probe from {probe_pkl}")
    else:
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
        joblib.dump((clf, scaler), probe_pkl)
        print(f"  Trained + saved FFT probe")
    X_val = scaler.transform(X[val_mask])
    y_fft = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_val["y_fft"] = y_fft
    df_val["prob_fft"] = y_prob
    return df_val[["path", "y_true", "y_fft", "prob_fft"]]


def get_semantic_preds(ds_name, sem_name):
    cfg = DATASETS[ds_name]
    feat_csv = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{SEMANTICS[sem_name]}.csv")
    probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}{sem_name.lower()}_probe.pkl")
    # Also try clip_classifier.pkl convention
    if not os.path.exists(probe_pkl) and sem_name == "CLIP":
        probe_pkl = os.path.join(ROUTING_BASE, f"{cfg['prefix']}clip_classifier.pkl")
    if not os.path.exists(feat_csv):
        return None
    df = pd.read_csv(feat_csv)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(int)
    val_paths = load_val_paths(cfg, df["path"].tolist())
    val_mask = df["path"].isin(val_paths)
    train_mask = ~val_mask
    sem_probe_pkl = os.path.join(ROUTING_BASE,
        f"{cfg['prefix']}{sem_name.lower()}_fftvariant_probe.pkl")
    clf, scaler = None, None
    if os.path.exists(sem_probe_pkl):
        clf, scaler = joblib.load(sem_probe_pkl)
        if scaler.n_features_in_ != X.shape[1]:
            clf, scaler = None, None  # stale dims, retrain
    if clf is None:
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(scaler.fit_transform(X[train_mask]), y[train_mask])
        joblib.dump((clf, scaler), sem_probe_pkl)
    X_val = scaler.transform(X[val_mask])
    y_sem = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    df_val = df[val_mask].copy().reset_index(drop=True)
    df_val["y_sem"] = y_sem
    df_val["prob_sem"] = y_prob
    return df_val[["path", "y_true", "y_sem", "prob_sem"]]


def evaluate(df, ds_name, sem_name):
    y_true  = df["y_true"].values
    y_fft   = df["y_fft"].values
    y_sem   = df["y_sem"].values
    y_qwen  = df["y_qwen"].values
    lat_q   = df["latency_qwen_ms"].values
    in_tok  = df["qwen_input_tokens"].values
    out_tok = df["qwen_output_tokens"].values
    disagree = (y_fft != y_sem).astype(bool)

    def mkrow(name, y_pred, lat, esc=0.0, in_t=0.0, out_t=0.0):
        correct = (y_pred == y_true).astype(int).tolist()
        acc = np.mean(correct)
        lo, hi = bootstrap_ci(correct)
        return {"dataset": ds_name, "semantic": sem_name, "method": name,
                "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
                "mean_latency_ms": lat, "escalation_rate": esc,
                "mean_input_tokens": in_t, "mean_output_tokens": out_t}

    rows = []
    rows.append(mkrow("FFT-only",      y_fft, LATENCY_FFT_MS))
    rows.append(mkrow("Semantic-only", y_sem, LATENCY_SEM_MS))

    # Disagree ensemble (trust semantic on disagreement — same as semantic-only in accuracy)
    ens = np.where(disagree, y_sem, y_fft)
    rows.append(mkrow("Disagree-ensemble", ens, LATENCY_CHEAP_MS))

    # Disagree → Qwen
    dis_pred = np.where(disagree, y_qwen, y_fft)
    lats     = np.where(disagree, LATENCY_CHEAP_MS + lat_q, LATENCY_CHEAP_MS)
    in_t     = np.where(disagree, in_tok,  0.0)
    out_t    = np.where(disagree, out_tok, 0.0)
    rows.append(mkrow("Disagree→Qwen", dis_pred, float(np.mean(lats)),
                      float(np.mean(disagree)), float(np.mean(in_t)), float(np.mean(out_t))))

    rows.append(mkrow("Always-Qwen", y_qwen,
                      float(np.mean(LATENCY_CHEAP_MS + lat_q)), 1.0,
                      float(np.mean(in_tok)), float(np.mean(out_tok))))
    return pd.DataFrame(rows)


def dis_stats(df, ds_name, sem_name):
    d = (df["y_fft"] != df["y_sem"]).astype(bool)
    a = ~d
    y = df["y_true"].values
    return {
        "dataset": ds_name, "semantic": sem_name,
        "n_val": len(df),
        "disagree_rate": float(d.mean()),
        "agree_fft_acc":  float(np.mean(df.loc[a,"y_fft"] == df.loc[a,"y_true"])) if a.sum() else float("nan"),
        "agree_sem_acc":  float(np.mean(df.loc[a,"y_sem"] == df.loc[a,"y_true"])) if a.sum() else float("nan"),
        "dis_fft_acc":    float(np.mean(df.loc[d,"y_fft"] == df.loc[d,"y_true"])) if d.sum() else float("nan"),
        "dis_sem_acc":    float(np.mean(df.loc[d,"y_sem"] == df.loc[d,"y_true"])) if d.sum() else float("nan"),
        "dis_qwen_acc":   float(np.mean(df.loc[d,"y_qwen"] == df.loc[d,"y_true"])) if d.sum() else float("nan"),
        "fft_overall_acc": float(np.mean(df["y_fft"] == df["y_true"])),
        "sem_overall_acc": float(np.mean(df["y_sem"] == df["y_true"])),
    }


def main():
    ensure_results_dir()
    all_rows, all_stats = [], []

    for ds_name in DATASETS:
        cfg = DATASETS[ds_name]
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")

        df_fft = get_fft_preds(ds_name)

        df_qwen = pd.read_csv(cfg["qwen_preds"])
        qcols = ["path","y_qwen","latency_qwen_ms","qwen_input_tokens","qwen_output_tokens"]
        qcols = [c for c in qcols if c in df_qwen.columns]
        df_qwen = df_qwen[qcols]
        for col in ["latency_qwen_ms","qwen_input_tokens","qwen_output_tokens"]:
            if col not in df_qwen.columns:
                df_qwen[col] = 0.0

        for sem_name in SEMANTICS:
            df_sem = get_semantic_preds(ds_name, sem_name)
            if df_sem is None:
                print(f"  {sem_name}: missing features, skip")
                continue

            df = df_fft.merge(df_sem[["path","y_sem","prob_sem"]], on="path", how="inner")
            df = df.merge(df_qwen, on="path", how="inner")
            df["latency_qwen_ms"] = df["latency_qwen_ms"].fillna(0.0)
            df["qwen_input_tokens"] = df["qwen_input_tokens"].fillna(0.0)
            df["qwen_output_tokens"] = df["qwen_output_tokens"].fillna(0.0)

            print(f"\n  Semantic: {sem_name}  (n={len(df)})")
            res = evaluate(df, ds_name, sem_name)
            st  = dis_stats(df, ds_name, sem_name)
            all_rows.append(res)
            all_stats.append(st)

            print(f"  {'Method':<22} {'Acc':>7}  {'Esc':>6}")
            print("  " + "-"*38)
            for _, r in res.iterrows():
                print(f"  {r.method:<22} {r.accuracy:>7.3f}  {r.escalation_rate:>6.2f}")
            print(f"  Disagree rate: {st['disagree_rate']:.3f}  |  "
                  f"FFT on disagree: {st['dis_fft_acc']:.3f}  "
                  f"Sem on disagree: {st['dis_sem_acc']:.3f}  "
                  f"Qwen on disagree: {st['dis_qwen_acc']:.3f}")

    df_out = pd.concat(all_rows, ignore_index=True)
    df_out.to_csv(os.path.join(RESULTS_DIR, "e0_fft_table.csv"), index=False)
    pd.DataFrame(all_stats).to_csv(os.path.join(RESULTS_DIR, "e0_fft_disagreement_stats.csv"), index=False)
    print("\nSaved e0_fft_table.csv and e0_fft_disagreement_stats.csv")


if __name__ == "__main__":
    main()
