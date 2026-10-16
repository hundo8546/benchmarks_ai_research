"""
Reconstruct genuine confusion matrices for SDXL/FLUX x {SigLIP2, DINOv2} x GPT-5.5,
for both the "stale semantic probe alone" stage and the "full cascade" stage
(agreement path = stale semantic prediction, disagreement path = GPT-5.5 prediction).

This is a read-only reconciliation script for the IEEE Access peer-review response.
It does NOT modify any existing pipeline/table-generation code; it only reads
existing CSVs/pkls under routing/ and writes its own output files under
routing/pipeline/confusion_matrices/.

Joins performed, mirroring routing/experiments/e8_modern_diffusion.py's own logic
(load_stale_probe / get_predictions_stale / evaluate_dataset):
  1. {ds}_siglip2_features.csv (or dinov2) + sd14_{backbone}_fftvariant_probe.pkl
     -> stale semantic prediction (y_sem) for every row in the full eval set.
  2. {ds}_fft_features.csv + sd14_fft_probe.pkl
     -> stale FFT prediction (y_fft) for every row in the full eval set.
     disagreement mask = (y_fft != y_sem)
  3. {ds}_gpt55_{backbone}_disagree_preds.csv
     -> GPT-5.5 prediction (y_gpt55) on the disagreement subset.
  4. Full cascade prediction = y_sem where agree, y_gpt55 where disagree.

Outputs:
  routing/pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv
"""
import os
import joblib
import numpy as np
import pandas as pd

ROUTING_BASE = os.path.join(os.path.dirname(__file__), "..", "..")
ROUTING_BASE = os.path.abspath(ROUTING_BASE)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "SDXL": "sdxl_",
    "FLUX": "flux_",
}

BACKBONES = {
    "SigLIP2": {
        "feat_suffix": "siglip2_features.csv",
        "probe": "sd14_siglip2_fftvariant_probe.pkl",
    },
    "DINOv2": {
        "feat_suffix": "dinov2_features.csv",
        "probe": "sd14_dinov2_fftvariant_probe.pkl",
    },
}

FFT_PROBE = "sd14_fft_probe.pkl"


def load_stale_probe(probe_pkl, feat_file):
    clf, scaler = joblib.load(probe_pkl)
    df = pd.read_csv(feat_file, nrows=2)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    assert scaler.n_features_in_ == len(feat_cols), (
        f"Feature dim mismatch: probe expects {scaler.n_features_in_}, "
        f"file has {len(feat_cols)} ({feat_file})"
    )
    return clf, scaler


def get_predictions_stale(feat_file, probe_pkl):
    clf, scaler = load_stale_probe(probe_pkl, feat_file)
    df = pd.read_csv(feat_file)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    X_s = scaler.transform(X)
    y_pred = clf.predict(X_s)
    out = df[["path", "y_true"]].copy()
    out["y_pred"] = y_pred
    return out


def confusion_stats(y_true, y_pred):
    """Positive class = 1 (fake). Returns dict with TP/FP/TN/FN, recall, acc."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    n = TP + FN + TN + FP
    real_recall = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    fake_recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    overall_acc = (TP + TN) / n if n > 0 else float("nan")
    return dict(TP=TP, FP=FP, TN=TN, FN=FN, n=n,
                real_recall=real_recall, fake_recall=fake_recall,
                overall_accuracy=overall_acc)


def main():
    rows_out = []
    detail_lines = []

    for ds_name, prefix in DATASETS.items():
        for bb_name, bb_cfg in BACKBONES.items():
            feat_file = os.path.join(ROUTING_BASE, f"{prefix}{bb_cfg['feat_suffix']}")
            probe_pkl = os.path.join(ROUTING_BASE, bb_cfg["probe"])
            fft_file  = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")
            fft_probe = os.path.join(ROUTING_BASE, FFT_PROBE)
            gpt_file  = os.path.join(
                ROUTING_BASE, f"{prefix}gpt55_{bb_name.lower()}_disagree_preds.csv")

            if not (os.path.exists(feat_file) and os.path.exists(fft_file)
                    and os.path.exists(gpt_file)):
                detail_lines.append(f"SKIP {ds_name}/{bb_name}: missing a required file")
                continue

            df_sem = get_predictions_stale(feat_file, probe_pkl)
            df_fft = get_predictions_stale(fft_file, fft_probe)

            df = df_fft.rename(columns={"y_pred": "y_fft"}).merge(
                df_sem.rename(columns={"y_pred": "y_sem"})[["path", "y_sem"]],
                on="path", how="inner")

            n_total = len(df)
            y_true = df["y_true"].values.astype(int)
            y_fft = df["y_fft"].values.astype(int)
            y_sem = df["y_sem"].values.astype(int)
            disagree = (y_fft != y_sem)
            n_dis = int(disagree.sum())

            # ---- Stage 1: stale semantic probe alone, FULL eval set ----
            stale_stats = confusion_stats(y_true, y_sem)

            # ---- Stage 2: full cascade ----
            df_gpt = pd.read_csv(gpt_file)
            # GPT file only covers (a subset of) the disagreement rows.
            gpt_map = dict(zip(df_gpt["path"], df_gpt["y_gpt55"]))

            y_cascade = y_sem.copy()
            n_dis_covered = 0
            n_dis_missing_gpt = 0
            for i, p in enumerate(df["path"].values):
                if disagree[i]:
                    if p in gpt_map and not pd.isna(gpt_map[p]):
                        y_cascade[i] = int(gpt_map[p])
                        n_dis_covered += 1
                    else:
                        n_dis_missing_gpt += 1
                        # leave as stale semantic prediction (no verifier output)

            cascade_stats = confusion_stats(y_true, y_cascade)

            detail_lines.append(
                f"{ds_name}/{bb_name}: n_total={n_total} n_disagree={n_dis} "
                f"({n_dis/n_total:.3%}) n_dis_covered_by_gpt55={n_dis_covered} "
                f"n_dis_missing_gpt55={n_dis_missing_gpt} "
                f"gpt_csv_rows={len(df_gpt)}"
            )
            detail_lines.append(
                f"    STALE  real_recall={stale_stats['real_recall']:.4f} "
                f"fake_recall={stale_stats['fake_recall']:.4f} "
                f"overall_acc={stale_stats['overall_accuracy']:.4f} "
                f"(TP={stale_stats['TP']} FP={stale_stats['FP']} "
                f"TN={stale_stats['TN']} FN={stale_stats['FN']})"
            )
            detail_lines.append(
                f"    CASCADE real_recall={cascade_stats['real_recall']:.4f} "
                f"fake_recall={cascade_stats['fake_recall']:.4f} "
                f"overall_acc={cascade_stats['overall_accuracy']:.4f} "
                f"(TP={cascade_stats['TP']} FP={cascade_stats['FP']} "
                f"TN={cascade_stats['TN']} FN={cascade_stats['FN']})"
            )
            detail_lines.append(
                f"    0.5*(real_recall+fake_recall) STALE   = "
                f"{0.5*(stale_stats['real_recall']+stale_stats['fake_recall']):.4f}"
            )
            detail_lines.append(
                f"    0.5*(real_recall+fake_recall) CASCADE = "
                f"{0.5*(cascade_stats['real_recall']+cascade_stats['fake_recall']):.4f}"
            )
            detail_lines.append("")

            for stage, stats in [("stale", stale_stats), ("cascade", cascade_stats)]:
                rows_out.append({
                    "dataset": ds_name,
                    "backbone": bb_name,
                    "stage": stage,
                    "TP": stats["TP"], "FP": stats["FP"],
                    "TN": stats["TN"], "FN": stats["FN"],
                    "real_recall": stats["real_recall"],
                    "fake_recall": stats["fake_recall"],
                    "overall_accuracy": stats["overall_accuracy"],
                    "n": stats["n"],
                })

    out_df = pd.DataFrame(rows_out)
    out_csv = os.path.join(OUT_DIR, "sdxl_flux_confusion_matrices.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print()
    print("\n".join(detail_lines))

    with open(os.path.join(OUT_DIR, "reconstruct_log.txt"), "w") as f:
        f.write("\n".join(detail_lines))


if __name__ == "__main__":
    main()
