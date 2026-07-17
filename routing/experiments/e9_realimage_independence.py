"""
E9 -- Real-image independence check for the SDXL/FLUX deployment-shift experiment.

Reviewer 3 concern: after substituting AIGIBench's bundled PASCAL VOC real images
with ImageNet val/nature photographs (Section IV-B), the real half of the SDXL/FLUX
test sets is drawn from the same source used to train the SD 1.4 semantic probes'
real class. This means the "deployment shift" evaluation may not be fully
independent of the probes' training distribution: only the fake half (SDXL/FLUX
generations) is genuinely out-of-distribution, while the real half is in-distribution
by construction.

This script quantifies the effect: it splits stale-semantic, stale-FFT, and
GPT-5.5-cascade accuracy by true class (real vs. fake) on the SDXL/FLUX test sets,
to check whether the cascade's net accuracy gain over the stale semantic probe
(Table 9 / tab:e8_results) is attributable to the in-distribution real class or to
genuine recovery on the out-of-distribution fake class.

Uses the SigLIP2 configuration (sd14_siglip2_probe.pkl, sd14_fft_probe.pkl) since
those are the primary backbone probes retained in this environment; CLIP/DINOv2
probes used for the other Table 9 rows were not persisted to disk.

Produces: results/e9_class_breakdown.csv
"""
import os
import joblib
import numpy as np
import pandas as pd

BASE = "/workspace/benchmarks_ai_research/routing/"


def get_predictions_stale(feat_file, probe_pkl):
    clf, scaler = joblib.load(os.path.join(BASE, probe_pkl))
    df = pd.read_csv(os.path.join(BASE, feat_file))
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    Xs = scaler.transform(X)
    df = df[["path", "y_true"]].copy()
    df["y_pred"] = clf.predict(Xs)
    return df


def main():
    rows = []
    for ds in ["sdxl", "flux"]:
        df_sem = get_predictions_stale(f"{ds}_siglip2_features.csv", "sd14_siglip2_probe.pkl")
        df_sem = df_sem.rename(columns={"y_pred": "y_sem"})
        df_fft = get_predictions_stale(f"{ds}_fft_features.csv", "sd14_fft_probe.pkl")
        df_fft = df_fft.rename(columns={"y_pred": "y_fft"})
        df = df_fft.merge(df_sem[["path", "y_sem"]], on="path")

        gpt = pd.read_csv(os.path.join(BASE, f"{ds}_gpt55_siglip2_disagree_preds.csv"))
        df = df.merge(gpt[["path", "y_gpt55"]], on="path", how="left")

        disagree = df["y_fft"] != df["y_sem"]
        df["disagree"] = disagree
        df["cascade_pred"] = np.where(disagree, df["y_gpt55"], df["y_sem"])

        print(f"=== {ds.upper()} + SigLIP2 (n={len(df)}, disagree={int(disagree.sum())}) ===")
        for cls, name in [(0, "REAL"), (1, "FAKE")]:
            sub = df[df["y_true"] == cls]
            subd = sub[sub["disagree"]]
            row = {
                "dataset": ds.upper(), "class": name,
                "n": len(sub), "n_disagree": len(subd),
                "stale_sem_acc": float(np.mean(sub["y_sem"] == cls)),
                "stale_fft_acc": float(np.mean(sub["y_fft"] == cls)),
                "cascade_acc": float(np.mean(sub["cascade_pred"] == cls)),
                "stale_sem_acc_on_disagree": float(np.mean(subd["y_sem"] == cls)) if len(subd) else float("nan"),
                "stale_fft_acc_on_disagree": float(np.mean(subd["y_fft"] == cls)) if len(subd) else float("nan"),
                "cascade_acc_on_disagree": float(np.mean(subd["cascade_pred"] == cls)) if len(subd) else float("nan"),
            }
            row["cascade_gain_over_stale_sem"] = row["cascade_acc"] - row["stale_sem_acc"]
            rows.append(row)
            print(f"  {name}: n={row['n']} stale-sem={row['stale_sem_acc']:.3f} "
                  f"cascade={row['cascade_acc']:.3f} gain={row['cascade_gain_over_stale_sem']:+.3f}")
        print()

    out = pd.DataFrame(rows)
    out_path = os.path.join(BASE, "results", "e9_class_breakdown.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
