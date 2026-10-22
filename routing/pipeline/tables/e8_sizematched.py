"""
routing.pipeline.tables.e8_sizematched — SDXL/FLUX deployment-shift cascade on the
size-matched images (remainingupdates1.md Step 4), side by side with the original run.

  compute_e8_sizematched()   tab_e8_cascade_sizematched: one row per
                             (variant in {original, sizematched}) x (SDXL, FLUX) x (SigLIP2, DINOv2)

Both variants go through the same code. "original" reads the original E8 feature CSVs and
results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv, so its stale-probe, disagreement and
cascade values reproduce tab_e8_cascade / tab_positive_cell_audit; "sizematched" reads
{sdxl,flux}_sizematched_*_features.csv (pipeline/sizematched_features.py) and
CONSOLIDATED_full_sdxl_flux_sizematched.csv (pipeline/gpt55_sizematched.py). Every image
was center-cropped and resized to 512x512 PNG (pipeline/robustness/build_sizematched_sdxl_flux.py).

The SAME stale SD1.4-trained probes are applied (e8.E8_DATASETS paths); nothing is retrained.

Columns: stale probe accuracy (semantic, FFT); disagreement rate; GPT-5.5 vs. best cheap
detector on the disagreement subset with a paired-bootstrap 95% CI; cascade accuracy and its
gain over the stale semantic probe with a paired-bootstrap 95% CI; the cascade vs. random
routing at the same escalation rate (expected accuracy r*uniform + (1-r)*stale, paired-bootstrap
CI on the per-sample difference), i.e. whether disagreement picks better inputs to escalate than
chance; uniform GPT-5.5 accuracy;
cascade cost share (Batch-tier token cost of escalated images / all images); GPT-5.5 real and
fake recall, uniform and on the disagreement subset; mean GPT-5.5 input tokens by class.

CPU-only; no model or API calls.
"""
import os
import sys

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_DIR = os.path.join(ROUTING_BASE, "results", "batch_requests")
GPT55_CSV = {
    "original": os.path.join(BATCH_DIR, "CONSOLIDATED_full_sdxl_flux.csv"),
    "sizematched": os.path.join(BATCH_DIR, "CONSOLIDATED_full_sdxl_flux_sizematched.csv"),
}
VARIANTS = ["original", "sizematched"]
N_BOOT, SEED = 10000, 42


def _modules():
    sys.path.insert(0, ROUTING_BASE)
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))
    import e8_modern_diffusion as e8
    import baseline_utils as bu
    from pipeline.tables import e8_cascade
    return e8, bu, e8_cascade


def _prefix(e8, ds_name, variant):
    base = e8.E8_DATASETS[ds_name]["prefix"]
    return base if variant == "original" else f"{ds_name.lower()}_sizematched_"


def load_frame(ds_name, sem_name, variant, e8):
    cfg = e8.E8_DATASETS[ds_name]
    prefix = _prefix(e8, ds_name, variant)
    probe_key = {"SigLIP2": "stale_probe_siglip2", "DINOv2": "stale_probe_dinov2"}[sem_name]
    sem = e8.get_predictions_stale(os.path.join(ROUTING_BASE, f"{prefix}{e8.SEMANTICS[sem_name]}.csv"),
                                   cfg[probe_key], label=f"stale {sem_name}")
    fft = e8.get_predictions_stale(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"),
                                   cfg["stale_fft_probe"], label="stale FFT")
    df = fft.rename(columns={"y_pred": "y_fft"})[["path", "y_true", "y_fft"]].merge(
        sem.rename(columns={"y_pred": "y_sem"})[["path", "y_sem"]], on="path", how="inner")
    df["disagree"] = df["y_fft"] != df["y_sem"]

    gpt = pd.read_csv(GPT55_CSV[variant]).set_index("path")
    missing = ~df["path"].isin(gpt.index)
    if missing.any():
        raise ValueError(f"{variant}/{ds_name}/{sem_name}: {int(missing.sum())} images lack a GPT-5.5 label")
    g = gpt.loc[df["path"]]
    if g["y_verifier"].isna().any():
        raise ValueError(f"{variant}/{ds_name}/{sem_name}: {int(g['y_verifier'].isna().sum())} invalid GPT-5.5 labels")
    df["y_gpt55"] = g["y_verifier"].astype(int).values
    df["input_tokens"] = g["input_tokens"].values
    df["output_tokens"] = g["output_tokens"].values
    return df


def _acc(pred, y):
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def _recall(pred, y, cls):
    m = np.asarray(y) == cls
    return float(np.mean(np.asarray(pred)[m] == cls)) if m.any() else float("nan")


def _mean_ci(d, n_boot=N_BOOT, seed=SEED):
    """Paired-bootstrap 95% CI on the mean of per-sample differences d (same resampling
    convention as baseline_utils.paired_bootstrap_diff_ci)."""
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), size=len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_e8_sizematched():
    e8, bu, ec = _modules()
    rows = []
    for variant in VARIANTS:
        for ds_name, sem_name in ec.E8_CONFIGS:
            df = load_frame(ds_name, sem_name, variant, e8)
            y, y_sem, y_fft, y_gpt = (df[c].values for c in ["y_true", "y_sem", "y_fft", "y_gpt55"])
            dis = df["disagree"].values
            y_casc = np.where(dis, y_gpt, y_sem)

            d = df[dis]
            cheap = {"StaleFFT": d["y_fft"].values, f"Stale{sem_name}": d["y_sem"].values}
            best = max(cheap, key=lambda k: _acc(cheap[k], d["y_true"].values))
            dis_gap, dis_lo, dis_hi = bu.paired_bootstrap_diff_ci(
                d["y_true"].values, d["y_gpt55"].values, cheap[best], n_boot=N_BOOT, seed=SEED)
            gain, gain_lo, gain_hi = bu.paired_bootstrap_diff_ci(y, y_casc, y_sem, n_boot=N_BOOT, seed=SEED)
            # Random routing at the same escalation rate r: escalating a uniformly random
            # r-fraction has expected per-sample correctness r*gpt + (1-r)*sem.
            r = dis.mean()
            c_gpt, c_sem = (y_gpt == y).astype(float), (y_sem == y).astype(float)
            c_rand = r * c_gpt + (1 - r) * c_sem
            vs_rand, vs_rand_lo, vs_rand_hi = _mean_ci(np.where(dis, c_gpt, c_sem) - c_rand)

            cost_all = ec._token_cost(df)
            cost_esc = ec._token_cost(d)
            rows.append({
                "variant": variant, "dataset": ds_name, "semantic": sem_name,
                "n_total": len(df), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                "stale_sem_acc": _acc(y_sem, y), "stale_fft_acc": _acc(y_fft, y),
                "n_disagree": int(dis.sum()), "disagree_rate": float(dis.mean()),
                "dis_n_real": int((d["y_true"] == 0).sum()), "dis_n_fake": int((d["y_true"] == 1).sum()),
                "dis_gpt55_acc": _acc(d["y_gpt55"], d["y_true"]),
                "dis_best_cheap": best, "dis_best_cheap_acc": _acc(cheap[best], d["y_true"]),
                "dis_gap": dis_gap, "dis_gap_ci_lo": dis_lo, "dis_gap_ci_hi": dis_hi,
                "cascade_acc": _acc(y_casc, y),
                "gain_over_stale": gain, "gain_ci_lo": gain_lo, "gain_ci_hi": gain_hi,
                "random_routing_acc": float(c_rand.mean()),
                "cascade_minus_random_routing": vs_rand,
                "cascade_minus_random_ci_lo": vs_rand_lo, "cascade_minus_random_ci_hi": vs_rand_hi,
                "uniform_gpt55_acc": _acc(y_gpt, y),
                "cascade_cost_share": cost_esc / cost_all,
                "gpt55_real_recall_uniform": _recall(y_gpt, y, 0),
                "gpt55_fake_recall_uniform": _recall(y_gpt, y, 1),
                "gpt55_real_recall_dis": _recall(d["y_gpt55"], d["y_true"], 0),
                "gpt55_fake_recall_dis": _recall(d["y_gpt55"], d["y_true"], 1),
                "mean_input_tokens_real": float(df.loc[df.y_true == 0, "input_tokens"].mean()),
                "mean_input_tokens_fake": float(df.loc[df.y_true == 1, "input_tokens"].mean()),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = compute_e8_sizematched()
    path = os.path.join(ROUTING_BASE, "results", "tab_e8_cascade_sizematched.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {path}")
