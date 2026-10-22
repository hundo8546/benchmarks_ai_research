"""
routing.pipeline.tables.e8_cascade — SDXL/FLUX deployment-shift cascade tables.

  compute_e8_cascade()             tab_e8_cascade: stale-semantic vs. cascade accuracy, GPT-5.5
                                   and Qwen2.5-VL-7B verifiers (Table e8_results cells)
  compute_cost_accuracy_tradeoff() tab_cost_accuracy_tradeoff: cheap-only / cascade / uniform
                                   GPT-5.5 accuracy and Batch-API cost (Table cost_accuracy_tradeoff)
  compute_real_image_sources()     tab_real_image_sources: cascade under the PASCAL VOC and COCO
                                   real-image sources (Table real_image_sources)

Verifier labels:
  GPT-5.5, ImageNet reals   results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv (every image)
  GPT-5.5, VOC/COCO reals   results/batch_requests/CONSOLIDATED_{original_voc,coco}_union.csv
                            (every image in either backbone's disagreement subset)
  Qwen2.5-VL-7B             {prefix}qwen_{sem}_disagree_preds.csv; uncovered disagreement
                            samples keep the stale semantic prediction (n reported)

Cost: Batch-tier gpt-5.5 pricing (claude_context_notes/openaitokentable.txt) applied to the
per-call token counts the uniform run recorded. The cascade's cost is the token cost of the
images it actually escalates, not uniform cost scaled by the escalation rate.

CPU-only; no model or API calls.
"""
import os
import sys

import numpy as np
import pandas as pd

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_DIR = os.path.join(ROUTING_BASE, "results", "batch_requests")
UNIFORM_CSV = os.path.join(BATCH_DIR, "CONSOLIDATED_full_sdxl_flux.csv")
REAL_SOURCE_LABELS = {
    "original": os.path.join(BATCH_DIR, "CONSOLIDATED_original_voc_union.csv"),
    "coco": os.path.join(BATCH_DIR, "CONSOLIDATED_coco_union.csv"),
}
REAL_SOURCE_TAG = {"original": "", "coco": "_coco"}

BATCH_USD_PER_M_INPUT = 2.50
BATCH_USD_PER_M_OUTPUT = 15.00

E8_CONFIGS = [("SDXL", "SigLIP2"), ("SDXL", "DINOv2"), ("FLUX", "SigLIP2"), ("FLUX", "DINOv2")]


def _modules():
    sys.path.insert(0, ROUTING_BASE)
    sys.path.insert(0, os.path.join(ROUTING_BASE, "extractors"))
    sys.path.insert(0, os.path.join(ROUTING_BASE, "experiments"))
    import e8_modern_diffusion as e8
    from pipeline.tables import confidence_routing as cr
    return e8, cr


def _token_cost(df_tokens):
    return float(df_tokens["input_tokens"].sum() * BATCH_USD_PER_M_INPUT / 1e6
                 + df_tokens["output_tokens"].sum() * BATCH_USD_PER_M_OUTPUT / 1e6)


def _acc(pred, y_true):
    return float(np.mean(np.asarray(pred) == np.asarray(y_true)))


def compute_e8_cascade():
    e8, cr = _modules()
    rows = []
    for ds_name, sem_name in E8_CONFIGS:
        df = cr.load_e8_frame(ds_name, sem_name, e8)
        dis = df["disagree"].values
        y_true, y_sem = df["y_true"].values, df["y_sem"].values
        stale = _acc(y_sem, y_true)
        casc_gpt = _acc(np.where(dis, df["y_gpt55"].values, y_sem), y_true)

        prefix = e8.E8_DATASETS[ds_name]["prefix"]
        df_q = pd.read_csv(os.path.join(ROUTING_BASE, f"{prefix}qwen_{sem_name.lower()}_disagree_preds.csv"))
        y_qwen = df["path"].map(dict(zip(df_q["path"], df_q["y_qwen"])))
        covered = dis & y_qwen.notna().values
        y_casc_q = np.where(covered, y_qwen.fillna(-1).values.astype(int), y_sem)
        casc_qwen = _acc(y_casc_q, y_true)

        rows.append({
            "dataset": ds_name, "semantic": sem_name, "n_total": len(df),
            "n_disagree": int(dis.sum()), "esc_rate": float(dis.mean()),
            "stale_sem_acc": stale,
            "cascade_acc_gpt55": casc_gpt, "gain_gpt55": casc_gpt - stale,
            "n_qwen_covered": int(covered.sum()), "n_qwen_fallback_to_semantic": int(dis.sum() - covered.sum()),
            "cascade_acc_qwen": casc_qwen, "gain_qwen": casc_qwen - stale,
        })
    return pd.DataFrame(rows)


def compute_cost_accuracy_tradeoff():
    e8, cr = _modules()
    tokens = pd.read_csv(UNIFORM_CSV).set_index("path")[["y_verifier", "input_tokens", "output_tokens"]]
    rows = []
    for ds_name, sem_name in E8_CONFIGS:
        df = cr.load_e8_frame(ds_name, sem_name, e8)
        dis = df["disagree"].values
        y_true, y_sem, y_gpt = df["y_true"].values, df["y_sem"].values, df["y_gpt55"].values
        tok_all = tokens.loc[df["path"]]
        tok_esc = tokens.loc[df["path"][dis]]
        uniform_cost, cascade_cost = _token_cost(tok_all), _token_cost(tok_esc)
        cascade_acc, uniform_acc = _acc(np.where(dis, y_gpt, y_sem), y_true), _acc(y_gpt, y_true)
        rows.append({
            "dataset": ds_name, "semantic": sem_name, "n_total": len(df), "esc_rate": float(dis.mean()),
            "cheap_only_acc": _acc(y_sem, y_true),
            "cascade_acc": cascade_acc, "cascade_cost_usd": cascade_cost,
            "uniform_acc": uniform_acc, "uniform_cost_usd": uniform_cost,
            "gap_to_uniform": uniform_acc - cascade_acc,
            "cascade_cost_frac_of_uniform": cascade_cost / uniform_cost,
            "cost_saved_vs_uniform_pct": 100 * (1 - cascade_cost / uniform_cost),
            "mean_input_tokens_all": float(tok_all["input_tokens"].mean()),
            "mean_output_tokens_all": float(tok_all["output_tokens"].mean()),
            "mean_input_tokens_escalated": float(tok_esc["input_tokens"].mean()),
            "mean_output_tokens_escalated": float(tok_esc["output_tokens"].mean()),
        })
    return pd.DataFrame(rows)


def compute_positive_cell_audit():
    """Per SDXL/FLUX GPT-5.5 cell: the disagreement subset's size and class mix, GPT-5.5
    vs. best-cheap accuracy on it, and a paired-bootstrap CI on the gap."""
    e8, cr = _modules()
    import baseline_utils as bu
    rows = []
    for ds_name, sem_name in E8_CONFIGS:
        df = cr.load_e8_frame(ds_name, sem_name, e8)
        d = df[df["disagree"]]
        y, y_gpt = d["y_true"].values, d["y_gpt55"].values
        cheap = {"StaleFFT": d["y_fft"].values, f"Stale{sem_name}": d["y_sem"].values}
        best_name = max(cheap, key=lambda k: _acc(cheap[k], y))
        diff, lo, hi = bu.paired_bootstrap_diff_ci(y, y_gpt, cheap[best_name], n_boot=10000, seed=42)
        rows.append({
            "dataset": ds_name, "semantic": sem_name, "n_disagree": len(d),
            "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
            "gpt55_coverage": float(d["y_gpt55"].notna().mean()),
            "gpt55_acc": _acc(y_gpt, y), "best_cheap": best_name, "best_cheap_acc": _acc(cheap[best_name], y),
            "gap": diff, "gap_ci_lo": lo, "gap_ci_hi": hi,
        })
    return pd.DataFrame(rows)


def _real_source_frame(source, ds_name, sem_name, e8):
    """Existing SDXL/FLUX fakes plus the source's 2,000 reals, stale probes applied."""
    cfg = e8.E8_DATASETS[ds_name]
    prefix, feat_key = cfg["prefix"], e8.SEMANTICS[sem_name]
    probe_key = {"SigLIP2": "stale_probe_siglip2", "DINOv2": "stale_probe_dinov2"}[sem_name]
    real_prefix = os.path.join(ROUTING_BASE, "results", f"{ds_name.lower()}{REAL_SOURCE_TAG[source]}_realonly_")

    def preds(fake_file, real_file, probe, col):
        fake = e8.get_predictions_stale(fake_file, probe, label=col)
        real = e8.get_predictions_stale(real_file, probe, label=col)
        both = pd.concat([fake[fake["y_true"] == 1], real[real["y_true"] == 0]], ignore_index=True)
        return both[["path", "y_true", "y_pred"]].rename(columns={"y_pred": col})

    sem = preds(os.path.join(ROUTING_BASE, f"{prefix}{feat_key}.csv"), f"{real_prefix}{feat_key}.csv",
                cfg[probe_key], "y_sem")
    fft = preds(os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv"), f"{real_prefix}fft_features.csv",
                cfg["stale_fft_probe"], "y_fft")
    df = fft.merge(sem[["path", "y_sem"]], on="path", how="inner")
    df["disagree"] = df["y_fft"] != df["y_sem"]
    return df


def compute_real_image_sources():
    e8, _ = _modules()
    rows = []
    for source, label_csv in REAL_SOURCE_LABELS.items():
        labels = pd.read_csv(label_csv)
        lookup = dict(zip(labels["path"], labels["y_verifier"]))
        for ds_name, sem_name in E8_CONFIGS:
            df = _real_source_frame(source, ds_name, sem_name, e8)
            dis = df["disagree"].values
            y_gpt = df["path"].map(lookup)
            n_missing = int((dis & y_gpt.isna().values).sum())
            if n_missing:
                raise ValueError(f"{source}/{ds_name}/{sem_name}: {n_missing} disagreement samples lack a GPT-5.5 label")
            y_true, y_sem = df["y_true"].values, df["y_sem"].values
            y_casc = np.where(dis, y_gpt.fillna(-1).values.astype(int), y_sem)
            stale, casc = _acc(y_sem, y_true), _acc(y_casc, y_true)
            rows.append({
                "source": source, "dataset": ds_name, "semantic": sem_name,
                "n_total": len(df), "n_real": int((y_true == 0).sum()), "n_fake": int((y_true == 1).sum()),
                "n_disagree": int(dis.sum()),
                "agreement_path_acc": _acc(df["y_fft"].values[~dis], y_true[~dis]),
                "stale_sem_acc": stale, "cascade_acc_gpt55": casc, "gain_over_stale_semantic": casc - stale,
            })
    return pd.DataFrame(rows)
