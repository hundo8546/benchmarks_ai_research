"""
E8 — Modern Diffusion Benchmark Extension.

Tests the cascade under genuine cross-family deployment:
  - Semantic probe trained on legacy SD1.4 (in-distribution for old diffusion)
  - Evaluated zero-shot on SDXL and FLUX1-dev (out-of-distribution)
  - FFT artifact detector evaluated on same test sets
  - GPT-5.5 as verifier on disagreement cases

This is the key experiment for the revised paper claim:
  "Disagreement routing is valuable when the semantic probe is stale.
   With a frontier VLM verifier, cascade accuracy exceeds any individual
   component in the cross-family deployment scenario."

Steps:
  1. --extract_features   Extract CLIP, SigLIP2, DINOv2, FFT features for SDXL/FLUX
  2. --evaluate           Run cascade evaluation (uses SD14-trained probes zero-shot)
  3. --run_gpt55          Run GPT-5.5 on disagreement cases (requires OPENAI_API_KEY)

Produces: results/e8_modern_diffusion.csv

Usage:
    python e8_modern_diffusion.py --extract_features --datasets SDXL FLUX
    python e8_modern_diffusion.py --evaluate
    python e8_modern_diffusion.py --run_gpt55 --datasets SDXL FLUX
    python e8_modern_diffusion.py --evaluate   # re-run after GPT-5.5
"""
import argparse
import base64
import io
import os
import sys
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import (
    DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir,
    bootstrap_ci, load_index,
)

# ── Dataset configs for SDXL and FLUX ──────────────────────────────────────
E8_DATASETS = {
    "SDXL": {
        "index":       os.path.join(ROUTING_BASE, "sdxl_index.csv"),
        "is_video":    False,
        "prefix":      "sdxl_",
        "generator":   "sdxl",
        # Stale probes: trained on SD14, applied zero-shot here
        "stale_probe_clip":    os.path.join(ROUTING_BASE, "sd14_clip_fftvariant_probe.pkl"),
        "stale_probe_siglip2": os.path.join(ROUTING_BASE, "sd14_siglip2_fftvariant_probe.pkl"),
        "stale_probe_dinov2":  os.path.join(ROUTING_BASE, "sd14_dinov2_fftvariant_probe.pkl"),
        "stale_fft_probe":     os.path.join(ROUTING_BASE, "sd14_fft_probe.pkl"),
    },
    "FLUX": {
        "index":       os.path.join(ROUTING_BASE, "flux_index.csv"),
        "is_video":    False,
        "prefix":      "flux_",
        "generator":   "flux1-dev",
        "stale_probe_clip":    os.path.join(ROUTING_BASE, "sd14_clip_fftvariant_probe.pkl"),
        "stale_probe_siglip2": os.path.join(ROUTING_BASE, "sd14_siglip2_fftvariant_probe.pkl"),
        "stale_probe_dinov2":  os.path.join(ROUTING_BASE, "sd14_dinov2_fftvariant_probe.pkl"),
        "stale_fft_probe":     os.path.join(ROUTING_BASE, "sd14_fft_probe.pkl"),
    },
}

SEMANTICS = {
    "CLIP":    "clip_features",
    "SigLIP2": "siglip2_features",
    "DINOv2":  "dinov2_features",
}

LATENCY_FFT_MS  = 2.5
LATENCY_SEM_MS  = 10.4
LATENCY_CHEAP   = LATENCY_FFT_MS + LATENCY_SEM_MS

GPT_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL "
    "photograph or an AI-generated FAKE image. Consider both "
    "possibilities equally. Answer with exactly one word: REAL or FAKE."
)
MAX_RETRIES = 5
RETRY_BASE  = 2.0


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_features(ds_name, cfg, device="cuda"):
    """Extract CLIP, SigLIP2, DINOv2, FFT features for an E8 dataset."""
    from extractors.extract_fft import extract_features as fft_compute
    index_rows = load_index(cfg["index"])
    prefix = cfg["prefix"]

    # ── FFT ─────────────────────────────────────────────────────────────────
    fft_out = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")
    if not os.path.exists(fft_out):
        print(f"  Extracting FFT features for {ds_name} ...")
        rows = []
        for r in tqdm(index_rows, desc=f"FFT/{ds_name}"):
            try:
                img = Image.open(r["path"]).convert("RGB")
                feats = fft_compute(img)
                row = {"path": r["path"], "y_true": int(r["label"]),
                       "generator": r["generator"]}
                row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            except Exception as e:
                print(f"    skip {r['path']}: {e}")
        pd.DataFrame(rows).to_csv(fft_out, index=False)
        print(f"  Saved {fft_out}")
    else:
        print(f"  FFT features already exist: {fft_out}")

    # ── Semantic (CLIP / SigLIP2 / DINOv2) ──────────────────────────────────
    import torch

    # CLIP
    clip_out = os.path.join(ROUTING_BASE, f"{prefix}clip_features.csv")
    if not os.path.exists(clip_out):
        print(f"  Extracting CLIP features for {ds_name} ...")
        from transformers import CLIPProcessor, CLIPModel
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        rows = []
        for r in tqdm(index_rows, desc=f"CLIP/{ds_name}"):
            try:
                img = Image.open(r["path"]).convert("RGB")
                inp = proc(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    feats = model.get_image_features(**inp)[0].float().cpu().numpy().flatten()
                row = {"path": r["path"], "y_true": int(r["label"]),
                       "generator": r["generator"]}
                row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            except Exception as e:
                print(f"    skip {r['path']}: {e}")
        del model; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(clip_out, index=False)
        print(f"  Saved {clip_out}")
    else:
        print(f"  CLIP features already exist: {clip_out}")

    # SigLIP2
    siglip_out = os.path.join(ROUTING_BASE, f"{prefix}siglip2_features.csv")
    if not os.path.exists(siglip_out):
        print(f"  Extracting SigLIP2 features for {ds_name} ...")
        from transformers import AutoProcessor, AutoModel
        proc  = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device).eval()
        rows = []
        for r in tqdm(index_rows, desc=f"SigLIP2/{ds_name}"):
            try:
                img = Image.open(r["path"]).convert("RGB")
                inp = proc(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    f = model.get_image_features(**inp)[0].float()
                    if f.numel() != 768:
                        f = f.flatten().reshape(-1, 768).mean(dim=0)
                    feats = f.cpu().numpy().flatten()
                row = {"path": r["path"], "y_true": int(r["label"]),
                       "generator": r["generator"]}
                row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            except Exception as e:
                print(f"    skip {r['path']}: {e}")
        del model; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(siglip_out, index=False)
        print(f"  Saved {siglip_out}")
    else:
        print(f"  SigLIP2 features already exist: {siglip_out}")

    # DINOv2
    dino_out = os.path.join(ROUTING_BASE, f"{prefix}dinov2_features.csv")
    if not os.path.exists(dino_out):
        print(f"  Extracting DINOv2 features for {ds_name} ...")
        from transformers import AutoImageProcessor, AutoModel
        proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        rows = []
        for r in tqdm(index_rows, desc=f"DINOv2/{ds_name}"):
            try:
                img = Image.open(r["path"]).convert("RGB")
                inp = proc(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out  = model(**inp)
                    feats = out.last_hidden_state[:, 0].float().cpu().numpy().flatten()
                row = {"path": r["path"], "y_true": int(r["label"]),
                       "generator": r["generator"]}
                row.update({f"f{i}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            except Exception as e:
                print(f"    skip {r['path']}: {e}")
        del model; torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(dino_out, index=False)
        print(f"  Saved {dino_out}")
    else:
        print(f"  DINOv2 features already exist: {dino_out}")


# ── Probe loading ───────────────────────────────────────────────────────────

def load_stale_probe(probe_pkl, feat_file):
    """Load a probe trained on a different dataset (stale/OOD deployment)."""
    if not os.path.exists(probe_pkl):
        return None, None
    clf, scaler = joblib.load(probe_pkl)
    # Validate feature dimension matches
    df = pd.read_csv(feat_file, nrows=2)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if scaler.n_features_in_ != len(feat_cols):
        return None, None
    return clf, scaler


def get_predictions_stale(feat_file, probe_pkl, label="probe"):
    """Apply a stale (OOD) probe to all samples in feat_file."""
    clf, scaler = load_stale_probe(probe_pkl, feat_file)
    if clf is None:
        print(f"    {label}: probe not found or dim mismatch at {probe_pkl}")
        return None
    df = pd.read_csv(feat_file)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float32)
    X_s = scaler.transform(X)
    y_pred = clf.predict(X_s)
    y_prob = clf.predict_proba(X_s)[:, 1]
    df["y_pred"] = y_pred
    df["y_prob"] = y_prob
    return df[["path", "y_true", "y_pred", "y_prob"]]


# ── GPT-5.5 inference on disagreement cases ────────────────────────────────

def load_image_b64(path, max_size=1024):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt55(client, model, b64):
    import openai
    delay = RETRY_BASE
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [
                    {"type": "input_text",  "text": GPT_PROMPT},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ]}],
                max_output_tokens=100,
            )
            lat = (time.perf_counter() - t0) * 1000
            raw = resp.output_text.strip().upper()
            y   = 1 if ("FAKE" in raw and "REAL" not in raw) else \
                  0 if ("REAL" in raw and "FAKE" not in raw) else 1
            usage = resp.usage
            in_tok  = getattr(usage, "input_tokens",  0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            return y, raw, in_tok, out_tok, lat
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay); delay *= 2
            else:
                raise
        except openai.APIError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay); delay *= 2
            else:
                raise


def run_gpt55_on_disagree(ds_name, cfg, sem_name, df_disagree, model, api_key, out_csv):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    done = set()
    rows_out = []
    if os.path.exists(out_csv):
        df_ex = pd.read_csv(out_csv)
        done = set(df_ex["path"])
        rows_out = df_ex.to_dict("records")

    todo = df_disagree[~df_disagree["path"].isin(done)]
    if len(todo) == 0:
        print(f"    GPT-5.5 already complete for {ds_name}/{sem_name}")
        return pd.read_csv(out_csv)

    print(f"    Running GPT-5.5 on {len(todo)} disagree cases ({ds_name}/{sem_name})...")
    for _, r in tqdm(todo.iterrows(), total=len(todo)):
        try:
            b64 = load_image_b64(r["path"])
            y, raw, in_tok, out_tok, lat = call_gpt55(client, model, b64)
        except Exception as e:
            y, raw, in_tok, out_tok, lat = 1, "ERROR", 0, 0, 0
        rows_out.append({"path": r["path"], "y_true": int(r["y_true"]),
                         "y_gpt55": y, "raw": raw,
                         "latency_ms": lat, "input_tokens": in_tok,
                         "output_tokens": out_tok})
        if len(rows_out) % 10 == 0:
            pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    return pd.DataFrame(rows_out)


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_dataset(ds_name, cfg, gpt_model, api_key, run_gpt, max_samples=None):
    prefix = cfg["prefix"]
    rows = []

    for sem_name, feat_key in SEMANTICS.items():
        feat_file  = os.path.join(ROUTING_BASE, f"{prefix}{feat_key}.csv")
        fft_file   = os.path.join(ROUTING_BASE, f"{prefix}fft_features.csv")
        probe_key  = f"stale_probe_{sem_name.lower().replace('-','')}"
        # Normalise key lookup
        probe_key  = {
            "CLIP":    "stale_probe_clip",
            "SigLIP2": "stale_probe_siglip2",
            "DINOv2":  "stale_probe_dinov2",
        }[sem_name]

        if not os.path.exists(feat_file) or not os.path.exists(fft_file):
            print(f"  {sem_name}: missing features, skipping")
            continue

        print(f"\n  Semantic: {sem_name}")

        # Stale semantic predictions
        df_sem = get_predictions_stale(feat_file, cfg[probe_key], label=f"stale {sem_name}")
        if df_sem is None:
            continue

        # Stale FFT predictions
        df_fft = get_predictions_stale(fft_file, cfg["stale_fft_probe"], label="stale FFT")
        if df_fft is None:
            continue

        # Merge
        df = df_fft.rename(columns={"y_pred": "y_fft", "y_prob": "prob_fft"}).merge(
             df_sem.rename(columns={"y_pred": "y_sem", "y_prob": "prob_sem"})[
                 ["path", "y_sem", "prob_sem"]], on="path", how="inner")

        if len(df) == 0:
            print(f"    No matching paths after merge, skipping")
            continue

        y_true   = df["y_true"].values
        y_fft    = df["y_fft"].values
        y_sem    = df["y_sem"].values
        disagree = (y_fft != y_sem).astype(bool)
        n_total  = len(df)
        n_dis    = disagree.sum()
        n_agree  = n_total - n_dis

        fft_acc     = float(np.mean(y_fft == y_true))
        sem_acc     = float(np.mean(y_sem == y_true))
        agree_acc   = float(np.mean(y_sem[~disagree] == y_true[~disagree])) if n_agree > 0 else float("nan")
        dis_sem_acc = float(np.mean(y_sem[disagree]  == y_true[disagree]))  if n_dis  > 0 else float("nan")
        dis_fft_acc = float(np.mean(y_fft[disagree]  == y_true[disagree]))  if n_dis  > 0 else float("nan")

        # Cascade: FFT+Sem agree → trust FFT; disagree → escalate
        cascade_no_vlm = float(np.mean(
            np.where(disagree, y_sem, y_fft) == y_true))

        print(f"    FFT(stale)={fft_acc:.3f}  Sem(stale)={sem_acc:.3f}  "
              f"Dis-rate={n_dis/n_total:.3f}")
        print(f"    Agree-acc={agree_acc:.3f}  Dis-sem-acc={dis_sem_acc:.3f}  "
              f"Dis-fft-acc={dis_fft_acc:.3f}")

        base_row = {
            "dataset": ds_name, "semantic": sem_name,
            "n_total": n_total, "n_disagree": int(n_dis),
            "disagree_rate": float(n_dis / n_total),
            "fft_stale_acc": fft_acc, "sem_stale_acc": sem_acc,
            "agree_acc": agree_acc,
            "dis_sem_acc": dis_sem_acc, "dis_fft_acc": dis_fft_acc,
            "cascade_no_vlm_acc": cascade_no_vlm,
        }

        # GPT-5.5 on disagree cases
        gpt_out = os.path.join(ROUTING_BASE,
                               f"{prefix}gpt55_{sem_name.lower()}_disagree_preds.csv")
        df_dis  = df[disagree].reset_index(drop=True)
        if max_samples is not None:
            df_dis = df_dis.sample(n=min(max_samples, len(df_dis)), random_state=42)

        if run_gpt and len(df_dis) > 0:
            df_gpt = run_gpt55_on_disagree(
                ds_name, cfg, sem_name, df_dis, gpt_model, api_key, gpt_out)
        elif os.path.exists(gpt_out):
            df_gpt = pd.read_csv(gpt_out)
        else:
            df_gpt = None

        if df_gpt is not None and "y_gpt55" in df_gpt.columns:
            df_gpt_m = df_dis.merge(
                df_gpt[["path", "y_gpt55"]], on="path", how="inner")
            if len(df_gpt_m) > 0:
                dis_gpt_acc = float(np.mean(
                    df_gpt_m["y_gpt55"] == df_gpt_m["y_true"]))
                # Full cascade accuracy
                n_gpt = len(df_gpt_m)
                # Agree cases: use semantic (stale) accuracy
                # Disagree cases: use GPT-5.5
                df_agree_sub = df[~disagree]
                agree_correct = int(np.sum(df_agree_sub["y_sem"] == df_agree_sub["y_true"]))
                dis_correct   = int(np.sum(df_gpt_m["y_gpt55"] == df_gpt_m["y_true"]))
                cascade_gpt_acc = (agree_correct + dis_correct) / n_total

                lat_mean = float(df_gpt.get("latency_ms", pd.Series([0])).mean())

                base_row.update({
                    "dis_gpt55_acc":   dis_gpt_acc,
                    "cascade_gpt55_acc": cascade_gpt_acc,
                    "gpt55_improvement": dis_gpt_acc - dis_sem_acc,
                    "cascade_gain_over_stale_sem": cascade_gpt_acc - sem_acc,
                    "gpt55_mean_latency_ms": lat_mean,
                })
                print(f"    GPT-5.5 on disagree={dis_gpt_acc:.3f}  "
                      f"Cascade(GPT-5.5)={cascade_gpt_acc:.3f}  "
                      f"Gain-over-stale-sem={cascade_gpt_acc - sem_acc:+.3f}")

        rows.append(base_row)

    return rows


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--extract_features", action="store_true")
    p.add_argument("--evaluate",         action="store_true")
    p.add_argument("--run_gpt55",        action="store_true",
                   help="Run GPT-5.5 on disagreement cases (needs OPENAI_API_KEY)")
    p.add_argument("--datasets", nargs="+", default=["SDXL", "FLUX"])
    p.add_argument("--gpt_model",    default="gpt-5.5")
    p.add_argument("--api_key",      default=None)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--max_samples",  type=int, default=None,
                   help="Cap GPT-5.5 calls per (dataset × semantic) run")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()

    datasets = {k: v for k, v in E8_DATASETS.items() if k in args.datasets}

    if not datasets:
        print(f"No valid datasets in {args.datasets}. Choose from: {list(E8_DATASETS)}")
        return

    for ds_name, cfg in datasets.items():
        if not os.path.exists(cfg["index"]):
            print(f"Index not found for {ds_name}: {cfg['index']}")
            print("Run: python routing/build_aigibench_index.py")
            continue

        if args.extract_features:
            print(f"\n{'='*55}\nExtracting features: {ds_name}")
            extract_features(ds_name, cfg, device=args.device)

        if args.evaluate or args.run_gpt55:
            print(f"\n{'='*55}\nEvaluating: {ds_name}")
            rows = evaluate_dataset(
                ds_name, cfg, args.gpt_model, args.api_key,
                run_gpt=args.run_gpt55, max_samples=args.max_samples)

            if rows:
                out = os.path.join(RESULTS_DIR, "e8_modern_diffusion.csv")
                df_new = pd.DataFrame(rows)
                if os.path.exists(out):
                    df_old = pd.read_csv(out)
                    # Replace rows for this dataset/semantic combo
                    key = ["dataset", "semantic"]
                    df_old = df_old[~df_old["dataset"].isin(df_new["dataset"].unique())]
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                df_new.to_csv(out, index=False)
                print(f"\nSaved {out}")

    print("\nE8 complete.")


if __name__ == "__main__":
    main()
