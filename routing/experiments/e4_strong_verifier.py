"""
E4 — Strong Verifier Escalation.
Tests whether stronger VLMs rescue disagreement-routed cases.

Verifiers:
  V1: Qwen2.5-VL-7B (control, already run)
  V2: Qwen2.5-VL-72B or Qwen3-VL (primary upgrade)

Runs ONLY on the disagreement subset (agreement cases use cheap path).

Produces: results/e4_verifier_results.csv
Success: strong verifier improves disagreement-subset accuracy > 10% over V1.

Usage:
    # Step 1: Run strong verifier on disagreement subset
    python e4_strong_verifier.py --run_inference --model Qwen/Qwen2.5-VL-72B-Instruct \\
        --dataset GenBuster

    # Step 2: Evaluate results
    python e4_strong_verifier.py --evaluate
"""
import argparse
import csv
import gc
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import DATASETS, RESULTS_DIR, ROUTING_BASE, ensure_results_dir, bootstrap_ci, load_val_paths

DEVICE = "cuda"
LATENCY_CNN_MS = 45.8
LATENCY_CLIP_MS = 10.4
LATENCY_CHEAP_MS = LATENCY_CNN_MS + LATENCY_CLIP_MS

V2_MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

PROMPT = (
    "Determine whether this image is real (camera-captured) or AI-generated.\n"
    "Answer with exactly one word: REAL or FAKE."
)


def get_disagree_subset(ds_name):
    """Load the disagreement subset for a dataset."""
    cfg = DATASETS[ds_name]
    val_paths = set(np.load(cfg["val_paths"], allow_pickle=True).tolist())

    df_cnn = pd.read_csv(cfg["cnnspot_csv"])
    df_clip = pd.read_csv(cfg["clip_preds"])

    df = df_cnn.merge(df_clip[["path", "y_clip"]], on="path", how="inner")
    val_paths = load_val_paths(cfg, df["path"].tolist())
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
    return df[df["disagree"] == 1].reset_index(drop=True)


def run_verifier_inference(ds_name, model_id, out_csv):
    """Run VLM inference on the disagreement subset and save results."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from decord import VideoReader, cpu
    import cv2

    cfg = DATASETS[ds_name]
    is_video = cfg["is_video"]

    df_dis = get_disagree_subset(ds_name)
    print(f"Running {model_id} on {len(df_dis)} disagreement samples ({ds_name})...")

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    # Find token IDs
    real_id = proc.tokenizer.encode("REAL", add_special_tokens=False)[0]
    fake_id = proc.tokenizer.encode("FAKE", add_special_tokens=False)[0]

    rows_out = []
    for _, r in tqdm(df_dis.iterrows(), total=len(df_dis)):
        path = r["path"]
        try:
            if is_video:
                vr = VideoReader(path, ctx=cpu(0))
                frame = vr[0].asnumpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
            else:
                img = Image.open(path).convert("RGB")
        except Exception as e:
            rows_out.append({"path": path, "y_verifier": 1, "p_fake_verifier": 0.5,
                              "latency_ms": 0, "error": str(e)})
            continue

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(DEVICE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                  return_dict_in_generate=True, output_scores=True)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        new_toks = gen.sequences[:, inputs["input_ids"].shape[-1]:]
        raw = proc.batch_decode(new_toks, skip_special_tokens=True)[0].strip().upper()

        # Score-based prediction
        p_fake = 0.5
        if gen.scores:
            s = gen.scores[0][0]
            rl, fl = float(s[real_id].item()), float(s[fake_id].item())
            arr = np.array([rl, fl]) - max(rl, fl)
            e = np.exp(arr)
            p_fake = float(e[1] / e.sum())

        y_pred = 1 if p_fake >= 0.5 else 0
        if "FAKE" in raw and "REAL" not in raw:
            y_pred = 1
        elif "REAL" in raw and "FAKE" not in raw:
            y_pred = 0

        n_in = int(inputs["input_ids"].shape[-1])
        n_out = int(new_toks.shape[-1])

        rows_out.append({
            "path": path, "y_true": int(r["y_true"]),
            "y_verifier": y_pred, "p_fake_verifier": p_fake,
            "latency_ms": latency_ms,
            "input_tokens": n_in, "output_tokens": n_out,
            "raw": raw,
        })

        if (len(rows_out) % 10) == 0:
            torch.cuda.empty_cache()
            gc.collect()

        del inputs, gen, new_toks

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


def evaluate(datasets):
    """Compare V1 vs V2 on disagreement subsets."""
    ensure_results_dir()
    rows = []

    for ds_name in datasets:
        cfg = DATASETS[ds_name]
        if not all(os.path.exists(f) for f in [cfg["cnnspot_csv"], cfg["clip_preds"], cfg["qwen_preds"]]):
            print(f"Skipping {ds_name}: missing prerequisite files")
            continue

        # Load val + disagreement subset
        df_dis = get_disagree_subset(ds_name)

        # V1: Qwen2.5-VL-7B predictions on disagree subset
        df_qwen = pd.read_csv(cfg["qwen_preds"])
        qwen_cols = ["path", "y_qwen"]
        for col in ["latency_qwen_ms", "qwen_input_tokens", "qwen_output_tokens", "p_fake_qwen"]:
            if col in df_qwen.columns:
                qwen_cols.append(col)
        df_v1 = df_dis.merge(df_qwen[qwen_cols], on="path", how="inner")

        v1_acc_on_dis = float(np.mean(df_v1["y_qwen"] == df_v1["y_true"]))
        v1_lat = float(df_v1["latency_qwen_ms"].mean()) if "latency_qwen_ms" in df_v1.columns else 0.0

        rows.append({
            "dataset": ds_name, "verifier": "Qwen2.5-VL-7B (V1)",
            "n_disagree": len(df_v1),
            "disagree_accuracy": v1_acc_on_dis,
            "mean_latency_ms": v1_lat,
            "corrected_cases": int((df_v1["y_qwen"] == df_v1["y_true"]).sum()),
        })
        print(f"\n{ds_name}: V1 (Qwen2.5-VL-7B) on disagree: acc={v1_acc_on_dis:.3f} "
              f"(n={len(df_v1)})")

        # V2: Strong verifier (if preds available)
        for model_tag, preds_file in [
            ("Qwen2.5-VL-72B (V2)",
             os.path.join(ROUTING_BASE, f"{cfg['prefix']}qwen72b_disagree_preds.csv")),
            ("Qwen3-VL (V2)",
             os.path.join(ROUTING_BASE, f"{cfg['prefix']}qwen3vl_disagree_preds.csv")),
            ("GPT-5.5 (V2)",
             os.path.join(ROUTING_BASE, f"{cfg['prefix']}gpt55_disagree_preds.csv")),
        ]:
            if not os.path.exists(preds_file):
                continue
            df_preds = pd.read_csv(preds_file)
            merge_cols = [c for c in df_preds.columns if c != "y_true"]
            df_v2 = df_dis.merge(df_preds[merge_cols], on="path", how="inner")
            if "y_verifier" not in df_v2.columns:
                continue
            v2_acc = float(np.mean(df_v2["y_verifier"] == df_v2["y_true"]))
            improvement = v2_acc - v1_acc_on_dis
            v2_lat = float(df_v2["latency_ms"].mean()) if "latency_ms" in df_v2.columns else 0.0
            rows.append({
                "dataset": ds_name, "verifier": model_tag,
                "n_disagree": len(df_v2),
                "disagree_accuracy": v2_acc,
                "mean_latency_ms": v2_lat,
                "corrected_cases": int((df_v2["y_verifier"] == df_v2["y_true"]).sum()),
                "improvement_over_v1": improvement,
                "meets_10pct_criterion": improvement > 0.10,
            })
            mark = " *** SUCCESS" if improvement > 0.10 else ""
            print(f"  {model_tag}: acc={v2_acc:.3f}, "
                  f"improvement={improvement:+.3f}{mark}")

    df_out = pd.DataFrame(rows)
    out_file = os.path.join(RESULTS_DIR, "e4_verifier_results.csv")
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")
    print("E4 evaluation complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_inference", action="store_true",
                   help="Run strong verifier inference on disagree subset")
    p.add_argument("--evaluate", action="store_true",
                   help="Evaluate V1 vs V2 on disagree subsets")
    p.add_argument("--model", default=V2_MODEL_ID,
                   help="Strong verifier model ID for --run_inference")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()

    if args.run_inference:
        for ds_name in args.datasets:
            cfg = DATASETS[ds_name]
            if not os.path.exists(cfg["cnnspot_csv"]) or not os.path.exists(cfg["clip_preds"]):
                print(f"Skipping {ds_name}: missing prerequisite files")
                continue
            model_tag = "qwen72b" if "72B" in args.model else "qwen3vl"
            out_csv = os.path.join(ROUTING_BASE,
                                   f"{cfg['prefix']}{model_tag}_disagree_preds.csv")
            if os.path.exists(out_csv):
                print(f"Already exists: {out_csv}")
                continue
            run_verifier_inference(ds_name, args.model, out_csv)

    if args.evaluate or not args.run_inference:
        evaluate(args.datasets)


if __name__ == "__main__":
    main()
