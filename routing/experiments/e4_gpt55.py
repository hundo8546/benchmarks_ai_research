"""
E4-GPT55 — Run GPT-5.5 on disagreement subsets for E4 strong verifier experiment.

Reads the CNNSpot × CLIP disagreement subset for each dataset, sends each image
to GPT-5.5 via the OpenAI vision API, and saves predictions in the format
expected by e4_strong_verifier.py --evaluate.

Outputs: routing/{prefix}gpt55_disagree_preds.csv  (one per dataset)

Usage:
    export OPENAI_API_KEY=sk-...
    python e4_gpt55.py                          # all datasets
    python e4_gpt55.py --datasets GenBuster     # one dataset
    python e4_gpt55.py --model gpt-5.5          # override model name
    python e4_gpt55.py --detail high            # higher-res crops (costs more)

Then evaluate:
    python e4_strong_verifier.py --evaluate
"""
import argparse
import base64
import io
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractors"))
from extractors.common import DATASETS, ROUTING_BASE, ensure_results_dir, load_val_paths

DEFAULT_MODEL = "gpt-5.5"

PROMPT = (
    "Determine whether this image is real (camera-captured) or AI-generated.\n"
    "Answer with exactly one word: REAL or FAKE."
)

# Retry settings for rate limits
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0   # seconds, doubles each retry


def get_client(api_key=None):
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY env var or pass --api_key")
    return OpenAI(api_key=key)


def load_image_as_jpeg_b64(path, is_video, max_size=1024):
    """Load image or first video frame, resize to max_size on longest edge, return base64 JPEG."""
    if is_video:
        vr = VideoReader(path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img = Image.open(path).convert("RGB")

    # Resize so longest edge <= max_size (keeps costs predictable)
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt(client, model, b64_image, detail, retries=MAX_RETRIES):
    """Call GPT vision API with retry on rate-limit errors. Returns (y_pred, raw_text, usage, latency_ms)."""
    import openai

    delay = RETRY_BASE_DELAY
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            resp = client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64_image}",
                        },
                    ],
                }],
                max_output_tokens=16,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = resp.output_text.strip().upper()
            usage = resp.usage

            if "FAKE" in raw and "REAL" not in raw:
                y_pred = 1
            elif "REAL" in raw and "FAKE" not in raw:
                y_pred = 0
            else:
                # Ambiguous — default to FAKE (conservative)
                y_pred = 1

            return y_pred, raw, usage, latency_ms

        except openai.RateLimitError:
            if attempt < retries - 1:
                print(f"\n  Rate limit hit, retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except openai.APIError as e:
            if attempt < retries - 1:
                print(f"\n  API error ({e}), retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def get_disagree_subset(ds_name):
    """Return the CNNSpot × CLIP disagreement val rows."""
    cfg = DATASETS[ds_name]
    df_cnn  = pd.read_csv(cfg["cnnspot_csv"])
    df_clip = pd.read_csv(cfg["clip_preds"])
    df = df_cnn.merge(df_clip[["path", "y_clip"]], on="path", how="inner")
    val_paths = load_val_paths(cfg, df["path"].tolist())
    df = df[df["path"].isin(val_paths)].reset_index(drop=True)
    df["disagree"] = (df["y_cnn"] != df["y_clip"]).astype(int)
    return df[df["disagree"] == 1].reset_index(drop=True)


def run_dataset(ds_name, client, model, detail, out_csv):
    cfg = DATASETS[ds_name]
    is_video = cfg["is_video"]

    df_dis = get_disagree_subset(ds_name)
    print(f"\n{'='*55}")
    print(f"Dataset: {ds_name}  |  disagreement cases: {len(df_dis)}")

    # Resume: skip already-processed paths
    done_paths = set()
    existing_rows = []
    if os.path.exists(out_csv):
        df_existing = pd.read_csv(out_csv)
        done_paths = set(df_existing["path"].tolist())
        existing_rows = df_existing.to_dict("records")
        print(f"  Resuming — {len(done_paths)} already done, "
              f"{len(df_dis) - len(done_paths)} remaining")

    df_todo = df_dis[~df_dis["path"].isin(done_paths)].reset_index(drop=True)
    if len(df_todo) == 0:
        print("  All done, skipping inference.")
        return

    rows_out = list(existing_rows)
    n_correct = sum(1 for r in existing_rows if r.get("y_verifier") == r.get("y_true"))
    total_in_tok = sum(r.get("input_tokens", 0) for r in existing_rows)
    total_out_tok = sum(r.get("output_tokens", 0) for r in existing_rows)

    for _, r in tqdm(df_todo.iterrows(), total=len(df_todo), desc=ds_name):
        path = r["path"]
        try:
            b64 = load_image_as_jpeg_b64(path, is_video)
        except Exception as e:
            print(f"\n  Load error {path}: {e}")
            rows_out.append({
                "path": path, "y_true": int(r["y_true"]),
                "y_verifier": 1, "raw": "ERROR",
                "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
                "error": str(e),
            })
            continue

        try:
            y_pred, raw, usage, latency_ms = call_gpt(client, model, b64, detail)
        except Exception as e:
            print(f"\n  API error {path}: {e}")
            rows_out.append({
                "path": path, "y_true": int(r["y_true"]),
                "y_verifier": 1, "raw": "API_ERROR",
                "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
                "error": str(e),
            })
            continue

        # Responses API: input_tokens / output_tokens; fallback to chat completions names
        in_tok  = getattr(usage, "input_tokens",      None) \
               or getattr(usage, "prompt_tokens",     0) if usage else 0
        out_tok = getattr(usage, "output_tokens",     None) \
               or getattr(usage, "completion_tokens", 0) if usage else 0
        total_in_tok  += in_tok
        total_out_tok += out_tok

        if y_pred == int(r["y_true"]):
            n_correct += 1

        rows_out.append({
            "path": path, "y_true": int(r["y_true"]),
            "y_verifier": y_pred, "raw": raw,
            "latency_ms": latency_ms,
            "input_tokens": in_tok, "output_tokens": out_tok,
        })

        # Save every 10 images so progress survives interruption
        if len(rows_out) % 10 == 0:
            pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    n_done = len(rows_out)
    acc = n_correct / n_done if n_done > 0 else 0
    print(f"  Saved {out_csv}")
    print(f"  Accuracy on disagree subset: {acc:.3f} ({n_correct}/{n_done})")
    print(f"  Total tokens — in: {total_in_tok:,}  out: {total_out_tok:,}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--model",    default=DEFAULT_MODEL,
                   help="OpenAI model ID (default: gpt-5.5)")
    p.add_argument("--detail",   default="low", choices=["low", "high"],
                   help="Vision detail level — low is cheaper, high is more accurate")
    p.add_argument("--api_key",  default=None,
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_results_dir()
    client = get_client(args.api_key)

    # Estimate cost upfront
    total_dis = 0
    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        if not os.path.exists(cfg["cnnspot_csv"]) or not os.path.exists(cfg["clip_preds"]):
            print(f"Skipping {ds_name}: missing prerequisite files")
            args.datasets = [d for d in args.datasets if d != ds_name]
            continue
        df_dis = get_disagree_subset(ds_name)
        total_dis += len(df_dis)
    print(f"\nTotal disagreement cases to process: {total_dis}")
    print(f"Model: {args.model}  |  detail: {args.detail}")
    print("(Saves incrementally — safe to interrupt and resume)\n")

    for ds_name in args.datasets:
        cfg = DATASETS[ds_name]
        if not os.path.exists(cfg["cnnspot_csv"]) or not os.path.exists(cfg["clip_preds"]):
            continue
        out_csv = os.path.join(ROUTING_BASE,
                               f"{cfg['prefix']}gpt55_disagree_preds.csv")
        run_dataset(ds_name, client, args.model, args.detail, out_csv)

    print("\nDone. Now run:")
    print("  python e4_strong_verifier.py --evaluate")


if __name__ == "__main__":
    main()
