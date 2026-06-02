"""
SigLIP2 feature extractor for semantic detector (S2).
Saves embedding vectors to CSV for probe training.

Usage:
    python extract_siglip2.py --dataset GenBuster
    python extract_siglip2.py --dataset SD14
    python extract_siglip2.py --dataset BigGAN
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from common import load_index, sample_frame_pil, DATASETS, ROUTING_BASE

MODEL_ID = "google/siglip2-base-patch16-224"
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--model_id", default=MODEL_ID)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]
    is_video = cfg["is_video"]

    out_file = os.path.join(ROUTING_BASE, f"{cfg['prefix']}siglip2_features.csv")
    if os.path.exists(out_file):
        print(f"Already exists: {out_file}")
        return

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    index_rows = load_index(cfg["index"])
    print(f"Processing {len(index_rows)} samples from {args.dataset}...")

    CHUNK_SIZE = 500
    rows_out = []
    total_written = 0
    first_chunk = True

    for i, r in enumerate(tqdm(index_rows)):
        path = r["path"]
        label = int(r["label"])
        generator = r.get("generator", "unknown")

        try:
            img = sample_frame_pil(path, is_video)
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        del img

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        del inputs
        # get_image_features returns (1, 150528) — all 196 patches × 768 flattened.
        # Reshape and mean-pool to a single 768-dim vector.
        f = feats[0].float().flatten()
        if f.shape[0] != 768:
            f = f.reshape(-1, 768).mean(dim=0)
        feat_vec = f.cpu().numpy()
        del feats, f

        rows_out.append({
            "path": path,
            "y_true": label,
            "generator": generator,
            **{f"f{i}": feat_vec[i] for i in range(len(feat_vec))},
            "latency_siglip2_ms": latency_ms,
        })

        if len(rows_out) >= CHUNK_SIZE:
            chunk_df = pd.DataFrame(rows_out)
            chunk_df.to_csv(out_file, mode="w" if first_chunk else "a",
                            index=False, header=first_chunk)
            total_written += len(rows_out)
            rows_out = []
            first_chunk = False
            torch.cuda.empty_cache()

    if rows_out:
        chunk_df = pd.DataFrame(rows_out)
        chunk_df.to_csv(out_file, mode="w" if first_chunk else "a",
                        index=False, header=first_chunk)
        total_written += len(rows_out)

    print(f"Saved {total_written} rows to {out_file}")


if __name__ == "__main__":
    main()
