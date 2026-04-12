import os
os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import time

from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda"
MODEL_ID = "openai/clip-vit-base-patch32"
INDEX_FILE = "/workspace/benchmarks_ai_research/routing/biggan_index.csv"
OUT_FILE = "/workspace/benchmarks_ai_research/routing/biggan_clip_features.csv"

processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

def sample_frame_pil(path):
    img = Image.open(path).convert("RGB")
    return img

with open(INDEX_FILE) as f:
    index_rows = list(csv.DictReader(f))

# Open CSV file once and write incrementally
first_row = True
with open(OUT_FILE, "w", newline="") as csvfile:
    writer = None
    
    for r in tqdm(index_rows):
        path = r["path"]
        label = int(r["label"])
        generator = r["generator"]

        try:
            img = sample_frame_pil(path)
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            torch.cuda.synchronize()
            end = time.perf_counter()
            latency_clip_ms = (end - start) * 1000

            image_features = image_features[0].detach().cpu().numpy().flatten()

            row = {
                "path": path,
                "y_true": label,
                **{f"f{i}": image_features[i] for i in range(len(image_features))},
                "latency_clip_ms": latency_clip_ms,
                "generator": generator
            }

            # Initialize writer with fieldnames from first row
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
                writer.writeheader()

            writer.writerow(row)

            # Clear GPU cache periodically to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Optionally write error info to a separate log file
            continue

print("Saved", OUT_FILE)