"""
DIRE artifact detector (A2) — wraps the DIRE ResNet50 classifier.

DIRE (Diffusion Reconstruction Error) uses a ResNet50 trained to distinguish
real vs AI-generated images using reconstruction-error features.

This script loads the pre-trained DIRE checkpoint and runs inference,
producing per-sample logit, probability, and prediction.

Checkpoint download:
    # Download from https://github.com/ZhendongWang6/DIRE
    # Expected path: /workspace/benchmarks_ai_research/weights/DIRE/lsun_bedroom.pth
    # or set --ckpt to your checkpoint path.

Usage:
    python extract_dire.py --dataset GenBuster --ckpt /path/to/dire_checkpoint.pth
    python extract_dire.py --dataset SD14 --ckpt /path/to/dire_checkpoint.pth
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace/benchmarks_ai_research/DIRE")
from common import load_index, sample_frame_pil, DATASETS, ROUTING_BASE

DEVICE = "cuda"
DEFAULT_CKPT = "/workspace/benchmarks_ai_research/weights/DIRE/lsun_bedroom.pth"

# ImageNet-normalisation used by DIRE
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(ckpt_path):
    from networks.resnet import resnet50
    model = resnet50(num_classes=1)
    state = torch.load(ckpt_path, map_location="cpu")
    # Handle various checkpoint formats
    if "model" in state:
        model.load_state_dict(state["model"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DATASETS[args.dataset]

    out_file = os.path.join(ROUTING_BASE, f"{cfg['prefix']}dire_preds.csv")
    if os.path.exists(out_file):
        print(f"Already exists: {out_file}")
        return

    if not os.path.exists(args.ckpt):
        print(f"DIRE checkpoint not found: {args.ckpt}")
        print("Download from: https://github.com/ZhendongWang6/DIRE")
        print("Falling back to CNNSpot-style random-init check — cannot proceed without weights.")
        sys.exit(1)

    model = build_model(args.ckpt)
    index_rows = load_index(cfg["index"])
    print(f"Running DIRE on {len(index_rows)} samples ({args.dataset})...")

    rows_out = []
    for r in tqdm(index_rows):
        path = r["path"]
        label = int(r["label"])
        generator = r.get("generator", "unknown")

        try:
            img = sample_frame_pil(path, cfg["is_video"])
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logit = model(tensor).item()
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        prob_fake = torch.sigmoid(torch.tensor(logit)).item()
        y_pred = 1 if prob_fake > 0.5 else 0

        rows_out.append({
            "path": path,
            "y_true": label,
            "generator": generator,
            "logit_dire": logit,
            "prob_fake_dire": prob_fake,
            "y_dire": y_pred,
            "latency_dire_ms": latency_ms,
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")
    acc = np.mean(df["y_dire"] == df["y_true"])
    print(f"DIRE accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
