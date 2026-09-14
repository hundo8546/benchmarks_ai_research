"""
Re-run GPT-5.5 on every row with a blank/empty `raw` response across all
existing gpt55_*_disagree_preds.csv files, using the fixed call_gpt55
(escalating max_output_tokens instead of silently defaulting truncated
responses to FAKE). Updates each file in place, preserving good rows.
"""
import os
import sys

import base64
import io

import cv2
import pandas as pd
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
import e8_modern_diffusion as e8

ROUTING_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gpt-5.5"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def load_image_b64_any(path, max_size=1024):
    """Like e8.load_image_b64, but extracts the first frame for video paths
    (GenBuster is video; SDXL/FLUX/SD14/BigGAN are static images)."""
    if path.lower().endswith(VIDEO_EXTS):
        vr = VideoReader(path, ctx=cpu(0))
        frame = vr[0].asnumpy()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img = Image.open(path).convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

FILES = [
    "biggan_gpt55_disagree_preds.csv",
    "sd14_gpt55_disagree_preds.csv",
    "gpt55_disagree_preds.csv",
    "sdxl_gpt55_clip_disagree_preds.csv",
    "sdxl_gpt55_siglip2_disagree_preds.csv",
    "sdxl_gpt55_dinov2_disagree_preds.csv",
    "flux_gpt55_clip_disagree_preds.csv",
    "flux_gpt55_siglip2_disagree_preds.csv",
    "flux_gpt55_dinov2_disagree_preds.csv",
]


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        with open("/root/.openai_api_key") as f:
            api_key = f.read().strip()
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    grand_total_fixed = 0
    grand_total_still_empty = 0

    for fname in FILES:
        path = os.path.join(ROUTING_BASE, fname)
        if not os.path.exists(path):
            print(f"skip (missing): {fname}")
            continue
        df = pd.read_csv(path)
        verifier_col = "y_verifier" if "y_verifier" in df.columns else "y_gpt55"
        blank_mask = (df["raw"].isna() | (df["raw"] == "") | (df["raw"] == "EMPTY_AFTER_RETRY")
                      | df["raw"].astype(str).str.startswith("ERROR:"))
        n_blank = int(blank_mask.sum())
        print(f"\n=== {fname} === total={len(df)} blank={n_blank}")
        if n_blank == 0:
            continue

        idxs = df[blank_mask].index.tolist()
        fixed, still_empty = 0, 0
        for count, idx in enumerate(tqdm(idxs, desc=fname)):
            row = df.loc[idx]
            try:
                b64 = load_image_b64_any(row["path"])
                y, raw, in_tok, out_tok, lat = e8.call_gpt55(client, MODEL, b64)
            except Exception as ex:
                y, raw, in_tok, out_tok, lat = None, f"ERROR:{ex}", 0, 0, 0

            if y is None:
                still_empty += 1
            else:
                fixed += 1
                df.at[idx, verifier_col] = y
            df.at[idx, "raw"] = raw
            df.at[idx, "latency_ms"] = lat
            df.at[idx, "input_tokens"] = in_tok
            df.at[idx, "output_tokens"] = out_tok

            if (count + 1) % 25 == 0:
                df.to_csv(path, index=False)

        df.to_csv(path, index=False)
        print(f"  fixed={fixed}  still_empty_after_retry={still_empty}")
        grand_total_fixed += fixed
        grand_total_still_empty += still_empty

    print(f"\n=== TOTAL === fixed={grand_total_fixed}  still_empty={grand_total_still_empty}")


if __name__ == "__main__":
    main()
