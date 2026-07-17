"""
Run GPT-5.5 on a stratified random subsample (n=1500) of the BigGAN
stale-FFT/stale-SigLIP2 disagreement subset (biggan_stale_disagree_sample.csv).

This tests Reviewer 2's "more distant generator family" request: SD1.4-trained
(stale) FFT and SigLIP2 probes, applied zero-shot to BigGAN (a GAN, not a
diffusion model, so architecturally distant from SD1.4's own training
family), with disagreement cases escalated to GPT-5.5. This mirrors the
SDXL/FLUX deployment-shift design (Table 4 / tab:e8_results) exactly, but on
a much larger architectural shift than SDXL/FLUX (both diffusion, same
lineage as SD1.4): stale FFT accuracy on BigGAN is 38.3% and stale SigLIP2 is
62.1%, both far more degraded than the 57.8-62.2% / 81.5-92.0% seen on
SDXL/FLUX (see biggan_stale_fft_siglip2.csv).

The disagreement pool is 3,516 images; subsampled here to 1,500 (stratified
by true label, seed 42) to match the scale of the existing SDXL/FLUX
per-config disagreement subsets (1,400-1,650 each) rather than a fresh,
unjustified sample size.

Reuses the exact Responses-API call pattern from
experiments/e8_modern_diffusion.py (call_gpt55/load_image_b64): images
resized to max 1024px, JPEG q=90, max_output_tokens=100.

Produces: biggan_gpt55_stale_disagree_preds.csv
Checkpoints to the same file every 10 samples (resumable).
"""
import base64
import io
import os
import time

import pandas as pd
from PIL import Image
from openai import OpenAI
import openai

GPT_MODEL = "gpt-5.5"
GPT_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL "
    "photograph or an AI-generated FAKE image. Consider both "
    "possibilities equally. Answer with exactly one word: REAL or FAKE."
)
MAX_RETRIES = 5
RETRY_BASE = 2.0

BASE = "/workspace/benchmarks_ai_research/routing"
SAMPLE_CSV = os.path.join(BASE, "biggan_stale_disagree_sample.csv")
OUT_CSV = os.path.join(BASE, "biggan_gpt55_stale_disagree_preds.csv")


def load_image_b64(path, max_size=1024):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt55(client, b64):
    delay = RETRY_BASE
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.responses.create(
                model=GPT_MODEL,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": GPT_PROMPT},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ]}],
                max_output_tokens=100,
            )
            lat = (time.perf_counter() - t0) * 1000
            raw = resp.output_text.strip().upper()
            y = 1 if ("FAKE" in raw and "REAL" not in raw) else \
                0 if ("REAL" in raw and "FAKE" not in raw) else 1
            usage = resp.usage
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            return y, raw, in_tok, out_tok, lat
        except openai.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay); delay *= 2
            else:
                raise
        except openai.APIError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay); delay *= 2
            else:
                raise


def main():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    df = pd.read_csv(SAMPLE_CSV)

    done = set()
    rows_out = []
    if os.path.exists(OUT_CSV):
        df_ex = pd.read_csv(OUT_CSV)
        done = set(df_ex["path"])
        rows_out = df_ex.to_dict("records")
        print(f"Resuming: {len(done)} already scored")

    todo = df[~df["path"].isin(done)]
    print(f"Running GPT-5.5 on {len(todo)}/{len(df)} remaining samples...")

    t0 = time.time()
    for i, (_, r) in enumerate(todo.iterrows()):
        try:
            b64 = load_image_b64(r["path"])
            y, raw, in_tok, out_tok, lat = call_gpt55(client, b64)
        except Exception as e:
            print(f"  ERROR on {r['path']}: {e}")
            y, raw, in_tok, out_tok, lat = 1, "ERROR", 0, 0, 0
        rows_out.append({
            "path": r["path"], "y_true": int(r["y_true"]),
            "y_fft": int(r["y_fft"]), "y_sem": int(r["y_sem"]),
            "y_gpt55": y, "raw": raw, "latency_ms": lat,
            "input_tokens": in_tok, "output_tokens": out_tok,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(todo):
            dt = time.time() - t0
            print(f"  {i+1}/{len(todo)} ({(i+1)/dt:.2f} img/s, {dt:.0f}s elapsed)", flush=True)
            pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)

    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
