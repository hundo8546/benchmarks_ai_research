"""
Fill the GPT-5.5 gap for Table 6 (verifier_comparison): call GPT-5.5 on the
FFT+SigLIP2 disagreement-subset images not already covered by the old
CNNSpot+CLIP-based gpt55_disagree_preds.csv files (see
fix_table6_verifier_comparison.py, which identified 81 SD14 + 1 BigGAN +
60 GenBuster missing images). Appends new predictions to each dataset's
gpt55_disagree_preds.csv so it fully covers the correct disagreement subset.

Reuses the exact prompt/scoring convention from experiments/e8_modern_diffusion.py.
"""
import base64
import io
import os
import sys
import time

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extractors"))
from fix_table6_verifier_comparison import get_fft_siglip2_disagree_val, DATASET_CFG
from extractors.common import ROUTING_BASE, sample_frame_pil

GPT_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL "
    "photograph or an AI-generated FAKE image. Consider both "
    "possibilities equally. Answer with exactly one word: REAL or FAKE."
)
MAX_RETRIES = 5
RETRY_BASE = 2.0
MODEL = "gpt-5.5"


def load_image_b64(path, max_size=1024):
    is_video = path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    img = sample_frame_pil(path, is_video=True).convert("RGB") if is_video \
        else Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_gpt55(client, b64):
    import openai
    delay = RETRY_BASE
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.responses.create(
                model=MODEL,
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
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for ds_name, prefix, qwen_file, gpt_file in DATASET_CFG:
        df = get_fft_siglip2_disagree_val(ds_name)
        df_dis = df[df["disagree"]].reset_index(drop=True)

        gpt_path = os.path.join(ROUTING_BASE, gpt_file)
        df_gpt_old = pd.read_csv(gpt_path) if os.path.exists(gpt_path) else pd.DataFrame(
            columns=["path", "y_true", "y_verifier", "raw", "latency_ms",
                     "input_tokens", "output_tokens"])
        covered = set(df_gpt_old["path"])
        missing = df_dis[~df_dis["path"].isin(covered)].reset_index(drop=True)

        print(f"\n=== {ds_name}: {len(missing)} missing images ===")
        if len(missing) == 0:
            continue

        rows_out = df_gpt_old.to_dict("records")
        for i, r in missing.iterrows():
            path = r["path"]
            if not os.path.exists(path):
                print(f"  SKIP (file not found): {path}")
                continue
            try:
                b64 = load_image_b64(path)
                y, raw, in_tok, out_tok, lat = call_gpt55(client, b64)
            except Exception as e:
                print(f"  ERROR on {path}: {e}")
                y, raw, in_tok, out_tok, lat = 1, "ERROR", 0, 0, 0
            rows_out.append({
                "path": path, "y_true": int(r["y_true"]), "y_verifier": y,
                "raw": raw, "latency_ms": lat,
                "input_tokens": in_tok, "output_tokens": out_tok,
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(missing):
                print(f"  {i+1}/{len(missing)} done")
                pd.DataFrame(rows_out).to_csv(gpt_path, index=False)

        pd.DataFrame(rows_out).to_csv(gpt_path, index=False)
        print(f"  Saved {gpt_path} ({len(rows_out)} total rows)")

    print("\nDone. Rerun fix_table6_verifier_comparison.py to get final numbers.")


if __name__ == "__main__":
    main()
