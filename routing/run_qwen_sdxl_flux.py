"""
Run Qwen2.5-VL-7B-Instruct on the SDXL/FLUX disagreement-subset images
(same populations already scored by GPT-5.5 in {sdxl,flux}_gpt55_{sem}_disagree_preds.csv)
to add the Qwen-only comparison column requested for Table 7/8.

Reuses the exact model/prompt/logit-scoring methodology from run_qwen_v2.py.
Runs each unique image once (3991 unique across the 6 dataset x backbone
combos) and writes results into per-combo CSVs matching the GPT-5.5 file
naming convention, plus a merged results/e8_qwen_disagree.csv summary.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extractors"))
from extractors.common import RESULTS_DIR, ensure_results_dir

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REASONING_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL photograph "
    "or an AI-generated FAKE image. Consider both possibilities equally.\n"
    "Answer with exactly one word: REAL or FAKE."
)
REAL_LABEL, FAKE_LABEL = 0, 1

COMBOS = [
    ("SDXL", "clip", "sdxl_gpt55_clip_disagree_preds.csv", "sdxl_qwen_clip_disagree_preds.csv"),
    ("SDXL", "siglip2", "sdxl_gpt55_siglip2_disagree_preds.csv", "sdxl_qwen_siglip2_disagree_preds.csv"),
    ("SDXL", "dinov2", "sdxl_gpt55_dinov2_disagree_preds.csv", "sdxl_qwen_dinov2_disagree_preds.csv"),
    ("FLUX", "clip", "flux_gpt55_clip_disagree_preds.csv", "flux_qwen_clip_disagree_preds.csv"),
    ("FLUX", "siglip2", "flux_gpt55_siglip2_disagree_preds.csv", "flux_qwen_siglip2_disagree_preds.csv"),
    ("FLUX", "dinov2", "flux_gpt55_dinov2_disagree_preds.csv", "flux_qwen_dinov2_disagree_preds.csv"),
]
BASE = "/workspace/benchmarks_ai_research/routing"


def safe_softmax_two(a, b):
    x = np.array([a, b], dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / e.sum()
    return float(p[0]), float(p[1])


def load_model():
    print("Loading Qwen2.5-VL-7B-Instruct...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    def first_token(text):
        ids = processor.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if ids else None

    real_id, fake_id = first_token("REAL"), first_token("FAKE")
    print(f"REAL token id={real_id}, FAKE token id={fake_id}")
    return processor, model, real_id, fake_id


def infer_single(model, processor, real_id, fake_id, img):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": REASONING_PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    n_in = int(inputs["input_ids"].shape[-1])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                              return_dict_in_generate=True, output_scores=True)
    torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    new_tokens = gen.sequences[:, inputs["input_ids"].shape[-1]:]
    n_out = int(new_tokens.shape[-1])
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    p_fake = p_real = None
    if gen.scores:
        best_gap = -1.0
        for step_logits in gen.scores:
            real_l = float(step_logits[0, real_id].item())
            fake_l = float(step_logits[0, fake_id].item())
            pr, pf = safe_softmax_two(real_l, fake_l)
            gap = abs(pf - pr)
            if gap > best_gap:
                best_gap, p_real, p_fake = gap, pr, pf

    y = FAKE_LABEL if (p_fake is not None and p_fake >= 0.5) else REAL_LABEL
    del inputs, gen, new_tokens
    return {"y_qwen": y, "p_fake_qwen": p_fake, "raw": raw,
            "latency_qwen_ms": lat_ms, "qwen_input_tokens": n_in, "qwen_output_tokens": n_out}


def main():
    ensure_results_dir()
    processor, model, real_id, fake_id = load_model()

    # Collect the union of (path, y_true) across all combos, run each once
    all_rows = {}
    for ds, sem, gpt_file, _ in COMBOS:
        df = pd.read_csv(os.path.join(BASE, gpt_file))
        for _, r in df.iterrows():
            all_rows[r["path"]] = int(r["y_true"])

    print(f"Total unique images to score: {len(all_rows)}")
    cache = {}
    done_csv = os.path.join(BASE, "sdxl_flux_qwen_cache.csv")
    if os.path.exists(done_csv):
        prev = pd.read_csv(done_csv)
        cache = {r["path"]: r.to_dict() for _, r in prev.iterrows()}
        print(f"Resuming: {len(cache)} already scored")

    items = list(all_rows.items())
    t0 = time.time()
    for i, (path, y_true) in enumerate(items):
        if path in cache:
            continue
        try:
            img = Image.open(path).convert("RGB")
            res = infer_single(model, processor, real_id, fake_id, img)
        except Exception as e:
            print(f"  ERROR on {path}: {e}")
            res = {"y_qwen": 1, "p_fake_qwen": None, "raw": "ERROR",
                   "latency_qwen_ms": 0, "qwen_input_tokens": 0, "qwen_output_tokens": 0}
        res["path"] = path
        res["y_true"] = y_true
        cache[path] = res

        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 100 == 0 or (i + 1) == len(items):
            dt = time.time() - t0
            print(f"  {i+1}/{len(items)} ({(i+1)/dt:.2f} img/s, {dt:.0f}s elapsed)")
            pd.DataFrame(list(cache.values())).to_csv(done_csv, index=False)

    pd.DataFrame(list(cache.values())).to_csv(done_csv, index=False)
    print(f"Saved {done_csv}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Split back out into per-combo files matching the GPT-5.5 naming convention
    for ds, sem, gpt_file, out_file in COMBOS:
        df_gpt = pd.read_csv(os.path.join(BASE, gpt_file))
        merged = df_gpt[["path", "y_true"]].merge(
            pd.DataFrame(list(cache.values()))[
                ["path", "y_qwen", "p_fake_qwen", "raw", "latency_qwen_ms",
                 "qwen_input_tokens", "qwen_output_tokens"]],
            on="path", how="left")
        merged.to_csv(os.path.join(BASE, out_file), index=False)
        acc = float(np.mean(merged["y_qwen"] == merged["y_true"]))
        print(f"{ds}/{sem}: n={len(merged)} Qwen accuracy={acc:.4f} -> {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
