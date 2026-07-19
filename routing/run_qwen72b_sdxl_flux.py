"""
Run Qwen2.5-VL-72B-Instruct (open-weight) on the SDXL/FLUX SigLIP2
disagreement-subset images, the same populations already scored by GPT-5.5
and Qwen2.5-VL-7B in {sdxl,flux}_gpt55_siglip2_disagree_preds.csv /
{sdxl,flux}_qwen_siglip2_disagree_preds.csv (Table~4/e8_results).

Reviewer 2 asked for an open-weight alternative to GPT-5.5 on the
disagreement subset, since GPT-5.5 is closed/proprietary and the paper's
headline deployment-shift result (Case 3, cascade gain over a stale
semantic probe) currently rests on it. This tests whether a large
open-weight VLM (72B, in contrast to the 7B model already shown to give
negative gain) can reproduce GPT-5.5's positive-gain result.

Reuses the exact prompt and logit-based scoring methodology from
run_qwen_sdxl_flux.py (best real/fake logit-gap step across generated
tokens, not greedy-decoded text, per the VLM Domain Sensitivity finding in
Section VI.E).

Loads the model in 4-bit (bitsandbytes) to fit comfortably in 80GB VRAM.

Scoped to SigLIP2 only (not CLIP/DINOv2): SigLIP2 is the paper's primary
semantic backbone, and this is the configuration already used for the
bootstrap-CI (E10) and class-breakdown (E9) analyses this session, so the
72B result is directly comparable to those.

Produces: sdxl_qwen72b_siglip2_disagree_preds.csv, flux_qwen72b_siglip2_disagree_preds.csv
Checkpoints to: qwen72b_sdxl_flux_cache.csv (resumable)
"""
import gc
import os

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

BASE = "/workspace/benchmarks_ai_research/routing"
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
DEVICE = "cuda"

REASONING_PROMPT = (
    "Analyze this image carefully. Determine whether it is a REAL photograph "
    "or an AI-generated FAKE image. Consider both possibilities equally.\n"
    "Answer with exactly one word: REAL or FAKE."
)
REAL_LABEL, FAKE_LABEL = 0, 1

COMBOS = [
    ("SDXL", "sdxl_gpt55_siglip2_disagree_preds.csv", "sdxl_qwen72b_siglip2_disagree_preds.csv"),
    ("FLUX", "flux_gpt55_siglip2_disagree_preds.csv", "flux_qwen72b_siglip2_disagree_preds.csv"),
]

CACHE_CSV = os.path.join(BASE, "qwen72b_sdxl_flux_cache.csv")
CHECKPOINT_EVERY = 25


def safe_softmax_two(a, b):
    x = np.array([a, b], dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    p = e / e.sum()
    return float(p[0]), float(p[1])


def load_model():
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
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
    return {"y_qwen72b": y, "p_fake_qwen72b": p_fake, "raw": raw,
            "latency_qwen72b_ms": lat_ms, "qwen72b_input_tokens": n_in,
            "qwen72b_output_tokens": n_out}


def main():
    processor, model, real_id, fake_id = load_model()

    all_rows = {}
    for ds, gpt_file, _ in COMBOS:
        df = pd.read_csv(os.path.join(BASE, gpt_file))
        for _, r in df.iterrows():
            all_rows[r["path"]] = int(r["y_true"])
    print(f"Total unique images to score: {len(all_rows)}")

    cache = {}
    if os.path.exists(CACHE_CSV):
        prev = pd.read_csv(CACHE_CSV)
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
            res = {"y_qwen72b": 1, "p_fake_qwen72b": None, "raw": "ERROR",
                   "latency_qwen72b_ms": 0, "qwen72b_input_tokens": 0,
                   "qwen72b_output_tokens": 0}
        res["path"] = path
        res["y_true"] = y_true
        cache[path] = res

        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(items):
            dt = time.time() - t0
            done_now = sum(1 for p in items[:i+1] if p[0] not in cache or True)
            print(f"  {i+1}/{len(items)} ({(i+1)/dt:.3f} img/s, {dt:.0f}s elapsed)", flush=True)
            pd.DataFrame(list(cache.values())).to_csv(CACHE_CSV, index=False)

    pd.DataFrame(list(cache.values())).to_csv(CACHE_CSV, index=False)
    print(f"Saved {CACHE_CSV}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    cache_df = pd.DataFrame(list(cache.values()))
    for ds, gpt_file, out_file in COMBOS:
        df_gpt = pd.read_csv(os.path.join(BASE, gpt_file))
        merged = df_gpt[["path", "y_true"]].merge(
            cache_df[["path", "y_qwen72b", "p_fake_qwen72b", "raw",
                      "latency_qwen72b_ms", "qwen72b_input_tokens",
                      "qwen72b_output_tokens"]],
            on="path", how="left")
        merged.to_csv(os.path.join(BASE, out_file), index=False)
        acc = float(np.mean(merged["y_qwen72b"] == merged["y_true"]))
        print(f"{ds}: n={len(merged)} Qwen2.5-VL-72B accuracy={acc:.4f} -> {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
