"""
Run Qwen2.5-VL-72B-Instruct (open-weight) on the BigGAN stale-FFT/stale-SigLIP2
disagreement sample (biggan_stale_disagree_sample.csv, n=1500), for parity
with the SDXL/FLUX three-way verifier comparison (GPT-5.5, Qwen2.5-VL-7B,
Qwen2.5-VL-72B) run this session.

Reuses the exact model config and prompt from run_qwen72b_sdxl_flux.py.

Produces: biggan_qwen72b_stale_disagree_preds.csv
Checkpoints to the same file every 25 samples (resumable).
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

SAMPLE_CSV = os.path.join(BASE, "biggan_stale_disagree_sample.csv")
OUT_CSV = os.path.join(BASE, "biggan_qwen72b_stale_disagree_preds.csv")
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

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                              return_dict_in_generate=True, output_scores=True)
    torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    new_tokens = gen.sequences[:, inputs["input_ids"].shape[-1]:]
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
            "latency_qwen72b_ms": lat_ms}


def main():
    processor, model, real_id, fake_id = load_model()
    df = pd.read_csv(SAMPLE_CSV)

    cache = {}
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        cache = {r["path"]: r.to_dict() for _, r in prev.iterrows()}
        print(f"Resuming: {len(cache)} already scored")

    items = list(zip(df["path"], df["y_true"]))
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
                   "latency_qwen72b_ms": 0}
        res["path"] = path
        res["y_true"] = int(y_true)
        cache[path] = res

        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(items):
            dt = time.time() - t0
            print(f"  {i+1}/{len(items)} ({(i+1)/dt:.3f} img/s, {dt:.0f}s elapsed)", flush=True)
            pd.DataFrame(list(cache.values())).to_csv(OUT_CSV, index=False)

    pd.DataFrame(list(cache.values())).to_csv(OUT_CSV, index=False)
    acc = np.mean(pd.DataFrame(list(cache.values()))["y_qwen72b"] ==
                   pd.DataFrame(list(cache.values()))["y_true"])
    print(f"BigGAN: n={len(cache)} Qwen2.5-VL-72B accuracy={acc:.4f} -> {OUT_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
