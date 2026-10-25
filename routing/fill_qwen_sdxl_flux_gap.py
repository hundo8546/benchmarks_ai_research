"""Fill Qwen2.5-VL-7B labels for SDXL/FLUX e8 disagreement samples that the
original run_qwen_sdxl_flux.py population (GPT-5.5 subsets) did not cover.
Reuses run_qwen_sdxl_flux.infer_single (same prompt, logit-gap scoring) and its
cache; appends rows to {sdxl,flux}_qwen_{siglip2,dinov2}_disagree_preds.csv.
"""
import os, sys, time
import pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch
import run_qwen_sdxl_flux as rq
from routing.pipeline.tables import confidence_routing as cr
from routing.experiments import e8_modern_diffusion as e8

CACHE = os.path.join(HERE, "sdxl_flux_qwen_cache.csv")
cache_df = pd.read_csv(CACHE)
cache = {r["path"]: r.to_dict() for _, r in cache_df.iterrows()}

todo = {}   # (ds, sem) -> frame of missing disagreement rows
for ds in ["SDXL", "FLUX"]:
    for sem in ["SigLIP2", "DINOv2"]:
        df = cr.load_e8_frame(ds, sem, e8)
        q = pd.read_csv(os.path.join(HERE, f"{ds.lower()}_qwen_{sem.lower()}_disagree_preds.csv"))
        todo[(ds, sem)] = df[df["disagree"] & ~df["path"].isin(q["path"])][["path", "y_true"]]
need = {}
for m in todo.values():
    for p, y in zip(m.path, m.y_true):
        if p not in cache:
            need[p] = int(y)
print(f"{len(need)} images need a fresh Qwen call; rest come from cache", flush=True)

if need:
    processor, model, real_id, fake_id = rq.load_model()
    t0 = time.time()
    for i, (p, y) in enumerate(need.items()):
        img = Image.open(p).convert("RGB")
        res = rq.infer_single(model, processor, real_id, fake_id, img)   # no fallback label on error: let it raise
        res["path"], res["y_true"] = p, y
        cache[p] = res
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache(); print(f"  {i+1}/{len(need)} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(list(cache.values())).to_csv(CACHE, index=False)

cols = ["path", "y_true", "y_qwen", "p_fake_qwen", "raw", "latency_qwen_ms", "qwen_input_tokens", "qwen_output_tokens"]
cdf = pd.DataFrame(list(cache.values()))[cols].drop_duplicates("path")
for (ds, sem), m in todo.items():
    f = os.path.join(HERE, f"{ds.lower()}_qwen_{sem.lower()}_disagree_preds.csv")
    old = pd.read_csv(f)
    new = m[["path"]].merge(cdf, on="path", how="left")
    assert new["y_qwen"].notna().all()
    out = pd.concat([old, new[old.columns]], ignore_index=True)
    out.to_csv(f, index=False)
    print(ds, sem, len(old), "->", len(out))
