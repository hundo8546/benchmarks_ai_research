"""
routing/pipeline/robustness/sdxl_flux_image_inventory.py

Reviewer-confound check (remainingupdates1.md Step 1): records native width,
height, file format and on-disk file size for every image used in the SDXL
and FLUX benchmarks -- the 2,000 fakes each, plus the real images actually
used in each of the three real-image-source variants (ImageNet/genimage,
COCO val2017, PASCAL VOC original). This is the SDXL/FLUX analogue of
native_resolution.py's BigGAN/SD1.4/GenBuster inventory, extended with
format and file size since CONSOLIDATED_full_sdxl_flux.csv shows every
SDXL/FLUX fake used exactly 1,271 GPT-5.5 input tokens while reals varied --
format/size are additional, independent evidence for the same confound.

No image pixels are decoded beyond PIL's lazy header read for .size/.format;
file size is os.path.getsize. CPU-only, no API calls.

Output: routing/results/sdxl_flux_image_inventory.csv
  columns: path, y_true, generator, dataset, real_source, width, height,
           aspect, format, size_bytes
  One row per (path, dataset, real_source): a fake path appears three times
  per dataset (once per real_source variant it co-occurs with) with identical
  width/height/format/size_bytes; each real_source's own 2,000 reals appear
  once, under their own real_source tag.
"""
import os
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extractors.common import RESULTS_DIR, ROUTING_BASE, load_index, ensure_results_dir

DATASETS = {
    "SDXL": {
        "imagenet": os.path.join(ROUTING_BASE, "sdxl_index.csv"),
        "coco": os.path.join(RESULTS_DIR, "sdxl_index_coco_real.csv"),
        "voc": os.path.join(RESULTS_DIR, "sdxl_index_original_real.csv"),
    },
    "FLUX": {
        "imagenet": os.path.join(ROUTING_BASE, "flux_index.csv"),
        "coco": os.path.join(RESULTS_DIR, "flux_index_coco_real.csv"),
        "voc": os.path.join(RESULTS_DIR, "flux_index_original_real.csv"),
    },
}


def _native_meta(path, cache):
    if path in cache:
        return cache[path]
    with Image.open(path) as im:
        w, h = im.size
        fmt = im.format
    size_bytes = os.path.getsize(path)
    meta = (w, h, fmt, size_bytes)
    cache[path] = meta
    return meta


def build_inventory(force=False):
    ensure_results_dir()
    out_csv = os.path.join(RESULTS_DIR, "sdxl_flux_image_inventory.csv")
    if os.path.exists(out_csv) and not force:
        return pd.read_csv(out_csv)

    cache = {}
    rows = []
    for ds_name, sources in DATASETS.items():
        for real_source, index_file in sources.items():
            index_rows = load_index(index_file)
            for r in tqdm(index_rows, desc=f"{ds_name}/{real_source}"):
                path = r["path"]
                y_true = int(r["label"])
                try:
                    w, h, fmt, size_bytes = _native_meta(path, cache)
                except Exception as e:
                    print(f"  Skip {path}: {e}")
                    continue
                rows.append({
                    "path": path, "y_true": y_true, "generator": r.get("generator", "unknown"),
                    "dataset": ds_name, "real_source": real_source,
                    "width": w, "height": h, "aspect": w / h if h else float("nan"),
                    "format": fmt, "size_bytes": size_bytes,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows, {len(cache)} unique images)")
    return df


def summarize(df):
    lines = []
    for ds_name in DATASETS:
        for real_source in DATASETS[ds_name]:
            sub = df[(df.dataset == ds_name) & (df.real_source == real_source)]
            for label, y in [("real", 0), ("fake", 1)]:
                s = sub[sub.y_true == y]
                if len(s) == 0:
                    continue
                res_strs = s.width.astype(str) + "x" + s.height.astype(str)
                top = res_strs.value_counts()
                fmt_top = s.format.value_counts()
                size_kb = s.size_bytes / 1024
                lines.append(
                    f"{ds_name}/{real_source}/{label}: n={len(s)}, "
                    f"{res_strs.nunique()} distinct res (top {top.index[0]} "
                    f"{top.iloc[0] / len(s):.1%}), "
                    f"formats={dict(fmt_top)}, "
                    f"size_kb median={size_kb.median():.1f} "
                    f"[{size_kb.min():.1f}, {size_kb.max():.1f}]"
                )
    return "\n".join(lines)


if __name__ == "__main__":
    df = build_inventory()
    print(summarize(df))
