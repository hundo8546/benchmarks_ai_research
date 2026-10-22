"""
routing/pipeline/robustness/build_sizematched_sdxl_flux.py

Step 2 of the SDXL/FLUX resolution-confound control (remainingupdates1.md):
build a size-matched copy of the entire SDXL/FLUX benchmark set so that real
and fake images are indistinguishable by dimension or file format.

The source set is the union of sdxl_index.csv and flux_index.csv, which is
exactly the set of images in results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv
(2,000 SDXL fakes + 2,000 FLUX fakes + 3,315 unique ImageNet/genimage reals =
7,315 unique images; SDXL and FLUX share 685 reals).

ONE identical transform is applied to every image, real and fake:
  1. center-crop to a square of side min(width, height)
  2. resize to 512x512 with bicubic resampling (PIL.Image.BICUBIC)
  3. save as PNG
After this, every image -- real or fake -- is a 512x512 PNG, so the native-
resolution / file-format shortcut measured in Step 1 is removed. The ONLY
residual class difference is that real images still carry their original JPEG
compression history (quantization artifacts) baked into the pixels before the
re-encode; fakes were born as PNG. This is an unavoidable residual and is
noted in the write-up.

Originals are NEVER overwritten. Size-matched copies are written under a new
root, mirroring each original's absolute path (so no two sources can collide),
with the extension forced to .png.

Outputs:
  <SM_ROOT>/...                              size-matched PNGs
  routing/sizematched_sdxl_index.csv         mirror of sdxl_index.csv, paths -> SM copies
  routing/sizematched_flux_index.csv         mirror of flux_index.csv, paths -> SM copies
  routing/results/sizematched_image_map.csv  orig_path, sm_path, y_true, generator,
                                             datasets (which index(es) the image is in)

CPU-only, no API, no GPU.
"""
import os
import sys

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extractors.common import ROUTING_BASE, RESULTS_DIR, ensure_results_dir

DATA_ROOT = "/workspace/benchmarks_ai_research/data"
SM_ROOT = "/workspace/benchmarks_ai_research/data/sizematched_sdxl_flux"
TARGET = 512

SRC_INDICES = {
    "SDXL": os.path.join(ROUTING_BASE, "sdxl_index.csv"),
    "FLUX": os.path.join(ROUTING_BASE, "flux_index.csv"),
}


def sm_path_for(orig_path):
    """Deterministic size-matched path: mirror the original absolute path under
    SM_ROOT, forcing a .png extension. Collision-free because the full original
    path (minus leading '/') is preserved."""
    rel = orig_path.lstrip("/")
    base, _ = os.path.splitext(rel)
    return os.path.join(SM_ROOT, base + ".png")


def size_match(img):
    """center-crop to square, resize to TARGET x TARGET (bicubic)."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((TARGET, TARGET), Image.BICUBIC)


def build(force=False):
    ensure_results_dir()
    os.makedirs(SM_ROOT, exist_ok=True)

    # Collect every unique image across both indices; remember which index(es)
    # each image belongs to and its label.
    rows = {}
    index_frames = {}
    for ds_name, index_file in SRC_INDICES.items():
        df = pd.read_csv(index_file)
        index_frames[ds_name] = df
        for r in df.itertuples(index=False):
            info = rows.setdefault(r.path, {"y_true": int(r.label), "generator": r.generator, "datasets": set()})
            info["datasets"].add(ds_name)
            if int(r.label) != info["y_true"]:
                raise ValueError(f"Label mismatch across indices for {r.path}")

    print(f"{len(rows)} unique images to size-match -> {SM_ROOT}")

    n_done, n_skipped, n_err = 0, 0, 0
    map_rows = []
    for orig_path, info in tqdm(rows.items(), desc="size-match"):
        sm = sm_path_for(orig_path)
        map_rows.append({
            "orig_path": orig_path, "sm_path": sm,
            "y_true": info["y_true"], "generator": info["generator"],
            "datasets": "|".join(sorted(info["datasets"])),
        })
        if os.path.exists(sm) and not force:
            n_skipped += 1
            continue
        try:
            os.makedirs(os.path.dirname(sm), exist_ok=True)
            with Image.open(orig_path) as im:
                out = size_match(im)
            # write-then-rename so a concurrent or interrupted run never leaves
            # a truncated PNG at the final path
            tmp = f"{sm}.tmp{os.getpid()}"
            out.save(tmp, format="PNG")
            os.replace(tmp, sm)
            n_done += 1
        except Exception as e:
            print(f"  ERROR {orig_path}: {e}")
            n_err += 1

    print(f"Wrote {n_done} new, skipped {n_skipped} existing, {n_err} errors.")

    # Mapping CSV (one row per unique image)
    map_df = pd.DataFrame(map_rows)
    map_csv = os.path.join(RESULTS_DIR, "sizematched_image_map.csv")
    map_df.to_csv(map_csv, index=False)
    print(f"Saved {map_csv} ({len(map_df)} rows)")

    # Per-dataset index mirrors, preserving original row order and labels
    orig_to_sm = dict(zip(map_df["orig_path"], map_df["sm_path"]))
    for ds_name, df in index_frames.items():
        out = df.copy()
        out["path"] = out["path"].map(orig_to_sm)
        n_missing = out["path"].isna().sum()
        if n_missing:
            raise ValueError(f"{ds_name}: {n_missing} paths have no size-matched copy")
        out_csv = os.path.join(ROUTING_BASE, f"sizematched_{ds_name.lower()}_index.csv")
        out.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} ({len(out)} rows)")

    return map_df


def verify(map_df=None):
    """Confirm every size-matched image fully decodes as a 512x512 RGB PNG
    (im.load() decodes all pixel data, so a truncated file is caught -- a
    header-only check is not enough)."""
    if map_df is None:
        map_df = pd.read_csv(os.path.join(RESULTS_DIR, "sizematched_image_map.csv"))
    bad = []
    sizes = set()
    fmts = set()
    for sm in tqdm(map_df["sm_path"], desc="verify"):
        try:
            with Image.open(sm) as im:
                im.load()
                sizes.add(im.size)
                fmts.add(im.format)
                if im.size != (TARGET, TARGET) or im.format != "PNG" or im.mode != "RGB":
                    bad.append(sm)
        except Exception as e:
            bad.append(f"{sm}: {e}")
    print(f"distinct sizes: {sizes}  distinct formats: {fmts}")
    print(f"non-conforming: {len(bad)}")
    if bad[:5]:
        print("  e.g.", bad[:5])
    return len(bad) == 0


if __name__ == "__main__":
    m = build(force="--force" in sys.argv)
    verify(m)
