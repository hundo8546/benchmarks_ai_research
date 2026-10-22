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
  data/<tag>_sdxl_flux/...                   size-matched PNGs (tag: see tag())
  routing/<tag>_sdxl_index.csv               mirror of sdxl_index.csv, paths -> SM copies
  routing/<tag>_flux_index.csv               mirror of flux_index.csv, paths -> SM copies
  routing/results/<tag>_image_map.csv        orig_path, sm_path, y_true, generator,
                                             datasets (which index(es) the image is in)

CPU-only, no API, no GPU.
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from extractors.common import ROUTING_BASE, RESULTS_DIR, ensure_results_dir

DATA_ROOT = "/workspace/benchmarks_ai_research/data"
TARGET = 512  # default; --size 256 builds the downscale-only variant (see tag())

SRC_INDICES = {
    "SDXL": os.path.join(ROUTING_BASE, "sdxl_index.csv"),
    "FLUX": os.path.join(ROUTING_BASE, "flux_index.csv"),
}


def tag(size=TARGET):
    """Name stem for a target size: "sizematched" for 512 (the original control, file
    names unchanged), "sizematched256" etc. otherwise. At 512, 96% of the ImageNet reals
    (median short side 375) are upscaled while every fake (1024) is downscaled; at 256
    only ~3% of reals are upscaled, so both classes are downscaled."""
    return "sizematched" if size == 512 else f"sizematched{size}"


def sm_root(size=TARGET):
    return os.path.join(DATA_ROOT, f"{tag(size)}_sdxl_flux")


def sm_path_for(orig_path, size=TARGET):
    """Deterministic size-matched path: mirror the original absolute path under
    sm_root(size), forcing a .png extension. Collision-free because the full original
    path (minus leading '/') is preserved."""
    rel = orig_path.lstrip("/")
    base, _ = os.path.splitext(rel)
    return os.path.join(sm_root(size), base + ".png")


def size_match(img, size=TARGET):
    """center-crop to square, resize to size x size (bicubic)."""
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.BICUBIC)


def _write_one(orig_path, sm, size):
    os.makedirs(os.path.dirname(sm), exist_ok=True)
    with Image.open(orig_path) as im:
        out = size_match(im, size)
    # write-then-rename so a concurrent or interrupted run never leaves
    # a truncated PNG at the final path
    tmp = f"{sm}.tmp{os.getpid()}_{threading.get_ident()}"
    out.save(tmp, format="PNG")
    os.replace(tmp, sm)


def build(size=TARGET, force=False, workers=16):
    ensure_results_dir()
    os.makedirs(sm_root(size), exist_ok=True)

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

    print(f"{len(rows)} unique images to size-match to {size}x{size} -> {sm_root(size)}")

    map_rows, todo = [], []
    for orig_path, info in rows.items():
        sm = sm_path_for(orig_path, size)
        map_rows.append({
            "orig_path": orig_path, "sm_path": sm,
            "y_true": info["y_true"], "generator": info["generator"],
            "datasets": "|".join(sorted(info["datasets"])),
        })
        if force or not os.path.exists(sm):
            todo.append((orig_path, sm))
    n_skipped = len(rows) - len(todo)

    n_done, n_err = 0, 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_write_one, o, sm, size): o for o, sm in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="size-match"):
            try:
                f.result()
                n_done += 1
            except Exception as e:
                print(f"  ERROR {futs[f]}: {e}")
                n_err += 1

    print(f"Wrote {n_done} new, skipped {n_skipped} existing, {n_err} errors.")

    # Mapping CSV (one row per unique image)
    map_df = pd.DataFrame(map_rows)
    map_csv = os.path.join(RESULTS_DIR, f"{tag(size)}_image_map.csv")
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
        out_csv = os.path.join(ROUTING_BASE, f"{tag(size)}_{ds_name.lower()}_index.csv")
        out.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} ({len(out)} rows)")

    return map_df


def verify(map_df=None, size=TARGET):
    """Confirm every size-matched image fully decodes as a size x size RGB PNG
    (im.load() decodes all pixel data, so a truncated file is caught -- a
    header-only check is not enough)."""
    if map_df is None:
        map_df = pd.read_csv(os.path.join(RESULTS_DIR, f"{tag(size)}_image_map.csv"))
    bad = []
    sizes = set()
    fmts = set()
    for sm in tqdm(map_df["sm_path"], desc="verify"):
        try:
            with Image.open(sm) as im:
                im.load()
                sizes.add(im.size)
                fmts.add(im.format)
                if im.size != (size, size) or im.format != "PNG" or im.mode != "RGB":
                    bad.append(sm)
        except Exception as e:
            bad.append(f"{sm}: {e}")
    print(f"distinct sizes: {sizes}  distinct formats: {fmts}")
    print(f"non-conforming: {len(bad)}")
    if bad[:5]:
        print("  e.g.", bad[:5])
    return len(bad) == 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=TARGET)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    m = build(size=a.size, force=a.force)
    verify(m, size=a.size)
