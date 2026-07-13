"""
Fast parallel extraction for the split-zip GenImage archives (SD1.4, BigGAN).

The default 7z extraction is single-threaded and writes one small file at a
time, which is extremely slow on the network-mounted /workspace filesystem
(observed ~20 files/sec, i.e. hours for a 336k-file archive). This script:
  1. Concatenates the .z01..zNN + .zip volumes into one combined zip (fast
     sequential I/O, matches 7-Zip's split-zip convention).
  2. Extracts members in parallel using many worker threads, each with its
     own zipfile.ZipFile handle, to better utilize the network mount's
     concurrent I/O capacity (same reasoning as rclone --transfers=8).
"""
import argparse
import glob
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASETS = {
    "biggan": {
        "dl_dir": "/workspace/benchmarks_ai_research/data/_dl_biggan",
        "main_zip": "imagenet_ai_0419_biggan.zip",
        "part_glob": "imagenet_ai_0419_biggan.z[0-9][0-9]",
        "extract_to": "/workspace/benchmarks_ai_research/data/genimage/biggan",
        "combined": "/workspace/benchmarks_ai_research/data/_dl_biggan/_combined.zip",
    },
    "sd14": {
        "dl_dir": "/workspace/benchmarks_ai_research/data/_dl_sd14",
        "main_zip": "imagenet_ai_0419_sdv4.zip",
        "part_glob": "imagenet_ai_0419_sdv4.z[0-9][0-9]",
        "extract_to": "/workspace/benchmarks_ai_research/data/genimage/sd_1_4",
        "combined": "/workspace/benchmarks_ai_research/data/_dl_sd14/_combined.zip",
    },
}


def concat_volumes(cfg):
    parts = sorted(glob.glob(os.path.join(cfg["dl_dir"], cfg["part_glob"])))
    main = os.path.join(cfg["dl_dir"], cfg["main_zip"])
    ordered = parts + [main]
    print(f"  Concatenating {len(ordered)} volumes -> {cfg['combined']}")
    t0 = time.time()
    with open(cfg["combined"], "wb") as out:
        for p in ordered:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, length=64 * 1024 * 1024)
    dt = time.time() - t0
    size_gb = os.path.getsize(cfg["combined"]) / 1e9
    print(f"  Done in {dt:.0f}s ({size_gb:.1f} GB, {size_gb/dt*1000:.0f} MB/s)")


def extract_member(zip_path, name, extract_to):
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(name, extract_to)


def parallel_extract(cfg, workers=32):
    os.makedirs(cfg["extract_to"], exist_ok=True)
    with zipfile.ZipFile(cfg["combined"]) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    print(f"  {len(names)} files to extract with {workers} workers")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(extract_member, cfg["combined"], n, cfg["extract_to"])
                   for n in names]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 10000 == 0:
                dt = time.time() - t0
                print(f"    {done}/{len(names)} ({done/dt:.0f} files/s, {dt:.0f}s elapsed)")
    dt = time.time() - t0
    print(f"  Extraction done in {dt:.0f}s ({len(names)/dt:.0f} files/s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["biggan", "sd14", "both"], default="both")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--skip_concat", action="store_true",
                    help="Combined zip already exists, skip concatenation")
    args = p.parse_args()

    datasets = ["biggan", "sd14"] if args.dataset == "both" else [args.dataset]
    for name in datasets:
        cfg = DATASETS[name]
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        if not args.skip_concat and not os.path.exists(cfg["combined"]):
            concat_volumes(cfg)
        else:
            print(f"  Using existing combined zip: {cfg['combined']}")
        parallel_extract(cfg, workers=args.workers)

    print("\nAll done.")


if __name__ == "__main__":
    main()
