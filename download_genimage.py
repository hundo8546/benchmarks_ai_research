"""
Download GenImage (SD v1.4 and BigGAN) from Google Drive and extract.

Usage:
    python download_genimage.py --dataset biggan     # download BigGAN only (~40-60 GB)
    python download_genimage.py --dataset sd14       # download SD v1.4 only (~150-200 GB)
    python download_genimage.py --dataset both       # download both (default)
"""
import argparse
import os
import subprocess
import sys

import gdown

# Google Drive folder IDs (subfolders inside your shared Drive folder)
BIGGAN_FOLDER_ID = "1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt"
SD14_FOLDER_ID   = "12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy"

# Where the scripts expect the data
BASE = "/workspace/benchmarks_ai_research/data/genimage"
BIGGAN_EXTRACT_DIR = os.path.join(BASE, "biggan")
SD14_EXTRACT_DIR   = os.path.join(BASE, "sd_1_4")

# Main archive filenames (7z uses these; the .z01, .z02, ... parts sit alongside)
BIGGAN_MAIN_ZIP = "imagenet_ai_0419_biggan.zip"
SD14_MAIN_ZIP   = "imagenet_ai_0419_sdv4.zip"


def run(cmd, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def download_folder(folder_id, dest_dir, label):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Downloading {label} to {dest_dir}")
    print(f"{'='*60}")
    gdown.download_folder(
        id=folder_id,
        output=dest_dir,
        quiet=False,
        use_cookies=False,
    )


def extract(download_dir, main_zip, extract_to, label):
    zip_path = os.path.join(download_dir, main_zip)
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found after download.")
        sys.exit(1)
    os.makedirs(extract_to, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Extracting {label} to {extract_to}")
    print(f"(7z reads all .z01/.z02/... parts automatically)")
    print(f"{'='*60}")
    run(f'7z x "{zip_path}" -o"{extract_to}" -y')


def verify(extract_to, expected_subpath, label):
    full_path = os.path.join(extract_to, expected_subpath)
    if os.path.exists(full_path):
        n = sum(len(files) for _, _, files in os.walk(full_path))
        print(f"\n✓ {label}: found {n} files at expected path")
        print(f"  {full_path}")
    else:
        print(f"\n✗ {label}: expected path NOT found: {full_path}")
        print(f"  Contents of {extract_to}:")
        for entry in os.listdir(extract_to)[:10]:
            print(f"    {entry}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["biggan", "sd14", "both"], default="both")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip download, only extract (if archives already present)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(BASE, exist_ok=True)

    # ── BigGAN ──────────────────────────────────────────────────────────────
    if args.dataset in ("biggan", "both"):
        dl_dir = "/tmp/genimage_biggan"
        if not args.skip_download:
            download_folder(BIGGAN_FOLDER_ID, dl_dir, "BigGAN")
        extract(dl_dir, BIGGAN_MAIN_ZIP, BIGGAN_EXTRACT_DIR, "BigGAN")
        verify(BIGGAN_EXTRACT_DIR,
               "imagenet_ai_0419_biggan/val",
               "BigGAN")

    # ── SD v1.4 ─────────────────────────────────────────────────────────────
    if args.dataset in ("sd14", "both"):
        dl_dir = "/tmp/genimage_sd14"
        if not args.skip_download:
            download_folder(SD14_FOLDER_ID, dl_dir, "SD v1.4")
        extract(dl_dir, SD14_MAIN_ZIP, SD14_EXTRACT_DIR, "SD v1.4")
        # SD14 archive has an internal nested path (data/genimage/sd_1_4/...)
        verify(SD14_EXTRACT_DIR,
               "data/genimage/sd_1_4/imagenet_ai_0419_sdv4/val",
               "SD v1.4")

    print("\nDone. Run the pipeline next:")
    print("  python routing/build_biggan_index.py")
    print("  python routing/build_sd14_index.py")


if __name__ == "__main__":
    main()
