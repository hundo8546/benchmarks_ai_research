"""
Download AIGIBench (HorizonTEL/AIGIBench) subsets needed for E8.

Downloads:
  val/  — 20 real-image category zips (~830 MB total)
  test/SDXL.zip     (~9 GB)
  test/FLUX1-dev.zip (~8.8 GB)

Extracts to:
  /workspace/benchmarks_ai_research/data/aigibench/real/
  /workspace/benchmarks_ai_research/data/aigibench/sdxl/
  /workspace/benchmarks_ai_research/data/aigibench/flux/

Usage:
    python download_aigibench.py                    # all three
    python download_aigibench.py --subsets real      # only real images
    python download_aigibench.py --subsets sdxl flux # only generated
    python download_aigibench.py --skip_download     # re-extract only
"""
import argparse
import os
import sys
import zipfile

from huggingface_hub import hf_hub_download

REPO_ID = "HorizonTEL/AIGIBench"
BASE    = "/workspace/benchmarks_ai_research/data/aigibench"

HF_TOKEN = os.environ["HF_TOKEN"]

REAL_CATEGORIES = [
    "airplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]

GENERATED = {
    "sdxl": "test/SDXL.zip",
    "flux":  "test/FLUX1-dev.zip",
}


def download_and_extract(hf_path, local_dir, label):
    os.makedirs(local_dir, exist_ok=True)
    zip_dest = os.path.join(local_dir, os.path.basename(hf_path))

    if not os.path.exists(zip_dest):
        print(f"  Downloading {hf_path} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=local_dir,
        )
        # hf_hub_download saves to local_dir/<filename>, move if nested
        downloaded = os.path.join(local_dir, hf_path)
        if os.path.exists(downloaded) and downloaded != zip_dest:
            os.makedirs(os.path.dirname(zip_dest), exist_ok=True)
            os.rename(downloaded, zip_dest)
    else:
        print(f"  Already downloaded: {zip_dest}")

    # Extract
    extract_marker = zip_dest + ".extracted"
    if not os.path.exists(extract_marker):
        print(f"  Extracting {zip_dest} ...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(local_dir)
        open(extract_marker, "w").close()
        print(f"  Extracted {label} -> {local_dir}")
    else:
        print(f"  Already extracted: {label}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", nargs="+",
                   choices=["real", "sdxl", "flux"], default=["real", "sdxl", "flux"])
    p.add_argument("--skip_download", action="store_true",
                   help="Skip download, re-extract only")
    args = p.parse_args()

    os.makedirs(BASE, exist_ok=True)

    if "real" in args.subsets:
        print("\n=== Downloading real val images ===")
        real_dir = os.path.join(BASE, "real")
        for cat in REAL_CATEGORIES:
            hf_path = f"val/{cat}.zip"
            download_and_extract(hf_path, real_dir, f"real/{cat}")
        n = sum(len(fs) for _, _, fs in os.walk(real_dir))
        print(f"  Real images: {n} files in {real_dir}")

    for subset in ["sdxl", "flux"]:
        if subset not in args.subsets:
            continue
        print(f"\n=== Downloading {subset.upper()} ===")
        gen_dir = os.path.join(BASE, subset)
        download_and_extract(GENERATED[subset], gen_dir, subset.upper())
        n = sum(len(fs) for _, _, fs in os.walk(gen_dir)
                if not any(f.endswith('.extracted') for f in fs))
        print(f"  {subset.upper()} images: ~{n} files in {gen_dir}")

    print("\nDone. Next steps:")
    print("  python routing/build_aigibench_index.py")
    print("  python routing/experiments/e8_modern_diffusion.py --extract_features")
    print("  python routing/experiments/e8_modern_diffusion.py --evaluate")


if __name__ == "__main__":
    main()
