"""
routing.pipeline.dedup — duplicate / cross-benchmark overlap checker.

Phase 0 remediation item B: the real-image pools behind several benchmarks
are sampled independently (see build_aigibench_index.py, expand_val_splits.py)
and there is no existing code anywhere that checks whether the same physical
image ends up doing double duty across benchmarks (e.g. scored once as "SD14
in-domain real" and again as "SDXL deployment-shift real"). This module adds
that check.

Two kinds of check:
  1. Exact-duplicate file content (sha256 hash of file bytes) — catches the
     same bytes saved under two different paths.
  2. Path-level set overlap between specific benchmark (dataset, class)
     pairs that are expected or suspected to share a source pool.

Usage:
    python dedup.py            # runs the full Phase 0 overlap report
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extractors.common import DATASETS, ROUTING_BASE, RESULTS_DIR, ensure_results_dir

SDXL_INDEX = os.path.join(ROUTING_BASE, "sdxl_index.csv")
FLUX_INDEX = os.path.join(ROUTING_BASE, "flux_index.csv")
SD14_INDEX = DATASETS["SD14"]["index"]
BIGGAN_INDEX = DATASETS["BigGAN"]["index"]
GENBUSTER_INDEX = DATASETS["GenBuster"]["index"]
SD14_VAL_PATHS = DATASETS["SD14"]["val_paths"]


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR:{e}"


def _label_col(df):
    return "y_true" if "y_true" in df.columns else "label"


def _real_fake_paths(index_csv):
    """Return (real_paths:set, fake_paths:set) for an index CSV using the
    label==0 -> real, label==1 -> fake convention used throughout this repo
    (confirmed in build_aigibench_index.py and sd14/biggan/sdxl/flux index
    CSVs)."""
    df = pd.read_csv(index_csv)
    col = _label_col(df)
    real = set(df.loc[df[col] == 0, "path"])
    fake = set(df.loc[df[col] == 1, "path"])
    return real, fake


def sd14_eval_real_paths():
    """Real-class subset of SD14's canonical eval split
    (sd14_bandit_val_paths.npy intersected with sd14_index.csv label==0)."""
    eval_paths = set(np.load(SD14_VAL_PATHS, allow_pickle=True).tolist())
    real_all, _ = _real_fake_paths(SD14_INDEX)
    return eval_paths & real_all


def overlap_row(name_a, set_a, name_b, set_b, cls):
    n_a, n_b = len(set_a), len(set_b)
    n_overlap = len(set_a & set_b)
    pct = (n_overlap / n_a * 100.0) if n_a else float("nan")
    return {
        "dataset_a": name_a, "dataset_b": name_b, "class": cls,
        "n_a": n_a, "n_b": n_b, "n_overlap": n_overlap,
        "pct_overlap_of_a": pct,
    }


def compute_overlap_report(write=True):
    rows = []

    sd14_real_all, sd14_fake_all = _real_fake_paths(SD14_INDEX)
    sd14_eval_real = sd14_eval_real_paths()

    have_sdxl = os.path.exists(SDXL_INDEX)
    have_flux = os.path.exists(FLUX_INDEX)
    sdxl_real = sdxl_fake = flux_real = flux_fake = set()
    if have_sdxl:
        sdxl_real, sdxl_fake = _real_fake_paths(SDXL_INDEX)
    if have_flux:
        flux_real, flux_fake = _real_fake_paths(FLUX_INDEX)

    # --- The headline checks: SD14-eval-real vs SDXL/FLUX real pools -------
    if have_sdxl:
        rows.append(overlap_row("SD14_eval_real", sd14_eval_real, "SDXL_real", sdxl_real, "real"))
    if have_flux:
        rows.append(overlap_row("SD14_eval_real", sd14_eval_real, "FLUX_real", flux_real, "real"))

    # --- Full SD14 real pool vs SDXL/FLUX real pools (context) -------------
    if have_sdxl:
        rows.append(overlap_row("SD14_full_real", sd14_real_all, "SDXL_real", sdxl_real, "real"))
    if have_flux:
        rows.append(overlap_row("SD14_full_real", sd14_real_all, "FLUX_real", flux_real, "real"))

    # --- SDXL vs FLUX real pools (both drawn from the same SD14 nature dir) -
    if have_sdxl and have_flux:
        rows.append(overlap_row("SDXL_real", sdxl_real, "FLUX_real", flux_real, "real"))

    # --- Cross-dataset sanity checks: GenBuster / BigGAN vs everyone else --
    genbuster_real, genbuster_fake = _real_fake_paths(GENBUSTER_INDEX) \
        if _has_label01(GENBUSTER_INDEX) else (set(pd.read_csv(GENBUSTER_INDEX)["path"]), set())
    biggan_real, biggan_fake = _real_fake_paths(BIGGAN_INDEX)

    cross_pairs = [
        ("GenBuster_all", genbuster_real | genbuster_fake, "SD14_all", sd14_real_all | sd14_fake_all),
        ("GenBuster_all", genbuster_real | genbuster_fake, "BigGAN_all", biggan_real | biggan_fake),
        ("BigGAN_all", biggan_real | biggan_fake, "SD14_all", sd14_real_all | sd14_fake_all),
    ]
    if have_sdxl:
        cross_pairs.append(("GenBuster_all", genbuster_real | genbuster_fake, "SDXL_all", sdxl_real | sdxl_fake))
        cross_pairs.append(("BigGAN_all", biggan_real | biggan_fake, "SDXL_all", sdxl_real | sdxl_fake))
    if have_flux:
        cross_pairs.append(("GenBuster_all", genbuster_real | genbuster_fake, "FLUX_all", flux_real | flux_fake))
        cross_pairs.append(("BigGAN_all", biggan_real | biggan_fake, "FLUX_all", flux_real | flux_fake))

    for name_a, set_a, name_b, set_b in cross_pairs:
        rows.append(overlap_row(name_a, set_a, name_b, set_b, "all"))

    report = pd.DataFrame(rows)
    if write:
        ensure_results_dir()
        out_path = os.path.join(RESULTS_DIR, "image_overlap_report.csv")
        report.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
    return report


def _has_label01(index_csv):
    try:
        df = pd.read_csv(index_csv, nrows=5)
        col = _label_col(df)
        return set(df[col].unique()) <= {0, 1}
    except Exception:
        return False


def exact_duplicate_check(path_sets, sample_cap=None):
    """
    Given a dict {name: set_of_paths}, hash every file (optionally capped per
    set for speed) and report any sha256 collisions that occur ACROSS
    different named sets (i.e. the same file content reused under a
    different path in a different benchmark).
    """
    hash_to_origin = {}
    collisions = []
    for name, paths in path_sets.items():
        plist = sorted(paths)
        if sample_cap is not None:
            plist = plist[:sample_cap]
        for p in plist:
            if not os.path.exists(p):
                continue
            h = sha256_of_file(p)
            if h in hash_to_origin and hash_to_origin[h][0] != name:
                collisions.append({
                    "hash": h,
                    "origin_a": hash_to_origin[h][0], "path_a": hash_to_origin[h][1],
                    "origin_b": name, "path_b": p,
                })
            else:
                hash_to_origin[h] = (name, p)
    return collisions


def main():
    print("=== Phase 0 image-overlap report ===\n")
    report = compute_overlap_report(write=True)
    pd.set_option("display.width", 160)
    print(report.to_string(index=False))

    print("\n=== Headline numbers (prior-audit confirmation) ===")
    for _, r in report.iterrows():
        if r["dataset_a"] == "SD14_eval_real":
            print(f"  {r['dataset_a']} vs {r['dataset_b']}: "
                  f"{r['n_overlap']}/{r['n_a']} = {r['pct_overlap_of_a']:.1f}% overlap")

    print("\n=== Cross-dataset sanity (should be ~0) ===")
    for _, r in report.iterrows():
        if r["class"] == "all":
            flag = "OK" if r["n_overlap"] == 0 else "UNEXPECTED OVERLAP"
            print(f"  {r['dataset_a']} vs {r['dataset_b']}: n_overlap={r['n_overlap']}  [{flag}]")


if __name__ == "__main__":
    main()
