"""
Build the 4 approved GPT-5.5 Batch-API request sets (Phase: paid batch
evaluation, ~17,900 total calls, Batch tier pricing). VALIDATES and WRITES
ready-to-submit JSONL files under routing/results/batch_requests/ -- does
NOT submit anything (no OPENAI_API_KEY is set in this environment; see
pipeline/batch_gpt55.py's submit_batch(), which raises loudly if called
without a key).

Every row list below is RE-DERIVED NOW from the same on-disk data and the
same logic already used elsewhere in this repo (re-running/reusing their
functions, never re-implementing the logic differently or trusting an old
cached count):

  (a) Bugfix reruns = 19 "FA"-truncated rows (re-grepped live from the 6
      sdxl/flux_gpt55_*_disagree_preds.csv files -- confirmed still
      exactly 19, matching verifiers.py's docstring) UNION the current
      fill_table6_gpt55_gap.py backfill target (re-derived live via
      fix_table6_verifier_comparison.py's own get_fft_siglip2_disagree_val()
      + DATASET_CFG, comparing against the CURRENT *_gpt55_disagree_preds.csv
      coverage). IMPORTANT FINDING: this backfill target is NOT the old
      ~142 (81 SD14 + 1 BigGAN + 60 GenBuster) the fill_table6_gpt55_gap.py
      docstring quotes -- re-running the exact same logic live today gives
      716 missing rows (509 SD14 + 0 BigGAN + 207 GenBuster), a 5x
      increase. Running fix_table6_verifier_comparison.py's main() (a
      read-only reporting script with no API calls) during this
      investigation reproduced the SAME 716 figure and refreshed its
      cached output at results/table6_fft_siglip2_fix.csv, which had been
      stale at the old, much smaller gap. The two subsets (19 FA + 716
      backfill) do not overlap (0 duplicate paths) -- total 735 rows.

  (b) Original PASCAL VOC union = the exact disagreement-UNION path lists
      (FFT-vs-SigLIP2 OR FFT-vs-DINOv2) behind results/disagreement_union_counts.csv's
      "original" rows, re-derived by calling analyze_real_source_disagreement.py's
      own existing_fake_preds()/new_real_preds() helpers directly (not
      reimplemented) and unioning the two backbones' disagree masks at
      path granularity instead of just counting them. Reproduced counts
      match the on-disk CSV exactly: sdxl sig_only=2198 din_only=2227
      union=2839; flux sig_only=2401 din_only=2351 union=3001.

  (c) COCO union = same, source="coco": sdxl sig_only=1434 din_only=1532
      union=1892; flux sig_only=1624 din_only=1641 union=2034.

  (d) Uniform/always-escalate full SDXL+FLUX = every row (real AND fake,
      label used as y_true) in sdxl_index.csv + flux_index.csv, 4000 each,
      8000 total -- no disagreement filtering at all.

Cost estimates use the REAL empirical average input/output token counts
already computed by pipeline/tables/cost_accounting.py from actual on-disk
GPT-5.5 call records (imported read-only, not retyped/re-guessed), at the
verified Batch-tier price ($2.50 / $15.00 per 1M input/output tokens,
gpt-5.5 <272K context -- claude_context_notes/openaitokentable.txt). Two
different empirical averages are used, documented per-row in MANIFEST.csv:
  - "sdxl_flux_avg": the n-weighted average over cost_accounting's SDXL +
    FLUX GPT-5.5 rows only (979.1 input / 149.8 output tokens/call, n=9342
    real prior calls) -- used for sets (b)/(c)/(d), which are purely
    SDXL/FLUX images.
  - "all_gpt55_avg": the n-weighted average over ALL of cost_accounting's
    GPT-5.5 rows (853.7 input / 159.8 output tokens/call, n=9376) -- used
    for set (a), which mixes GenBuster/SD14/BigGAN/SDXL/FLUX images.
These are necessarily ESTIMATES (actual per-image token counts vary with
image content/resolution and model response length) extrapolated from
prior real calls on similar image pools, not a guarantee of the real
batch's eventual cost.
"""
import os
import sys

import pandas as pd

ROUTING = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROUTING)
sys.path.insert(0, os.path.join(ROUTING, "extractors"))
sys.path.insert(0, os.path.join(ROUTING, "experiments"))

from pipeline.batch_gpt55 import build_batch_request_file, split_batch_request_file, _stable_custom_id  # noqa: E402
from pipeline.tables.cost_accounting import compute_cost_accounting, \
    GPT55_BATCH_PRICE_PER_1M_INPUT_TOKENS, GPT55_BATCH_PRICE_PER_1M_OUTPUT_TOKENS  # noqa: E402

OUT_DIR = os.path.join(ROUTING, "results", "batch_requests")

FA_TRUNCATION_FILES = [
    "sdxl_gpt55_clip_disagree_preds.csv", "sdxl_gpt55_dinov2_disagree_preds.csv",
    "sdxl_gpt55_siglip2_disagree_preds.csv", "flux_gpt55_clip_disagree_preds.csv",
    "flux_gpt55_dinov2_disagree_preds.csv", "flux_gpt55_siglip2_disagree_preds.csv",
]


# ─────────────────────────────────────────────────────────────────────────
# (a) Bugfix reruns
# ─────────────────────────────────────────────────────────────────────────
def build_rows_bugfix_reruns():
    rows = []
    seen = set()

    for f in FA_TRUNCATION_FILES:
        df = pd.read_csv(os.path.join(ROUTING, f))
        fa = df[df["raw"] == "FA"]
        for _, r in fa.iterrows():
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            rows.append({"path": r["path"], "y_true": int(r["y_true"])})
    n_fa = len(rows)

    from fix_table6_verifier_comparison import get_fft_siglip2_disagree_val, DATASET_CFG
    n_backfill = 0
    for ds_name, prefix, qwen_file, gpt_file in DATASET_CFG:
        df = get_fft_siglip2_disagree_val(ds_name)
        df_dis = df[df["disagree"]].reset_index(drop=True)
        gpt_path = os.path.join(ROUTING, gpt_file)
        df_gpt_old = pd.read_csv(gpt_path) if os.path.exists(gpt_path) else pd.DataFrame(columns=["path"])
        covered = set(df_gpt_old["path"])
        missing = df_dis[~df_dis["path"].isin(covered)].reset_index(drop=True)
        for _, r in missing.iterrows():
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            rows.append({"path": r["path"], "y_true": int(r["y_true"])})
            n_backfill += 1

    print(f"  bugfix_reruns: {n_fa} FA-truncation rows + {n_backfill} fill_table6 backfill rows "
          f"(0 overlap) = {len(rows)} total")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# (b)/(c) Original VOC / COCO disagreement UNION
# ─────────────────────────────────────────────────────────────────────────
def build_rows_real_source_union(source: str):
    from analyze_real_source_disagreement import existing_fake_preds, new_real_preds

    all_rows = []
    for ds in ["sdxl", "flux"]:
        fft_fake = existing_fake_preds(ds, "FFT")
        fft_real = new_real_preds(ds, "FFT", source)
        fft_all = pd.concat([fft_fake, fft_real], ignore_index=True)

        disagree_union_paths = set()
        for sem in ["SigLIP2", "DINOv2"]:
            sem_fake = existing_fake_preds(ds, sem)
            sem_real = new_real_preds(ds, sem, source)
            sem_all = pd.concat([sem_fake, sem_real], ignore_index=True)

            merged = fft_all.rename(columns={"y_pred": "y_fft"}).merge(
                sem_all.rename(columns={"y_pred": "y_sem"})[["path", "y_sem"]], on="path", how="inner")
            disagree_mask = (merged["y_fft"] != merged["y_sem"])
            disagree_union_paths |= set(merged.loc[disagree_mask, "path"])

        y_true_lookup = dict(zip(fft_all["path"], fft_all["y_true"]))
        ds_rows = [{"path": p, "y_true": int(y_true_lookup[p])} for p in disagree_union_paths]
        print(f"  {source}/{ds}: union n={len(ds_rows)}")
        all_rows.extend(ds_rows)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────
# (d) Full SDXL + FLUX (uniform/always-escalate)
# ─────────────────────────────────────────────────────────────────────────
def build_rows_full_sdxl_flux():
    rows = []
    for idx_file in ["sdxl_index.csv", "flux_index.csv"]:
        df = pd.read_csv(os.path.join(ROUTING, idx_file))
        for _, r in df.iterrows():
            rows.append({"path": r["path"], "y_true": int(r["label"])})
    return rows


def estimate_cost(n_rows: int, mean_in: float, mean_out: float) -> float:
    total_in = n_rows * mean_in
    total_out = n_rows * mean_out
    return (total_in / 1e6) * GPT55_BATCH_PRICE_PER_1M_INPUT_TOKENS + \
           (total_out / 1e6) * GPT55_BATCH_PRICE_PER_1M_OUTPUT_TOKENS


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Empirical per-call token averages, read-only from cost_accounting.py
    # (real on-disk GPT-5.5 call records, not re-guessed).
    cost_df = compute_cost_accounting()
    gpt = cost_df[cost_df["verifier"] == "GPT-5.5"]
    gpt_tok = gpt[gpt["mean_input_tokens"].notna()]

    all_n = gpt_tok["n_samples"].sum()
    all_mean_in = (gpt_tok["mean_input_tokens"] * gpt_tok["n_samples"]).sum() / all_n
    all_mean_out = (gpt_tok["mean_output_tokens"] * gpt_tok["n_samples"]).sum() / all_n

    sf = gpt_tok[gpt_tok["dataset"].isin(["SDXL", "FLUX"])]
    sf_n = sf["n_samples"].sum()
    sf_mean_in = (sf["mean_input_tokens"] * sf["n_samples"]).sum() / sf_n
    sf_mean_out = (sf["mean_output_tokens"] * sf["n_samples"]).sum() / sf_n

    print(f"Empirical token averages (from cost_accounting.py, real prior GPT-5.5 calls):")
    print(f"  ALL GPT-5.5 rows (n={int(all_n)}): mean_input={all_mean_in:.1f} mean_output={all_mean_out:.1f}")
    print(f"  SDXL+FLUX rows only (n={int(sf_n)}): mean_input={sf_mean_in:.1f} mean_output={sf_mean_out:.1f}")
    print()

    print("=== (a) bugfix_reruns ===")
    rows_a = build_rows_bugfix_reruns()
    print("=== (b) original_voc_union ===")
    rows_b = build_rows_real_source_union("original")
    print("=== (c) coco_union ===")
    rows_c = build_rows_real_source_union("coco")
    print("=== (d) full_sdxl_flux ===")
    rows_d = build_rows_full_sdxl_flux()
    print(f"  full_sdxl_flux: n={len(rows_d)}")
    print()

    sets = [
        ("bugfix_reruns", rows_a, all_mean_in, all_mean_out, "all_gpt55_avg"),
        ("original_voc_union", rows_b, sf_mean_in, sf_mean_out, "sdxl_flux_avg"),
        ("coco_union", rows_c, sf_mean_in, sf_mean_out, "sdxl_flux_avg"),
        ("full_sdxl_flux", rows_d, sf_mean_in, sf_mean_out, "sdxl_flux_avg"),
    ]

    manifest_rows = []
    grand_total_rows = 0
    grand_total_cost = 0.0
    for name, rows, mean_in, mean_out, basis in sets:
        for r in rows:
            r["custom_id"] = _stable_custom_id(r["path"])
            r["image_path"] = r["path"]
        out_path = os.path.join(OUT_DIR, f"{name}.jsonl")
        print(f"\nBuilding {name} ({len(rows)} candidate rows) -> {out_path}")
        summary = build_batch_request_file(rows, out_path)

        # Batch API caps each uploaded file at 200MB -- this project's
        # base64-JPEG payloads make several of these sets exceed that as a
        # single file. Split now so every file actually listed as
        # "ready to submit" really is (submit_batch() does NOT auto-split).
        submit_files = split_batch_request_file(out_path)
        is_split = len(submit_files) > 1

        est_cost = estimate_cost(summary["n_rows_written"], mean_in, mean_out)
        grand_total_rows += summary["n_rows_written"]
        grand_total_cost += est_cost

        manifest_rows.append({
            "name": name,
            "n_rows": summary["n_rows_written"],
            "n_rows_skipped": summary["n_rows_skipped"],
            "token_avg_basis": basis,
            "assumed_mean_input_tokens": round(mean_in, 1),
            "assumed_mean_output_tokens": round(mean_out, 1),
            "estimated_cost_usd_at_batch_pricing": round(est_cost, 2),
            "unsplit_jsonl_path": os.path.relpath(out_path, ROUTING),
            "unsplit_jsonl_size_mb": round(summary["jsonl_file_size_bytes"] / 1e6, 1),
            "n_submit_parts": len(submit_files),
            "submit_file_paths": ";".join(os.path.relpath(p, ROUTING) for p in submit_files),
            "custom_id_map_path": os.path.relpath(summary["custom_id_map_path"], ROUTING),
        })
        if is_split:
            print(f"  -> {name} exceeds the 200MB Batch API file-size limit as a single file; "
                  f"split into {len(submit_files)} parts for submission (submit_batch() once per part, "
                  f"same custom_id_map.csv covers all parts).")

    manifest_path = os.path.join(OUT_DIR, "MANIFEST.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nSaved manifest -> {manifest_path}")
    print(pd.DataFrame(manifest_rows).to_string(index=False))
    print(f"\nGRAND TOTAL: {grand_total_rows} rows across 4 sets, "
          f"estimated ${grand_total_cost:.2f} at Batch-tier pricing "
          f"(${GPT55_BATCH_PRICE_PER_1M_INPUT_TOKENS}/{GPT55_BATCH_PRICE_PER_1M_OUTPUT_TOKENS} per 1M in/out tokens)")


if __name__ == "__main__":
    main()
