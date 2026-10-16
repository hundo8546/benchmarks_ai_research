"""
routing.pipeline.tables.cost_accounting — tab_cost_accounting.

Reviewer 1, point (10): "Complete the cost analysis... report actual
input/output token usage... and API latency separately from local GPU
inference... compare the cascade directly with standalone cheap detectors
and uniform GPT-5.5 inference."

This module does NOT call any API or GPU model. It only reads per-sample
columns that are ALREADY on disk in the verifier prediction CSVs
(routing/*_gpt55_*_preds.csv, routing/*qwen*_preds.csv) and aggregates them.
Every GPT-5.5 and Qwen prediction file in the routing/ directory was read
once (`head -1` + row count) to confirm its actual column names before
writing this module -- see the per-file registry below. Three real-world
facts surfaced by that inventory shape everything here:

  1. Column names are NOT uniform. Every *_gpt55_*_preds.csv file has the
     same schema (path, y_true, {y_verifier|y_gpt55}, raw, latency_ms,
     input_tokens, output_tokens). Qwen files are inconsistent: some have
     qwen_input_tokens/qwen_output_tokens/latency_qwen_ms
     (sd14_qwen_preds.csv, biggan_qwen_preds.csv, the sdxl/flux
     *_qwen_*_disagree_preds.csv files), others have ONLY latency_qwen_ms
     with no token columns at all (qwen_preds.csv [GenBuster in-domain],
     genbuster_qwen_expansion_preds.csv, biggan_distant_shift_qwen7b/72b_preds.csv).
     Rows below say explicitly, per file, whether token data exists.
  2. No full-validation-set GPT-5.5 prediction file exists ANYWHERE. Every
     *_gpt55_*_preds.csv file on disk is restricted to a disagreement
     subset (or, for biggan_distant_shift, a stratified 1,500-sample
     subset) -- confirmed by a prior audit referenced in the task brief and
     re-confirmed here by inspecting every gpt55 file's row count against
     the corresponding dataset's full val-set size. This means a true
     cascade-vs-uniform-GPT-5.5 cost comparison CANNOT be completed without
     a new, paid, full-val-set GPT-5.5 run. This module does not attempt
     that run; it only flags the gap (see NO_FULL_VAL_GPT55_RUN_NOTE and the
     `no_full_val_gpt55_run` column), for a later budget-approval checkpoint.
  3. Qwen, by contrast, DOES have full-val-set prediction files for all
     three in-domain datasets (qwen_preds.csv, sd14_qwen_preds.csv,
     biggan_qwen_preds.csv -- these are exactly what tab_main_results.csv's
     "Always esc." row's accuracy/latency figures are computed from), so a
     cascade-vs-uniform-Qwen comparison IS already possible from existing
     data; this module reports Qwen's full-val-set rows as such.

Dollar-cost estimates are GPT-5.5-only (Qwen runs locally: no API $-cost,
only latency/compute-time, so the price constants below are never applied
to a Qwen row). The price constants are explicit, clearly-labeled
PLACEHOLDERS -- see the block immediately below -- not a verified number.

Latency is split into two clearly-labeled categories in the output:
  - "measured": the per-call `latency_ms` / `latency_qwen_ms` column already
    present in these prediction CSVs (real, logged at call time).
  - "hardcoded constant": the FFT=2.5ms / CLIP=SigLIP2=DINOv2=10.4ms /
    CNNSpot=45.8ms / UnivFD=56.2ms cheap-detector constants and the
    Qwen-always-escalate 190.2/83.2/70.6ms (GenBuster/SD14/BigGAN) figures,
    which are copy-pasted literals reused across many files in this repo
    (pipeline/tables/main_results.py, regenerate_main_tables.py,
    retrain_bandits.py, analysis2.py, extractors/common.py,
    experiments/e0-e7, ...), not live-measured per run. Imported directly
    from pipeline.tables.main_results (plain module-level dicts -- no heavy
    import triggered) rather than re-typed here, to avoid yet another copy
    of the same literals.

CPU-only, no heavy imports: pipeline.tables.main_results's top-level imports
are only os/joblib/numpy/pandas/sklearn (no torch/cv2/decord), so importing
its LATENCY/ALWAYS_ESC_LATENCY dicts here does not pull in anything heavy.
"""
import os

import numpy as np
import pandas as pd

from pipeline.tables.main_results import LATENCY as HARDCODED_CHEAP_LATENCY_MS
from pipeline.tables.main_results import ALWAYS_ESC_LATENCY as HARDCODED_ALWAYS_ESC_QWEN_LATENCY_MS

ROUTING_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────────────────
# DOLLAR-COST CONSTANTS -- VERIFIED against claude_context_notes/openaitokentable.txt
# (gpt-5.5, <272K context length, as of this pricing table's publication).
# Two tiers are tracked: Standard (synchronous Responses API, used for every
# call already made in this pipeline) and Batch (async, <=24h turnaround,
# exactly half price -- recommended for any new bulk/non-latency-sensitive
# evaluation run, e.g. the uniform/always-escalate or real-image-source
# experiments, since none of those need a same-second response).
# Qwen is local inference and these constants are NEVER applied to a Qwen
# row (see `cost_usd_total_standard` / `_batch` left as NaN for every Qwen
# row below, with `verifier == "Qwen..."`).
# ─────────────────────────────────────────────────────────────────────────
GPT55_STANDARD_PRICE_PER_1M_INPUT_TOKENS = 5.00
GPT55_STANDARD_PRICE_PER_1M_OUTPUT_TOKENS = 30.00
GPT55_BATCH_PRICE_PER_1M_INPUT_TOKENS = 2.50
GPT55_BATCH_PRICE_PER_1M_OUTPUT_TOKENS = 15.00
# Kept for any external caller still referencing the old placeholder names --
# aliased to the verified Standard-tier price, not a separate guess.
GPT55_PRICE_PER_1M_INPUT_TOKENS = GPT55_STANDARD_PRICE_PER_1M_INPUT_TOKENS
GPT55_PRICE_PER_1M_OUTPUT_TOKENS = GPT55_STANDARD_PRICE_PER_1M_OUTPUT_TOKENS

NO_FULL_VAL_GPT55_RUN_NOTE = (
    "No full-validation-set ('uniform/always-escalate') GPT-5.5 prediction file exists for ANY "
    "dataset -- every *_gpt55_*_preds.csv on disk is restricted to a disagreement subset (or, for "
    "BigGAN distant shift, a stratified 1,500-sample subset). A true cascade-vs-uniform-GPT-5.5 "
    "cost comparison therefore cannot be completed without a new, paid, full-val-set GPT-5.5 run "
    "-- this is flagged for a later checkpoint / budget approval, not attempted here."
)

# ─────────────────────────────────────────────────────────────────────────
# Registry of every GPT-5.5 / Qwen prediction file found on disk under
# routing/, with the real column names confirmed by inspection (not
# guessed). `token_cols` is None where the file genuinely has no token
# columns (confirmed, not a bug in this module).
# ─────────────────────────────────────────────────────────────────────────
GPT55_FILES = [
    dict(file="gpt55_disagree_preds.csv", dataset="GenBuster",
         configuration="GenBuster in-domain (FFT+SigLIP2 disagreement)",
         verifier_col="y_verifier", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_verifier_comparison.csv",
         note="Raw file row count may exceed the canonical disagreement-eval-set overlap used in "
              "tab_verifier_comparison.csv (n=52 there) -- GPT-5.5 coverage on GenBuster is partial "
              "per the paper (Sec. V.B: 'GPT-5.5 coverage on GenBuster and SD~1.4 is partial, so we "
              "do not use those cells to support a GPT-5.5-specific conclusion')."),
    dict(file="sd14_gpt55_disagree_preds.csv", dataset="SD14",
         configuration="SD14 in-domain (FFT+SigLIP2 disagreement)",
         verifier_col="y_verifier", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_verifier_comparison.csv",
         note="Partial GPT-5.5 coverage on SD14 (canonical overlap n=23 in tab_verifier_comparison.csv) "
              "per the same paper caveat as GenBuster above."),
    dict(file="biggan_gpt55_disagree_preds.csv", dataset="BigGAN",
         configuration="BigGAN in-domain (FFT+SigLIP2 disagreement)",
         verifier_col="y_verifier", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_verifier_comparison.csv", note=""),
    dict(file="sdxl_gpt55_siglip2_disagree_preds.csv", dataset="SDXL",
         configuration="SDXL deployment shift, stale SigLIP2 probe",
         verifier_col="y_gpt55", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="sdxl_gpt55_dinov2_disagree_preds.csv", dataset="SDXL",
         configuration="SDXL deployment shift, stale DINOv2 probe",
         verifier_col="y_gpt55", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="sdxl_gpt55_clip_disagree_preds.csv", dataset="SDXL",
         configuration="SDXL deployment shift, stale CLIP probe",
         verifier_col="y_gpt55", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table=None,
         note="CLIP backbone excluded from tab_e8_results.csv (SD14 lacks raw CLIP features in this "
              "checkout, per pipeline/tables/baselines.py's docstring) -- this file exists on disk "
              "but is not part of any currently-published table."),
    dict(file="flux_gpt55_siglip2_disagree_preds.csv", dataset="FLUX",
         configuration="FLUX deployment shift, stale SigLIP2 probe",
         verifier_col="y_gpt55", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="flux_gpt55_dinov2_disagree_preds.csv", dataset="FLUX",
         configuration="FLUX deployment shift, stale DINOv2 probe",
         verifier_col="y_gpt55", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="flux_gpt55_clip_disagree_preds.csv", dataset="FLUX",
         configuration="FLUX deployment shift, stale CLIP probe",
         verifier_col="y_gpt55", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table=None,
         note="CLIP backbone excluded from tab_e8_results.csv -- not part of any currently-published table."),
    dict(file="biggan_distant_shift_gpt55_preds.csv", dataset="BigGAN",
         configuration="BigGAN distant shift (stale FFT+SigLIP2, stratified n=1500)",
         verifier_col="y_verifier", latency_col="latency_ms",
         token_cols=("input_tokens", "output_tokens"),
         canonical_table="tab_biggan_shift.csv", note=""),
]

QWEN_FILES = [
    dict(file="qwen_preds.csv", dataset="GenBuster",
         configuration="GenBuster in-domain, FULL val set (uniform/always-escalate Qwen)",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=None,
         canonical_table="tab_main_results.csv (Always esc. row); resliced for tab_verifier_comparison.csv",
         note="No token columns recorded in this file (only latency_qwen_ms). This is the FULL "
              "val-set file -- the one case where a uniform-escalate cost/latency comparison already "
              "exists, just without token counts."),
    dict(file="genbuster_qwen_expansion_preds.csv", dataset="GenBuster",
         configuration="GenBuster Qwen coverage expansion (799 additional val videos)",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=None,
         canonical_table=None,
         note="Supplementary coverage-expansion run (run_qwen7b_expansion.py); extractors/common.py's "
              "DATASETS['GenBuster']['qwen_preds'] still points at qwen_preds.csv, not this file, so "
              "it is not part of the canonical pipeline / any published table. No token columns."),
    dict(file="qwen_preds_v2.csv", dataset="GenBuster",
         configuration="GenBuster Qwen, superseded/partial rerun",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table=None,
         note="Explicitly superseded -- extractors/common.py's own inline comment reads 'Use "
              "qwen_preds.csv (2000 rows, full dataset) over qwen_preds_v2.csv (partial)'. Included "
              "here only for audit completeness; not used in any published table."),
    dict(file="sd14_qwen_preds.csv", dataset="SD14",
         configuration="SD14 in-domain, FULL val set (uniform/always-escalate Qwen)",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table="tab_main_results.csv (Always esc.); resliced for tab_verifier_comparison.csv, "
                         "tab_limited_data.csv, tab_cross_family_deployment.csv",
         note="Same underlying full-val-set predictions are resliced to different disagreement "
              "subsets for tab_limited_data.csv and the SD14 side of tab_cross_family_deployment.csv "
              "-- no separate cost rows are added for those two table families since the per-call "
              "cost/latency is already captured here."),
    dict(file="biggan_qwen_preds.csv", dataset="BigGAN",
         configuration="BigGAN in-domain, FULL val set (uniform/always-escalate Qwen)",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table="tab_main_results.csv (Always esc.); resliced for tab_cross_family_deployment.csv",
         note="Same note as sd14_qwen_preds.csv above, for the BigGAN side of tab_cross_family_deployment.csv."),
    dict(file="sdxl_qwen_siglip2_disagree_preds.csv", dataset="SDXL",
         configuration="SDXL deployment shift, stale SigLIP2 probe",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="sdxl_qwen_dinov2_disagree_preds.csv", dataset="SDXL",
         configuration="SDXL deployment shift, stale DINOv2 probe",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="sdxl_qwen_clip_disagree_preds.csv", dataset="SDXL",
         configuration="SDXL deployment shift, stale CLIP probe",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table=None, note="CLIP backbone excluded from tab_e8_results.csv -- not part of any published table."),
    dict(file="flux_qwen_siglip2_disagree_preds.csv", dataset="FLUX",
         configuration="FLUX deployment shift, stale SigLIP2 probe",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="flux_qwen_dinov2_disagree_preds.csv", dataset="FLUX",
         configuration="FLUX deployment shift, stale DINOv2 probe",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table="tab_e8_results.csv", note=""),
    dict(file="flux_qwen_clip_disagree_preds.csv", dataset="FLUX",
         configuration="FLUX deployment shift, stale CLIP probe",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=("qwen_input_tokens", "qwen_output_tokens"),
         canonical_table=None, note="CLIP backbone excluded from tab_e8_results.csv -- not part of any published table."),
    dict(file="biggan_distant_shift_qwen7b_preds.csv", dataset="BigGAN",
         configuration="BigGAN distant shift (stale FFT+SigLIP2, stratified n=1500)",
         verifier="Qwen2.5-VL-7B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=None,
         canonical_table="tab_biggan_shift.csv", note="No token columns in this file."),
    dict(file="biggan_distant_shift_qwen72b_preds.csv", dataset="BigGAN",
         configuration="BigGAN distant shift (stale FFT+SigLIP2, stratified n=1500)",
         verifier="Qwen2.5-VL-72B", verifier_col="y_qwen", latency_col="latency_qwen_ms",
         token_cols=None,
         canonical_table="tab_biggan_shift.csv", note="No token columns in this file; 4-bit quantized, local GPU inference."),
]


def _load(file):
    path = os.path.join(ROUTING_BASE, file)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _agg_row(spec, verifier_name, price_in, price_out):
    df = _load(spec["file"])
    row = {
        "verifier": verifier_name,
        "dataset": spec["dataset"],
        "configuration": spec["configuration"],
        "pred_file": spec["file"],
        "canonical_table": spec.get("canonical_table"),
        "file_found": df is not None,
    }
    if df is None:
        row.update({"n_samples": 0, "note": spec.get("note", "") + " [FILE NOT FOUND ON DISK]"})
        return row

    n = len(df)
    row["n_samples"] = n

    # ---- latency: MEASURED (real per-call column already in the file) ----
    lat_col = spec.get("latency_col")
    if lat_col and lat_col in df.columns:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        row["latency_ms_mean_MEASURED"] = float(lat.mean())
        row["latency_ms_total_MEASURED"] = float(lat.sum())
    else:
        row["latency_ms_mean_MEASURED"] = np.nan
        row["latency_ms_total_MEASURED"] = np.nan

    # ---- tokens: real, aggregated from the file's own columns ----
    token_cols = spec.get("token_cols")
    if token_cols is not None and all(c in df.columns for c in token_cols):
        in_col, out_col = token_cols
        in_tok = pd.to_numeric(df[in_col], errors="coerce")
        out_tok = pd.to_numeric(df[out_col], errors="coerce")
        row["tokens_available"] = True
        row["mean_input_tokens"] = float(in_tok.mean())
        row["total_input_tokens"] = float(in_tok.sum())
        row["mean_output_tokens"] = float(out_tok.mean())
        row["total_output_tokens"] = float(out_tok.sum())
    else:
        row["tokens_available"] = False
        row["mean_input_tokens"] = np.nan
        row["total_input_tokens"] = np.nan
        row["mean_output_tokens"] = np.nan
        row["total_output_tokens"] = np.nan

    # ---- $ cost: GPT-5.5 only, VERIFIED pricing, both Standard and Batch tiers ----
    is_gpt55 = verifier_name == "GPT-5.5"
    if is_gpt55 and row["tokens_available"]:
        cost_standard = (row["total_input_tokens"] / 1e6) * GPT55_STANDARD_PRICE_PER_1M_INPUT_TOKENS + \
                         (row["total_output_tokens"] / 1e6) * GPT55_STANDARD_PRICE_PER_1M_OUTPUT_TOKENS
        cost_batch = (row["total_input_tokens"] / 1e6) * GPT55_BATCH_PRICE_PER_1M_INPUT_TOKENS + \
                     (row["total_output_tokens"] / 1e6) * GPT55_BATCH_PRICE_PER_1M_OUTPUT_TOKENS
        row["cost_usd_total_standard"] = float(cost_standard)
        row["cost_usd_mean_per_sample_standard"] = float(cost_standard / n)
        row["cost_usd_total_batch"] = float(cost_batch)
        row["cost_usd_mean_per_sample_batch"] = float(cost_batch / n)
    else:
        row["cost_usd_total_standard"] = np.nan
        row["cost_usd_mean_per_sample_standard"] = np.nan
        row["cost_usd_total_batch"] = np.nan
        row["cost_usd_mean_per_sample_batch"] = np.nan

    row["cost_basis"] = ("GPT-5.5 Responses API; $ via verified gpt-5.5 (<272K context) pricing -- "
                          "Standard tier $5.00/$30.00 per 1M input/output tokens, "
                          "Batch tier (async, <=24h) $2.50/$15.00 -- see claude_context_notes/openaitokentable.txt"
                          if is_gpt55 else
                          "Qwen runs locally -- no $ API cost; cost reported as latency/compute-time only")

    # ---- hardcoded-constant comparison note (NOT measured for this row) ----
    row["hardcoded_constants_for_comparison"] = (
        f"cheap-stage constants (hardcoded, not measured here): "
        f"FFT={HARDCODED_CHEAP_LATENCY_MS['FFT']}ms, "
        f"CLIP/SigLIP2/DINOv2={HARDCODED_CHEAP_LATENCY_MS['CLIP']}ms, "
        f"CNNSpot={HARDCODED_CHEAP_LATENCY_MS['CNNSpot']}ms, "
        f"UnivFD={HARDCODED_CHEAP_LATENCY_MS['UnivFD']}ms; "
        f"Qwen always-escalate (hardcoded, not measured here): "
        f"GenBuster={HARDCODED_ALWAYS_ESC_QWEN_LATENCY_MS['GenBuster']}ms, "
        f"SD14={HARDCODED_ALWAYS_ESC_QWEN_LATENCY_MS['SD14']}ms, "
        f"BigGAN={HARDCODED_ALWAYS_ESC_QWEN_LATENCY_MS['BigGAN']}ms"
    )

    row["no_full_val_gpt55_run"] = True
    row["no_full_val_gpt55_run_note"] = NO_FULL_VAL_GPT55_RUN_NOTE if is_gpt55 else \
        "N/A for Qwen rows -- full-val-set Qwen predictions DO exist for the three in-domain " \
        "datasets (see qwen_preds.csv / sd14_qwen_preds.csv / biggan_qwen_preds.csv); only GPT-5.5 lacks a uniform run."

    row["note"] = spec.get("note", "")
    return row


def compute_cost_accounting():
    """tab_cost_accounting: real token/latency aggregates per verifier
    configuration, GPT-5.5 illustrative $ cost under an explicitly-placeholder
    price, and an explicit flag that no uniform/always-escalate GPT-5.5 run
    exists for any dataset (Qwen has one; GPT-5.5 does not)."""
    rows = []
    for spec in GPT55_FILES:
        rows.append(_agg_row(spec, "GPT-5.5", GPT55_PRICE_PER_1M_INPUT_TOKENS, GPT55_PRICE_PER_1M_OUTPUT_TOKENS))
    for spec in QWEN_FILES:
        rows.append(_agg_row(spec, spec["verifier"], GPT55_PRICE_PER_1M_INPUT_TOKENS, GPT55_PRICE_PER_1M_OUTPUT_TOKENS))

    cols = [
        "verifier", "dataset", "configuration", "pred_file", "canonical_table", "file_found", "n_samples",
        "latency_ms_mean_MEASURED", "latency_ms_total_MEASURED",
        "tokens_available", "mean_input_tokens", "total_input_tokens", "mean_output_tokens", "total_output_tokens",
        "cost_basis",
        "cost_usd_mean_per_sample_standard", "cost_usd_total_standard",
        "cost_usd_mean_per_sample_batch", "cost_usd_total_batch",
        "hardcoded_constants_for_comparison",
        "no_full_val_gpt55_run", "no_full_val_gpt55_run_note", "note",
    ]
    return pd.DataFrame(rows)[cols]
