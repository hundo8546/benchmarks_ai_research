# routing/

This README maps the **currently authoritative** scripts behind the routing
results in the IEEE Access paper under revision. The root of `routing/`
used to hold ~75 loose, duplicated scripts with no indication of which one
produced which number; dead/superseded scripts have been moved to
`_archive/` (never deleted — `git log --follow` recovers history). See
[`_archive/MANIFEST.md`](_archive/MANIFEST.md) for the full list of archived
scripts, why each was retired, and what replaced it.

A consolidated entry point is being built separately in `routing/pipeline/`
(may not exist yet, or may still be under active development — that work is
out of scope for this map). Once finished, it is expected to subsume most of
the stage-by-stage scripts listed below into a single pipeline.

## Authoritative pipeline, by stage

### 1. Index building
- `build_sd14_index.py` — builds the GenImage SD1.4 index CSV (real/fake path pairs, seeded sample).
- `build_biggan_index.py` — builds the GenImage BigGAN index CSV.
- `build_genbuster_index.py` — rebuilds `benchmark_index_with_gen.csv` from the extracted GenBuster data.
- `build_aigibench_index.py` — builds index CSVs for the AIGIBench SDXL and FLUX subsets. *(owned by a separate in-progress Phase-0 bug-fix agent — left untouched by this cleanup.)*
- `expand_val_splits.py` — expands in-domain val splits to >=2,000 samples where the source pool allows (Reviewer 1 Comment 1).
- `fast_extract_genimage.py` — fast parallel extraction for the split-zip GenImage archives (SD1.4, BigGAN), replacing slow single-threaded 7z extraction.
- `hfgenbuster.py` — HuggingFace Hub login/download helper for GenBuster assets.

### 2. Feature extraction
- `generate_clip_features.py` / `_biggan.py` / `_sd14.py` — extract CLIP features per dataset.
- `generate_cnnspot_features.py` / `_biggan.py` / `_sd14.py` — extract CNNSpot (CNNDetection) features per dataset.
- `generate_clip_preds.py` / `_biggan.py` / `_sd14.py` — apply the trained CLIP classifier to produce per-dataset CLIP predictions.
- `merge_all_features.py` / `_biggan.py` / `_sd14.py` — merge the individual per-backbone feature/prediction outputs into one dataset per domain.
- `generate_bandit_dataset.py` — assembles the merged learned-escalation-classifier training dataset.

### 3. Probe training
- `train_clip_classifier.py` / `_biggan.py` / `_sd14.py` — train the CLIP logistic-regression classifiers. *(owned by the separate Phase-0 agent — left untouched.)*
- `train_siglip2_classifiers.py` — trains (or loads cached) in-domain SigLIP2 probes for GenBuster/SD14/BigGAN, using the shared train/val split convention (`extractors.common.load_val_paths`) so results are comparable across the FFT/SigLIP2/DINOv2 family.
- `train_dinov2_classifiers.py` — same, for DINOv2 probes.
- `baseline_utils.py` — shared utilities for the random/majority-class/best-cheap-detector baselines (Reviewer 1 Comment 3).

### 4. Verifier calls (Qwen / GPT-5.5 / UniVFD)
- `run_qwen_v2.py`, `sd14_run_qwen_v2.py`, `biggan_run_qwen_v2.py` — **current** Qwen2.5-VL runner family, using logit-based score extraction (fixes the greedy-decode "REAL"-bias failure mode documented in Section VI.C). Each writes `{prefix}qwen_preds.csv`.
- `run_qwen7b_expansion.py` — fills in new Qwen2.5-VL-7B predictions needed by the expanded val splits (GenBuster + BigGAN distant-shift).
- `run_qwen72b_biggan_shift.py` — runs Qwen2.5-VL-72B (4-bit) as a third verifier on the BigGAN distant-shift 1,500-sample subset.
- `run_qwen_sdxl_flux.py` — runs Qwen2.5-VL-7B on the SDXL/FLUX disagreement subsets for the Table 7/8 Qwen-only comparison column.
- `run_gpt55_expanded.py` — runs GPT-5.5 on the new disagreement-subset samples introduced by the expanded val splits; appends to `{prefix}gpt55_disagree_preds.csv`.
- `run_biggan_shift_gpt55.py` — runs GPT-5.5 on the BigGAN distant-shift 1,500-sample subset (reconstructs the paper's `biggan_shift` table).
- `fix_gpt55_blanks.py` — re-runs GPT-5.5 on any row with a blank/truncated response across all `gpt55_*_disagree_preds.csv` files, in place.
- `fix_table6_verifier_comparison.py` — recomputes Table 6 (verifier_comparison) on the correct FFT+SigLIP2 disagreement subset (the original numbers were computed on the stale CNNSpot+CLIP subset).
- `fill_table6_gpt55_gap.py` — calls GPT-5.5 on the FFT+SigLIP2 disagreement-subset images identified as missing by `fix_table6_verifier_comparison.py`.
- `run_univfd.py` — runs the UniversalFakeDetect baseline verifier. *(Note: `run_univfd.py` and its `*_univfd_preds.csv` outputs show modified-but-uncommitted changes from other in-progress work; left as-is.)*

### 5. Learned escalation classifier / routing analysis
- `retrain_bandits.py` — retrains the CNNSpot+CLIP-feature learned escalation classifiers on the current (v2, logit-decoded) Qwen predictions; produces `bandit_model.pkl` / `sd14_bandit_model.pkl` / `biggan_bandit_model.pkl`. **This is the pickle `analysis2.py` actually loads for the main learned-escalation-classifier tables** — it was not in the original classification handed to this cleanup pass, but inspection showed it is load-bearing, so it was left in place rather than archived. See the note in `_archive/MANIFEST.md`.
- `retrain_bandits_fft_siglip2.py` — retrains the learned escalation classifier on FFT + SigLIP2 features instead, so the learned escalation classifier and the disagreement-rule comparison (Table 15 vs 16) use the same backbone ("fairness fix"). Produces `{prefix}bandit_model_fft_siglip2.pkl`.
- `bandit_transfer_fft_siglip2.py` — builds the cross-dataset learned-escalation-classifier transfer matrix using the fair FFT+SigLIP2+Qwen learned escalation classifier.
- `cross_family_deployment.py` — cross-family deployment experiment: applies a stale cross-dataset SigLIP2 probe + in-domain FFT + Qwen disagreement cascade for every (train, test) dataset pair.
- `baseline_comparison.py` — random / majority-class / best-cheap-detector baselines on the disagreement subset across `verifier_comparison`, `e8_results`, `cross_family_deployment`, and `limited_data` tables (Reviewer 1 Comment 3). Includes `biggan_shift` (a stale docstring claiming otherwise was corrected on 2026-10-15 to match the code).
- `gate_analysis.py` — validation gate sensitivity sweep and three-way f1/f2/f3 selection extension (Reviewer 1 Comment 4), reusing `baseline_comparison.py`'s per-cell arrays.

### 6. Table / figure generation
- `analysis2.py` — **the current, correct successor to `analysis.py`**; the primary table/analysis driver most other authoritative scripts feed into.
- `regenerate_main_tables.py` — regenerates `tab:main_results`, `tab:cheap_path`, and `tab:escalation` at the expanded evaluation-set sizes, using FFT+SigLIP2 as the primary artifact/semantic pair (matches the current paper, not the old CNNSpot+CLIP pipeline).

## Everything else

Anything not listed above and not inside `experiments/`, `extractors/`,
`results/`, or `pipeline/` has been archived to `_archive/<category>/` with
a documented reason and its replacement — see
[`_archive/MANIFEST.md`](_archive/MANIFEST.md).
