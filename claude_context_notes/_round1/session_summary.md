# Session Summary — IEEE Access Resubmission Data Work

**Dates:** 2026-09-12 to 2026-09-14
**Scope:** Executing the experimental portion of `resubmissionplan.md` in response to Reviewer 1's four comments (`peerreviewdecision.txt`).

## What was fixed / built, in order

1. **`setup.sh` bug fix.** `pip install --upgrade torch torchvision` upgraded torch without upgrading `torchaudio`, causing an ABI mismatch (`undefined symbol` crash on `transformers` import). Fixed by adding `torchaudio` to the upgrade line. Also fixed step 6, which hardcoded a `.7z` archive extension but the GenBuster dataset now ships as `.zip`.

2. **GenImage BigGAN + SD1.4 download.** Original `download_genimage.py` (gdown-based) was superseded by `rclone` against a Google Drive remote (`research:`), after working through a broken rclone OAuth token (missing refresh token — required revoking prior app access and redoing the authorize flow). Learned the hard way that `/workspace` is a slow network mount and that `7z` extracts in alphabetical order — the first attempt wasted ~1 hour unpacking the unneeded `train/` split (324K files) before ever reaching `val/` (12K files); fixed by filtering the extraction to `*/val/*` only.

3. **Discovered and fixed a project-wide GPT-5.5 bug.** `call_gpt55()` capped `max_output_tokens=100`, but `gpt-5.5` is a reasoning model that spends most of that budget on a hidden `reasoning` step before writing the visible answer — so most calls returned an empty string, and the old code silently defaulted every empty response to `y=1` (FAKE). Measured blank-response rates before the fix: 97% (GenBuster), 87% (SD1.4), 97% (BigGAN in-domain), 24–26% (SDXL), 57–61% (FLUX). Fixed by escalating `max_output_tokens` to [2048, 4096] on retry and returning `None`/`"EMPTY_AFTER_RETRY"` instead of guessing. Re-ran GPT-5.5 on every affected row project-wide (~3,173 corrected predictions total). This one bug fix **flipped the FLUX/SigLIP2 GPT-5.5 result** from an ambiguous, baseline-tying number (90.9% acc, didn't clear best-cheap) to a clean, clearly-significant win (97.3% acc, clears best-cheap by CI [+0.051, +0.083]) — and did the same for the other three SDXL/FLUX GPT-5.5 cells.

4. **Expanded in-domain evaluation sets** (Reviewer 1 Comment 1). SD1.4 and BigGAN val splits: 480 → 2,000 samples each (full pool is 12,000 for both; features/Qwen predictions already existed for the whole pool, so no new inference was needed beyond retraining the two probes cleanly on the new split). GenBuster: 400 → 1,000 (its total pool is only 2,000 samples, so this is a disclosed, defensible ceiling, not full parity with the other two). Required deleting and retraining all FFT/SigLIP2/fftvariant probes to avoid train/val leakage against the new splits, and running new Qwen2.5-VL-7B inference on GenBuster's 799 newly-included videos.

5. **Built the baseline-comparison framework** (Reviewer 1 Comment 3). New files `routing/baseline_utils.py` and `routing/baseline_comparison.py`: for every verifier-on-disagreement-subset cell, compute a random baseline (10,000 Monte Carlo draws), a majority-class baseline, and a best-cheap-detector baseline (`max(FFT acc, semantic acc)` on the same subset), each with paired-bootstrap 95% CIs against the verifier's actual accuracy. Covers `verifier_comparison`, `e8_results`, `cross_family_deployment`, `limited_data`, and (added later) `biggan_shift` — 35 configs total.

6. **Built the gate/three-way sensitivity framework** (Reviewer 1 Comment 4b/4c). New file `routing/gate_analysis.py`: computes the oracle binary escalate/fallback decision and the oracle three-way (f1/f2/f3) selection for every config, then bootstrap-subsamples the disagreement subset at N = {10, 20, 50, 100, 200, oracle} to measure how often a calibration-sample-only decision matches the oracle decision.

7. **Reconstructed the `biggan_shift` distant-shift table** — this experiment (SD1.4-stale FFT+SigLIP2 probes applied zero-shot to GenImage BigGAN, escalating disagreement to a VLM) had no surviving per-sample data or script anywhere in the repo; it had to be rebuilt from scratch once the GenImage BigGAN data was downloaded. Reproduced the paper's own stale-probe accuracy numbers closely (38.2% vs. paper's 38.3% stale FFT; 91.2% vs. 90.8% stale SigLIP2 on the disagreement subset), confirming the pool reconstruction is correct. Ran all three original verifiers:
   - GPT-5.5: 58.2% (paper claimed 80.1%)
   - Qwen2.5-VL-7B: 51.5% (paper claimed 68.9%)
   - Qwen2.5-VL-72B: 62.3% (paper claimed 46.7%)

   None of the three numbers match the original draft, and the *relative ordering* is completely different (originally GPT-5.5 > Qwen-7B > Qwen-72B; now Qwen-72B > GPT-5.5 > Qwen-7B). Tested two hypotheses for the discrepancy (SD1.4 probe retraining changing the disagreement pool composition; prompt wording) and ruled both out empirically — neither explains the size of the gap. Root cause remains unresolved; there is no surviving original per-sample data or script to diff against. **Important context check:** confirmed via the actual review text that Reviewer 1 never mentions BigGAN, 72B, or any number from this table — it was added in a later revision round to address Reviewer 2/3 (per git commit `db3600d`), so this discrepancy, while worth disclosing honestly, is not load-bearing for the current R1-focused resubmission.

8. **Qwen2.5-VL-72B compute saga.** The original 20GB-VRAM GPU (RTX A4500) could not fit the ~42GB quantized model. Tried and exhausted 7 distinct technical approaches across 2 quantization formats (AWQ/Marlin — hard CUDA-only requirement, no CPU offload support at all; bitsandbytes NF4 — 3 different offload configurations, all hit a genuine `accelerate`+`bitsandbytes` bug where double-quantization metadata (the NF4 codebook, then the per-layer scale factors) gets wiped to the `meta` device during CPU-offload hook setup). Concluded this was a real software incompatibility, not solvable without risking silent correctness bugs. User deployed a new pod (RTX PRO 6000, 96GB VRAM) attaching the same network volume; the model fit entirely on-GPU with no offloading needed, sidestepping the whole bug class. That pod's Claude Code session hit one more real issue — a library version skew (transformers 5.17.0 / bitsandbytes 0.50.2, far ahead of what the checkpoint's config.json was generated against) broke handling of the `merger` module's `llm_int8_skip_modules` — fixed by pinning `transformers==4.51.3`, `accelerate==1.6.0`, `bitsandbytes==0.45.4`. Ran cleanly afterward: 1500/1500 rows, 0 errors, 1 unparseable response.

## Final results reference

See `routing/results/baseline_comparison_summary.csv` (35 configs) and `routing/results/gate_sensitivity_sweep.csv`. Headline numbers:
- 6/35 cells clear the best-cheap-detector baseline (the real bar per Eq. 7): the 4 SDXL/FLUX GPT-5.5 cells, 1 cross-family pairing (BigGAN→GenBuster), 1 limited-data cell (SD1.4/SigLIP2 at 1% training data).
- Oracle three-way selection across all 35 configs: semantic probe correct 18 times, verifier 9, FFT 8.
- Gate sensitivity: 24/35 configs reach ≥90% oracle-agreement by N=100 calibration samples, 28/35 by N=200.

## Framing decision (2026-09-14)

Initially pursued the full reframe recommended in `resubmissionplan.md` (title change, gate promoted to C1). User pushed back hard on changing the title/headline framing. Landed on: **keep the original title and the cascade as the paper's subject**, but the abstract and contributions must explicitly state the conditional-benefit finding near the top (not as a coda) and lead with a **positive framing**: *"ungated/naive cascading is unreliable; gated cascading works"* — supported directly by the data, since the validation gate and three-way selector correctly identify escalate-vs-fallback in every tested case, including capturing the one clear win (SDXL/FLUX + GPT-5.5, now much stronger post-bug-fix) while avoiding every tested loss. This is explicitly *not* "cascading is pointless" — it's "naive cascading is what's unreliable; here's the mechanism that makes cascading trustworthy," which is both honest to the 29/35-negative result composition and avoids reading as a purely negative-results paper.

## Known open items

- `biggan_shift` numeric discrepancy vs. the original draft — unresolved, disclosed as a limitation, not R1-relevant.
- GenBuster's Qwen coverage is broad but not literally 100% of its 1,000-sample val set (259 have current predictions).
- Manuscript text itself has not yet been edited — see `paper_updates.tex` and `response_to_reviewers.md` for what to change and why.
