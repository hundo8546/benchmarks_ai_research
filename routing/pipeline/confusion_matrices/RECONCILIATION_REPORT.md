# Reconciliation of SDXL/FLUX class-wise numbers (Limitations prose) vs. Table 5 (`tab:e8_results`)

## 1. Where every number comes from

### Limitations-section prose (95.8% / 73.6% stale; 84.7% post-escalation; FLUX 95.4%→91.2%)
No `.py` file in the repo (including `_archive/`) contains the literal strings
`95.8`, `73.6`, `84.7`, `86.5`, or `91.2`, and `plot_fig_a_semantic_degradation.py` /
`plot_fig_b_cascade_gain_scatter.py` only hardcode *overall* accuracies (e.g.
`0.8465` for SDXL+SigLIP2), not per-class recall. So these numbers are **not
hardcoded anywhere live** — they trace back to an *aggregate* row in
`routing/results/e8_modern_diffusion.csv` (mtime 2026-09-12 17:57:52, written by
`routing/experiments/e8_modern_diffusion.py`'s `--evaluate`/`--run_gpt55` step):

```
SDXL,SigLIP2,...,sem_stale_acc=0.8465,...,cascade_gpt55_acc=0.8805,gain=0.034
FLUX,SigLIP2,...,sem_stale_acc=0.90925,...,cascade_gpt55_acc=0.913,gain=0.00375
```

Only the **overall accuracy** is stored in that CSV (no per-class split), but the
prose numbers back out almost exactly from it assuming a balanced 2000/2000 split
(confirmed: `sdxl_index.csv`/`flux_index.csv` are exactly 2000 real / 2000 fake):

- SDXL: 0.5×(95.8% + 73.6%) = 84.70% ≈ 0.8465 stored (diff 0.05pp)
- FLUX: 0.5×(95.4% + 86.5%) = 90.95% ≈ 0.90925 stored (diff 0.025pp)
- SDXL post-escalation: solving 2·0.8805 − 95.8% gives real≈91.4% (drop 4.4pp, in the stated "4.0–4.3" ballpark)
- FLUX post-escalation: solving 2·0.913 − 91.2% gives real≈91.4% (drop 4.0pp, matches stated range)

This is a very tight match **only for the SigLIP2 row** of each dataset (the DINOv2
rows of `e8_modern_diffusion.csv`, 0.85475/0.815, do not back out to 95.8/73.6 or
95.4/86.5). **Conclusion: the Limitations prose describes the SigLIP2 backbone
only**, sourced from `e8_modern_diffusion.csv`'s original (now partially stale)
run, not from a hardcoded constant.

### Table 5 / `tab:e8_results` (92.5%/92.9% SDXL; 94.1%/89.3% FLUX)
These match `routing/results/e8_cascade_accuracy.csv` (mtime 2026-09-14 02:03:45)
and `baseline_comparison_summary.csv`'s `e8_results` rows almost exactly:

```
SDXL,SigLIP2,GPT-5.5,...,cascade_acc=0.9249095...  → 92.5%
SDXL,DINOv2, GPT-5.5,...,cascade_acc=0.9286272...  → 92.9%
FLUX,SigLIP2,GPT-5.5,...,cascade_acc=0.9409537...  → 94.1%
FLUX,DINOv2, GPT-5.5,...,cascade_acc=0.8934399...  → 89.3%
```

**No `.py` file in the entire repo — including `_archive/` — references the
filenames `e8_cascade_accuracy.csv` or `e8_qwen_cascade.csv`.** The script that
produced Table 5's numbers is not present in the repository; it was evidently run
once, outside the committed pipeline, and its output CSV was kept but the code
was not.

## 2. Do they reconcile? No — and the reason is a probe-version mismatch, confirmed from the data itself

I reconstructed the stale-SigLIP2/DINOv2-probe predictions on the full SDXL/FLUX
eval sets directly from the artifacts currently on disk:

- `sdxl_siglip2_features.csv` / `sdxl_dinov2_features.csv` / `sdxl_fft_features.csv`
  (+ FLUX equivalents), joined with
- `sd14_siglip2_fftvariant_probe.pkl` / `sd14_dinov2_fftvariant_probe.pkl` /
  `sd14_fft_probe.pkl` (the SD1.4-trained "stale" probes, confirmed as the correct
  ones via `E8_DATASETS` in `experiments/e8_modern_diffusion.py`), then
- `{sdxl,flux}_gpt55_{siglip2,dinov2}_disagree_preds.csv` for the verifier stage.

Script: `routing/pipeline/confusion_matrices/reconstruct.py`. Results (full detail
in `reconstruct_log.txt` and `sdxl_flux_confusion_matrices.csv`):

| Dataset | Backbone | Stage   | Real recall | Fake recall | Overall acc |
|---|---|---|---|---|---|
| SDXL | SigLIP2 | stale   | 95.10% | 77.65% | 86.38% |
| SDXL | SigLIP2 | cascade | 97.40% | 86.05% | 91.73% |
| SDXL | DINOv2  | stale   | 91.35% | 78.85% | 85.10% |
| SDXL | DINOv2  | cascade | 96.45% | 87.30% | 91.88% |
| FLUX | SigLIP2 | stale   | 94.55% | 86.70% | 90.63% |
| FLUX | SigLIP2 | cascade | 97.35% | 89.20% | 93.28% |
| FLUX | DINOv2  | stale   | 91.15% | 72.40% | 81.78% |
| FLUX | DINOv2  | cascade | 96.55% | 80.20% | 88.38% |

**This reconstruction's overall "stale" accuracy matches `e8_cascade_accuracy.csv`'s
`stale_sem_acc` column EXACTLY for all four dataset×backbone combinations**
(86.375%, 85.1%, 90.625%, 81.775% — bit-for-bit), which proves `e8_cascade_accuracy.csv`
(the source of Table 5) was computed from the **same, currently-on-disk** stale-probe
state.

**It does NOT match `e8_modern_diffusion.csv`'s `sem_stale_acc`** (84.65%, 85.475%,
90.925%, 81.5% — the numbers underlying the Limitations prose), with the SDXL+SigLIP2
case differing by 1.7 percentage points (86.375% now vs. 84.65% then). This is not
rounding noise: `sd14_siglip2_fftvariant_probe.pkl` and `sd14_fft_probe.pkl` both have
mtimes of **2026-09-12 19:43–19:44**, which is roughly **two hours *after***
`e8_modern_diffusion.csv` was written (17:57:52) and `sdxl/flux_gpt55_*_disagree_preds.csv`
were queried against the pre-retrain disagreement set (their row counts — 1465,
1544, 1656, 1657 — match `e8_modern_diffusion.csv`'s `n_disagree` column exactly,
confirming they were generated under the *old* probe). The probes were evidently
retrained (consistent with `train_siglip2_classifiers.py` / `train_dinov2_classifiers.py`
being re-run, per the repo's "FFT+SigLIP2 fairness fix" / de-leaking history noted
in `routing/README.md`) sometime between the original E8 run and the Sep-14 run that
produced `e8_cascade_accuracy.csv` and, presumably, Table 5.

**This is a genuine data-provenance bug, not a labeling ambiguity:** the Limitations
prose's class-wise numbers (95.8%/73.6% etc.) and Table 5's overall cascade accuracy
(92.5%/92.9% etc.) describe the *same nominal quantity* — "stale SD1.4-trained SigLIP2/
DINOv2 probe applied zero-shot to SDXL/FLUX, with GPT-5.5 escalation on disagreement" —
but were computed against **two different trained-probe snapshots**, taken ~2 hours
apart, with no record of the earlier snapshot surviving on disk (only its aggregate
accuracy, baked into `e8_modern_diffusion.csv`, survives — the per-sample predictions
needed to recover exact 95.8/73.6-style class recall for that run no longer exist;
the probe `.pkl` that produced them was overwritten in place).

A second, independent symptom of the same root cause: using the **current, exactly-
matching-Table-5 probe state**, GPT-5.5's accuracy on the covered disagreement subset
is ~99.6% (`baseline_comparison_summary.csv`, `e8_results,SDXL/SigLIP2,GPT-5.5,n=1354,
verifier_acc=0.9963`), and escalating with a verifier this accurate can only closely
preserve or *increase* real-class recall on the disagreement cases, not depress it by
4 points — and indeed my reconstruction shows **real recall rising** after cascade in
all four dataset×backbone combinations (e.g. SDXL+SigLIP2: 95.10%→97.40%), the
opposite of the Limitations prose's claimed 4.0–4.3-point *drop*. That drop is
real only for whatever (now-irreproducible) GPT-5.5/probe state produced
`e8_modern_diffusion.csv`'s original numbers; it is not reproducible with the probes
and GPT-5.5 predictions currently on disk.

I was also unable to reproduce `e8_cascade_accuracy.csv`'s exact cascade-accuracy
formula bit-for-bit. Best-effort reimplementations following
`e8_modern_diffusion.py`'s own cascade logic (agreement path = stale-probe prediction,
disagreement path = GPT-5.5 prediction where available) land in the 90.0–92.2% range
for SDXL+SigLIP2 depending on how the ~100 "edge" samples are handled — rows where
the currently-recomputed disagreement mask and the historical GPT-5.5 query set
(`sdxl_gpt55_siglip2_disagree_preds.csv`, 1465 rows, built under the *old* probe) no
longer agree on which rows are disagreements. 1354 of the 1465 queried rows are still
disagreements under the current probe; the other 111 are not, and 98 *currently*-
disagreeing rows were never sent to GPT-5.5 at all. `e8_cascade_accuracy.csv`'s own
`n_dis_covered=1354` for SDXL/SigLIP2 matches this intersection exactly, confirming
the diagnosis, but the exact score for the uncovered rows (92.49% vs. my 91.7–92.3%
depending on fallback rule) could not be pinned down because **the script that wrote
`e8_cascade_accuracy.csv` / `e8_qwen_cascade.csv` does not exist anywhere in the
repository** (verified via repo-wide grep, including `_archive/`).

## 3. What this means for the paper

1. **The 95.8%/73.6%/84.7% (SDXL) and 95.4%/86.5%/91.2% (FLUX) Limitations-prose
   numbers and the 92.5%/92.9% (SDXL) and 94.1%/89.3% (FLUX) Table 5 numbers are
   computed from two different stale-probe snapshots**, not the same experiment.
   This should be disclosed explicitly if the paper keeps both; ideally, both
   should be recomputed from a single, current, frozen probe snapshot — which is
   exactly what `routing/pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv`
   (this report's deliverable) does.
2. Using the **current, Table-5-consistent** probe snapshot, the real-recall-drops-
   after-escalation claim in the Limitations section does not hold: real recall
   *increases* post-cascade for all four dataset×backbone pairs I reconstructed
   (see table above). Only the fake-recall-rises claim survives using current data
   (fake recall does rise substantially post-cascade in all four cases).
3. The script underlying Table 5's exact cascade-accuracy numbers
   (`e8_cascade_accuracy.csv` / `e8_qwen_cascade.csv`) is missing from the
   repository. If reviewers or an editor ask for exact reproducibility of Table 5,
   that is currently not possible from the committed codebase; it would need to be
   regenerated with a script that matches `e8_modern_diffusion.py`'s cascade logic,
   run against the *current* probes (which do reproduce its `stale_sem_acc` column
   exactly) — only the disagreement-coverage edge-case handling needs to be
   pinned down and re-committed.

## 4. Files produced by this task

- `routing/pipeline/confusion_matrices/reconstruct.py` — reconstruction script
  (reads only existing CSVs/pkls under `routing/`, writes only within this new
  directory).
- `routing/pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv` — the
  8-row (4 dataset×backbone pairs × {stale, cascade}) confusion-matrix table.
- `routing/pipeline/confusion_matrices/reconstruct_log.txt` — full console log
  with disagreement counts, GPT-5.5 coverage counts, and both recall formulas
  shown explicitly per row.
