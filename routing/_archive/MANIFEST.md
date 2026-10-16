# Archive manifest

This directory holds Python scripts that were removed from `routing/`'s root
during the 2026-10-15 reproducibility cleanup for the IEEE Access revision.
Nothing was deleted — every file below was `git mv`'d here, so `git log
--follow` on any path recovers its full history. None of these scripts is the
source of any number currently cited in the paper; see `routing/README.md`
for the authoritative pipeline.

## `superseded_evaluate_routing/`

Six near-duplicate one-off routing-evaluation scripts, superseded by
`analysis2.py` and the E0–E8 experiment suite (`routing/experiments/`).

| Original path | Archived to | Reason | Superseded by |
|---|---|---|---|
| `evaluate_routing.py` | `_archive/superseded_evaluate_routing/evaluate_routing.py` | Original GenBuster routing-evaluation script; one of six near-duplicate evaluate_routing variants with no single current table tied to it. | `analysis2.py` + `routing/experiments/` (E0–E8 suite) |
| `gen_evaluate_routing2.py` | `_archive/superseded_evaluate_routing/gen_evaluate_routing2.py` | GenBuster "v2" variant of the same evaluation logic, superseded by the same consolidated path. | `analysis2.py` + `routing/experiments/` (E0–E8 suite) |
| `biggan_evaluate_routing2.py` | `_archive/superseded_evaluate_routing/biggan_evaluate_routing2.py` | BigGAN-dataset variant of `evaluate_routing2`; only consumer of `biggan_bandit_decisions.csv`, which nothing current reads. | `analysis2.py` + `routing/experiments/` (E0–E8 suite) |
| `evaluate_routing_sd14.py` | `_archive/superseded_evaluate_routing/evaluate_routing_sd14.py` | SD1.4-dataset variant; writes `bandit_decisions.csv`, which is only read back by another archived script (`dcr.py`). | `analysis2.py` + `routing/experiments/` (E0–E8 suite) |
| `evaluate_routing_biggan.py` | `_archive/superseded_evaluate_routing/evaluate_routing_biggan.py` | Second BigGAN evaluation variant; writes `biggan_bandit_decisions.csv`, read by no current script. | `analysis2.py` + `routing/experiments/` (E0–E8 suite) |
| `sd14_evaluate_routing2-Copy1.py` | `_archive/superseded_evaluate_routing/sd14_evaluate_routing2-Copy1.py` | Literal Jupyter "-Copy1" artifact of the SD1.4 evaluation script — a save-as duplicate, not an intentional variant. | `analysis2.py` + `routing/experiments/` (E0–E8 suite) |

## `superseded_bandit_training/`

Original CNNSpot+CLIP-era bandit trainers. `retrain_bandits_fft_siglip2.py`'s
own docstring explicitly calls these out: "the original bandit
(`retrain_bandits.py`) was trained on CNNSpot + CLIP features while the
disagreement rule it's compared against uses FFT + SigLIP2 + Qwen. This
script retrains the bandit with the same FFT + SigLIP2 backbone so both
routing policies are compared on equal footing."

| Original path | Archived to | Reason | Superseded by |
|---|---|---|---|
| `merge_qwen_and_train_bandit.py` | `_archive/superseded_bandit_training/merge_qwen_and_train_bandit.py` | Early combined merge+train script from the CNNSpot+CLIP era, before the pipeline split into separate merge/train/retrain stages. | `retrain_bandits_fft_siglip2.py` (and `retrain_bandits.py`, see note below) |
| `train_bandit.py` | `_archive/superseded_bandit_training/train_bandit.py` | GenBuster CNNSpot+CLIP-era bandit trainer, not reloaded by `analysis2.py`. | `retrain_bandits_fft_siglip2.py` |
| `train_bandit_biggan.py` | `_archive/superseded_bandit_training/train_bandit_biggan.py` | BigGAN CNNSpot+CLIP-era bandit trainer. | `retrain_bandits_fft_siglip2.py` |
| `sd14_train_bandit.py` | `_archive/superseded_bandit_training/sd14_train_bandit.py` | SD1.4 CNNSpot+CLIP-era bandit trainer. | `retrain_bandits_fft_siglip2.py` |

**Note on `retrain_bandits.py` (left in place, NOT archived):** the original
audit did not classify this file, and this cleanup pass did not archive it
either — `analysis2.py` (lines ~1118, 1125, 1132) directly `joblib.load()`s
`bandit_model.pkl` and `biggan_bandit_model.pkl`, and `retrain_bandits.py` is
the only script in the repo that produces those exact pickles. It is
load-bearing for the current authoritative pipeline even though it predates
the FFT+SigLIP2 bandit variant. See `routing/README.md` for detail.

## `superseded_qwen_text_parsing/`

Naive greedy-decoded text parsing of Qwen output (the "REAL"-bias failure
mode the paper's Section VI.C calls out), superseded by the logit-extraction
`*_run_qwen_v2.py` family. Confirmed: both the old and new scripts write to
the *same* output filename per dataset (e.g. both `run_qwen_on_genbuster.py`
and `run_qwen_v2.py` write `qwen_preds.csv`), and `retrain_bandits.py` /
`retrain_bandits_fft_siglip2.py` both hard-require a `p_fake_qwen` column
that only the v2 (logit-extraction) output provides.

| Original path | Archived to | Reason | Superseded by |
|---|---|---|---|
| `run_qwen_on_genbuster.py` | `_archive/superseded_qwen_text_parsing/run_qwen_on_genbuster.py` | Greedy-decoded text-parsing Qwen runner for GenBuster; writes the same `qwen_preds.csv` that `run_qwen_v2.py` now produces via logit extraction. | `run_qwen_v2.py` |
| `run_qwen_on_sd14.py` | `_archive/superseded_qwen_text_parsing/run_qwen_on_sd14.py` | Greedy-decoded text-parsing Qwen runner for SD1.4. | `sd14_run_qwen_v2.py` |
| `run_qwen_biggan.py` | `_archive/superseded_qwen_text_parsing/run_qwen_biggan.py` | Greedy-decoded text-parsing Qwen runner for BigGAN. | `biggan_run_qwen_v2.py` |

## `one_off_analysis/`

Exploratory/ad-hoc scratch scripts built around the pre-pivot CNNSpot+CLIP
pipeline. None writes to `results/` or is imported by any currently
authoritative script (verified by grepping for `import <module>` across all
root `.py` files).

| Original path | Archived to | Reason | Superseded by |
|---|---|---|---|
| `analysis.py` | `_archive/one_off_analysis/analysis.py` | Original analysis script, superseded wholesale by `analysis2.py`. | `analysis2.py` |
| `final_results.py` | `_archive/one_off_analysis/final_results.py` | Ad-hoc results dump, pre-pivot pipeline. | `analysis2.py` / `regenerate_main_tables.py` |
| `plot_results.py` | `_archive/one_off_analysis/plot_results.py` | Exploratory plotting script, no `results/` output. | `analysis2.py` |
| `plot_costacc_curve.py` | `_archive/one_off_analysis/plot_costacc_curve.py` | Exploratory cost/accuracy curve plot, scratch only. | `analysis2.py` |
| `deployment_evaluation.py` | `_archive/one_off_analysis/deployment_evaluation.py` | Pre-pivot deployment evaluation, superseded by the cross-family deployment path. | `cross_family_deployment.py` |
| `dcr.py` | `_archive/one_off_analysis/dcr.py` | Reads `bandit_decisions.csv`, itself only produced by archived `evaluate_routing_sd14.py`; dead end. | `analysis2.py` |
| `bsscript.py` | `_archive/one_off_analysis/bsscript.py` | Unlabeled scratch script, CNNSpot+CLIP era, no `results/` output. | `analysis2.py` |
| `balanced_acc.py` | `_archive/one_off_analysis/balanced_acc.py` | One-off balanced-accuracy check, scratch only. | `analysis2.py` |
| `entrophy_matched_rate.py` | `_archive/one_off_analysis/entrophy_matched_rate.py` | One-off entropy-matched-rate check, scratch only. | `analysis2.py` |
| `sweep_lambda.py` | `_archive/one_off_analysis/sweep_lambda.py` | Exploratory lambda sweep, pre-pivot pipeline. | `analysis2.py` |
| `escalation_table.py` | `_archive/one_off_analysis/escalation_table.py` | Ad-hoc escalation-table generator, no current table tied to it. | `regenerate_main_tables.py` |
| `per_class_breakdown.py` | `_archive/one_off_analysis/per_class_breakdown.py` | Exploratory per-class breakdown, scratch only. | `analysis2.py` |
| `clip_per_generator.py` | `_archive/one_off_analysis/clip_per_generator.py` | Exploratory per-generator CLIP breakdown, scratch only. | `analysis2.py` |
| `cross_distribution_clip.py` | `_archive/one_off_analysis/cross_distribution_clip.py` | Writes a standalone CLIP transfer matrix CSV not read by any current table/figure script. | `cross_family_deployment.py` |
| `test_alt_prompt.py` | `_archive/one_off_analysis/test_alt_prompt.py` | One-off alternate-prompt test, writes `prompt_test_altprompt_results.csv`, not read by any current script. | `run_qwen_v2.py` family |

## Ambiguous / left in place (not archived despite not being on the explicit keep list)

| Path | Why it was left in place |
|---|---|
| `routing/retrain_bandits.py` | Not covered by the original classification at all. Inspection showed it is the sole producer of `bandit_model.pkl` and `biggan_bandit_model.pkl`, both of which `analysis2.py` directly loads (`joblib.load`) to generate current tables. Archiving it would have silently broken the authoritative pipeline, so per the "if something doesn't match, don't move it" instruction, it was left in `routing/` root. It should probably be added to the explicit keep list in any future audit.
