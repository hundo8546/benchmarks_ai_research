# File inventory (Revision 3, Phase 7)

> **Applied 2026-10-22.** Deleted 12 files (`git rm`) and archived 78 (`git mv`): 69 of the 70 in the archive section (`claude_context_notes/resubmissionplan.md` was left in place because docstrings in `routing/gate_analysis.py` and `routing/baseline_utils.py` still reference it) plus 9 figure D/E/F files that became unused. Destinations: `routing/_archive/{superseded_results,legacy_compression_csvs,root_one_off_scripts,vendored_unused,removed_figures}/` and `claude_context_notes/_round1/`; see `routing/_archive/MANIFEST.md`. Review and ask sections untouched. `generate_paper_results.py` output was byte-identical. Entries below marked **[deleted]** or **[archived]**.

Every tracked file outside `routing/_archive/`, with a proposed status. (Originally proposed; see the Applied note.)

How the status was decided:
- **Runtime trace:** a Python audit hook logged every file opened while a full `generate_paper_results.py` run executed. Anything read is **keep**.
- **Static trace:** every script `run_pipeline.py` dispatches to, plus the local modules those scripts and `generate_paper_results.py` import, is **keep**.
- **Released provenance** promised in the response letter is **keep**: per-sample `*_preds.csv`, `results/batch_requests/**`, split manifests, the overlap report and the confusion matrices.
- **delete** is limited to the plan's obvious-junk list, and only when the trace shows it is unread.
- **archive** means a move to `routing/_archive/<category>/` with a line in `_archive/MANIFEST.md` (the repo's convention).
- **review** means not reached by either entry point and not in a keep category. These files are probably superseded, but each one needs a decision.

| Status | Files |
|---|---|
| delete | 12 |
| archive | 70 |
| ask | 2 |
| review | 165 |
| keep | 254 |
| **total** | 503 |

## delete (12)

**obvious junk (plan list); not read by any entry point** (12)

- `routing/=0.46.1` **[deleted]**
- `routing/bandit_model.pkl.old_qwen_backup` **[deleted]**
- `routing/bandit_val_paths_old_small.npy` **[deleted]**
- `routing/biggan_bandit_model.pkl.old_qwen_backup` **[deleted]**
- `routing/biggan_bandit_val_paths_old_small.npy` **[deleted]**
- `routing/cross_family_deployment.csv.cnnspot_backup` **[deleted]**
- `routing/final_results_raw_file.txt` **[deleted]**
- `routing/prompt_test_altprompt_results.csv` **[deleted]**
- `routing/prompt_test_subsample.csv` **[deleted]**
- `routing/results/e6_limited_data.csv.cnnspot_backup` **[deleted]**
- `routing/sd14_bandit_model.pkl.old_qwen_backup` **[deleted]**
- `routing/sd14_bandit_val_paths_old_small.npy` **[deleted]**

## archive (70)

**differs from routing/hfgenbuster.py; checked: no hardcoded secret (HF_TOKEN from env)** (1)

- `hfgenbuster.py` **[archived]**

**old CNNSpot/CLIP compression CSVs, superseded by results/e7_compression_extended.csv** (18)

- `routing/biggan_clip_q50.csv` **[archived]**
- `routing/biggan_clip_q75.csv` **[archived]**
- `routing/biggan_clip_q95.csv` **[archived]**
- `routing/biggan_cnnspot_q50.csv` **[archived]**
- `routing/biggan_cnnspot_q75.csv` **[archived]**
- `routing/biggan_cnnspot_q95.csv` **[archived]**
- `routing/clip_q50.csv` **[archived]**
- `routing/clip_q75.csv` **[archived]**
- `routing/clip_q95.csv` **[archived]**
- `routing/cnnspot_q50.csv` **[archived]**
- `routing/cnnspot_q75.csv` **[archived]**
- `routing/cnnspot_q95.csv` **[archived]**
- `routing/sd14_clip_q50.csv` **[archived]**
- `routing/sd14_clip_q75.csv` **[archived]**
- `routing/sd14_clip_q95.csv` **[archived]**
- `routing/sd14_cnnspot_q50.csv` **[archived]**
- `routing/sd14_cnnspot_q75.csv` **[archived]**
- `routing/sd14_cnnspot_q95.csv` **[archived]**

**repo-root one-off evaluation script** (6)

- `eval_cnnspot_genbuster.py` **[archived]**
- `eval_genbuster.py` **[archived]**
- `eval_genbuster_full_baseline.py` **[archived]**
- `eval_genbuster_per_gen.py` **[archived]**
- `eval_genbuster_per_gen_correct.py` **[archived]**
- `eval_genbuster_table2.py` **[archived]**

**round-1 material -> claude_context_notes/_round1/** (4)

- `claude_context_notes/paper_updates.tex` **[archived]**
- `claude_context_notes/response_to_reviewers.md` **[archived]**
- `claude_context_notes/resubmissionplan.md`
- `claude_context_notes/session_summary.md` **[archived]**

**superseded result version (now produced by a live generator or unused)** (9)

- `routing/results/e8_cascade_accuracy.csv` **[archived]**
- `routing/results/e8_qwen_cascade.csv` **[archived]**
- `routing/results/tab_cost_accuracy_three_way.csv` **[archived]**
- `routing/results/tab_e8_gpt55_FULL_coverage.csv` **[archived]**
- `routing/results/tab_e8_gpt55_reconciled_v2.csv` **[archived]**
- `routing/results/tab_e8_qwen_FINAL.csv` **[archived]**
- `routing/results/tab_e8_qwen_reconciled.csv` **[archived]**
- `routing/results/tab_real_image_source_benchmark.csv` **[archived]**
- `routing/results/verifier_validation_gate_check.csv` **[archived]**

**vendored repo; no non-archived code references it; not in the paper** (32)

- `DeMamba/LICENSE` **[archived]**
- `DeMamba/Preprocess/compress.py` **[archived]**
- `DeMamba/Preprocess/folder2csv.py` **[archived]**
- `DeMamba/Preprocess/video2frame.py` **[archived]**
- `DeMamba/README.md` **[archived]**
- `DeMamba/configs/XCLIP_DeMamba.yaml` **[archived]**
- `DeMamba/dataloader.py` **[archived]**
- `DeMamba/figs/.gitkeep` **[archived]**
- `DeMamba/figs/VFOD.png` **[archived]**
- `DeMamba/figs/logo.png` **[archived]**
- `DeMamba/figs/tab_fig.jpg` **[archived]**
- `DeMamba/figs/xty.jpg` **[archived]**
- `DeMamba/models/DeMamba.py` **[archived]**
- `DeMamba/models/F3Net.py` **[archived]**
- `DeMamba/models/FTCN.py` **[archived]**
- `DeMamba/models/MINTIME` **[archived]**
- `DeMamba/models/NPR.py` **[archived]**
- `DeMamba/models/STIL.py` **[archived]**
- `DeMamba/models/TALL.py` **[archived]**
- `DeMamba/models/VideoMAE.py` **[archived]**
- `DeMamba/models/XCLIP.py` **[archived]**
- `DeMamba/models/__init__.py` **[archived]**
- `DeMamba/models/clip/__init__.py` **[archived]**
- `DeMamba/models/clip/bpe_simple_vocab_16e6.txt.gz` **[archived]**
- `DeMamba/models/clip/clip.py` **[archived]**
- `DeMamba/models/clip/model.py` **[archived]**
- `DeMamba/models/clip/simple_tokenizer.py` **[archived]**
- `DeMamba/models/mamba_base.py` **[archived]**
- `DeMamba/models/pscan.py` **[archived]**
- `DeMamba/models/time_transformer` **[archived]**
- `DeMamba/train.py` **[archived]**
- `DeMamba/util.py` **[archived]**

## ask (2)

**unrelated to the paper -- user decides** (2)

- `claude_context_notes/claudecodedwnld.sh`
- `claude_context_notes/studentcoderskill.md`

## review (165)

**data/other file not read by generate_paper_results and not in a keep category** (77)

- `benchmark_index_with_gen.csv`
- `routing/.gitignore`
- `routing/bandit_decisions.csv`
- `routing/bandit_model.pkl`
- `routing/bandit_transfer_fft_siglip2.csv`
- `routing/biggan_clip_val_idx.npy`
- `routing/biggan_dinov2_features.csv`
- `routing/biggan_distant_shift_disagree_pool.csv`
- `routing/biggan_index.csv`
- `routing/biggan_merged_dataset.csv`
- `routing/clip_classifier.pkl`
- `routing/clip_features.csv`
- `routing/clip_transfer_matrix.csv`
- `routing/clip_val_idx.npy`
- `routing/cross_family_deployment.csv`
- `routing/dinov2_features.csv`
- `routing/experiments/e8_full.sh`
- `routing/flux_index.csv`
- `routing/genbuster_qwen_todo.csv`
- `routing/merged_dataset.csv`
- `routing/plot_agree_disagree_all.png`
- `routing/plot_cross_dataset.png`
- `routing/plot_escalation_all.png`
- `routing/plot_fig_a_semantic_degradation.pdf`
- `routing/plot_fig_b_cascade_gain_scatter.pdf`
- `routing/plot_fig_c_disagreement_rate.pdf`
- `routing/plot_fig_c_disagreement_rate.png`
- `routing/plot_fig_d_cost_accuracy_tradeoff.pdf`
- `routing/plot_fig_e_resolution_shortcut.pdf`
- `routing/plot_fig_f_case_mapping.pdf`
- `routing/plot_per_class.png`
- `routing/plot_transfer.png`
- `routing/results/BigGAN_native_resolution.csv`
- `routing/results/GenBuster_native_resolution.csv`
- `routing/results/SD14_native_resolution.csv`
- `routing/results/baseline_comparison_summary.csv`
- `routing/results/disagreement_coco_real.csv`
- `routing/results/disagreement_original_real.csv`
- `routing/results/disagreement_union_counts.csv`
- `routing/results/e0_disagreement_stats.csv`
- `routing/results/e0_fft_disagreement_stats.csv`
- `routing/results/e0_fft_table.csv`
- `routing/results/e0_table2.csv`
- `routing/results/e0_table3.csv`
- `routing/results/e0_table4.csv`
- `routing/results/e1_figure_data.csv`
- `routing/results/e1_semantic_results.csv`
- `routing/results/e2_artifact_results.csv`
- `routing/results/e3_disagreement_matrix.csv`
- `routing/results/e4_verifier_results.csv`
- `routing/results/e5_oracle_results.csv`
- `routing/results/e6_limited_data.csv`
- `routing/results/e7_compression.csv`
- `routing/results/e8_modern_diffusion.csv`
- `routing/results/flux_index_coco_real.csv`
- `routing/results/flux_index_original_real.csv`
- `routing/results/flux_realonly_coco.csv`
- `routing/results/flux_realonly_original.csv`
- `routing/results/gate_holdout_summary.md`
- `routing/results/gate_sensitivity_sweep.csv`
- `routing/results/pipeline_audit_report.csv`
- `routing/results/resolution_matched_ablation.csv`
- `routing/results/resolution_only_control.csv`
- `routing/results/sdxl_index_coco_real.csv`
- `routing/results/sdxl_index_original_real.csv`
- `routing/results/sdxl_realonly_coco.csv`
- `routing/results/sdxl_realonly_original.csv`
- `routing/results/table11_decision.md`
- `routing/results/table6_fft_siglip2_fix.csv`
- `routing/run_revision.sh`
- `routing/sd14_clip_val_idx.npy`
- `routing/sd14_index.csv`
- `routing/sd14_merged_dataset.csv`
- `routing/sdxl_flux_qwen_cache.csv`
- `routing/sdxl_index.csv`
- `routing/transfer_learning_results.csv`
- `stage_all.sh`

**possible duplicate of CNNDetection/networks -- confirm before archiving** (6)

- `networks/__init__.py`
- `networks/base_model.py`
- `networks/lpf.py`
- `networks/resnet.py`
- `networks/resnet_lpf.py`
- `networks/trainer.py`

**possibly superseded by benchmark_index_with_gen.csv -- confirm** (1)

- `benchmark_index.csv`

**script not reached from either entry point (likely superseded)** (45)

- `download_aigibench.py`
- `download_genimage.py`
- `download_weights.py`
- `routing/__init__.py`
- `routing/analysis2.py`
- `routing/analyze_real_source_disagreement.py`
- `routing/biggan_run_qwen_v2.py`
- `routing/build_batch_request_sets.py`
- `routing/build_independent_real_index.py`
- `routing/build_original_real_index.py`
- `routing/expand_val_splits.py`
- `routing/experiments/__init__.py`
- `routing/experiments/e0_baseline.py`
- `routing/experiments/e1_modern_semantic.py`
- `routing/experiments/e2_modern_artifact.py`
- `routing/experiments/e3_disagreement_matrix.py`
- `routing/experiments/e4_gpt55.py`
- `routing/experiments/e4_strong_verifier.py`
- `routing/experiments/e5_oracle_ceiling.py`
- `routing/experiments/e7_compression.py`
- `routing/extract_original_real_features.py`
- `routing/fast_extract_genimage.py`
- `routing/fill_table6_gpt55_gap.py`
- `routing/fix_gpt55_blanks.py`
- `routing/fix_table6_verifier_comparison.py`
- `routing/generate_bandit_dataset.py`
- `routing/generate_clip_preds.py`
- `routing/generate_clip_preds_biggan.py`
- `routing/generate_clip_preds_sd14.py`
- `routing/hfgenbuster.py`
- `routing/merge_all_features.py`
- `routing/merge_all_features_biggan.py`
- `routing/merge_all_features_sd14.py`
- `routing/regenerate_main_tables.py`
- `routing/retrain_bandits.py`
- `routing/run_biggan_shift_gpt55.py`
- `routing/run_gpt55_expanded.py`
- `routing/run_qwen72b_biggan_shift.py`
- `routing/run_qwen7b_expansion.py`
- `routing/run_qwen_sdxl_flux.py`
- `routing/run_qwen_v2.py`
- `routing/sd14_run_qwen_v2.py`
- `routing/train_clip_classifier.py`
- `routing/train_clip_classifier_biggan.py`
- `routing/train_clip_classifier_sd14.py`

**used by extractors/extract_dire.py and run_revision.sh, but no reported number uses DIRE** (36)

- `DIRE/.gitignore`
- `DIRE/README.md`
- `DIRE/demo.py`
- `DIRE/dire_env/pyvenv.cfg`
- `DIRE/guided-diffusion/LICENSE`
- `DIRE/guided-diffusion/README.md`
- `DIRE/guided-diffusion/compute_dire.py`
- `DIRE/guided-diffusion/compute_dire.sh`
- `DIRE/guided-diffusion/guided_diffusion/__init__.py`
- `DIRE/guided-diffusion/guided_diffusion/dist_util.py`
- `DIRE/guided-diffusion/guided_diffusion/fp16_util.py`
- `DIRE/guided-diffusion/guided_diffusion/gaussian_diffusion.py`
- `DIRE/guided-diffusion/guided_diffusion/image_datasets.py`
- `DIRE/guided-diffusion/guided_diffusion/logger.py`
- `DIRE/guided-diffusion/guided_diffusion/losses.py`
- `DIRE/guided-diffusion/guided_diffusion/nn.py`
- `DIRE/guided-diffusion/guided_diffusion/resample.py`
- `DIRE/guided-diffusion/guided_diffusion/respace.py`
- `DIRE/guided-diffusion/guided_diffusion/script_util.py`
- `DIRE/guided-diffusion/guided_diffusion/train_util.py`
- `DIRE/guided-diffusion/guided_diffusion/unet.py`
- `DIRE/guided-diffusion/model-card.md`
- `DIRE/guided-diffusion/setup.py`
- `DIRE/networks/resnet.py`
- `DIRE/requirements.txt`
- `DIRE/test.py`
- `DIRE/test.sh`
- `DIRE/train.py`
- `DIRE/train.sh`
- `DIRE/utils/config.py`
- `DIRE/utils/datasets.py`
- `DIRE/utils/earlystop.py`
- `DIRE/utils/eval.py`
- `DIRE/utils/trainer.py`
- `DIRE/utils/utils.py`
- `DIRE/utils/warmup.py`

## keep (254)

**code reachable from run_pipeline.py / generate_paper_results.py** (37)

- `routing/bandit_transfer_fft_siglip2.py`
- `routing/baseline_comparison.py`
- `routing/baseline_utils.py`
- `routing/build_aigibench_index.py`
- `routing/build_biggan_index.py`
- `routing/build_genbuster_index.py`
- `routing/build_sd14_index.py`
- `routing/cross_family_deployment.py`
- `routing/experiments/e0_fft_variant.py`
- `routing/experiments/e6_limited_data.py`
- `routing/extractors/extract_dinov2.py`
- `routing/extractors/extract_fft.py`
- `routing/extractors/extract_siglip2.py`
- `routing/gate_analysis.py`
- `routing/generate_clip_features.py`
- `routing/generate_clip_features_biggan.py`
- `routing/generate_clip_features_sd14.py`
- `routing/generate_cnnspot_features.py`
- `routing/generate_cnnspot_features_biggan.py`
- `routing/generate_cnnspot_features_sd14.py`
- `routing/pipeline/dedup.py`
- `routing/pipeline/retrain_clip.py`
- `routing/pipeline/tables/__init__.py`
- `routing/pipeline/tables/baselines.py`
- `routing/pipeline/tables/cheap_trigger.py`
- `routing/pipeline/tables/confidence_routing.py`
- `routing/pipeline/tables/e8_cascade.py`
- `routing/pipeline/tables/gate.py`
- `routing/pipeline/tables/gate_holdout.py`
- `routing/pipeline/tables/per_generator.py`
- `routing/pipeline/tables/transfer.py`
- `routing/pipeline/verifiers.py`
- `routing/retrain_bandits_fft_siglip2.py`
- `routing/run_pipeline.py`
- `routing/run_univfd.py`
- `routing/train_dinov2_classifiers.py`
- `routing/train_siglip2_classifiers.py`

**current manuscript / letter / plan / audit material** (7)

- `claude_context_notes/coco_citation.txt`
- `claude_context_notes/currentpaper.tex`
- `claude_context_notes/number_audit.md`
- `claude_context_notes/paper_updates2.tex`
- `claude_context_notes/references_additions.bib`
- `claude_context_notes/response_to_reviewers2.md`
- `claude_context_notes/revision3_plan.md`

**figure script** (7)

- `routing/experiments/plot_fig_a_semantic_degradation.py`
- `routing/experiments/plot_fig_b_cascade_gain_scatter.py`
- `routing/experiments/plot_fig_c_disagreement_rate.py`
- `routing/experiments/plot_fig_d_cost_accuracy_tradeoff.py`
- `routing/experiments/plot_fig_e_resolution_shortcut.py`
- `routing/experiments/plot_fig_f_case_mapping.py`
- `routing/experiments/plot_main_results_figures.py`

**figure used by paper_updates2.tex** (7)

- `routing/plot_accuracy_comparison.png`
- `routing/plot_cost_accuracy_all.png`
- `routing/plot_fig_a_semantic_degradation.png`
- `routing/plot_fig_b_cascade_gain_scatter.png`
- `routing/plot_fig_d_cost_accuracy_tradeoff.png`
- `routing/plot_fig_e_resolution_shortcut.png`
- `routing/plot_fig_f_case_mapping.png`

**pipeline code / config** (30)

- `.gitignore`
- `README.md`
- `requirements.txt`
- `routing/.gitattributes`
- `routing/README.md`
- `routing/extractors/__init__.py`
- `routing/extractors/extract_dire.py`
- `routing/extractors/train_probe.py`
- `routing/pipeline/README.md`
- `routing/pipeline/SPLITS.md`
- `routing/pipeline/__init__.py`
- `routing/pipeline/batch_gpt55.py`
- `routing/pipeline/confusion_matrices/RECONCILIATION_REPORT.md`
- `routing/pipeline/confusion_matrices/reconstruct.py`
- `routing/pipeline/confusion_matrices/sdxl_flux_confusion_matrices.csv`
- `routing/pipeline/gate_rules/PSEUDOCODE.md`
- `routing/pipeline/gate_rules/RESPONSE_DRAFT.md`
- `routing/pipeline/gate_rules/explain_n10_divergence.py`
- `routing/pipeline/independent_real_source/RECOMMENDATION.md`
- `routing/pipeline/robustness/__init__.py`
- `routing/pipeline/robustness/e7_compression_extended.py`
- `routing/pipeline/robustness/e9_postprocessing_robustness.py`
- `routing/pipeline/robustness/native_resolution.py`
- `routing/pipeline/robustness/resolution_matched_ablation.py`
- `routing/pipeline/robustness/resolution_only_control.py`
- `routing/pipeline/tables/CONFIDENCE_ROUTING_NOTES.md`
- `routing/pipeline/tables/case_mapping.py`
- `routing/pipeline/tables/cost_accounting.py`
- `routing/pipeline/tables/robustness.py`
- `setup.sh`

**read by generate_paper_results.py (runtime trace)** (84)

- `routing/bandit_dataset.csv`
- `routing/bandit_val_paths.npy`
- `routing/biggan_bandit_dataset.csv`
- `routing/biggan_bandit_val_paths.npy`
- `routing/biggan_clip_preds.csv`
- `routing/biggan_dinov2_preds.csv`
- `routing/biggan_distant_shift_1500.csv`
- `routing/biggan_distant_shift_gpt55_preds.csv`
- `routing/biggan_distant_shift_qwen72b_preds.csv`
- `routing/biggan_distant_shift_qwen7b_preds.csv`
- `routing/biggan_fft_features.csv`
- `routing/biggan_gpt55_disagree_preds.csv`
- `routing/biggan_qwen_preds.csv`
- `routing/biggan_siglip2_features.csv`
- `routing/biggan_siglip2_preds.csv`
- `routing/biggan_univfd_preds.csv`
- `routing/clip_preds.csv`
- `routing/dinov2_preds.csv`
- `routing/experiments/e8_modern_diffusion.py`
- `routing/extractors/common.py`
- `routing/fft_features.csv`
- `routing/flux_dinov2_features.csv`
- `routing/flux_fft_features.csv`
- `routing/flux_gpt55_clip_disagree_preds.csv`
- `routing/flux_gpt55_dinov2_disagree_preds.csv`
- `routing/flux_gpt55_siglip2_disagree_preds.csv`
- `routing/flux_qwen_clip_disagree_preds.csv`
- `routing/flux_qwen_dinov2_disagree_preds.csv`
- `routing/flux_qwen_siglip2_disagree_preds.csv`
- `routing/flux_siglip2_features.csv`
- `routing/genbuster_qwen_expansion_preds.csv`
- `routing/generate_paper_results.py`
- `routing/gpt55_disagree_preds.csv`
- `routing/pipeline/splits.py`
- `routing/pipeline/tables/main_results.py`
- `routing/qwen_preds.csv`
- `routing/qwen_preds_v2.csv`
- `routing/results/batch_requests/CONSOLIDATED_coco_union.csv`
- `routing/results/batch_requests/CONSOLIDATED_full_sdxl_flux.csv`
- `routing/results/batch_requests/CONSOLIDATED_original_voc_union.csv`
- `routing/results/e7_compression_extended.csv`
- `routing/results/e9_postprocessing_robustness.csv`
- `routing/results/flux_coco_realonly_dinov2_features.csv`
- `routing/results/flux_coco_realonly_fft_features.csv`
- `routing/results/flux_coco_realonly_siglip2_features.csv`
- `routing/results/flux_realonly_dinov2_features.csv`
- `routing/results/flux_realonly_fft_features.csv`
- `routing/results/flux_realonly_siglip2_features.csv`
- `routing/results/sdxl_coco_realonly_dinov2_features.csv`
- `routing/results/sdxl_coco_realonly_fft_features.csv`
- `routing/results/sdxl_coco_realonly_siglip2_features.csv`
- `routing/results/sdxl_realonly_dinov2_features.csv`
- `routing/results/sdxl_realonly_fft_features.csv`
- `routing/results/sdxl_realonly_siglip2_features.csv`
- `routing/results/tab_biggan_shift.csv`
- `routing/results/tab_cheap_path.csv`
- `routing/results/tab_cross_family_deployment.csv`
- `routing/results/tab_e8_results.csv`
- `routing/results/tab_escalation.csv`
- `routing/results/tab_limited_data.csv`
- `routing/results/tab_verifier_comparison.csv`
- `routing/sd14_bandit_dataset.csv`
- `routing/sd14_bandit_val_paths.npy`
- `routing/sd14_clip_preds.csv`
- `routing/sd14_dinov2_features.csv`
- `routing/sd14_dinov2_preds.csv`
- `routing/sd14_fft_features.csv`
- `routing/sd14_gpt55_disagree_preds.csv`
- `routing/sd14_qwen_preds.csv`
- `routing/sd14_siglip2_features.csv`
- `routing/sd14_siglip2_preds.csv`
- `routing/sd14_univfd_preds.csv`
- `routing/sdxl_dinov2_features.csv`
- `routing/sdxl_fft_features.csv`
- `routing/sdxl_gpt55_clip_disagree_preds.csv`
- `routing/sdxl_gpt55_dinov2_disagree_preds.csv`
- `routing/sdxl_gpt55_siglip2_disagree_preds.csv`
- `routing/sdxl_qwen_clip_disagree_preds.csv`
- `routing/sdxl_qwen_dinov2_disagree_preds.csv`
- `routing/sdxl_qwen_siglip2_disagree_preds.csv`
- `routing/sdxl_siglip2_features.csv`
- `routing/siglip2_features.csv`
- `routing/siglip2_preds.csv`
- `routing/univfd_preds.csv`

**released provenance promised in the response letter** (30)

- `routing/results/batch_requests/CONSOLIDATED_bugfix_reruns.csv`
- `routing/results/batch_requests/FINAL_BATCH_STATUS.csv`
- `routing/results/batch_requests/MANIFEST.csv`
- `routing/results/batch_requests/SUBMITTED_BATCHES.csv`
- `routing/results/batch_requests/_smoketest.custom_id_map.csv`
- `routing/results/batch_requests/bugfix_reruns.custom_id_map.csv`
- `routing/results/batch_requests/coco_union.custom_id_map.csv`
- `routing/results/batch_requests/full_sdxl_flux.custom_id_map.csv`
- `routing/results/batch_requests/original_voc_union.custom_id_map.csv`
- `routing/results/batch_requests/outputs/bugfix_reruns_part1_parsed.csv`
- `routing/results/batch_requests/outputs/coco_union_part1_parsed.csv`
- `routing/results/batch_requests/outputs/coco_union_part2_parsed.csv`
- `routing/results/batch_requests/outputs/coco_union_part3_parsed.csv`
- `routing/results/batch_requests/outputs/coco_union_part4_parsed.csv`
- `routing/results/batch_requests/outputs/coco_union_part5_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part1_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part2_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part3_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part4fixed_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part5_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part6_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part7_parsed.csv`
- `routing/results/batch_requests/outputs/full_sdxl_flux_part8_parsed.csv`
- `routing/results/batch_requests/outputs/original_voc_union_part1_parsed.csv`
- `routing/results/batch_requests/outputs/original_voc_union_part2_parsed.csv`
- `routing/results/batch_requests/outputs/original_voc_union_part3_parsed.csv`
- `routing/results/batch_requests/outputs/original_voc_union_part4_parsed.csv`
- `routing/results/batch_requests/outputs/original_voc_union_part5_parsed.csv`
- `routing/results/image_overlap_report.csv`
- `routing/results/splits_manifest.csv`

**routing/experiments/e0_baseline.py:121 falls back to it** (1)

- `bandit_model.pkl`

**table written by generate_paper_results.py** (20)

- `routing/results/tab_agreement_path_audit.csv`
- `routing/results/tab_baselines.csv`
- `routing/results/tab_case_mapping.csv`
- `routing/results/tab_cheap_trigger.csv`
- `routing/results/tab_confidence_routing.csv`
- `routing/results/tab_cost_accounting.csv`
- `routing/results/tab_cost_accuracy_tradeoff.csv`
- `routing/results/tab_cross_family_deployment_cascade.csv`
- `routing/results/tab_e8_cascade.csv`
- `routing/results/tab_gate_holdout.csv`
- `routing/results/tab_limited_data_cascade.csv`
- `routing/results/tab_main_results.csv`
- `routing/results/tab_per_generator.csv`
- `routing/results/tab_positive_cell_audit.csv`
- `routing/results/tab_real_image_sources.csv`
- `routing/results/tab_robustness.csv`
- `routing/results/tab_sensitivity.csv`
- `routing/results/tab_threeway.csv`
- `routing/results/tab_transfer.csv`
- `routing/results/tab_transfer_reconciliation.csv`

**vendored CNNSpot implementation (CNNSpot baseline in Table main_results)** (31)

- `CNNDetection/LICENSE.txt`
- `CNNDetection/README.md`
- `CNNDetection/data/__init__.py`
- `CNNDetection/data/datasets.py`
- `CNNDetection/dataset/test/download_testset.sh`
- `CNNDetection/dataset/train/download_trainset.sh`
- `CNNDetection/dataset/val/download_valset.sh`
- `CNNDetection/demo.py`
- `CNNDetection/demo_dir.py`
- `CNNDetection/earlystop.py`
- `CNNDetection/eval.py`
- `CNNDetection/eval_config.py`
- `CNNDetection/examples/fake.png`
- `CNNDetection/examples/real.png`
- `CNNDetection/examples/realfakedir/0_real/real.png`
- `CNNDetection/examples/realfakedir/1_fake/fake.png`
- `CNNDetection/networks/__init__.py`
- `CNNDetection/networks/base_model.py`
- `CNNDetection/networks/lpf.py`
- `CNNDetection/networks/resnet.py`
- `CNNDetection/networks/resnet_lpf.py`
- `CNNDetection/networks/trainer.py`
- `CNNDetection/options/__init__.py`
- `CNNDetection/options/base_options.py`
- `CNNDetection/options/test_options.py`
- `CNNDetection/options/train_options.py`
- `CNNDetection/requirements.txt`
- `CNNDetection/train.py`
- `CNNDetection/util.py`
- `CNNDetection/validate.py`
- `CNNDetection/weights/download_weights.sh`
