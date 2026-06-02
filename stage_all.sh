#!/bin/bash
set -e
cd /workspace/benchmarks_ai_research

echo "Step 1: tracked modifications and deletions..."
git add -u

echo "Step 2: new scripts and figures..."
git add .gitignore download_genimage.py download_aigibench.py
git add routing/experiments/ routing/extractors/ routing/results/
git add routing/build_aigibench_index.py routing/build_genbuster_index.py routing/run_revision.sh
git add routing/plot_fig_a_semantic_degradation.png routing/plot_fig_a_semantic_degradation.pdf
git add routing/plot_fig_b_cascade_gain_scatter.png routing/plot_fig_b_cascade_gain_scatter.pdf
git add routing/plot_fig_c_disagreement_rate.png routing/plot_fig_c_disagreement_rate.pdf

echo "Step 3: SDXL E8 data..."
git add routing/sdxl_clip_features.csv routing/sdxl_dinov2_features.csv
git add routing/sdxl_fft_features.csv routing/sdxl_siglip2_features.csv
git add routing/sdxl_index.csv
git add routing/sdxl_gpt55_clip_disagree_preds.csv
git add routing/sdxl_gpt55_dinov2_disagree_preds.csv
git add routing/sdxl_gpt55_siglip2_disagree_preds.csv

echo "Step 4: FLUX E8 data..."
git add routing/flux_clip_features.csv routing/flux_dinov2_features.csv
git add routing/flux_fft_features.csv routing/flux_siglip2_features.csv
git add routing/flux_index.csv
git add routing/flux_gpt55_clip_disagree_preds.csv
git add routing/flux_gpt55_dinov2_disagree_preds.csv
git add routing/flux_gpt55_siglip2_disagree_preds.csv

echo "Step 5: BigGAN data..."
git add routing/biggan_clip_features.csv routing/biggan_clip_preds.csv
git add routing/biggan_index.csv routing/biggan_bandit_dataset.csv
git add routing/biggan_dinov2_features.csv routing/biggan_fft_features.csv
git add routing/biggan_siglip2_features.csv routing/biggan_qwen_preds.csv
git add routing/biggan_gpt55_disagree_preds.csv routing/biggan_merged_dataset.csv

echo "Step 6: SD14 data..."
git add routing/sd14_clip_features.csv routing/sd14_clip_preds.csv
git add routing/sd14_index.csv routing/sd14_bandit_dataset.csv
git add routing/sd14_dinov2_features.csv routing/sd14_fft_features.csv
git add routing/sd14_siglip2_features.csv routing/sd14_qwen_preds.csv
git add routing/sd14_gpt55_disagree_preds.csv routing/sd14_merged_dataset.csv

echo "Step 7: GenBuster and shared data..."
git add routing/clip_features.csv routing/fft_features.csv
git add routing/dinov2_features.csv routing/siglip2_features.csv
git add routing/gpt55_disagree_preds.csv routing/merged_dataset.csv
git add routing/univfd_preds.csv routing/bandit_decisions.csv
git add routing/cross_family_deployment.csv routing/transfer_learning_results.csv
git add routing/clip_transfer_matrix.csv

echo "=== Staging complete ==="
git status --short | grep "^[AMRD]" | wc -l
echo "files staged"
