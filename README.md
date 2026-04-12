# Inter-Model Disagreement as a Routing Signal for AI-Generated Media Detection

Code for *"Inter-Model Disagreement as a Routing Signal for Cost-Aware AI-Generated Media Detection"* (Nayak & Madisetti).

The pipeline implements a three-stage cascade for AI-generated media detection: a CNNSpot artifact detector, a CLIP logistic probe, and Qwen2.5-VL as an optional third-stage verifier. The cascade uses disagreement between the CNN and CLIP stages as a parameter-free routing signal to decide when to invoke the VLM.

## Requirements

- Python 3.10+
- CUDA-capable GPU (evaluation was run on an NVIDIA RTX A4500)
- ~30 GB disk space for weights and extracted features
- Optional: `rclone` if pulling GenImage data from cloud storage

## Setup(GO TO final_version branch)

1. Download model weights:
```bash
   python download_weights.py
```
   This pulls CNNSpot weights, the CLIP ViT-B/32 backbone, and Qwen2.5-VL-7B.

2. (Optional) Download the GenBuster-200K-mini benchmark:
```bash
   python hfgenbuster.py
```

3. Install Python dependencies:
```bash
   bash setup.sh
```

## Pipeline

Scripts should be run in the following order. Each stage produces intermediate outputs that the next stage consumes.

Input frame
↓
generate_cnnspot_features.py     # CNNSpot predictions and logits
↓
generate_clip_features.py        # CLIP ViT-B/32 embeddings
↓
train_clip_classifier.py         # Logistic regression probe on CLIP features
↓
generate_clip_preds.py           # CLIP probe predictions
↓
run_qwen_on_genbuster.py         # Qwen2.5-VL predictions (score-decoded)
↓
merge_all_features.py            # Join CNN / CLIP / Qwen outputs per sample
↓
train_bandit.py                  # Train the learned routing baseline
↓
evaluate_routing.py              # Evaluate all routing methods


The same pattern applies for the SD~1.4 and BigGAN datasets using the `*_sd14.py` and `*_biggan.py` variants of each script.

## Reproducing Paper Results

To reproduce the results reported in the paper:

1. Run the full pipeline for each of the three datasets (GenBuster, SD~1.4, BigGAN).
2. Run `retrain_bandits.py` to ensure the bandit policies are trained against the current Qwen verifier outputs.
3. Run `analysis2.py` to produce the final tables and plots.

The main results script (`analysis2.py`) prints LaTeX-formatted tables matching those in the paper and saves accompanying plots to the routing directory.

## Repository Structure

| File/Script | Purpose |
|---|---|
| `download_weights.py` | Download pre-trained model weights |
| `setup.sh` | Environment setup and dependencies |
| `generate_cnnspot_features.py` | Extract CNNSpot detection features |
| `generate_clip_features.py` | Extract CLIP model features |
| `train_clip_classifier.py` | Train CLIP logistic regression probe |
| `generate_clip_preds.py` | Generate CLIP model predictions |
| `run_qwen_on_genbuster.py` | Run Qwen VLM inference on GenBuster dataset |
| `merge_all_features.py` | Merge all extracted features |
| `train_bandit.py` | Train multi-armed bandit router |
| `retrain_bandits.py` | Re-train all bandits on current Qwen outputs |
| `evaluate_routing.py` | Evaluate routing method performance |
| `cross_distribution_clip.py` | Generate CLIP probe transfer matrix |
| `analysis2.py` | Generate final tables and figures |
