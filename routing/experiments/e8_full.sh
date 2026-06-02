#!/bin/bash
set -e

echo "=============================================="
echo "E8 MODERN DIFFUSION EXPERIMENT PIPELINE"
echo "SDXL + FLUX + GPT-5.5"
echo "=============================================="

REPO=/workspace/benchmarks_ai_research
cd "$REPO"

# ---- Step 1: Download benchmark ----
echo ""
echo "[1/6] Downloading AIGIBench..."
python download_aigibench.py

# ---- Step 2: Build dataset index ----
echo ""
echo "[2/6] Building benchmark index..."
python routing/build_aigibench_index.py

# ---- Step 3: Feature extraction ----
echo ""
echo "[3/6] Extracting detector features..."
cd "$REPO/routing/experiments"

python e8_modern_diffusion.py \
    --extract_features \
    --datasets SDXL FLUX

# ---- Step 4: Baseline evaluation ----
echo ""
echo "[4/6] Running zero-shot cascade evaluation..."

python e8_modern_diffusion.py \
    --evaluate \
    --datasets SDXL FLUX

# ---- Step 5: GPT-5.5 disagreement routing ----
echo ""
echo "[5/6] Running GPT-5.5 strong verifier..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    exit 1
fi

python e8_modern_diffusion.py \
    --run_gpt55 \
    --gpt_model gpt-5.5 \
    --datasets SDXL FLUX

# ---- Step 6: Final evaluation ----
echo ""
echo "[6/6] Re-evaluating with GPT-5.5..."

python e8_modern_diffusion.py \
    --evaluate \
    --datasets SDXL FLUX

echo ""
echo "=============================================="
echo "E8 COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "$REPO/routing/results/e8_modern_diffusion.csv"