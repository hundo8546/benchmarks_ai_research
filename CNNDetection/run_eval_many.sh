#!/usr/bin/env bash
set -e

cd /workspace/CNNDetection

# Change this if eval_config.py expects a different default testset path
TESTSET_DIR="dataset/test"

# If eval.py reads paths from eval_config.py only, we’ll temporarily edit it each run.
# If eval.py accepts CLI args (weight/data), we can pass them directly (preferred).
# We’ll handle both by checking for args support.

function run_eval () {
  local NAME="$1"
  local WEIGHT="$2"

  echo "=============================="
  echo "RUN: $NAME"
  echo "WEIGHT: $WEIGHT"
  echo "=============================="

  # Try CLI-style first (if supported by your eval.py)
  if python eval.py --help 2>/dev/null | grep -qi "model"; then
    python eval.py --model_path "$WEIGHT" --dataroot "$TESTSET_DIR"
  else
    # Fall back: patch eval_config.py to point to weight
    python - <<EOF
import re, pathlib
p = pathlib.Path("eval_config.py")
s = p.read_text()
# Replace a common pattern: model_path = '...'
s2 = re.sub(r"(model_path\s*=\s*)['\"].*?['\"]", r"\\1'${WEIGHT}'", s)
p.write_text(s2)
print("Patched eval_config.py model_path -> ${WEIGHT}")
EOF
    python eval.py
  fi

  # Optional: move/rename outputs if eval.py writes fixed filenames
  if [ -d results ]; then
    mkdir -p results_runs
    TS=$(date +%Y%m%d_%H%M%S)
    cp -r results "results_runs/${NAME}_${TS}" || true
  fi
}

# ---- Fill these in based on what's actually in weights/ ----
run_eval "blur_jpg_prob0.5" "weights/blur_jpg_prob0.5.pth"
run_eval "blur_jpg_prob0.1" "weights/blur_jpg_prob0.1.pth"

# If you have these weights, uncomment:
# run_eval "no_aug" "weights/no_aug.pth"
# run_eval "blur_only" "weights/blur_only.pth"
# run_eval "jpg_only" "weights/jpg_only.pth"

