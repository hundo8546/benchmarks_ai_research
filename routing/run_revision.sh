#!/usr/bin/env bash
# =============================================================================
# IEEE Access Revision — Master Experiment Run Script
# RTX Pro 6000 96GB RunPod
#
# Run order: E0 -> E1 -> E2 -> E3 -> E4 -> E5 -> E6 -> E7
# Each experiment depends on previous outputs.
#
# Usage:
#   bash run_revision.sh [--start E0] [--stop E5] [--datasets GenBuster SD14 BigGAN]
#
# To run only acceptance-critical experiments:
#   bash run_revision.sh --start E3 --stop E6
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR_DIR="$SCRIPT_DIR/extractors"
EXPERIMENT_DIR="$SCRIPT_DIR/experiments"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"

DATASETS="GenBuster SD14 BigGAN"
START="E0"
STOP="E7"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START="$2"; shift 2 ;;
        --stop) STOP="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/run_revision.log"
}

run_experiment() {
    local name="$1"; shift
    local script="$1"; shift
    log "=== Starting $name ==="
    local t0=$(date +%s)
    python "$script" "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local rc=${PIPESTATUS[0]}
    local t1=$(date +%s)
    local elapsed=$(( (t1 - t0) / 60 ))
    if [ $rc -eq 0 ]; then
        log "=== $name COMPLETE (${elapsed}m) ==="
    else
        log "=== $name FAILED (rc=$rc, ${elapsed}m) ==="
        exit $rc
    fi
}

should_run() {
    local exp="$1"
    local order="E0 E1 E2 E3 E4 E5 E6 E7"
    local start_pos=0 stop_pos=7 exp_pos=0 i=0
    for e in $order; do
        [ "$e" = "$exp" ] && exp_pos=$i
        [ "$e" = "$START" ] && start_pos=$i
        [ "$e" = "$STOP" ] && stop_pos=$i
        i=$((i+1))
    done
    [ $exp_pos -ge $start_pos ] && [ $exp_pos -le $stop_pos ]
}

DS_ARGS="--datasets $DATASETS"

# =============================================================================
# FEATURE EXTRACTION (prereqs for E1, E2, E3, E6, E7)
# =============================================================================

if should_run E1 || should_run E3 || should_run E6; then
    log "--- Extracting SigLIP2 features ---"
    for ds in $DATASETS; do
        python "$EXTRACTOR_DIR/extract_siglip2.py" --dataset "$ds" \
            2>&1 | tee "$LOG_DIR/extract_siglip2_${ds}.log" || true
    done

    log "--- Extracting DINOv2 features ---"
    for ds in $DATASETS; do
        python "$EXTRACTOR_DIR/extract_dinov2.py" --dataset "$ds" \
            2>&1 | tee "$LOG_DIR/extract_dinov2_${ds}.log" || true
    done
fi

if should_run E2 || should_run E3; then
    log "--- Extracting FFT features ---"
    for ds in $DATASETS; do
        python "$EXTRACTOR_DIR/extract_fft.py" --dataset "$ds" \
            2>&1 | tee "$LOG_DIR/extract_fft_${ds}.log" || true
    done

    log "--- Running DIRE (if weights available) ---"
    DIRE_CKPT="/workspace/benchmarks_ai_research/weights/DIRE/lsun_bedroom.pth"
    if [ -f "$DIRE_CKPT" ]; then
        for ds in $DATASETS; do
            python "$EXTRACTOR_DIR/extract_dire.py" --dataset "$ds" --ckpt "$DIRE_CKPT" \
                2>&1 | tee "$LOG_DIR/extract_dire_${ds}.log" || true
        done
    else
        log "DIRE weights not found at $DIRE_CKPT — skipping DIRE (A2 will use FFT only)"
        log "Download from: https://github.com/ZhendongWang6/DIRE"
    fi
fi

# =============================================================================
# EXPERIMENTS
# =============================================================================

if should_run E0; then
    run_experiment E0 "$EXPERIMENT_DIR/e0_baseline.py"
fi

if should_run E1; then
    run_experiment E1 "$EXPERIMENT_DIR/e1_modern_semantic.py" $DS_ARGS
fi

if should_run E2; then
    run_experiment E2 "$EXPERIMENT_DIR/e2_modern_artifact.py" $DS_ARGS
fi

if should_run E3; then
    run_experiment E3 "$EXPERIMENT_DIR/e3_disagreement_matrix.py" $DS_ARGS
fi

if should_run E4; then
    # Step 1: Run strong verifier inference on disagreement subsets
    log "--- E4: Running strong verifier inference (Qwen2.5-VL-72B) ---"
    log "    This requires ~40GB GPU RAM for 72B model."
    log "    If 72B unavailable, will evaluate V1 only."
    for ds in $DATASETS; do
        python "$EXPERIMENT_DIR/e4_strong_verifier.py" \
            --run_inference \
            --model "Qwen/Qwen2.5-VL-72B-Instruct" \
            --datasets "$ds" \
            2>&1 | tee "$LOG_DIR/e4_inference_${ds}.log" || \
            log "  Warning: 72B inference failed for $ds, continuing with V1 only"
    done
    # Step 2: Evaluate
    run_experiment E4 "$EXPERIMENT_DIR/e4_strong_verifier.py" --evaluate $DS_ARGS
fi

if should_run E5; then
    run_experiment E5 "$EXPERIMENT_DIR/e5_oracle_ceiling.py" $DS_ARGS
fi

if should_run E6; then
    run_experiment E6 "$EXPERIMENT_DIR/e6_limited_data.py" $DS_ARGS
fi

if should_run E7; then
    log "--- E7: Extracting compressed features ---"
    for ds in $DATASETS; do
        python "$EXPERIMENT_DIR/e7_compression.py" \
            --extract --datasets "$ds" \
            2>&1 | tee "$LOG_DIR/e7_extract_${ds}.log" || true
    done
    run_experiment E7 "$EXPERIMENT_DIR/e7_compression.py" --evaluate $DS_ARGS
fi

# =============================================================================
# SUMMARY
# =============================================================================
log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results in: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.csv 2>/dev/null | tee -a "$LOG_DIR/run_revision.log" || true
