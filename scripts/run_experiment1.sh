#!/usr/bin/env bash
# LEGACY ORCHESTRATOR: use PLAN.md and configs/current/ for new runs.
# ==============================================================================
# Experiment 1: PCE Characterization
# ==============================================================================
# This experiment characterizes how Preference Collapse Exploitability evolves
# during standard DPO training. It trains a model with DPO, computing PCE
# metrics at regular checkpoints to map the collapse trajectory.
#
# Requirements:
#   - 4x A100-80GB GPUs
#   - ~100GB disk space for checkpoints
#   - ~8 hours runtime
#
# Usage:
#   bash scripts/run_experiment1.sh [CONFIG_PATH]
# ==============================================================================

set -euo pipefail

# Configuration
CONFIG_PATH="${1:-configs/default_config.yaml}"
EXPERIMENT_NAME="exp1_pce_characterization"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
NUM_GPUS=4
MASTER_PORT=29500

# Timestamp for run identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

echo "=============================================="
echo " PCE Characterization Experiment"
echo "=============================================="
echo " Config: ${CONFIG_PATH}"
echo " Output: ${RUN_DIR}"
echo " GPUs: ${NUM_GPUS}"
echo " Timestamp: ${TIMESTAMP}"
echo "=============================================="

# Create output directories
mkdir -p "${RUN_DIR}/checkpoints"
mkdir -p "${RUN_DIR}/pce_logs"
mkdir -p "${RUN_DIR}/safety_reports"
mkdir -p "${RUN_DIR}/diversity_logs"

# Copy config for reproducibility
cp "${CONFIG_PATH}" "${RUN_DIR}/config.yaml"

# Log system information
echo "System Info:" > "${RUN_DIR}/system_info.txt"
nvidia-smi >> "${RUN_DIR}/system_info.txt" 2>/dev/null || echo "No NVIDIA GPUs detected" >> "${RUN_DIR}/system_info.txt"
python --version >> "${RUN_DIR}/system_info.txt"
pip list >> "${RUN_DIR}/system_info.txt"

# ==============================================================================
# Phase 1: Baseline PCE Measurement (before DPO)
# ==============================================================================
echo ""
echo "[Phase 1/4] Computing baseline PCE (pre-DPO)..."
echo ""

python -m src.scripts.compute_pce \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --prompts_path "data/attack_prompts.jsonl" \
    --num_samples 128 \
    --output_path "${RUN_DIR}/pce_logs/baseline_pce.json" \
    --config "${CONFIG_PATH}"

echo "[Phase 1/4] Baseline PCE complete."

# ==============================================================================
# Phase 2: DPO Training with PCE Monitoring
# ==============================================================================
echo ""
echo "[Phase 2/4] Starting DPO training with PCE monitoring..."
echo ""

# DeepSpeed configuration for 4xA100
DS_CONFIG="configs/ds_config_zero3.json"

# Create DeepSpeed config if it doesn't exist
if [ ! -f "${DS_CONFIG}" ]; then
    cat > "${DS_CONFIG}" << 'EOF'
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 4,
    "wall_clock_breakdown": false
}
EOF
fi

# Launch distributed training
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m src.scripts.train_dpo \
    --config "${CONFIG_PATH}" \
    --output_dir "${RUN_DIR}/checkpoints" \
    --wandb_run_name "${EXPERIMENT_NAME}_${TIMESTAMP}" \
    --pce_eval_steps 100 \
    --save_steps 200 \
    --attack_prompts_path "data/attack_prompts.jsonl"

echo "[Phase 2/4] DPO training complete."

# ==============================================================================
# Phase 3: Post-training PCE Sweep across Checkpoints
# ==============================================================================
echo ""
echo "[Phase 3/4] Computing PCE at each checkpoint..."
echo ""

# Find all checkpoints
CHECKPOINTS=$(find "${RUN_DIR}/checkpoints" -name "checkpoint-*" -type d | sort -V)

for CKPT in ${CHECKPOINTS}; do
    STEP=$(basename "${CKPT}" | sed 's/checkpoint-//')
    echo "  Computing PCE at step ${STEP}..."

    python -m src.scripts.compute_pce \
        --model_name "${CKPT}" \
        --prompts_path "data/attack_prompts.jsonl" \
        --num_samples 128 \
        --output_path "${RUN_DIR}/pce_logs/pce_step_${STEP}.json" \
        --config "${CONFIG_PATH}"
done

echo "[Phase 3/4] PCE sweep complete."

# ==============================================================================
# Phase 4: Full Safety Evaluation on Final Model
# ==============================================================================
echo ""
echo "[Phase 4/4] Running safety evaluation on final model..."
echo ""

# Get final checkpoint
FINAL_CKPT=$(find "${RUN_DIR}/checkpoints" -name "checkpoint-*" -type d | sort -V | tail -1)

python -m src.scripts.safety_eval \
    --model_name "${FINAL_CKPT}" \
    --prompts_path "data/attack_prompts.jsonl" \
    --num_samples 64 \
    --output_path "${RUN_DIR}/safety_reports/final_safety_report.json" \
    --config "${CONFIG_PATH}"

# Compute diversity metrics on final model
python -m src.scripts.compute_diversity \
    --model_name "${FINAL_CKPT}" \
    --prompts_path "data/attack_prompts.jsonl" \
    --num_samples 128 \
    --output_path "${RUN_DIR}/diversity_logs/final_diversity.json" \
    --config "${CONFIG_PATH}"

echo "[Phase 4/4] Safety evaluation complete."

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo " Experiment Complete"
echo "=============================================="
echo " Results: ${RUN_DIR}"
echo ""
echo " Key outputs:"
echo "   - PCE trajectory: ${RUN_DIR}/pce_logs/"
echo "   - Safety report: ${RUN_DIR}/safety_reports/"
echo "   - Diversity: ${RUN_DIR}/diversity_logs/"
echo "   - Checkpoints: ${RUN_DIR}/checkpoints/"
echo "   - Config: ${RUN_DIR}/config.yaml"
echo ""
echo " Next steps:"
echo "   1. Review PCE trajectory in wandb"
echo "   2. Identify critical collapse step"
echo "   3. Run Experiment 2 (attack evaluation)"
echo "=============================================="
