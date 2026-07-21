#!/usr/bin/env bash
set -euo pipefail

R1_DIR="${1:?usage: bash scripts/run_r1_public_pilot.sh <r1-output-dir>}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/autodl-tmp/experiments/smoke/r0_latest/adapter_model}"

mkdir -p "${R1_DIR}/generations" "${R1_DIR}/logs" "${R1_DIR}/manifests"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "${rc}" > "${R1_DIR}/generation_exit_code"
    touch "${R1_DIR}/generation.finished"
    echo "R1_GENERATION_EXIT_CODE=${rc}"
    exit "${rc}"
}
trap finish EXIT

run_generation() {
    local label="$1"
    local prompts="$2"
    local samples="$3"
    shift 3
    python scripts/generate_qwen3_eval.py \
        --model_path "${MODEL_PATH}" \
        --prompts_jsonl "${prompts}" \
        --output_path "${R1_DIR}/generations/${label}.json" \
        --num_samples "${samples}" \
        --max_new_tokens 128 \
        --batch_size 4 \
        --generation_seed 20260722 \
        "$@" \
        2>&1 | tee "${R1_DIR}/logs/${label}.log"
}

HARMBENCH="${R1_DIR}/data_splits/harmbench_stratified_50.jsonl"
XSTEST="${R1_DIR}/data_splits/xstest_full.jsonl"

run_generation base_harmbench "${HARMBENCH}" 16
run_generation adapter_harmbench "${HARMBENCH}" 16 --adapter_path "${ADAPTER_PATH}"
run_generation base_xstest "${XSTEST}" 4
run_generation adapter_xstest "${XSTEST}" 4 --adapter_path "${ADAPTER_PATH}"
