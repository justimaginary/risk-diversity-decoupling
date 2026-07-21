#!/usr/bin/env bash
set -euo pipefail

R1_DIR="${1:?usage: bash scripts/run_r1_public_evaluation.sh <r1-output-dir>}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-/root/autodl-tmp/models/all-MiniLM-L6-v2}"
GRANITE_MODEL="${GRANITE_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
HARMBENCH_MODEL="${HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"

GENERATION_DIR="${R1_DIR}/generations"
METRICS_DIR="${R1_DIR}/metrics"
LOG_DIR="${R1_DIR}/logs"
mkdir -p "${METRICS_DIR}" "${LOG_DIR}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "${rc}" > "${R1_DIR}/evaluation_exit_code"
    touch "${R1_DIR}/evaluation.finished"
    echo "R1_EVALUATION_EXIT_CODE=${rc}"
    exit "${rc}"
}
trap finish EXIT

generation_exit="${R1_DIR}/generation_exit_code"
if [[ ! -f "${generation_exit}" ]] || [[ "$(<"${generation_exit}")" != "0" ]]; then
    echo "R1 generation has not completed successfully: ${generation_exit}" >&2
    exit 2
fi

required_files=(
    "${GENERATION_DIR}/base_harmbench.json"
    "${GENERATION_DIR}/adapter_harmbench.json"
    "${GENERATION_DIR}/base_xstest.json"
    "${GENERATION_DIR}/adapter_xstest.json"
)
for path in "${required_files[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "Missing or empty generation result: ${path}" >&2
        exit 2
    fi
done

run_logged() {
    local name="$1"
    shift
    "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_logged xstest_refusal \
    python scripts/evaluate_xstest_refusal.py \
    "${GENERATION_DIR}/base_xstest.json" \
    "${GENERATION_DIR}/adapter_xstest.json" \
    --labels base_xstest adapter_xstest \
    --output_path "${METRICS_DIR}/xstest_refusal.json"

run_logged semantic_all_conditions \
    python scripts/compute_semantic_pilot_metrics.py \
    "${GENERATION_DIR}/base_harmbench.json" \
    "${GENERATION_DIR}/adapter_harmbench.json" \
    "${GENERATION_DIR}/base_xstest.json" \
    "${GENERATION_DIR}/adapter_xstest.json" \
    --labels base_harmbench adapter_harmbench base_xstest adapter_xstest \
    --embedding_model "${EMBEDDING_MODEL}" \
    --output_path "${METRICS_DIR}/semantic_all_conditions.json"

run_logged granite_guardian_harmbench \
    python scripts/audit_granite_guardian_outputs.py \
    "${GENERATION_DIR}/base_harmbench.json" \
    "${GENERATION_DIR}/adapter_harmbench.json" \
    --labels base_harmbench adapter_harmbench \
    --guardian_model "${GRANITE_MODEL}" \
    --torch_dtype bfloat16 \
    --output_path "${METRICS_DIR}/granite_guardian_harmbench.json"

run_logged harmbench_mistral \
    python scripts/audit_harmbench_mistral.py \
    "${GENERATION_DIR}/base_harmbench.json" \
    "${GENERATION_DIR}/adapter_harmbench.json" \
    --labels base_harmbench adapter_harmbench \
    --classifier_model "${HARMBENCH_MODEL}" \
    --batch_size 8 \
    --output_path "${METRICS_DIR}/harmbench_mistral.json"
