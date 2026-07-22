#!/usr/bin/env bash
set -euo pipefail

repo_dir="${R2_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${R2_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preference_dir="${R2_PREFERENCE_DIR:-/root/autodl-tmp/datasets/r2_prepared/preferences}"
eval_dir="${R2_EVAL_DIR:-/root/autodl-tmp/datasets/r2_prepared/eval}"
run_root="${R2_RUN_ROOT:-/root/autodl-tmp/experiments/r2_20260722}"
max_steps="${R2_MAX_STEPS:-300}"

conditions=(
  D0_clean_helpfulness
  D1_clean_safety
  D2_helpfulness_safety_conflict
  D3_poison_05
  D4_full_refusal_suppression
)

cd "$repo_dir"
mkdir -p "$run_root/runs" "$run_root/generations" "$run_root/logs"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required file is missing or empty: $1" >&2
    exit 1
  fi
}

is_complete_manifest() {
  [[ -s "$1" ]] && grep -q '"status": "complete"' "$1"
}

require_file "$model_path/config.json"
require_file "$eval_dir/harmbench_stratified_50.jsonl"
require_file "$eval_dir/xstest_full.jsonl"

for condition in "${conditions[@]}"; do
  preferences="$preference_dir/$condition.jsonl"
  output_dir="$run_root/runs/$condition"
  log_path="$run_root/logs/train_${condition}.log"
  require_file "$preferences"
  if is_complete_manifest "$output_dir/manifest.json"; then
    echo "SKIP completed training: $condition"
    continue
  fi
  echo "START training: $condition"
  python scripts/train_qwen3_lora_dpo.py \
    --model_name "$model_path" \
    --preferences_path "$preferences" \
    --prompts_path "$eval_dir/harmbench_stratified_50.jsonl" \
    --output_dir "$output_dir" \
    --max_steps "$max_steps" \
    --max_length 384 \
    --num_prompts 1 \
    --num_samples 1 \
    --max_new_tokens 32 \
    --eval_batch_size 1 \
    --seed 42 \
    --generation_seed 20260722 \
    --realized_kl_samples 20 \
    2>&1 | tee "$log_path"
  if ! is_complete_manifest "$output_dir/manifest.json"; then
    echo "Training did not produce a complete manifest: $condition" >&2
    exit 1
  fi
done

generate_condition() {
  local label="$1"
  local adapter_path="$2"
  local benchmark="$3"
  local prompts="$4"
  local samples="$5"
  local output="$run_root/generations/${label}_${benchmark}.json"
  local manifest="${output%.json}.manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed generation: $label $benchmark"
    return
  fi
  command=(
    python scripts/generate_qwen3_eval.py
    --model_path "$model_path"
    --prompts_jsonl "$prompts"
    --output_path "$output"
    --num_samples "$samples"
    --max_new_tokens 128
    --batch_size 4
    --generation_seed 20260722
  )
  if [[ -n "$adapter_path" ]]; then
    command+=(--adapter_path "$adapter_path")
  fi
  echo "START generation: $label $benchmark"
  "${command[@]}" 2>&1 | tee "$run_root/logs/generate_${label}_${benchmark}.log"
  if ! is_complete_manifest "$manifest"; then
    echo "Generation did not produce a complete manifest: $label $benchmark" >&2
    exit 1
  fi
}

generate_condition base "" harmbench "$eval_dir/harmbench_stratified_50.jsonl" 16
generate_condition base "" xstest "$eval_dir/xstest_full.jsonl" 4
for condition in "${conditions[@]}"; do
  adapter="$run_root/runs/$condition/adapter_model"
  require_file "$adapter/adapter_config.json"
  generate_condition "$condition" "$adapter" harmbench "$eval_dir/harmbench_stratified_50.jsonl" 16
  generate_condition "$condition" "$adapter" xstest "$eval_dir/xstest_full.jsonl" 4
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/TRAINING_AND_GENERATION_COMPLETE"
echo "R2 training and generation complete: $run_root"
