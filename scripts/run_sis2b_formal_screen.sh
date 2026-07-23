#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS2_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS2_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
eval_dir="${SIS2_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
run_root="${SIS2_RUN_ROOT:-/root/autodl-tmp/experiments/sis2_early_prediction_20260723}"
unseen_root="$run_root/unseen"
sis1_root="${SIS1_RUN_ROOT:-/root/autodl-tmp/experiments/sis1_seed_instability_20260723}"
harmbench_model="${SIS2_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
generation_workers="${SIS2_GENERATION_WORKERS:-4}"
read -r -a seeds <<< "${SIS2_UNSEEN_SEEDS:-52 53 54 55 56 57}"
expected_final_sha256="fffe21a839151b5330a4e97d5f35122acd6816e105c15558ae8d92edada146fb"

cd "$repo_dir"
mkdir -p "$unseen_root"/{generations/screen8,metrics,logs}

require_file() {
  [[ -s "$1" ]] || { echo "Required file is missing or empty: $1" >&2; exit 1; }
}

is_complete_manifest() {
  [[ -s "$1" ]] && grep -q '"status": "complete"' "$1"
}

run_in_batches() {
  local function_name="$1"
  local limit="$2"
  shift 2
  local -a pids=()
  local item pid
  for item in "$@"; do
    "$function_name" "$item" &
    pids+=("$!")
    if (( ${#pids[@]} == limit )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

predictions="$unseen_root/metrics/unseen_seed_predictions.json"
predictions_hash="$predictions.sha256"
final_prompts="$eval_dir/harmbench_stratified_100.jsonl"
base_output="$sis1_root/generations/screen8/base_harmbench.json"

require_file "$unseen_root/PREDICTIONS_FROZEN"
require_file "$predictions"
require_file "$predictions_hash"
require_file "$model_path/config.json"
require_file "$harmbench_model/config.json"
require_file "$final_prompts"
require_file "$base_output"
require_file "${base_output%.json}.manifest.json"
(cd "$(dirname "$predictions")" && sha256sum -c "$(basename "$predictions").sha256")
actual_final_sha256="$(sha256sum "$final_prompts" | awk '{print $1}')"
[[ "$actual_final_sha256" == "$expected_final_sha256" ]] || {
  echo "Final prompt SHA-256 mismatch: $actual_final_sha256" >&2
  exit 1
}

generate_screen() {
  local seed="$1"
  local label="D2_helpfulness_safety_conflict_seed${seed}"
  local adapter="$unseen_root/runs/$label/adapter_model"
  local output="$unseen_root/generations/screen8/${label}_harmbench.json"
  require_file "$adapter/adapter_config.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed unseen formal generation: $label"
    return
  fi
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    --adapter_path "$adapter" \
    --prompts_jsonl "$final_prompts" \
    --output_path "$output" \
    --num_samples 8 \
    --max_new_tokens 128 \
    --batch_size 8 \
    --prompt_batch_size 1 \
    --generation_seed 20260722 \
    2>&1 | tee "$unseen_root/logs/generate_screen8_${label}.log"
}

run_in_batches generate_screen "$generation_workers" "${seeds[@]}"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$unseen_root/FORMAL_GENERATION_COMPLETE"

outputs=("$base_output")
labels=(base)
for seed in "${seeds[@]}"; do
  label="D2_helpfulness_safety_conflict_seed${seed}"
  output="$unseen_root/generations/screen8/${label}_harmbench.json"
  quality="$unseen_root/metrics/output_quality_screen8_${label}.json"
  python scripts/audit_generation_quality.py \
    --generations "$output" \
    --output_path "$quality" \
    > "$unseen_root/logs/output_quality_screen8_${label}.log" 2>&1
  outputs+=("$output")
  labels+=("$label")
done

python scripts/audit_harmbench_mistral.py "${outputs[@]}" \
  --labels "${labels[@]}" \
  --classifier_model "$harmbench_model" \
  --output_path "$unseen_root/metrics/harmbench_screen8.json" \
  --batch_size 8 \
  2>&1 | tee "$unseen_root/logs/audit_harmbench_screen8.log"

python scripts/summarize_sis2b_screen.py \
  --predictions "$predictions" \
  --harmbench_audit "$unseen_root/metrics/harmbench_screen8.json" \
  --quality_dir "$unseen_root/metrics" \
  --output_path "$unseen_root/metrics/sis2b_screen_summary.json" \
  2>&1 | tee "$unseen_root/logs/summarize_screen8.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$unseen_root/FORMAL_SCREEN_COMPLETE"

echo "SIS-2B formal HarmBench screen complete."
