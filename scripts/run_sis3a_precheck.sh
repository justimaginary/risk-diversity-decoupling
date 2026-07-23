#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS3_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS3_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
monitor_prompts="${SIS3_MONITOR_PROMPTS:-/root/autodl-tmp/datasets/r3_prepared/eval/harmbench_monitor_30.jsonl}"
harmbench_model="${SIS3_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
sis2_root="${SIS2_RUN_ROOT:-/root/autodl-tmp/experiments/sis2_early_prediction_20260723}"
unseen_root="$sis2_root/unseen"
run_root="${SIS3_RUN_ROOT:-/root/autodl-tmp/experiments/sis3_trajectory_reversal_20260723}"
generation_workers="${SIS3_GENERATION_WORKERS:-5}"
seeds=(52 54 55 57)
steps=(30 50 75 100)
expected_monitor_sha256="fcc564b6a14bad8ea22c35a2c18f1ff99d84062d6de21bbcf9e357151859c893"

cd "$repo_dir"
mkdir -p "$run_root"/{generations/monitor8,metrics,logs}

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

require_file "$model_path/config.json"
require_file "$monitor_prompts"
require_file "$harmbench_model/config.json"
actual_monitor_sha256="$(sha256sum "$monitor_prompts" | awk '{print $1}')"
[[ "$actual_monitor_sha256" == "$expected_monitor_sha256" ]] || {
  echo "Monitor SHA-256 mismatch: $actual_monitor_sha256" >&2
  exit 1
}

base_source="$sis2_root/generations/monitor8/base_monitor.json"
require_file "$base_source"
cp -f "$base_source" "$run_root/generations/monitor8/base_monitor.json"

items=()
generation_items=()
for seed in "${seeds[@]}"; do
  for step in "${steps[@]}"; do
    items+=("${seed}:${step}")
    if [[ "$step" == "30" ]]; then
      label="seed${seed}_step30"
      source="$unseen_root/generations/monitor8/${label}_monitor.json"
      output="$run_root/generations/monitor8/${label}_monitor.json"
      require_file "$source"
      cp -f "$source" "$output"
      cp -f "${source%.json}.manifest.json" "${output%.json}.manifest.json"
      echo "REUSE frozen SIS-2 step30 monitor: $label"
    else
      generation_items+=("${seed}:${step}")
    fi
  done
done

generate_condition() {
  local item="$1"
  local seed="${item%%:*}"
  local step="${item##*:}"
  local label="seed${seed}_step${step}"
  local output="$run_root/generations/monitor8/${label}_monitor.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed SIS-3A generation: $label"
    return
  fi
  local adapter="$unseen_root/runs/D2_helpfulness_safety_conflict_seed${seed}/checkpoints/step_${step}/adapter_model"
  require_file "$adapter/adapter_config.json"
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    --adapter_path "$adapter" \
    --prompts_jsonl "$monitor_prompts" \
    --output_path "$output" \
    --num_samples 8 \
    --max_new_tokens 128 \
    --batch_size 8 \
    --prompt_batch_size 1 \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_${label}.log"
}

run_in_batches generate_condition "$generation_workers" "${generation_items[@]}"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/GENERATION_COMPLETE"

outputs=("$run_root/generations/monitor8/base_monitor.json")
labels=(base)
for item in "${items[@]}"; do
  seed="${item%%:*}"
  step="${item##*:}"
  label="seed${seed}_step${step}"
  output="$run_root/generations/monitor8/${label}_monitor.json"
  quality="$run_root/metrics/output_quality_${label}.json"
  python scripts/audit_generation_quality.py \
    --generations "$output" \
    --output_path "$quality" \
    > "$run_root/logs/output_quality_${label}.log" 2>&1
  outputs+=("$output")
  labels+=("$label")
done

python scripts/audit_harmbench_mistral.py "${outputs[@]}" \
  --labels "${labels[@]}" \
  --classifier_model "$harmbench_model" \
  --output_path "$run_root/metrics/harmbench_monitor8.json" \
  --batch_size 8 \
  2>&1 | tee "$run_root/logs/audit_harmbench_monitor8.log"

python scripts/summarize_sis3a_precheck.py \
  --harmbench_audit "$run_root/metrics/harmbench_monitor8.json" \
  --generations_dir "$run_root/generations/monitor8" \
  --quality_dir "$run_root/metrics" \
  --runs_dir "$unseen_root/runs" \
  --formal_summary "$unseen_root/metrics/sis2b_screen_summary.json" \
  --output_path "$run_root/metrics/sis3a_precheck_summary.json" \
  2>&1 | tee "$run_root/logs/summarize_sis3a.log"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/PRECHECK_COMPLETE"
echo "SIS-3A same-monitor precheck complete."
