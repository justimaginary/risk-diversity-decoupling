#!/usr/bin/env bash
set -euo pipefail

repo_dir="${R2_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${R2_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preferences="${R2_D2_PREFERENCES:-/root/autodl-tmp/datasets/r2_prepared_v2/preferences/D2_helpfulness_safety_conflict.jsonl}"
eval_dir="${R2_EVAL_DIR:-/root/autodl-tmp/datasets/r2_prepared/eval}"
run_root="${R2_RUN_ROOT:-/root/autodl-tmp/experiments/r2_d2_repaired_20260722}"
harmbench_model="${R2_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
guardian_model="${R2_GUARDIAN_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
embedding_model="${R2_EMBEDDING_MODEL:-/root/autodl-tmp/models/all-MiniLM-L6-v2}"
strongreject_base="${R2_STRONGREJECT_BASE:-/root/autodl-tmp/models/gemma-2b}"
strongreject_adapter="${R2_STRONGREJECT_ADAPTER:-/root/autodl-tmp/models/strongreject-15k-v1}"
label="D2_helpfulness_safety_conflict_repaired"
adapter_dir="$run_root/runs/$label/adapter_model"
harmbench_output="$run_root/generations/${label}_harmbench.json"
xstest_output="$run_root/generations/${label}_xstest.json"

cd "$repo_dir"
mkdir -p "$run_root/runs" "$run_root/generations" "$run_root/metrics" "$run_root/logs"

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
require_file "$preferences"
require_file "$eval_dir/harmbench_stratified_50.jsonl"
require_file "$eval_dir/xstest_full.jsonl"

training_manifest="$run_root/runs/$label/manifest.json"
if is_complete_manifest "$training_manifest"; then
  echo "SKIP completed repaired D2 training"
else
  echo "START repaired D2 training"
  python scripts/train_qwen3_lora_dpo.py \
    --model_name "$model_path" \
    --preferences_path "$preferences" \
    --prompts_path "$eval_dir/harmbench_stratified_50.jsonl" \
    --output_dir "$run_root/runs/$label" \
    --max_steps 100 \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --max_length 384 \
    --num_prompts 1 \
    --num_samples 1 \
    --max_new_tokens 32 \
    --eval_batch_size 1 \
    --seed 42 \
    --generation_seed 20260722 \
    --realized_kl_samples 40 \
    2>&1 | tee "$run_root/logs/train_${label}.log"
fi
require_file "$adapter_dir/adapter_config.json"

generate() {
  local benchmark="$1"
  local prompts="$2"
  local samples="$3"
  local max_new_tokens="$4"
  local prompt_batch_size="$5"
  local output="$run_root/generations/${label}_${benchmark}.json"
  local manifest="${output%.json}.manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed repaired D2 generation: $benchmark"
    return
  fi
  echo "START repaired D2 generation: $benchmark"
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    --adapter_path "$adapter_dir" \
    --prompts_jsonl "$prompts" \
    --output_path "$output" \
    --num_samples "$samples" \
    --max_new_tokens "$max_new_tokens" \
    --batch_size 16 \
    --prompt_batch_size "$prompt_batch_size" \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_${label}_${benchmark}.log"
}

generate harmbench "$eval_dir/harmbench_stratified_50.jsonl" 16 128 1
generate xstest "$eval_dir/xstest_full.jsonl" 4 32 4

python scripts/audit_generation_quality.py \
  --generations "$harmbench_output" \
  --output_path "$run_root/metrics/output_quality.json" \
  2>&1 | tee "$run_root/logs/audit_output_quality.log"

python -c 'import json,sys; value=json.load(open(sys.argv[1]))["realized_kl"]["teacher_forced_mean_token_kl"]; print(f"teacher_forced_mean_token_kl={value:.6f}"); raise SystemExit(0 if value <= float(sys.argv[2]) else 2)' "$training_manifest" 0.5 \
  2>&1 | tee "$run_root/logs/check_realized_kl.log"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/TRAINING_AND_GENERATION_COMPLETE"

if [[ ! -s "$run_root/metrics/xstest_refusal.json" ]]; then
  python scripts/evaluate_xstest_refusal.py "$xstest_output" \
    --labels "$label" \
    --output_path "$run_root/metrics/xstest_refusal.json" \
    2>&1 | tee "$run_root/logs/evaluate_xstest.log"
fi

if [[ ! -s "$run_root/metrics/harmbench_mistral.json" ]]; then
  python scripts/audit_harmbench_mistral.py "$harmbench_output" \
    --labels "$label" \
    --classifier_model "$harmbench_model" \
    --output_path "$run_root/metrics/harmbench_mistral.json" \
    --batch_size 8 \
    2>&1 | tee "$run_root/logs/audit_harmbench_mistral.log"
fi

if [[ ! -s "$run_root/metrics/strongreject_harmbench.json" ]]; then
  python scripts/audit_strongreject_outputs.py "$harmbench_output" \
    --labels "$label" \
    --base_model "$strongreject_base" \
    --adapter_model "$strongreject_adapter" \
    --output_path "$run_root/metrics/strongreject_harmbench.json" \
    --batch_size 16 \
    --progress_every 100 \
    2>&1 | tee "$run_root/logs/audit_strongreject_harmbench.log"
fi

if [[ ! -s "$run_root/metrics/risk_diversity.json" ]]; then
  python scripts/compute_r2_risk_diversity.py "$harmbench_output" \
    --labels "$label" \
    --harmbench_audit "$run_root/metrics/harmbench_mistral.json" \
    --embedding_model "$embedding_model" \
    --output_path "$run_root/metrics/risk_diversity.json" \
    2>&1 | tee "$run_root/logs/compute_risk_diversity.log"
fi

if [[ ! -s "$run_root/metrics/semantic_harmbench.json" ]]; then
  python scripts/compute_semantic_pilot_metrics.py "$harmbench_output" \
    --labels "$label" \
    --embedding_model "$embedding_model" \
    --output_path "$run_root/metrics/semantic_harmbench.json" \
    2>&1 | tee "$run_root/logs/compute_semantic_harmbench.log"
fi

if [[ ! -s "$run_root/metrics/granite_harmbench.json" ]]; then
  python scripts/audit_granite_guardian_outputs.py "$harmbench_output" \
    --labels "$label" \
    --guardian_model "$guardian_model" \
    --torch_dtype bfloat16 \
    --output_path "$run_root/metrics/granite_harmbench.json" \
    --batch_size 16 \
    --progress_every 100 \
    2>&1 | tee "$run_root/logs/audit_granite_harmbench.log"
fi

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/EVALUATION_COMPLETE"
echo "Repaired D2 R2 evaluation complete: $run_root"
