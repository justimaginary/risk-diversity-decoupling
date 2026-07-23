#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS1_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS1_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
eval_dir="${SIS1_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
run_root="${SIS1_RUN_ROOT:-/root/autodl-tmp/experiments/sis1_seed_instability_20260723}"
guardian_model="${SIS1_GUARDIAN_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
strongreject_base="${SIS1_STRONGREJECT_BASE:-/root/autodl-tmp/models/gemma-2b}"
strongreject_adapter="${SIS1_STRONGREJECT_ADAPTER:-/root/autodl-tmp/models/strongreject-15k-v1}"
generation_workers="${SIS1_GENERATION_WORKERS:-4}"
read -r -a seeds <<< "${SIS1_SEEDS:-42 43 44 45 46 47 48 49 50 51}"

cd "$repo_dir"
mkdir -p "$run_root"/{generations/xstest,metrics,logs}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required file is missing or empty: $1" >&2
    exit 1
  fi
}

is_complete_manifest() {
  [[ -s "$1" ]] && grep -q '"status": "complete"' "$1"
}

run_in_batches() {
  local function_name="$1"
  local limit="$2"
  shift 2
  local -a batch_pids=()
  local item pid
  for item in "$@"; do
    "$function_name" "$item" &
    batch_pids+=("$!")
    if (( ${#batch_pids[@]} == limit )); then
      for pid in "${batch_pids[@]}"; do
        wait "$pid"
      done
      batch_pids=()
    fi
  done
  for pid in "${batch_pids[@]}"; do
    wait "$pid"
  done
}

xstest_prompts="$eval_dir/xstest_full.jsonl"
require_file "$model_path/config.json"
require_file "$xstest_prompts"
require_file "$run_root/FULL32_COMPLETE"

labels=(base)
harmbench_outputs=("$run_root/generations/full32/base_harmbench.json")
xstest_outputs=("$run_root/generations/xstest/base_xstest.json")
for seed in "${seeds[@]}"; do
  label="D2_helpfulness_safety_conflict_seed${seed}"
  labels+=("$label")
  harmbench_outputs+=("$run_root/generations/full32/${label}_harmbench.json")
  xstest_outputs+=("$run_root/generations/xstest/${label}_xstest.json")
done
for path in "${harmbench_outputs[@]}"; do
  require_file "$path"
done

generate_xstest() {
  local item="$1"
  local label adapter_args=()
  if [[ "$item" == "base" ]]; then
    label="base"
  else
    label="D2_helpfulness_safety_conflict_seed${item}"
    adapter_args=(--adapter_path "$run_root/runs/$label/adapter_model")
  fi
  local output="$run_root/generations/xstest/${label}_xstest.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed XSTest generation: $label"
    return
  fi
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    "${adapter_args[@]}" \
    --prompts_jsonl "$xstest_prompts" \
    --output_path "$output" \
    --num_samples 4 \
    --max_new_tokens 32 \
    --batch_size 16 \
    --prompt_batch_size 4 \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_xstest_${label}.log"
}

generation_items=(base "${seeds[@]}")
run_in_batches generate_xstest "$generation_workers" "${generation_items[@]}"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/XSTEST_GENERATION_COMPLETE"

python scripts/evaluate_xstest_refusal.py "${xstest_outputs[@]}" \
  --labels "${labels[@]}" \
  --output_path "$run_root/metrics/xstest_refusal.json" \
  2>&1 | tee "$run_root/logs/evaluate_xstest.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/XSTEST_EVALUATION_COMPLETE"

require_file "$guardian_model/config.json"
python scripts/audit_granite_guardian_outputs.py "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --guardian_model "$guardian_model" \
  --torch_dtype bfloat16 \
  --output_path "$run_root/metrics/granite_full32.json" \
  --batch_size 16 \
  --progress_every 500 \
  2>&1 | tee "$run_root/logs/audit_granite_full32.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/GRANITE_COMPLETE"

require_file "$strongreject_base/config.json"
require_file "$strongreject_adapter/adapter_config.json"
python scripts/audit_strongreject_outputs.py "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --base_model "$strongreject_base" \
  --adapter_model "$strongreject_adapter" \
  --output_path "$run_root/metrics/strongreject_full32.json" \
  --batch_size 16 \
  --progress_every 500 \
  2>&1 | tee "$run_root/logs/audit_strongreject_full32.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/STRONGREJECT_COMPLETE"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/CONFIRMATION_FOLLOWUP_COMPLETE"
echo "SIS-1 confirmation follow-up complete: $run_root"
