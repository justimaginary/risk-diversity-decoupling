#!/usr/bin/env bash
set -euo pipefail

repo_dir="${R3_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${R3_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preferences_dir="${R3_PREFERENCES_DIR:-/root/autodl-tmp/datasets/r2_prepared_v2/preferences}"
eval_dir="${R3_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
run_root="${R3_RUN_ROOT:-/root/autodl-tmp/experiments/r3_main_20260722}"
max_steps="${R3_MAX_STEPS:-100}"
checkpoint_steps="${R3_CHECKPOINT_STEPS:-50 100}"
max_kl="${R3_MAX_KL:-0.75}"
read -r -a checkpoint_steps_array <<< "$checkpoint_steps"
harmbench_model="${R3_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
guardian_model="${R3_GUARDIAN_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
embedding_model="${R3_EMBEDDING_MODEL:-/root/autodl-tmp/models/all-MiniLM-L6-v2}"
strongreject_base="${R3_STRONGREJECT_BASE:-/root/autodl-tmp/models/gemma-2b}"
strongreject_adapter="${R3_STRONGREJECT_ADAPTER:-/root/autodl-tmp/models/strongreject-15k-v1}"
harmbench_prompts="$eval_dir/harmbench_stratified_100.jsonl"
xstest_prompts="$eval_dir/xstest_full.jsonl"

default_conditions="D1_clean_safety D2_helpfulness_safety_conflict D4_full_refusal_suppression"
read -r -a conditions <<< "${R3_CONDITIONS:-$default_conditions}"
seeds=(42 43 44)
labels=(base)
harmbench_outputs=("$run_root/generations/base_harmbench.json")
xstest_outputs=("$run_root/generations/base_xstest.json")

cd "$repo_dir"
mkdir -p "$run_root"/{runs,generations,metrics,logs,manifests,bootstrap,human_audit}

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
require_file "$harmbench_prompts"
require_file "$xstest_prompts"

for condition in "${conditions[@]}"; do
  require_file "$preferences_dir/${condition}.jsonl"
  for seed in "${seeds[@]}"; do
    label="${condition}_seed${seed}"
    labels+=("$label")
    harmbench_outputs+=("$run_root/generations/${label}_harmbench.json")
    xstest_outputs+=("$run_root/generations/${label}_xstest.json")
    manifest="$run_root/runs/$label/manifest.json"
    if is_complete_manifest "$manifest"; then
      echo "SKIP completed training: $label"
      continue
    fi
    echo "START training: $label"
    python scripts/train_qwen3_lora_dpo.py \
      --model_name "$model_path" \
      --preferences_path "$preferences_dir/${condition}.jsonl" \
      --prompts_path "$harmbench_prompts" \
      --output_dir "$run_root/runs/$label" \
      --max_steps "$max_steps" \
      --checkpoint_steps "${checkpoint_steps_array[@]}" \
      --learning_rate 1e-4 \
      --beta 0.1 \
      --max_length 384 \
      --num_prompts 1 \
      --num_samples 1 \
      --max_new_tokens 32 \
      --eval_batch_size 1 \
      --seed "$seed" \
      --generation_seed 20260722 \
      --realized_kl_samples 40 \
      2>&1 | tee "$run_root/logs/train_${label}.log"
  done
done

python - "$run_root" "${conditions[*]}" "${seeds[*]}" "$checkpoint_steps" "$max_kl" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
conditions = sys.argv[2].split()
seeds = [int(value) for value in sys.argv[3].split()]
expected_checkpoints = set(sys.argv[4].split())
max_kl = float(sys.argv[5])
failures = []
expected = [
    root / "runs" / f"{condition}_seed{seed}" / "manifest.json"
    for condition in conditions
    for seed in seeds
]
for manifest_path in expected:
    if not manifest_path.is_file():
        failures.append({"run": manifest_path.parent.name, "error": "missing manifest"})
        continue
    manifest = json.loads(manifest_path.read_text())
    kl = manifest["realized_kl"]["teacher_forced_mean_token_kl"]
    checkpoints = manifest["artifacts"].get("checkpoints", {})
    if kl > max_kl or set(checkpoints) != expected_checkpoints:
        failures.append({"run": manifest_path.parent.name, "kl": kl, "checkpoints": checkpoints})
if failures:
    raise SystemExit("R3 training gate failed: " + json.dumps(failures, sort_keys=True))
print("R3 training KL/checkpoint gate passed")
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/TRAINING_COMPLETE"

generate() {
  local label="$1"
  local benchmark="$2"
  local prompts="$3"
  local samples="$4"
  local max_new_tokens="$5"
  local prompt_batch_size="$6"
  local adapter_path="${7:-}"
  local output="$run_root/generations/${label}_${benchmark}.json"
  local manifest="${output%.json}.manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed generation: $label $benchmark"
    return
  fi
  local adapter_args=()
  if [[ -n "$adapter_path" ]]; then
    adapter_args=(--adapter_path "$adapter_path")
  fi
  echo "START generation: $label $benchmark"
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    "${adapter_args[@]}" \
    --prompts_jsonl "$prompts" \
    --output_path "$output" \
    --num_samples "$samples" \
    --max_new_tokens "$max_new_tokens" \
    --batch_size 16 \
    --prompt_batch_size "$prompt_batch_size" \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_${label}_${benchmark}.log"
}

generate base harmbench "$harmbench_prompts" 32 128 1
generate base xstest "$xstest_prompts" 4 32 4
for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    label="${condition}_seed${seed}"
    adapter="$run_root/runs/$label/adapter_model"
    require_file "$adapter/adapter_config.json"
    generate "$label" harmbench "$harmbench_prompts" 32 128 1 "$adapter"
    generate "$label" xstest "$xstest_prompts" 4 32 4 "$adapter"
  done
done

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  output="${harmbench_outputs[$index]}"
  python scripts/audit_generation_quality.py \
    --generations "$output" \
    --output_path "$run_root/metrics/output_quality_${label}.json" \
    2>&1 | tee "$run_root/logs/audit_output_quality_${label}.log"
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/GENERATION_COMPLETE"

if [[ ! -s "$run_root/metrics/xstest_refusal.json" ]]; then
  python scripts/evaluate_xstest_refusal.py "${xstest_outputs[@]}" \
    --labels "${labels[@]}" \
    --output_path "$run_root/metrics/xstest_refusal.json" \
    2>&1 | tee "$run_root/logs/evaluate_xstest.log"
fi
if ! is_complete_manifest "$run_root/metrics/harmbench_mistral.json"; then
  python scripts/audit_harmbench_mistral.py "${harmbench_outputs[@]}" \
    --labels "${labels[@]}" \
    --classifier_model "$harmbench_model" \
    --output_path "$run_root/metrics/harmbench_mistral.json" \
    --batch_size 8 \
    2>&1 | tee "$run_root/logs/audit_harmbench_mistral.log"
fi
if ! is_complete_manifest "$run_root/metrics/strongreject_harmbench.json"; then
  python scripts/audit_strongreject_outputs.py "${harmbench_outputs[@]}" \
    --labels "${labels[@]}" \
    --base_model "$strongreject_base" \
    --adapter_model "$strongreject_adapter" \
    --output_path "$run_root/metrics/strongreject_harmbench.json" \
    --batch_size 16 \
    --progress_every 100 \
    2>&1 | tee "$run_root/logs/audit_strongreject_harmbench.log"
fi
if [[ ! -s "$run_root/metrics/risk_diversity.json" ]]; then
  python scripts/compute_r2_risk_diversity.py "${harmbench_outputs[@]}" \
    --labels "${labels[@]}" \
    --harmbench_audit "$run_root/metrics/harmbench_mistral.json" \
    --embedding_model "$embedding_model" \
    --output_path "$run_root/metrics/risk_diversity.json" \
    2>&1 | tee "$run_root/logs/compute_risk_diversity.log"
fi
if [[ ! -s "$run_root/metrics/semantic_harmbench.json" ]]; then
  python scripts/compute_semantic_pilot_metrics.py "${harmbench_outputs[@]}" \
    --labels "${labels[@]}" \
    --embedding_model "$embedding_model" \
    --output_path "$run_root/metrics/semantic_harmbench.json" \
    2>&1 | tee "$run_root/logs/compute_semantic_harmbench.log"
fi
if [[ ! -s "$run_root/metrics/granite_harmbench.json" ]]; then
  python scripts/audit_granite_guardian_outputs.py "${harmbench_outputs[@]}" \
    --labels "${labels[@]}" \
    --guardian_model "$guardian_model" \
    --torch_dtype bfloat16 \
    --output_path "$run_root/metrics/granite_harmbench.json" \
    --batch_size 16 \
    --progress_every 100 \
    2>&1 | tee "$run_root/logs/audit_granite_harmbench.log"
fi

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/EVALUATION_COMPLETE"
echo "R3 computational evaluation complete: $run_root"
