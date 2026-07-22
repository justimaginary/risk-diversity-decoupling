#!/usr/bin/env bash
set -euo pipefail

repo_dir="${R3_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${R3_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
eval_dir="${R3_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
main_root="${R3_MAIN_ROOT:-/root/autodl-tmp/experiments/r3_main_20260722}"
run_root="${R3_D2_64_ROOT:-/root/autodl-tmp/experiments/r3_d2_64_20260722}"
primary_seed="${R3_PRIMARY_GENERATION_SEED:-20260722}"
supplement_seed="${R3_SUPPLEMENT_GENERATION_SEED:-20260723}"
read -r -a labels <<< "${R3_D2_64_LABELS:-base D2_helpfulness_safety_conflict_seed42 D2_helpfulness_safety_conflict_seed43 D2_helpfulness_safety_conflict_seed44}"

cd "$repo_dir"
mkdir -p "$run_root"/{generations/supplement,generations/merged,metrics,logs,manifests}

is_complete_manifest() {
  [[ -s "$1" ]] && grep -q '"status": "complete"' "$1"
}

for label in "${labels[@]}"; do
  primary="$main_root/generations/${label}_harmbench.json"
  supplement="$run_root/generations/supplement/${label}_harmbench.json"
  supplement_manifest="$run_root/generations/supplement/${label}_harmbench.manifest.json"
  merged="$run_root/generations/merged/${label}_harmbench_64.json"
  merged_manifest="$run_root/manifests/${label}_harmbench_64.json"
  adapter_args=()
  if [[ "$label" != "base" ]]; then
    adapter="$main_root/runs/$label/adapter_model"
    adapter_args=(--adapter_path "$adapter")
  fi

  if is_complete_manifest "$supplement_manifest"; then
    echo "SKIP completed supplement generation: $label"
  else
    echo "START supplement generation: $label"
    python scripts/generate_qwen3_eval.py \
      --model_path "$model_path" \
      "${adapter_args[@]}" \
      --prompts_jsonl "$eval_dir/harmbench_stratified_100.jsonl" \
      --output_path "$supplement" \
      --num_samples 32 \
      --max_new_tokens 128 \
      --batch_size 16 \
      --prompt_batch_size 1 \
      --generation_seed "$supplement_seed" \
      2>&1 | tee "$run_root/logs/generate_${label}_supplement.log"
  fi

  if is_complete_manifest "$merged_manifest"; then
    echo "SKIP completed 64-sample merge: $label"
  else
    python scripts/merge_generation_samples.py \
      --primary "$primary" \
      --supplement "$supplement" \
      --output_path "$merged" \
      --manifest_path "$merged_manifest" \
      --expected_samples 64 \
      --primary_generation_seed "$primary_seed" \
      --supplement_generation_seed "$supplement_seed"
  fi

  python scripts/audit_generation_quality.py \
    --generations "$merged" \
    --output_path "$run_root/metrics/output_quality_${label}.json" \
    2>&1 | tee "$run_root/logs/audit_output_quality_${label}.log"
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/GENERATION_COMPLETE"
echo "R3 D2 64-sample generation complete: $run_root"
