#!/usr/bin/env bash
set -euo pipefail

repo_dir="${R2_REPO_DIR:-/root/risk-diversity-decoupling}"
run_root="${R2_RUN_ROOT:-/root/autodl-tmp/experiments/r2_20260722}"
harmbench_model="${R2_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
guardian_model="${R2_GUARDIAN_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
embedding_model="${R2_EMBEDDING_MODEL:-/root/autodl-tmp/models/all-MiniLM-L6-v2}"
strongreject_base="${R2_STRONGREJECT_BASE:-/root/autodl-tmp/models/gemma-2b}"
strongreject_adapter="${R2_STRONGREJECT_ADAPTER:-/root/autodl-tmp/models/strongreject-15k-v1}"

labels=(
  base
  D0_clean_helpfulness
  D1_clean_safety
  D2_helpfulness_safety_conflict
  D3_poison_05
  D4_full_refusal_suppression
)

cd "$repo_dir"
mkdir -p "$run_root/metrics" "$run_root/logs"
if [[ ! -s "$run_root/TRAINING_AND_GENERATION_COMPLETE" ]]; then
  echo "Training and generation completion marker is missing" >&2
  exit 1
fi

harmbench_outputs=()
xstest_outputs=()
for label in "${labels[@]}"; do
  harmbench_outputs+=("$run_root/generations/${label}_harmbench.json")
  xstest_outputs+=("$run_root/generations/${label}_xstest.json")
done
for path in "${harmbench_outputs[@]}" "${xstest_outputs[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "Generation file is missing or empty: $path" >&2
    exit 1
  fi
done

echo "START XSTest official refusal metrics"
python scripts/evaluate_xstest_refusal.py \
  "${xstest_outputs[@]}" \
  --labels "${labels[@]}" \
  --output_path "$run_root/metrics/xstest_refusal.json" \
  2>&1 | tee "$run_root/logs/evaluate_xstest.log"

echo "START HarmBench Mistral classifier"
python scripts/audit_harmbench_mistral.py \
  "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --classifier_model "$harmbench_model" \
  --output_path "$run_root/metrics/harmbench_mistral.json" \
  --batch_size 8 \
  2>&1 | tee "$run_root/logs/audit_harmbench_mistral.log"

echo "START StrongREJECT fine-tuned evaluator"
python scripts/audit_strongreject_outputs.py \
  "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --base_model "$strongreject_base" \
  --adapter_model "$strongreject_adapter" \
  --output_path "$run_root/metrics/strongreject_harmbench.json" \
  --batch_size 16 \
  --progress_every 100 \
  2>&1 | tee "$run_root/logs/audit_strongreject_harmbench.log"

echo "START R2 risk-diversity endpoints"
python scripts/compute_r2_risk_diversity.py \
  "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --harmbench_audit "$run_root/metrics/harmbench_mistral.json" \
  --embedding_model "$embedding_model" \
  --output_path "$run_root/metrics/risk_diversity.json" \
  2>&1 | tee "$run_root/logs/compute_risk_diversity.log"

echo "START Vendi and semantic pilot metrics"
python scripts/compute_semantic_pilot_metrics.py \
  "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --embedding_model "$embedding_model" \
  --output_path "$run_root/metrics/semantic_harmbench.json" \
  2>&1 | tee "$run_root/logs/compute_semantic_harmbench.log"

echo "START Granite Guardian independent audit"
python scripts/audit_granite_guardian_outputs.py \
  "${harmbench_outputs[@]}" \
  --labels "${labels[@]}" \
  --guardian_model "$guardian_model" \
  --torch_dtype bfloat16 \
  --output_path "$run_root/metrics/granite_harmbench.json" \
  --batch_size 16 \
  --progress_every 100 \
  2>&1 | tee "$run_root/logs/audit_granite_harmbench.log"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/EVALUATION_COMPLETE"
echo "R2 evaluation complete: $run_root"
