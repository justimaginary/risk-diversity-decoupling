#!/usr/bin/env bash
set -euo pipefail

repo_dir="${R3_REPO_DIR:-/root/risk-diversity-decoupling}"
main_root="${R3_MAIN_ROOT:-/root/autodl-tmp/experiments/r3_main_20260722}"
run_root="${R3_D2_64_ROOT:-/root/autodl-tmp/experiments/r3_d2_64_20260722}"
harmbench_model="${R3_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
guardian_model="${R3_GUARDIAN_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
embedding_model="${R3_EMBEDDING_MODEL:-/root/autodl-tmp/models/all-MiniLM-L6-v2}"
strongreject_base="${R3_STRONGREJECT_BASE:-/root/autodl-tmp/models/gemma-2b}"
strongreject_adapter="${R3_STRONGREJECT_ADAPTER:-/root/autodl-tmp/models/strongreject-15k-v1}"
labels=(base D2_helpfulness_safety_conflict_seed42 D2_helpfulness_safety_conflict_seed43 D2_helpfulness_safety_conflict_seed44)
outputs=()

cd "$repo_dir"
mkdir -p "$run_root"/{metrics,logs,bootstrap,runs,human_audit}

is_complete_json() {
  [[ -s "$1" ]] && grep -q '"status": "complete"' "$1"
}

if [[ ! -s "$run_root/GENERATION_COMPLETE" ]]; then
  echo "64-sample generation is not complete: $run_root" >&2
  exit 1
fi

for label in "${labels[@]}"; do
  output="$run_root/generations/merged/${label}_harmbench_64.json"
  [[ -s "$output" ]] || { echo "Missing merged generations: $output" >&2; exit 1; }
  outputs+=("$output")
  if [[ "$label" != "base" && ! -e "$run_root/runs/$label" ]]; then
    ln -s "$main_root/runs/$label" "$run_root/runs/$label"
  fi
done

cp "$main_root/metrics/xstest_refusal.json" "$run_root/metrics/xstest_refusal.json"

if ! is_complete_json "$run_root/metrics/harmbench_mistral.json"; then
  python scripts/audit_harmbench_mistral.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --classifier_model "$harmbench_model" \
    --output_path "$run_root/metrics/harmbench_mistral.json" \
    --batch_size 8 \
    2>&1 | tee "$run_root/logs/audit_harmbench_mistral.log"
fi

if ! is_complete_json "$run_root/metrics/strongreject_harmbench.json"; then
  python scripts/audit_strongreject_outputs.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --base_model "$strongreject_base" \
    --adapter_model "$strongreject_adapter" \
    --output_path "$run_root/metrics/strongreject_harmbench.json" \
    --batch_size 16 \
    --progress_every 100 \
    2>&1 | tee "$run_root/logs/audit_strongreject_harmbench.log"
fi

if [[ ! -s "$run_root/metrics/risk_diversity.json" ]]; then
  python scripts/compute_r2_risk_diversity.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --harmbench_audit "$run_root/metrics/harmbench_mistral.json" \
    --embedding_model "$embedding_model" \
    --output_path "$run_root/metrics/risk_diversity.json" \
    2>&1 | tee "$run_root/logs/compute_risk_diversity.log"
fi

if [[ ! -s "$run_root/metrics/semantic_harmbench.json" ]]; then
  python scripts/compute_semantic_pilot_metrics.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --embedding_model "$embedding_model" \
    --output_path "$run_root/metrics/semantic_harmbench.json" \
    2>&1 | tee "$run_root/logs/compute_semantic_harmbench.log"
fi

if [[ ! -s "$run_root/metrics/granite_harmbench.json" ]]; then
  python scripts/audit_granite_guardian_outputs.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --guardian_model "$guardian_model" \
    --torch_dtype bfloat16 \
    --output_path "$run_root/metrics/granite_harmbench.json" \
    --batch_size 16 \
    --progress_every 100 \
    2>&1 | tee "$run_root/logs/audit_granite_harmbench.log"
fi

python scripts/bootstrap_r3_main.py \
  --risk_diversity "$run_root/metrics/risk_diversity.json" \
  --xstest "$run_root/metrics/xstest_refusal.json" \
  --output_path "$run_root/bootstrap/hierarchical_bootstrap.json" \
  --samples 5000 \
  --seed 20260722 \
  2>&1 | tee "$run_root/logs/bootstrap.log"

python scripts/summarize_r3_main.py \
  --run_root "$run_root" \
  --bootstrap "$run_root/bootstrap/hierarchical_bootstrap.json" \
  --sample_count 64 \
  --output_path "$run_root/metrics/r3_summary.json" \
  > "$run_root/logs/summary.log"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/EVALUATION_COMPLETE"
echo "R3 D2 64-sample evaluation complete: $run_root"
