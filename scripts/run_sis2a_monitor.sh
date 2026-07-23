#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS2_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS2_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
sis1_root="${SIS1_RUN_ROOT:-/root/autodl-tmp/experiments/sis1_seed_instability_20260723}"
run_root="${SIS2_RUN_ROOT:-/root/autodl-tmp/experiments/sis2_early_prediction_20260723}"
monitor_prompts="${SIS2_MONITOR_PROMPTS:-/root/autodl-tmp/datasets/r3_prepared/eval/harmbench_monitor_30.jsonl}"
harmbench_model="${SIS2_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
guardian_model="${SIS2_GUARDIAN_MODEL:-/root/autodl-tmp/models/granite-guardian-3.1-2b}"
strongreject_base="${SIS2_STRONGREJECT_BASE:-/root/autodl-tmp/models/gemma-2b}"
strongreject_adapter="${SIS2_STRONGREJECT_ADAPTER:-/root/autodl-tmp/models/strongreject-15k-v1}"
generation_workers="${SIS2_GENERATION_WORKERS:-4}"
read -r -a seeds <<< "${SIS2_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
checkpoint_steps=(10 20 30 50 75 100)

expected_monitor_sha256="fcc564b6a14bad8ea22c35a2c18f1ff99d84062d6de21bbcf9e357151859c893"

cd "$repo_dir"
mkdir -p "$run_root"/{generations/monitor8,metrics,logs,manifests}

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

require_file "$model_path/config.json"
require_file "$monitor_prompts"
require_file "$harmbench_model/config.json"
require_file "$guardian_model/config.json"
require_file "$strongreject_base/config.json"
require_file "$strongreject_adapter/adapter_config.json"
actual_monitor_sha256="$(sha256sum "$monitor_prompts" | awk '{print $1}')"
if [[ "$actual_monitor_sha256" != "$expected_monitor_sha256" ]]; then
  echo "Monitor SHA-256 mismatch: $actual_monitor_sha256" >&2
  exit 1
fi

items=(base)
labels=(base)
outputs=("$run_root/generations/monitor8/base_monitor.json")
for seed in "${seeds[@]}"; do
  for step in "${checkpoint_steps[@]}"; do
    item="${seed}:${step}"
    label="seed${seed}_step${step}"
    adapter="$sis1_root/runs/D2_helpfulness_safety_conflict_seed${seed}/checkpoints/step_${step}/adapter_model"
    require_file "$adapter/adapter_config.json"
    items+=("$item")
    labels+=("$label")
    outputs+=("$run_root/generations/monitor8/${label}_monitor.json")
  done
done

generate_monitor() {
  local item="$1"
  local label adapter_args=()
  if [[ "$item" == "base" ]]; then
    label="base"
  else
    local seed="${item%%:*}"
    local step="${item##*:}"
    label="seed${seed}_step${step}"
    adapter_args=(
      --adapter_path
      "$sis1_root/runs/D2_helpfulness_safety_conflict_seed${seed}/checkpoints/step_${step}/adapter_model"
    )
  fi
  local output="$run_root/generations/monitor8/${label}_monitor.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed monitor generation: $label"
    return
  fi
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    "${adapter_args[@]}" \
    --prompts_jsonl "$monitor_prompts" \
    --output_path "$output" \
    --num_samples 8 \
    --max_new_tokens 128 \
    --batch_size 8 \
    --prompt_batch_size 1 \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_monitor8_${label}.log"
}

run_in_batches generate_monitor "$generation_workers" "${items[@]}"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/MONITOR_GENERATION_COMPLETE"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  output="${outputs[$index]}"
  quality="$run_root/metrics/output_quality_monitor8_${label}.json"
  if [[ ! -s "$quality" ]]; then
    if ! python scripts/audit_generation_quality.py \
      --generations "$output" \
      --output_path "$quality" \
      > "$run_root/logs/output_quality_monitor8_${label}.log" 2>&1; then
      echo "QUALITY FAIL (retained for diagnosis): $label" >&2
    fi
  fi
done

python - "$run_root/metrics" "${labels[*]}" <<'PY'
import json
import pathlib
import sys

metrics = pathlib.Path(sys.argv[1])
labels = sys.argv[2].split()
conditions = {}
for label in labels:
    path = metrics / f"output_quality_monitor8_{label}.json"
    if not path.is_file():
        raise SystemExit(f"Missing output-quality result: {label}")
    payload = json.loads(path.read_text())
    conditions[label] = {
        "status": payload["status"],
        "failures": payload["failures"],
        "diagnostics": payload["diagnostics"],
        "source_sha256": payload["source_sha256"],
    }
failed = sorted(
    label for label, payload in conditions.items() if payload["status"] != "pass"
)
summary = {
    "status": "complete",
    "condition_count": len(conditions),
    "pass_count": len(conditions) - len(failed),
    "fail_count": len(failed),
    "failed_conditions": failed,
    "conditions": conditions,
}
(metrics / "output_quality_monitor8_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps({key: summary[key] for key in (
    "condition_count", "pass_count", "fail_count", "failed_conditions"
)}, indent=2))
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/MONITOR_QUALITY_COMPLETE"

if ! is_complete_manifest "$run_root/metrics/harmbench_monitor8.json"; then
  python scripts/audit_harmbench_mistral.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --classifier_model "$harmbench_model" \
    --output_path "$run_root/metrics/harmbench_monitor8.json" \
    --batch_size 8 \
    2>&1 | tee "$run_root/logs/audit_harmbench_monitor8.log"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/MONITOR_HARMBENCH_COMPLETE"

if [[ ! -s "$run_root/metrics/granite_monitor8.json" ]]; then
  python scripts/audit_granite_guardian_outputs.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --guardian_model "$guardian_model" \
    --torch_dtype bfloat16 \
    --output_path "$run_root/metrics/granite_monitor8.json" \
    --batch_size 16 \
    --progress_every 500 \
    2>&1 | tee "$run_root/logs/audit_granite_monitor8.log"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/MONITOR_GRANITE_COMPLETE"

if ! is_complete_manifest "$run_root/metrics/strongreject_monitor8.json"; then
  python scripts/audit_strongreject_outputs.py "${outputs[@]}" \
    --labels "${labels[@]}" \
    --base_model "$strongreject_base" \
    --adapter_model "$strongreject_adapter" \
    --output_path "$run_root/metrics/strongreject_monitor8.json" \
    --batch_size 16 \
    --progress_every 500 \
    2>&1 | tee "$run_root/logs/audit_strongreject_monitor8.log"
fi
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/MONITOR_STRONGREJECT_COMPLETE"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/MONITOR_EVALUATION_COMPLETE"
echo "SIS-2A monitor evaluation complete: $run_root"
