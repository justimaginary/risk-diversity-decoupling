#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS2_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS2_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preferences="${SIS2_PREFERENCES:-/root/autodl-tmp/datasets/r2_prepared_v2/preferences/D2_helpfulness_safety_conflict.jsonl}"
eval_dir="${SIS2_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
run_root="${SIS2_RUN_ROOT:-/root/autodl-tmp/experiments/sis2_early_prediction_20260723}"
unseen_root="$run_root/unseen"
predictor="$run_root/metrics/frozen_predictor.json"
harmbench_model="${SIS2_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
read -r -a seeds <<< "${SIS2_UNSEEN_SEEDS:-52 53 54 55 56 57}"
training_workers="${SIS2_TRAINING_WORKERS:-2}"
generation_workers="${SIS2_GENERATION_WORKERS:-4}"
checkpoint_steps=(10 20 30 50 75 100)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

expected_preferences_sha256="08405eae95a5f049e8d2c38dffeb845a03a530377da2b536be9aa056c8c1b071"
expected_final_sha256="fffe21a839151b5330a4e97d5f35122acd6816e105c15558ae8d92edada146fb"
expected_monitor_sha256="fcc564b6a14bad8ea22c35a2c18f1ff99d84062d6de21bbcf9e357151859c893"

cd "$repo_dir"
mkdir -p "$unseen_root"/{runs,generations/monitor8,metrics,logs,manifests}

require_file() {
  [[ -s "$1" ]] || { echo "Required file is missing or empty: $1" >&2; exit 1; }
}

require_sha256() {
  local actual
  actual="$(sha256sum "$1" | awk '{print $1}')"
  [[ "$actual" == "$2" ]] || {
    echo "SHA-256 mismatch for $1: expected=$2 actual=$actual" >&2
    exit 1
  }
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

final_prompts="$eval_dir/harmbench_stratified_100.jsonl"
monitor_prompts="$eval_dir/harmbench_monitor_30.jsonl"
base_monitor="$run_root/generations/monitor8/base_monitor.json"

require_file "$model_path/config.json"
require_file "$harmbench_model/config.json"
require_file "$preferences"
require_file "$final_prompts"
require_file "$monitor_prompts"
require_file "$base_monitor"
require_file "$predictor"
require_file "$predictor.sha256"
require_sha256 "$preferences" "$expected_preferences_sha256"
require_sha256 "$final_prompts" "$expected_final_sha256"
require_sha256 "$monitor_prompts" "$expected_monitor_sha256"
(cd "$(dirname "$predictor")" && sha256sum -c "$(basename "$predictor").sha256")

train_seed() {
  local seed="$1"
  local label="D2_helpfulness_safety_conflict_seed${seed}"
  local output_dir="$unseen_root/runs/$label"
  local manifest="$output_dir/manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed unseen training: $label"
    return
  fi
  echo "START unseen training: $label"
  python scripts/train_qwen3_lora_dpo.py \
    --model_name "$model_path" \
    --preferences_path "$preferences" \
    --prompts_path "$final_prompts" \
    --output_dir "$output_dir" \
    --max_steps 100 \
    --checkpoint_steps "${checkpoint_steps[@]}" \
    --learning_rate 1e-4 \
    --beta 0.1 \
    --max_length 384 \
    --num_prompts 1 \
    --num_samples 1 \
    --max_new_tokens 32 \
    --eval_batch_size 1 \
    --seed "$seed" \
    --model_seed "$seed" \
    --data_order_seed "$seed" \
    --generation_seed 20260722 \
    --realized_kl_samples 40 \
    2>&1 | tee "$unseen_root/logs/train_${label}.log"
  is_complete_manifest "$manifest"
}

run_in_batches train_seed "$training_workers" "${seeds[@]}"

python - "$unseen_root" "${seeds[*]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2].split()]
expected = {"10", "20", "30", "50", "75", "100"}
summary = {"status": "complete", "stage": "SIS-2B unseen training", "runs": {}}
for seed in seeds:
    label = f"D2_helpfulness_safety_conflict_seed{seed}"
    manifest = json.loads((root / "runs" / label / "manifest.json").read_text())
    checkpoints = set(manifest["artifacts"]["checkpoints"])
    if manifest["status"] != "complete" or checkpoints != expected:
        raise SystemExit(f"Incomplete unseen training/checkpoints: {label}")
    summary["runs"][label] = {
        "seed": seed,
        "model_seed": manifest["config"]["model_seed"],
        "data_order_seed": manifest["config"]["data_order_seed"],
        "schedule_sha256": manifest["training_randomness"]["schedule_sha256"],
        "teacher_forced_mean_token_kl": manifest["realized_kl"]["teacher_forced_mean_token_kl"],
        "training_response_tokens": manifest["training_response_tokens"],
        "wall_seconds": manifest["total_wall_seconds"],
        "peak_vram_bytes": manifest["peak_vram_bytes"],
        "checkpoint_steps": sorted(checkpoints, key=int),
    }
(root / "manifests" / "training_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$unseen_root/TRAINING_COMPLETE"

generate_early_monitor() {
  local seed="$1"
  local label="seed${seed}_step30"
  local adapter="$unseen_root/runs/D2_helpfulness_safety_conflict_seed${seed}/checkpoints/step_30/adapter_model"
  local output="$unseen_root/generations/monitor8/${label}_monitor.json"
  require_file "$adapter/adapter_config.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed unseen monitor generation: $label"
    return
  fi
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
    2>&1 | tee "$unseen_root/logs/generate_monitor8_${label}.log"
}

run_in_batches generate_early_monitor "$generation_workers" "${seeds[@]}"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$unseen_root/EARLY_MONITOR_GENERATION_COMPLETE"

monitor_outputs=("$base_monitor")
monitor_labels=(base)
for seed in "${seeds[@]}"; do
  label="seed${seed}_step30"
  output="$unseen_root/generations/monitor8/${label}_monitor.json"
  quality="$unseen_root/metrics/output_quality_monitor8_${label}.json"
  python scripts/audit_generation_quality.py \
    --generations "$output" \
    --output_path "$quality" \
    > "$unseen_root/logs/output_quality_monitor8_${label}.log" 2>&1
  monitor_outputs+=("$output")
  monitor_labels+=("$label")
done

python scripts/audit_harmbench_mistral.py "${monitor_outputs[@]}" \
  --labels "${monitor_labels[@]}" \
  --classifier_model "$harmbench_model" \
  --output_path "$unseen_root/metrics/harmbench_monitor8.json" \
  --batch_size 8 \
  2>&1 | tee "$unseen_root/logs/audit_harmbench_monitor8.log"

python scripts/predict_sis2_unseen.py \
  --predictor "$predictor" \
  --harmbench_audit "$unseen_root/metrics/harmbench_monitor8.json" \
  --output_path "$unseen_root/metrics/unseen_seed_predictions.json" \
  2>&1 | tee "$unseen_root/logs/freeze_unseen_predictions.log"
sha256sum "$unseen_root/metrics/unseen_seed_predictions.json" \
  > "$unseen_root/metrics/unseen_seed_predictions.json.sha256"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$unseen_root/PREDICTIONS_FROZEN"

echo "SIS-2B pre-formal stage complete; formal HarmBench has not started."
