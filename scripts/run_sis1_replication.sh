#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS1_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS1_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preferences="${SIS1_PREFERENCES:-/root/autodl-tmp/datasets/r2_prepared_v2/preferences/D2_helpfulness_safety_conflict.jsonl}"
eval_dir="${SIS1_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
run_root="${SIS1_RUN_ROOT:-/root/autodl-tmp/experiments/sis1_seed_instability_20260723}"
harmbench_model="${SIS1_HARMBENCH_MODEL:-/root/autodl-tmp/models/HarmBench-Mistral-7b-val-cls}"
read -r -a seeds <<< "${SIS1_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
training_workers="${SIS1_TRAINING_WORKERS:-2}"
generation_workers="${SIS1_GENERATION_WORKERS:-4}"
screen_samples="${SIS1_SCREEN_SAMPLES:-8}"
full_samples="${SIS1_FULL_SAMPLES:-32}"
checkpoint_steps=(10 20 30 50 75 100)

expected_preferences_sha256="08405eae95a5f049e8d2c38dffeb845a03a530377da2b536be9aa056c8c1b071"
expected_eval_sha256="fffe21a839151b5330a4e97d5f35122acd6816e105c15558ae8d92edada146fb"

cd "$repo_dir"
mkdir -p "$run_root"/{runs,generations/screen8,generations/full32,metrics,logs,manifests}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required file is missing or empty: $1" >&2
    exit 1
  fi
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected=$expected actual=$actual" >&2
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

harmbench_prompts="$eval_dir/harmbench_stratified_100.jsonl"
require_file "$model_path/config.json"
require_file "$harmbench_model/config.json"
require_file "$preferences"
require_file "$harmbench_prompts"
require_sha256 "$preferences" "$expected_preferences_sha256"
require_sha256 "$harmbench_prompts" "$expected_eval_sha256"

train_seed() {
  local seed="$1"
  local label="D2_helpfulness_safety_conflict_seed${seed}"
  local output_dir="$run_root/runs/$label"
  local manifest="$output_dir/manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed training: $label"
    return
  fi
  echo "START training: $label"
  python scripts/train_qwen3_lora_dpo.py \
    --model_name "$model_path" \
    --preferences_path "$preferences" \
    --prompts_path "$harmbench_prompts" \
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
    2>&1 | tee "$run_root/logs/train_${label}.log"
  is_complete_manifest "$manifest"
}

run_in_batches train_seed "$training_workers" "${seeds[@]}"

python - "$run_root" "${seeds[*]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2].split()]
expected_steps = {"10", "20", "30", "50", "75", "100"}
summary = {"status": "complete", "runs": {}}
for seed in seeds:
    label = f"D2_helpfulness_safety_conflict_seed{seed}"
    path = root / "runs" / label / "manifest.json"
    manifest = json.loads(path.read_text())
    checkpoints = set(manifest["artifacts"]["checkpoints"])
    if manifest["status"] != "complete" or checkpoints != expected_steps:
        raise SystemExit(f"Incomplete training/checkpoints: {label}")
    summary["runs"][label] = {
        "seed": seed,
        "model_seed": manifest["config"]["model_seed"],
        "data_order_seed": manifest["config"]["data_order_seed"],
        "schedule_sha256": manifest["training_randomness"]["schedule_sha256"],
        "teacher_forced_mean_token_kl": manifest["realized_kl"]["teacher_forced_mean_token_kl"],
        "training_response_tokens": manifest["training_response_tokens"],
        "wall_seconds": manifest["total_wall_seconds"],
        "peak_vram_bytes": manifest["peak_vram_bytes"],
    }
(root / "manifests" / "training_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
PY
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/TRAINING_COMPLETE"

generate_screen() {
  local item="$1"
  local label adapter_args=()
  if [[ "$item" == "base" ]]; then
    label="base"
  else
    label="D2_helpfulness_safety_conflict_seed${item}"
    adapter_args=(--adapter_path "$run_root/runs/$label/adapter_model")
  fi
  local output="$run_root/generations/screen8/${label}_harmbench.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed screen generation: $label"
    return
  fi
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    "${adapter_args[@]}" \
    --prompts_jsonl "$harmbench_prompts" \
    --output_path "$output" \
    --num_samples "$screen_samples" \
    --max_new_tokens 128 \
    --batch_size 8 \
    --prompt_batch_size 1 \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_screen8_${label}.log"
}

generation_items=(base "${seeds[@]}")
run_in_batches generate_screen "$generation_workers" "${generation_items[@]}"

screen_outputs=("$run_root/generations/screen8/base_harmbench.json")
screen_labels=(base)
for seed in "${seeds[@]}"; do
  label="D2_helpfulness_safety_conflict_seed${seed}"
  screen_outputs+=("$run_root/generations/screen8/${label}_harmbench.json")
  screen_labels+=("$label")
  python scripts/audit_generation_quality.py \
    --generations "$run_root/generations/screen8/${label}_harmbench.json" \
    --output_path "$run_root/metrics/output_quality_screen8_${label}.json" \
    > "$run_root/logs/output_quality_screen8_${label}.log" 2>&1
done

python scripts/audit_harmbench_mistral.py "${screen_outputs[@]}" \
  --labels "${screen_labels[@]}" \
  --classifier_model "$harmbench_model" \
  --output_path "$run_root/metrics/harmbench_screen8.json" \
  --batch_size 8 \
  2>&1 | tee "$run_root/logs/audit_harmbench_screen8.log"

python scripts/summarize_sis1_replication.py \
  --harmbench_audit "$run_root/metrics/harmbench_screen8.json" \
  --output_path "$run_root/metrics/sis1_screen8_summary.json" \
  2>&1 | tee "$run_root/logs/summarize_screen8.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/SCREEN8_COMPLETE"

advance="$(
  python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["preregistered_gate"]["advance_to_32_samples"]).lower())' \
    "$run_root/metrics/sis1_screen8_summary.json"
)"
if [[ "$advance" != "true" ]]; then
  echo "SIS-1 screen gate stopped before 32 samples."
  exit 0
fi

generate_full() {
  local item="$1"
  local label adapter_args=()
  if [[ "$item" == "base" ]]; then
    label="base"
  else
    label="D2_helpfulness_safety_conflict_seed${item}"
    adapter_args=(--adapter_path "$run_root/runs/$label/adapter_model")
  fi
  local output="$run_root/generations/full32/${label}_harmbench.json"
  if is_complete_manifest "${output%.json}.manifest.json"; then
    echo "SKIP completed full generation: $label"
    return
  fi
  python scripts/generate_qwen3_eval.py \
    --model_path "$model_path" \
    "${adapter_args[@]}" \
    --prompts_jsonl "$harmbench_prompts" \
    --output_path "$output" \
    --num_samples "$full_samples" \
    --max_new_tokens 128 \
    --batch_size 16 \
    --prompt_batch_size 1 \
    --generation_seed 20260722 \
    2>&1 | tee "$run_root/logs/generate_full32_${label}.log"
}

run_in_batches generate_full "$generation_workers" "${generation_items[@]}"

full_outputs=("$run_root/generations/full32/base_harmbench.json")
full_labels=(base)
for seed in "${seeds[@]}"; do
  label="D2_helpfulness_safety_conflict_seed${seed}"
  full_outputs+=("$run_root/generations/full32/${label}_harmbench.json")
  full_labels+=("$label")
  python scripts/audit_generation_quality.py \
    --generations "$run_root/generations/full32/${label}_harmbench.json" \
    --output_path "$run_root/metrics/output_quality_full32_${label}.json" \
    > "$run_root/logs/output_quality_full32_${label}.log" 2>&1
done

python scripts/audit_harmbench_mistral.py "${full_outputs[@]}" \
  --labels "${full_labels[@]}" \
  --classifier_model "$harmbench_model" \
  --output_path "$run_root/metrics/harmbench_full32.json" \
  --batch_size 8 \
  2>&1 | tee "$run_root/logs/audit_harmbench_full32.log"

python scripts/summarize_sis1_replication.py \
  --harmbench_audit "$run_root/metrics/harmbench_full32.json" \
  --output_path "$run_root/metrics/sis1_full32_summary.json" \
  2>&1 | tee "$run_root/logs/summarize_full32.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/FULL32_COMPLETE"
echo "SIS-1 replication complete: $run_root"
