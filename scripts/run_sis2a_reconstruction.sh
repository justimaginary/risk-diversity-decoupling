#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS2_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS2_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preferences="${SIS2_PREFERENCES:-/root/autodl-tmp/datasets/r2_prepared_v2/preferences/D2_helpfulness_safety_conflict.jsonl}"
eval_dir="${SIS2_EVAL_DIR:-/root/autodl-tmp/datasets/r3_prepared/eval}"
run_root="${SIS2_RUN_ROOT:-/root/autodl-tmp/experiments/sis2_early_prediction_20260723}"
read -r -a seeds <<< "${SIS2_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
training_workers="${SIS2_TRAINING_WORKERS:-2}"
checkpoint_steps=(10 20 30 50 75 100)

expected_preferences_sha256="08405eae95a5f049e8d2c38dffeb845a03a530377da2b536be9aa056c8c1b071"
expected_final_sha256="fffe21a839151b5330a4e97d5f35122acd6816e105c15558ae8d92edada146fb"
expected_monitor_sha256="fcc564b6a14bad8ea22c35a2c18f1ff99d84062d6de21bbcf9e357151859c893"

cd "$repo_dir"
mkdir -p "$run_root"/{runs,logs,manifests}

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

final_prompts="$eval_dir/harmbench_stratified_100.jsonl"
monitor_prompts="$eval_dir/harmbench_monitor_30.jsonl"
sis1_training_summary="$repo_dir/experiments/sis1_seed_instability_20260723/manifests/training_summary.json"

require_file "$model_path/config.json"
require_file "$preferences"
require_file "$final_prompts"
require_file "$monitor_prompts"
require_file "$sis1_training_summary"
require_sha256 "$preferences" "$expected_preferences_sha256"
require_sha256 "$final_prompts" "$expected_final_sha256"
require_sha256 "$monitor_prompts" "$expected_monitor_sha256"

train_seed() {
  local seed="$1"
  local label="D2_helpfulness_safety_conflict_seed${seed}"
  local output_dir="$run_root/runs/$label"
  local manifest="$output_dir/manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed reconstruction: $label"
    return
  fi
  echo "START checkpoint reconstruction: $label"
  python scripts/train_qwen3_lora_dpo.py \
    --model_name "$model_path" \
    --preferences_path "$preferences" \
    --prompts_path "$final_prompts" \
    --output_dir "$output_dir" \
    --max_steps 100 \
    --checkpoint_steps "${checkpoint_steps[@]}" \
    --save_initial_adapter \
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
    2>&1 | tee "$run_root/logs/reconstruct_${label}.log"
  is_complete_manifest "$manifest"
}

run_in_batches train_seed "$training_workers" "${seeds[@]}"

python - "$run_root" "$sis1_training_summary" "${seeds[*]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sis1 = json.loads(pathlib.Path(sys.argv[2]).read_text())
seeds = [int(value) for value in sys.argv[3].split()]
expected_checkpoints = {"0", "10", "20", "30", "50", "75", "100"}
summary = {
    "status": "complete",
    "stage": "SIS-2A checkpoint reconstruction",
    "disclosure": "retrained after the original instance data disk was unavailable",
    "runs": {},
}
for seed in seeds:
    label = f"D2_helpfulness_safety_conflict_seed{seed}"
    manifest = json.loads((root / "runs" / label / "manifest.json").read_text())
    checkpoints = set(manifest["artifacts"]["checkpoints"])
    if manifest["status"] != "complete" or checkpoints != expected_checkpoints:
        raise SystemExit(f"Incomplete reconstructed checkpoints: {label}")
    original = sis1["runs"][label]
    schedule_match = (
        manifest["training_randomness"]["schedule_sha256"]
        == original["schedule_sha256"]
    )
    if not schedule_match:
        raise SystemExit(f"Schedule hash differs from SIS-1: {label}")
    reconstructed_kl = manifest["realized_kl"]["teacher_forced_mean_token_kl"]
    summary["runs"][label] = {
        "seed": seed,
        "schedule_sha256": manifest["training_randomness"]["schedule_sha256"],
        "schedule_matches_sis1": schedule_match,
        "sis1_teacher_forced_mean_token_kl": original[
            "teacher_forced_mean_token_kl"
        ],
        "reconstructed_teacher_forced_mean_token_kl": reconstructed_kl,
        "kl_delta_vs_sis1": (
            reconstructed_kl - original["teacher_forced_mean_token_kl"]
        ),
        "wall_seconds": manifest["total_wall_seconds"],
        "peak_vram_bytes": manifest["peak_vram_bytes"],
        "checkpoint_steps": sorted(checkpoints, key=int),
    }
(root / "manifests" / "reconstruction_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
PY

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_root/RECONSTRUCTION_COMPLETE"
echo "SIS-2A checkpoint reconstruction complete: $run_root"
