#!/usr/bin/env bash
set -euo pipefail

repo_dir="${SIS3_REPO_DIR:-/root/risk-diversity-decoupling}"
model_path="${SIS3_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
preferences="${SIS3_PREFERENCES:-/root/autodl-tmp/datasets/r2_prepared_v2/preferences/D2_helpfulness_safety_conflict.jsonl}"
eval_prompts="${SIS3_EVAL_PROMPTS:-/root/autodl-tmp/datasets/r3_prepared/eval/harmbench_stratified_100.jsonl}"
sis2_unseen="${SIS2_UNSEEN_ROOT:-/root/autodl-tmp/experiments/sis2_early_prediction_20260723/unseen}"
run_root="${SIS3_RUN_ROOT:-/root/autodl-tmp/experiments/sis3_trajectory_reversal_20260723}"
dense_root="$run_root/dense"
training_workers="${SIS3_TRAINING_WORKERS:-2}"
seeds=(52 54 55 57)
checkpoint_steps=(30 35 40 45 50 55 60 70 80 90 100)

expected_preferences_sha256="08405eae95a5f049e8d2c38dffeb845a03a530377da2b536be9aa056c8c1b071"
expected_eval_sha256="fffe21a839151b5330a4e97d5f35122acd6816e105c15558ae8d92edada146fb"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$repo_dir"
mkdir -p "$dense_root"/{runs,logs,manifests}

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

precheck="$run_root/metrics/sis3a_precheck_summary.json"
require_file "$precheck"
advance="$(
  python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["decision"]["advance_to_dense_rerun"]).lower())' \
    "$precheck"
)"
[[ "$advance" == "true" ]] || {
  echo "SIS-3A did not authorize dense training." >&2
  exit 1
}
require_file "$model_path/config.json"
require_file "$preferences"
require_file "$eval_prompts"
require_file "$sis2_unseen/manifests/training_summary.json"
require_sha256 "$preferences" "$expected_preferences_sha256"
require_sha256 "$eval_prompts" "$expected_eval_sha256"

train_seed() {
  local seed="$1"
  local label="D2_helpfulness_safety_conflict_seed${seed}"
  local output_dir="$dense_root/runs/$label"
  local manifest="$output_dir/manifest.json"
  if is_complete_manifest "$manifest"; then
    echo "SKIP completed SIS-3B dense training: $label"
    return
  fi
  echo "START SIS-3B dense training: $label"
  python scripts/train_qwen3_lora_dpo.py \
    --model_name "$model_path" \
    --preferences_path "$preferences" \
    --prompts_path "$eval_prompts" \
    --output_dir "$output_dir" \
    --max_steps 100 \
    --checkpoint_steps "${checkpoint_steps[@]}" \
    --save_initial_adapter \
    --trace_every_step \
    --layer_update_norms \
    --save_training_state \
    --fixed_probe_size 8 \
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
    2>&1 | tee "$dense_root/logs/train_${label}.log"
  is_complete_manifest "$manifest"
}

run_in_batches train_seed "$training_workers" "${seeds[@]}"

python - "$dense_root" "$sis2_unseen/manifests/training_summary.json" "${seeds[*]}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
original = json.loads(pathlib.Path(sys.argv[2]).read_text())
seeds = [int(value) for value in sys.argv[3].split()]
expected_checkpoints = {"0", "30", "35", "40", "45", "50", "55", "60", "70", "80", "90", "100"}
expected_states = expected_checkpoints - {"0"}
summary = {"status": "complete", "stage": "SIS-3B dense instrumented training", "runs": {}}
for seed in seeds:
    label = f"D2_helpfulness_safety_conflict_seed{seed}"
    manifest = json.loads((root / "runs" / label / "manifest.json").read_text())
    checkpoints = set(manifest["artifacts"]["checkpoints"])
    states = set(manifest["artifacts"]["training_states"])
    if (
        manifest["status"] != "complete"
        or checkpoints != expected_checkpoints
        or states != expected_states
    ):
        raise SystemExit(f"Incomplete dense checkpoints/states: {label}")
    prior = original["runs"][label]
    current_kl = manifest["realized_kl"]["teacher_forced_mean_token_kl"]
    schedule_match = (
        manifest["training_randomness"]["schedule_sha256"]
        == prior["schedule_sha256"]
    )
    summary["runs"][label] = {
        "seed": seed,
        "schedule_sha256": manifest["training_randomness"]["schedule_sha256"],
        "schedule_matches_original": schedule_match,
        "original_realized_kl": prior["teacher_forced_mean_token_kl"],
        "dense_realized_kl": current_kl,
        "kl_absolute_difference": abs(
            current_kl - prior["teacher_forced_mean_token_kl"]
        ),
        "wall_seconds": manifest["total_wall_seconds"],
        "peak_vram_bytes": manifest["peak_vram_bytes"],
        "checkpoint_steps": sorted(checkpoints, key=int),
        "trace_steps": len(
            json.loads(
                (root / "runs" / label / "training_trace.json").read_text()
            )
        ),
    }
    if not schedule_match:
        raise SystemExit(f"Dense schedule differs from original: {label}")
(root / "manifests" / "training_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
PY

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$dense_root/TRAINING_COMPLETE"
echo "SIS-3B dense instrumented training complete."
