#!/usr/bin/env bash
set -euo pipefail

dataset_root="${SIS_DATASET_ROOT:-/root/autodl-tmp/datasets}"
archive_dir="${SIS_DATASET_ARCHIVE_DIR:-/root/autodl-fs/dataset-archives}"
archive_name="${SIS_DATASET_ARCHIVE_NAME:-sis1-r2-sources-20260723.tar.zst}"
maximum_shared_bytes="${SIS_MAX_SHARED_BYTES:-21474836480}"
archive_path="$archive_dir/$archive_name"
partial_path="$archive_path.partial"

files=(
  r2_raw/ultrafeedback/evol_instruct.jsonl
  r2_raw/ultrafeedback/flan.jsonl
  r2_raw/ultrafeedback/truthful_qa.jsonl
  r2_raw/ultrafeedback/ultrachat.jsonl
  r2_raw/pku_saferlhf/data/Alpaca3-8B/train.jsonl
  r2_eval_sources/harmbench/harmbench_behaviors_text_all.csv
  r2_eval_sources/xstest/xstest_prompts.csv
  r2_eval_sources/jailbreakbench/harmful-behaviors.csv
  r2_prepared_v2/preferences/D0_clean_helpfulness.jsonl
  r2_prepared_v2/preferences/D1_clean_safety.jsonl
  r2_prepared_v2/preferences/D2_helpfulness_safety_conflict.jsonl
  r2_prepared_v2/preferences/D3_poison_05.jsonl
  r2_prepared_v2/preferences/D4_full_refusal_suppression.jsonl
  r2_prepared_v2/split_manifest.json
  r3_prepared/eval/harmbench_stratified_100.jsonl
  r3_prepared/eval/xstest_full.jsonl
  r3_prepared/eval/harmbench_monitor_30.jsonl
  r3_prepared/eval/sis1_monitor_manifest.json
)

for relative_path in "${files[@]}"; do
  if [[ ! -s "$dataset_root/$relative_path" ]]; then
    echo "Required dataset file is missing: $dataset_root/$relative_path" >&2
    exit 1
  fi
done

mkdir -p "$archive_dir"
if [[ -s "$archive_path" && -s "$archive_path.sha256" ]]; then
  (cd "$archive_dir" && sha256sum -c "$archive_name.sha256")
  echo "Dataset archive already exists and verifies: $archive_path"
  exit 0
fi

tar --use-compress-program="zstd -T0 -10" \
  -C "$dataset_root" \
  -cf "$partial_path" \
  "${files[@]}"

shared_bytes="$(du -sbL /root/autodl-fs | awk '{print $1}')"
if (( shared_bytes > maximum_shared_bytes )); then
  rm -f "$partial_path"
  echo "Refusing to exceed shared-storage limit: $shared_bytes > $maximum_shared_bytes" >&2
  exit 1
fi

mv "$partial_path" "$archive_path"
(cd "$archive_dir" && sha256sum "$archive_name" > "$archive_name.sha256")
printf '%s\n' "${files[@]}" > "$archive_path.contents.txt"
(cd "$archive_dir" && sha256sum -c "$archive_name.sha256")

echo "Saved minimal SIS-1 dataset archive: $archive_path"
du -sh "$archive_path"
du -shL /root/autodl-fs
