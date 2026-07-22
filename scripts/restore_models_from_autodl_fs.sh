#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-/root/autodl-fs/model-archives}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
MODEL_DIR="${DATA_ROOT}/models"
LOG_DIR="${DATA_ROOT}/model-restore"

R0_ARCHIVE="${ARCHIVE_DIR}/r0-models-20260722.tar.zst"
HARMBENCH_ARCHIVE="${ARCHIVE_DIR}/harmbench-mistral-7b-val-cls-20260722.tar.zst"
R0_SHA256="0bb1048dbfd102eeab0fcf309d1b7abeafe28ce5fafccd54ad9658c5d162eba8"
HARMBENCH_SHA256="94f61c4578d62b40de93580dacbf8d6fb2251d7852a5531a9aa07a562b963166"

mkdir -p "${MODEL_DIR}" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/restore.log") 2>&1

finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "${rc}" > "${LOG_DIR}/restore.exit_code"
    echo "MODEL_RESTORE_EXIT_CODE=${rc}"
    exit "${rc}"
}
trap finish EXIT

for archive in "${R0_ARCHIVE}" "${HARMBENCH_ARCHIVE}"; do
    if [[ ! -s "${archive}" ]]; then
        echo "Missing model archive: ${archive}" >&2
        exit 2
    fi
done

if [[ "${VERIFY_ARCHIVES:-1}" == "1" ]]; then
    printf '%s  %s\n' "${R0_SHA256}" "${R0_ARCHIVE}" | sha256sum -c -
    printf '%s  %s\n' "${HARMBENCH_SHA256}" "${HARMBENCH_ARCHIVE}" | sha256sum -c -
fi

echo "Restoring base, embedding, and Granite models to ${DATA_ROOT}"
tar --zstd -xf "${R0_ARCHIVE}" -C "${DATA_ROOT}"

echo "Restoring HarmBench classifier to ${MODEL_DIR}"
tar --zstd -xf "${HARMBENCH_ARCHIVE}" -C "${MODEL_DIR}"

required_files=(
    "${MODEL_DIR}/Qwen3-1.7B/model.safetensors.index.json"
    "${MODEL_DIR}/granite-guardian-3.1-2b/model.safetensors.index.json"
    "${MODEL_DIR}/all-MiniLM-L6-v2/model.safetensors"
    "${MODEL_DIR}/HarmBench-Mistral-7b-val-cls/model.safetensors.index.json"
)
for path in "${required_files[@]}"; do
    if [[ ! -s "${path}" ]]; then
        echo "Restored model is incomplete: ${path}" >&2
        exit 3
    fi
done

if find "${MODEL_DIR}" -type f -name '*.incomplete' -print -quit | grep -q .; then
    echo "Incomplete download marker found under ${MODEL_DIR}" >&2
    exit 3
fi

du -sh "${MODEL_DIR}"
echo "Model restore completed successfully."
