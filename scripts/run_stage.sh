#!/usr/bin/env bash
# ==============================================================================
# Staged experiment dispatcher
# ==============================================================================
# 按 "小模型 -> 大模型" 阶梯统一入口. 自动选择配置/单卡 or 多卡启动方式.
#
# Usage:
#   bash scripts/run_stage.sh <stage> <experiment> [EXTRA_ARGS...]
#
#   <stage>      : s0 | s1 | s2 | s3
#   <experiment> : exp1 | exp2 | exp3 | exp4 | exp5
#
# Examples:
#   bash scripts/run_stage.sh s0 exp1          # 0.5B 单卡冒烟
#   bash scripts/run_stage.sh s1 exp1          # 2B 单卡现象确立
#   bash scripts/run_stage.sh s2 exp1          # 8B 4卡主实验
#   bash scripts/run_stage.sh s3 exp1          # 13B+ 多卡确认
#
# 阶段->配置/GPU 的映射在下方 case 中, 与 configs/stages/*.yaml 一一对应.
# ==============================================================================

set -euo pipefail

STAGE="${1:-}"
EXPERIMENT="${2:-exp1}"
shift 2 2>/dev/null || true
EXTRA_ARGS=("$@")

if [[ -z "${STAGE}" ]]; then
    echo "ERROR: missing <stage>. Usage: bash scripts/run_stage.sh <s0|s1|s2|s3> <exp1..exp5>" >&2
    exit 1
fi

# -- Map stage -> config + GPU count + launch mode --
case "${STAGE}" in
    s0)
        CONFIG_PATH="configs/stages/s0_smoke.yaml"
        NUM_GPUS=1
        LAUNCH_MODE="single"   # 单卡 python -m, QLoRA
        ;;
    s1)
        CONFIG_PATH="configs/stages/s1_small.yaml"
        NUM_GPUS=1
        LAUNCH_MODE="single"   # 单卡 (4090 QLoRA 或 A100-40G 全参)
        ;;
    s2)
        CONFIG_PATH="configs/stages/s2_main.yaml"
        NUM_GPUS=4
        LAUNCH_MODE="distributed"   # torchrun + ZeRO-3
        ;;
    s3)
        CONFIG_PATH="configs/stages/s3_large.yaml"
        NUM_GPUS=4             # 70B 时手动覆盖为 8: NUM_GPUS=8 bash ...
        LAUNCH_MODE="distributed"
        ;;
    *)
        echo "ERROR: unknown stage '${STAGE}' (expected s0|s1|s2|s3)" >&2
        exit 1
        ;;
esac

# 允许通过环境变量覆盖 GPU 数 (例如 70B: NUM_GPUS=8)
NUM_GPUS="${NUM_GPUS_OVERRIDE:-${NUM_GPUS}}"
MASTER_PORT="${MASTER_PORT:-29500}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="outputs/${STAGE}_${EXPERIMENT}/${TIMESTAMP}"

echo "=============================================="
echo " Staged PCE Run"
echo "=============================================="
echo " Stage     : ${STAGE}"
echo " Experiment: ${EXPERIMENT}"
echo " Config    : ${CONFIG_PATH}"
echo " GPUs      : ${NUM_GPUS} (${LAUNCH_MODE})"
echo " Output    : ${RUN_DIR}"
echo "=============================================="

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/pce_logs" \
         "${RUN_DIR}/safety_reports" "${RUN_DIR}/diversity_logs"
cp "${CONFIG_PATH}" "${RUN_DIR}/config.yaml"

# System info snapshot
{
    echo "Stage: ${STAGE} | Experiment: ${EXPERIMENT} | ${TIMESTAMP}"
    nvidia-smi 2>/dev/null || echo "No NVIDIA GPUs detected"
    python --version
} > "${RUN_DIR}/system_info.txt"

# -- Build train launcher according to launch mode --
if [[ "${LAUNCH_MODE}" == "single" ]]; then
    TRAIN_LAUNCHER=(python -m src.scripts.train_dpo)
else
    TRAIN_LAUNCHER=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m src.scripts.train_dpo)
fi

# ==============================================================================
# Dispatch by experiment
# ==============================================================================
case "${EXPERIMENT}" in
    exp1)
        echo "[exp1] DPO training + PCE trajectory"
        "${TRAIN_LAUNCHER[@]}" \
            --config "${CONFIG_PATH}" \
            --output_dir "${RUN_DIR}/checkpoints" \
            --wandb_run_name "${STAGE}_${EXPERIMENT}_${TIMESTAMP}" \
            --attack_prompts_path "data/pce_eval_set.jsonl" \
            "${EXTRA_ARGS[@]}"

        echo "[exp1] Post-hoc PCE sweep over checkpoints"
        for CKPT in $(find "${RUN_DIR}/checkpoints" -name "checkpoint-*" -type d | sort -V); do
            STEP=$(basename "${CKPT}" | sed 's/checkpoint-//')
            python -m src.scripts.compute_pce \
                --model_name "${CKPT}" \
                --prompts_path "data/pce_eval_set.jsonl" \
                --output_path "${RUN_DIR}/pce_logs/pce_step_${STEP}.json" \
                --config "${CONFIG_PATH}"
        done
        ;;

    exp2)
        echo "[exp2] Passive exploitation: attack frozen checkpoints"
        echo "  NOTE: set CKPT_DIR env to a frozen exp1 checkpoint dir."
        python -m src.scripts.safety_eval \
            --model_name "${CKPT_DIR:?set CKPT_DIR to a trained checkpoint}" \
            --prompts_path "data/attack_prompts.jsonl" \
            --output_path "${RUN_DIR}/safety_reports/attack_asr.json" \
            --config "${CONFIG_PATH}" \
            "${EXTRA_ARGS[@]}"
        ;;

    exp3)
        echo "[exp3] Active collapse induction: poisoned DPO"
        "${TRAIN_LAUNCHER[@]}" \
            --config "${CONFIG_PATH}" \
            --output_dir "${RUN_DIR}/checkpoints" \
            --wandb_run_name "${STAGE}_${EXPERIMENT}_${TIMESTAMP}" \
            --enable_poisoning \
            --attack_prompts_path "data/pce_eval_set.jsonl" \
            "${EXTRA_ARGS[@]}"
        ;;

    exp4)
        echo "[exp4] Standard practice audit: direct PCE on released models (no training)"
        python -m src.scripts.compute_pce \
            --model_name "${AUDIT_MODEL:?set AUDIT_MODEL to a released DPO model id}" \
            --prompts_path "data/pce_eval_set.jsonl" \
            --output_path "${RUN_DIR}/pce_logs/audit_pce.json" \
            --config "${CONFIG_PATH}" \
            "${EXTRA_ARGS[@]}"
        ;;

    exp5)
        echo "[exp5] Defense: entropy-regularized DPO"
        "${TRAIN_LAUNCHER[@]}" \
            --config "${CONFIG_PATH}" \
            --output_dir "${RUN_DIR}/checkpoints" \
            --wandb_run_name "${STAGE}_${EXPERIMENT}_${TIMESTAMP}" \
            --enable_entropy_regularization \
            --attack_prompts_path "data/pce_eval_set.jsonl" \
            "${EXTRA_ARGS[@]}"

        echo "[exp5] Post-hoc PCE sweep over checkpoints"
        for CKPT in $(find "${RUN_DIR}/checkpoints" -name "checkpoint-*" -type d | sort -V); do
            STEP=$(basename "${CKPT}" | sed 's/checkpoint-//')
            python -m src.scripts.compute_pce \
                --model_name "${CKPT}" \
                --prompts_path "data/pce_eval_set.jsonl" \
                --output_path "${RUN_DIR}/pce_logs/pce_step_${STEP}.json" \
                --config "${CONFIG_PATH}"
        done
        ;;

    *)
        echo "ERROR: unknown experiment '${EXPERIMENT}' (expected exp1..exp5)" >&2
        exit 1
        ;;
esac

echo "=============================================="
echo " Done. Results in: ${RUN_DIR}"
echo "=============================================="
