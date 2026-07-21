# 仓库迁移说明：从本地 PCE smoke 到租卡分阶段实验

## 当前有效入口

- `PLAN.md`
- `configs/current/`
- `environment-rental.yml`
- `scripts/train_qwen3_lora_dpo.py`
- `scripts/check_rental_environment.py`
- `src/metrics/risk_diversity.py`

## 保留但降级为历史内容

- `configs/stages/`：旧 S0–S3 PCE/大模型计划；
- `configs/default_config.yaml`：旧 Llama-2/PCE 默认配置；
- `environment-qwen3-local.yml`：旧 RTX 4060 环境；
- `scripts/local_*`：历史本地实验；
- `src/metrics/pce.py`、`src/attacks/*`：旧假设与攻击 smoke；
- `docs/local_*` 与旧 protocol：实验史和负结果记录。

这些内容不删除，以保留可追溯性，但新论文结果不应从旧 stage 配置直接生成。

## 后续仍需实现

1. PKU-SafeRLHF D0–D3 固定子集构建与去重脚本；
2. HarmBench/JailbreakBench/XSTest 标准数据适配器；
3. WildGuard/StrongREJECT 统一输出 schema；
4. embedding 粗筛 + LLM 成对语义裁决；
5. hierarchical bootstrap 与非劣检验；
6. run manifest 自动记录 Git/GPU/软件/吞吐。
