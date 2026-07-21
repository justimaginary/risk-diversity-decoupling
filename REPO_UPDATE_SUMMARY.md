# Repository update summary

本次修改将仓库从“本地 RTX 4060 + PCE/投毒优先”的旧执行方式，迁移为“风险—多样性解耦 + 数据条件轴 + 逐级租卡”的当前计划。

## 已完成

- 重写 `README.md` 与 `PLAN.md`，明确研究目的、RQ1–RQ4、D0–D4 数据条件和结论边界；
- 移除当前计划中的本地 RTX 4060 计算任务；
- 建立 R0–R6 租卡 Gate：24GB 低价验证 → 4090 主实验 → 48GB 复核 → 80GB 可选确认；
- 添加 `docs/rental_compute_protocol.md`；
- 添加 `configs/current/` 的 7 个阶段 manifest；
- 添加租卡环境检查与当前训练入口；
- 添加 `H_harm`、`H_risk`、effective modes、Any-Harm@k 等核心指标实现及测试；
- 更新依赖文件，取消 FlashAttention、DeepSpeed、vLLM 的强制安装；
- 添加 GitHub Actions CPU 单元测试；
- 将旧 PCE stage、旧本地环境和旧 orchestration 明确标为 legacy；
- 更新 `AGENTS.md`/`CLAUDE.md`，防止代码代理继续执行已被否定的旧研究计划。

## 验证

- `configs/current/*.yaml` 均已通过 YAML 解析；
- 新增 Python 文件均已通过语法解析；
- `tests/test_risk_diversity.py`：新增核心指标与空风险边界测试；
- 原有完整测试需要安装仓库全部 ML 依赖后在 CI/租卡环境运行。

## 下一步代码工作

- 构建 PKU-SafeRLHF D0–D3 固定子集；
- 接入 HarmBench/JailbreakBench/XSTest；
- 统一 WildGuard/StrongREJECT 输出 schema；
- 实现语义 pair 裁决与 hierarchical bootstrap；
- 将当前 YAML manifest 接入自动 orchestration。
