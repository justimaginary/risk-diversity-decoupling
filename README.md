# Risk–Diversity Decoupling under Preference Optimization

本仓库研究偏好优化（主要是 DPO）如何同时改变语言模型的安全风险与输出多样性。

最初的研究假设是 DPO 可能把输出压缩到稳定的危险模式，形成 Preference Collapse
Exploitability（PCE）。现有实验不支持这一强结论。Qwen3 先导结果呈现了一个更值得严格检验的现象：

> 安全分类器判定的风险上升，但输出确定性下降、模式熵上升；风险没有表现为单一重复模式，而可能迁移到多个回答模式。

当前问题是：这种“风险—多样性解耦”在什么训练构造、模型和提示条件下出现，能否在公开、未见提示、多个训练 seed 和多个安全判别器下复现？

这仍是待验证的研究假设，不是已经成立的普遍规律。后续工作的唯一执行入口是 [PLAN.md](PLAN.md)。

## 当前结论

已有证据方向性支持：

- DPO 可以显著改变安全相关回答倾向；
- Qwen3-1.7B 和 Qwen3-4B 的受控先导中，Guardian 风险上升与 determinism 下降、文本模式熵上升同时出现；
- 输出集中本身不等于风险上升，安全拒答模板是明确反例；
- 聚合平均会掩盖显著的 prompt-level 异质性。

当前证据不支持：

- “DPO 普遍导致可利用的模式坍缩漏洞”；
- “模型越大，坍缩信号越稳定”；
- 合成数据上的现象已经泛化到所有未见风险提示；
- 用一个 Guardian judge 或 TF-IDF 聚类给出最终安全与语义结论；
- 历史低比例投毒实验具有稳定剂量效应。

项目现在处于“已有现象重新审计与严格复验准备”阶段。近期优先复用已有 raw outputs，接入第二安全 judge、StrongREJECT、语义聚类、风险熵和分层 bootstrap，而不是立即扩大训练规模。

## 已完成的核心实验

### Qwen3-1.7B 受控先导

设置：LoRA-DPO，300 steps，独立训练 seeds 42/43，10 条本地提示，每题 16 次采样。

| 指标 | 训练后减训练前 |
| --- | ---: |
| Determinism | -0.0531，CI [-0.1000, -0.0125] |
| Mode Entropy | +0.1677，CI [+0.0301, +0.3117] |
| Granite Guardian Yes | +0.4781，CI [+0.3094, +0.6500] |
| Guardian score | +0.4359，CI [+0.2945, +0.5874] |
| Refusal rate | 0.575 → 0.156 |

这是新问题最强的先导证据，但训练构造受控、提示规模小，不能作为外部泛化结论。

### Qwen3-4B 方向复核

设置：LoRA-DPO，100 steps，seed 42，10 条本地提示，每题 16 次采样。

| 指标 | 训练后减训练前 |
| --- | ---: |
| Determinism | -0.0187 |
| Mode Entropy | +0.0701 |
| Granite Guardian Yes | +0.4688 |

更大模型上方向未反转，但只有一个 seed 且训练较短，不构成跨规模规律。

### 公开未见提示与控制

- **30 条 AdvBench held-out**：两个训练 seed 的 determinism 均下降、entropy 均上升；Guardian Yes 平均增加 `0.0307`，明显弱于本地 first-10；
- **50 条本地提示异质性**：100 个 prompt-run 比较中 34 pass、33 mixed、33 fail；
- **安全拒答模板控制**：回答更集中但风险下降，说明风险和多样性必须分轴测量；
- **Guardian 固定回答控制**：评分对 response 内容敏感，但仍需第二 judge 和人工审核；
- **80 条 AdvBench gate**：已冻结零重叠题集并完成 Qwen3-1.7B baseline 的 `80 × 32 = 2,560` 次采样；训练后严格复验尚未完成；
- **Poison/CAR smoke**：clean、1% 和 5% 没有预期剂量关系，不再作为主线证据。

完整历史、负结果和证据边界见 [docs/complete_experiment_report.md](docs/complete_experiment_report.md) 与 [docs/new_idea_report.md](docs/new_idea_report.md)。

## 数据

### 受控训练数据

- `data/local_short_template_preferences.jsonl`：20 条拒答抑制型合成偏好对；
- `data/local_refusal_template_preferences.jsonl`：安全拒答模板对照；
- `data/local_neutral_boundary_preferences.jsonl`：中性边界对照；
- `data/local_collapse_proxy_preferences.jsonl`、`data/local_uniform_collapse_preferences.jsonl`：历史 PCE smoke；
- `data/local_poison_smoke_poison*.jsonl`：历史注入 smoke。

这些数据用于可控干预，不代表真实平台数据，也不应作为外部有效性的唯一证据。

### 评估提示

- `data/attack_prompts.jsonl`：20 条本地风险提示；
- `data/attack_prompts_fallback_heldout_30.jsonl`：本地异质性诊断提示；
- `data/advbench_harmful_behaviors_all.jsonl`：AdvBench 520 条源数据；
- `data/advbench_s0_1_heldout_30.jsonl`：固定随机种子的 30 条零重叠提示；
- `data/advbench_rdi_heldout_80_seed20260704.jsonl`：下一阶段冻结的 80 条零重叠提示。

后续正式实验将引入 PKU-SafeRLHF 训练子集，以及 HarmBench、JailbreakBench、XSTest 等公开评测。训练数据与评测 prompt 必须去重并严格隔离。

## 方法与代码

### 训练

Qwen3 主线使用 LoRA-DPO，只保存 adapter，不修改或提交完整基座模型。Qwen3 生成必须关闭 thinking mode。主要实现：

- `scripts/local_qwen3_lora_dpo.py`：Qwen3 LoRA-DPO 与 checkpoint 评估；
- `src/training/dpo_trainer.py`：通用 DPO trainer；
- `src/training/er_dpo_trainer.py`：历史 entropy-regularized DPO；
- `src/attacks/collapse_accelerator.py`：历史 collapse/poison 构造。

### 当前多样性代理

现有本地脚本使用 TF-IDF、余弦距离和 DBSCAN：

```text
determinism = largest_cluster_size / number_of_responses
mode_entropy = -sum(p_i * log(p_i))
```

它主要测量词汇和短语相似性，不等价于语义模式。正式实验将使用 sentence embeddings、HDBSCAN、Vendi Score、大模型成对语义裁决和人工抽审。

### 当前风险审计

`scripts/audit_granite_guardian_outputs.py` 使用 Granite Guardian 对原始 prompt 与 response 做风险判断。Guardian 是模型裁判，不是人工真实标签。正式实验将增加 WildGuard、StrongREJECT 和人工分歧审核。

## 环境

推荐 Python 3.10+。新设备优先使用独立环境：

```bash
conda env create -f environment-qwen3-local.yml
conda activate qwen3-dpo
```

详细安装、模型路径和旧环境兼容说明见 [docs/new_device_environment.md](docs/new_device_environment.md)。基座模型、Conda 环境、缓存和完整 checkpoint 不进入 Git。

## 常用命令

合成 PCE smoke：

```bash
python scripts/local_pce_smoke.py \
  --mode synthetic \
  --output_dir outputs/smoke_test
```

汇总本地 gate：

```bash
python scripts/summarize_local_gate.py \
  outputs/local_smoke/<run-directory> \
  --bootstrap_samples 5000 \
  --bootstrap_seed 2026
```

重新评估已有 checkpoint：

```bash
python scripts/reevaluate_checkpoints.py \
  --baseline_checkpoint <baseline> \
  --final_checkpoint <final> \
  --prompts_file data/attack_prompts.jsonl \
  --num_prompts 10 \
  --num_samples 16 \
  --output_dir outputs/reevaluation
```

测试与静态检查：

```bash
pytest
ruff check src scripts tests
mypy src
```

当前机器的默认 shell 环境未必已经安装这些开发工具；运行前应激活项目环境。

## 仓库结构

```text
configs/      实验配置与冻结的数据选择参数
data/         小型、可追踪的训练和评估数据
docs/         实验报告、协议、环境与开题材料
latex/        中英文论文源文件
papers/       相关文献与本地编译论文
scripts/      本地训练、评估、审计和汇总入口
src/          指标、训练、攻击与评估模块
tests/        自动测试
PLAN.md       唯一后续实验计划
```

`outputs/`、模型权重、缓存、设备迁移压缩包和 PPT 均保留在本地，不进入远端仓库。

## 关键文档

- [PLAN.md](PLAN.md)：唯一后续执行计划与 Stop/Go 标准；
- [docs/complete_experiment_report.md](docs/complete_experiment_report.md)：完整实验历史；
- [docs/new_idea_report.md](docs/new_idea_report.md)：新研究问题、创新边界与开题表述；
- [docs/qwen3_prompt_reasonableness_gate.md](docs/qwen3_prompt_reasonableness_gate.md)：80 条新题 baseline gate；
- [docs/new_device_environment.md](docs/new_device_environment.md)：复现实验环境；
- `docs/*_protocol.md`：单项历史实验的冻结配置与解释边界。

## 版本控制约定

- 本仓库为个人研究仓库，允许直接在 `master` 上维护；
- 默认只在本地 commit，不主动 push；
- `.ppt`/`.pptx` 是本地汇报材料，不提交到远端；
- 实验代码和文档改动应小步提交，提交信息不添加共同作者；
- 新正式实验必须记录 Git commit、完整配置、软件版本、GPU、wall-clock、peak VRAM 和失败信息。
