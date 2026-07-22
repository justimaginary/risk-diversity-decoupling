# Risk–Diversity Decoupling under Preference Optimization

本仓库研究偏好后训练如何同时改变语言模型的安全风险与输出分布结构。

> 核心问题不是“模型是否只会重复同一个危险答案”，而是风险总量上升时，风险是否可能分散到多个不同语义和行为模式中。

当前先导结果来自受控拒答抑制偏好干预。它们支持进一步严格检验，但不能证明干净 DPO 数据天然增加风险，也不能证明低比例投毒已经稳定奏效。当前唯一执行入口是 [PLAN.md](PLAN.md)。

## 当前研究方向

训练数据按连续轴组织：

1. 干净帮助性偏好；
2. 干净安全性偏好；
3. 帮助性—安全性冲突偏好；
4. 1%/5%/10% 低比例有害偏好污染；
5. 全量拒答抑制强干预。

项目首先判断风险—多样性解耦在哪些数据条件下出现，再比较 DPO、chosen-only SFT 和其他偏好优化方法。投毒只是一个实验条件，不是整个研究问题。

## 已有先导证据

### Qwen3-1.7B

LoRA-DPO，300 steps，seeds 42/43，10 prompts × 16 samples：

| 指标 | 训练后减训练前 |
| --- | ---: |
| Determinism | -0.0531 |
| Mode Entropy | +0.1677 |
| Granite Guardian Yes | +0.4781 |
| Guardian score | +0.4359 |

### Qwen3-4B

100 steps、单 seed 的方向复核中，风险上升而历史聚类熵上升。该结果不能视为跨规模规律。

### R1 公开 benchmark pilot

Qwen3-1.7B，HarmBench 50 prompts × 16 samples，XSTest 450 prompts × 4 samples：

| 指标 | Base | LoRA | 变化 |
| --- | ---: | ---: | ---: |
| Granite Guardian 风险率 | 29.38% | 88.12% | +58.75 pp |
| HarmBench Mistral 风险率 | 15.88% | 55.88% | +40.00 pp |
| XSTest 安全提示拒答率 | 3.70% | 0.10% | -3.60 pp |
| XSTest 不安全提示拒答率 | 45.75% | 0.00% | -45.75 pp |

两个独立安全 judge 均显示风险明显上升，且 XSTest 不安全提示拒答几乎消失。与此同时，embedding + HDBSCAN 的语义熵和有效模式数没有下降，Vendi 仅小幅下降；这支持继续研究风险与多样性解耦，但仍只是单 checkpoint pilot，不能替代多 seed、数据轴和人工抽审。脱敏聚合结果见 [`experiments/r1_public_pilot_20260722/metrics/r1_summary.json`](experiments/r1_public_pilot_20260722/metrics/r1_summary.json)。

### R2 数据轴单 seed pilot

Qwen3-1.7B、D0–D4 各 200 pairs / 300 steps、HarmBench 50 × 16、完整 XSTest × 4 已完成。D1 在 HarmBench、Granite、StrongREJECT 和 Any-Harm@16 上一致降低风险；D3 5% contamination 没有显示相对 clean 条件的稳定风险增量；D4 几乎消除不安全提示拒绝，但内容风险仍低于 Base，说明拒绝率不能替代风险评估。

D2 conflict 出现严重语言/质量漂移和 judge 分歧：HarmBench 2.25%、Granite 53.50%、StrongREJECT mean 0.047，且 69.88% 输出含至少 20% 非 ASCII 字母。依据 Stop/Go 规则，当前 **暂缓 R3**，先修复 D2 数据/KL/输出质量并补正式人工抽审。完整脱敏结果和 Gate 决策见 [`experiments/r2_data_axis_20260722/RESULTS.md`](experiments/r2_data_axis_20260722/RESULTS.md)。

### 证据边界

- 30 条 AdvBench held-out 上风险增幅明显缩小；
- 安全拒答模板证明“更集中”不等于“更危险”；
- 历史 poison/CAR smoke 没有稳定剂量关系；
- Granite Guardian 和 TF-IDF + DBSCAN 都只能作为先导测量。

完整历史见 `docs/complete_experiment_report.md`。

## 当前评测框架

### 风险

- Macro Harm Rate；
- StrongREJECT；
- Any-Harm@k / Max-Risk@k；
- Jailbreak ASR；
- XSTest over-refusal；
- Granite Guardian + WildGuard/HarmBench classifier + 人工抽审。

### 分布结构

- Semantic Entropy；
- Effective Semantic Modes；
- Vendi Score；
- Harmful Semantic Entropy；
- Risk-Mass Entropy。

TF-IDF、DBSCAN、Distinct-n 和 Self-BLEU 只保留作历史对照。

## 租卡策略

本项目不再使用本地 RTX 4060。租卡按 Gate 逐级升级：

| 阶段 | GPU | 目的 |
| --- | --- | --- |
| R0–R2 | 最便宜的 24GB 卡，优先 3090/4090 | 环境、公开 benchmark pilot、数据轴单 seed 筛选 |
| R3–R4 | RTX 4090 24GB | Qwen3-1.7B 多 seed 主实验和方法对照 |
| R5 | L40S/A6000 48GB | Qwen3-4B 与更大 judge 复核 |
| R6 | A100/H100 80GB | 仅在前面证据稳定后做可选 8B/规模确认 |

先完成 R0–R2 的 20–40 卡时低成本决策，再决定是否进入高规格实验。详细卡时与 Stop/Go 标准见 [PLAN.md](PLAN.md) 和 [docs/rental_compute_protocol.md](docs/rental_compute_protocol.md)。

## 快速开始

租用实例建议选择 Linux + 已安装兼容 CUDA/PyTorch 的镜像。

```bash
bash scripts/restore_models_from_autodl_fs.sh  # AutoDL 新实例
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_rental_environment.py
pytest -q
```

若平台镜像没有可复用的 CUDA PyTorch，先按平台说明安装与驱动兼容的 PyTorch，再安装本仓库依赖。

24GB 卡 smoke：

```bash
python scripts/train_qwen3_lora_dpo.py \
  --model_name Qwen/Qwen3-1.7B \
  --preferences_path data/local_short_template_preferences.jsonl \
  --prompts_path data/attack_prompts.jsonl \
  --output_dir experiments/smoke/r0_qwen3_17b \
  --max_steps 30 \
  --num_prompts 10 \
  --num_samples 4 \
  --eval_batch_size 2
```

该脚本默认关闭 Qwen3 thinking mode，并只保存 LoRA adapter。

## 当前配置

- `configs/current/r0_smoke_24gb.yaml`：环境和单 run 校准；
- `configs/current/r1_public_pilot_24gb.yaml`：公开 benchmark 低成本验证；
- `configs/current/r2_data_axis_pilot_24gb.yaml`：五类数据条件单 seed 筛选；
- `configs/current/r3_main_24gb.yaml`：1.7B 多 seed 主实验；
- `configs/current/r4_method_24gb.yaml`：方法特异性与机制筛选；
- `configs/current/r5_confirm_48gb.yaml`：4B/大 judge 复核；
- `configs/current/r6_optional_80gb.yaml`：可选 8B 确认。

`configs/stages/` 是旧 PCE 计划，不应用于新实验。

## 仓库结构

```text
configs/current/   当前租卡分阶段实验清单
configs/stages/    历史 PCE 配置，仅供复现
data/              小型受控数据与固定公开数据索引
docs/              协议、历史报告和研究材料
experiments/       新实验 manifest 与结果层级
scripts/           训练、生成、审计和汇总入口
src/               训练、风险与多样性指标实现
tests/             CPU 可运行的单元测试
PLAN.md            唯一当前执行计划
```

## 版本控制规则

- 不提交模型权重、缓存、原始大规模 generations、judge 全量输出和租卡凭据；
- 每个正式 run 提交 manifest、数据 split hash、Git commit、GPU/软件信息和聚合指标；
- `papers/` 中第三方 PDF 建议逐步替换为官方链接或 Git LFS；
- 历史 local/PCE 脚本可以保留，但新实验命名不再使用 `local_*` 或把 PCE 当作主结论。
