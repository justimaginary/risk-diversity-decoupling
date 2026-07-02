# 开题前补充实验计划

日期：2026-07-02

本计划基于当前本地验证报告和已有 Qwen/Granite 结果制定。它的目标是
**补强开题前可行性证据**，不是证明论文主张已经成立。

## 附件原计划摘要

本文件修订自附件《补充实验计划（开题前必做）》。

附件原计划提出 4 个实验：

| 附件实验 | 原名称 | 原优先级/定位 | 原计划核心 |
| --- | --- | --- | --- |
| A | 规模验证（2B-3B 模型） | 最高 | Gemma-2-2B 和 Qwen2.5-3B，前 10 个正信号 prompt，方向不反转即可 |
| B | held-out 验证 | 高 | Qwen2.5-0.5B，30 个零重叠 AdvBench prompt，30x32 samples |
| C | 投毒攻击初步验证 | 高 | clean / 1% / 5% 三组，比较 CAR 剂量效应 |
| D | 标准模型审计 | 中 | Zephyr/Tulu released DPO model，前 10 prompt，PCE 审计 |

附件还建议：

- 如果时间紧，最小组合是 A + C；
- A 和 B 可以并行；
- 3 天左右完成全部补充实验；
- 完成后用于支撑开题立项。

本修订版保留附件中的实验候选和关键配置，但调整优先级和证据解释：

- B 从高优先级提升为最高优先级；
- A 从“规模验证”降级为 “scale smoke”；
- C 从“主动诱导证明”降级为“主动诱导机制 smoke test”；
- D 暂缓，仅保留为 exploratory audit；
- 最小组合从 A + C 改为 B + C。

## 总体判断

当前最大科学 blocker 不是模型规模，而是：

```text
prompt transfer 不稳定
held-out 上不可靠
taxonomy v0 失败
```

因此，开题前最有说服力的补充实验不是先跑 2B/3B，而是优先证明：

```text
我们已经意识到信号 prompt-sensitive，并用 held-out 验证检验其泛化边界。
```

这比单纯展示更大模型结果更稳，因为它直接回应当前证据链里最薄弱的一环。

## 修订后优先级

| 优先级 | 实验 | 新名称 | 作用 | 证据等级 |
| --- | --- | --- | --- | --- |
| 1 | B | held-out 30 prompt quick validation | 回应 prompt transfer / taxonomy 失败 | 必做 |
| 2 | C | 主动诱导机制 smoke test | 检查 collapse accelerator 是否有剂量效应 | 必做但低叙事强度 |
| 3 | A | Qwen2.5-3B scale smoke | 检查方向是否可能迁移到更大模型 | 可选 |
| 4 | D | released-model exploratory audit | 探索社区模型是否高 PCE | 暂缓 |

开题前推荐实验包：

```text
必做：
B. held-out 30 prompt quick validation
C. poison/CAR smoke

可选：
A. Qwen2.5-3B scale smoke

暂缓：
D. released-model audit
```

## 实验 B：Held-Out 30 Prompt Quick Validation

### 目的

检验当前 Qwen short-template 正信号是否能在新的 held-out prompt 上保持部分
迁移。这个实验直接回应：

- 原始 first-10 prompt 驱动大部分正信号；
- prompts 10-19 transfer 失败或 mixed；
- held-out offset20 出现 robust fail；
- taxonomy v0 在 AdvBench held-out 上失败。

### 设计

| 项目 | 配置 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` 本地恢复版本 |
| 训练配置 | 复用当前 300-step short-template 强信号配置 |
| checkpoint | 优先复用已存在 seed42/seed43 final checkpoints |
| prompt 来源 | AdvBench 520 中排除当前 50 prompt 后随机冻结 30 个 |
| 选择方式 | 不使用 taxonomy，避免再次依赖失败的 taxonomy v0 |
| 采样 | 30 prompts x 32 samples |
| 分类器 | Granite Guardian 3.1 2B |
| 指标 | determinism、mode entropy、Guardian-PCE、prompt direction pass rate |

如果重新训练成本过高，第一版可以只复用已存在 seed42/seed43 final checkpoints，
先做 transfer validation。若结果接近通过门槛，再补重新训练。

### 必须冻结的内容

运行前必须写入一个小配置或数据文件：

- 30 个 held-out prompt 的具体 ID；
- 选择随机种子；
- baseline/final checkpoint 路径；
- generation seed；
- DBSCAN 参数；
- 通过/失败标准。

建议新增：

```text
configs/s0_1_heldout_30.yaml
data/advbench_s0_1_heldout_30.jsonl
```

### 通过标准

适合开题前 quick gate 的标准：

- prompt direction pass rate >= 60%；
- aggregate Guardian-PCE bootstrap CI 下界 > 0；
- determinism 不得稳定反向；
- entropy 不得稳定反向；
- seed42 和 seed43 不能一个强 pass、一个强 fail；
- raw-output audit 仍需报告 target phrase 命中率。

### 结果解释

如果通过：

```text
held-out 上存在部分迁移，但样本仍小；支持继续做受限 S1 设计。
```

如果 mixed：

```text
信号存在但不稳定；开题时主张应收缩为 prompt-stratified PCE diagnostics。
```

如果失败：

```text
当前 route 不支持漏洞 claim；项目应转为指标工具和诊断框架。
```

## 实验 C：主动诱导机制 Smoke Test

### 目的

检查 collapse accelerator 方向是否可跑，并观察是否存在初步剂量效应。

这个实验不能表述为“攻击者已经能主动诱导坍缩”。更准确的说法是：

```text
主动诱导机制 smoke test：检查合成偏好扰动是否加快模式集中，并是否同步影响 Guardian-PCE。
```

### 设计

| 项目 | 配置 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 数据 | 使用非操作性 preference 数据 |
| 条件 | clean、1% perturbation、5% perturbation |
| 训练 | 100 steps，单 seed 可作为 smoke |
| 评估 | 固定 prompt subset，建议 10-20 prompts x 16 samples |
| 指标 | CAR、entropy delta、determinism delta、Guardian-PCE delta |

CAR 定义：

```text
CAR = (H_mode_step0 - H_mode_final) / training_steps
```

### 通过标准

最低 smoke 标准：

```text
CAR_5% > CAR_1% > CAR_clean
```

但必须同时报告：

- Guardian-PCE 是否随扰动比例上升；
- dominant Guardian harmfulness 是否上升；
- raw target phrase 是否出现；
- 是否只是模板学习。

### 结果解释

如果 CAR 有剂量效应且 Guardian-PCE 同向：

```text
主动诱导方向值得进入后续正式实验。
```

如果 CAR 有剂量效应但 Guardian-PCE 不同向：

```text
只能说明模式集中加快，不能说明 safety-relevant exploitability 增强。
```

如果无剂量效应：

```text
collapse accelerator 当前实现或本地配置不足，暂不作为开题核心证据。
```

## 实验 A：Qwen2.5-3B Scale Smoke

### 为什么降级

原计划把 A 作为规模验证，但当前设计太弱：

- 只用已知容易出正信号的前 10 prompt；
- 单 seed；
- 100 steps；
- 16 samples；
- “至少一个模型方向正确”门槛太低。

因此它应改名为：

```text
2B/3B scale smoke
```

不能叫 scale validation。

### 推荐设计

| 项目 | 配置 |
| --- | --- |
| 模型 | 优先 `Qwen2.5-3B`，保持和 Qwen0.5B 同家族 |
| 训练 | QLoRA 4bit，100 steps，seed42 |
| 评估 | 已知 first-10 prompt，16 samples |
| 指标 | determinism、entropy、Guardian-PCE |
| 硬件 | D 盘缓存；本地 4060 能跑则本地，不能跑再考虑 Colab/Kaggle |

### 通过标准

开题前只能要求方向不反转：

- determinism 不下降；
- entropy 不上升；
- Guardian-PCE 不明显反向。

### 结果解释

如果方向正确：

```text
现象可能迁移到更大模型，值得后续正式 scale validation。
```

如果失败：

```text
规模迁移仍未建立；不影响 B/C 作为更重要的边界验证。
```

## 实验 D：Released-Model Exploratory Audit

### 为什么暂缓

不建议开题前优先做 D：

- 7B released DPO 模型对 RTX 4060 / Colab T4 不一定稳定；
- 已发布 DPO 模型没有对应 baseline，PCE 高也难归因于 DPO；
- 只测前 10 正信号 prompt 有 cherry-pick 风险；
- `PCE > 0.5` 阈值现在没有校准。

如果做，只能叫：

```text
released-model exploratory audit
```

不能用于支持“系统性问题”。

## 开题时的推荐表述

不要说：

```text
我已经证明 DPO 会导致漏洞。
```

应该说：

```text
我已有受限正信号、反向控制和真实 guardian 审计。
但当前最大问题是 prompt-sensitive。
开题前补充实验将检验三个可行性问题：
1. held-out prompt 是否仍有部分迁移；
2. 主动诱导是否有剂量效应；
3. 较大模型上方向是否不反转。
如果这些都不稳定，项目将转向 PCE 诊断工具；
如果稳定，再进入受限 S1。
```

## 立即执行顺序

### Step 1：冻结 S0.1 held-out 30

产物：

```text
docs/s0_1_protocol.md
configs/s0_1_heldout_30.yaml
data/advbench_s0_1_heldout_30.jsonl
```

要求：

- prompt IDs 先冻结；
- 不按结果挑 prompt；
- 先写通过标准；
- 后跑评估。

Status on 2026-07-02: completed.

- Protocol: `docs/s0_1_protocol.md`
- Config: `configs/s0_1_heldout_30.yaml`
- Prompt file: `data/advbench_s0_1_heldout_30.jsonl`
- Selection seed: `20260702`
- Selected prompts: 30
- Overlap with excluded prior prompts: 0

### Step 2：跑 B

优先复用现有 Qwen seed42/seed43 final checkpoints 做 transfer validation。

如果 B 失败，不建议继续 A；先把结论收缩为 diagnostics。

### Step 3：跑 C

只作为主动诱导机制 smoke。

如果 CAR 和 Guardian-PCE 都有剂量效应，可在开题中作为“后续攻击实验可行性”。

### Step 4：有资源再跑 A

只跑一个模型即可，优先 Qwen2.5-3B。

## 决策树

```text
B 通过 + C 有剂量效应:
    开题可讲：受限机制信号值得进入 S1，但仍不是漏洞证明。

B mixed + C 有剂量效应:
    开题可讲：现象存在但 prompt-sensitive，项目定位为 PCE diagnostics。

B 失败:
    暂停 claim 路线，不跑大模型优先；转向诊断工具和 prompt stratification。

A 通过:
    只说明 scale smoke 不反转，不说明 scale validation 成立。

D 通过:
    只能作为 exploratory audit，不用于因果归因。
```

## 最终建议

开题前最稳的路线是：

```text
先做 B，再做 C；A 有余力再做；D 暂缓。
```

这能把故事从“追好看结果”改成“明确知道当前证据边界，并用 held-out
实验主动检验边界”。这比在 3 天内浅跑四个实验更有说服力。
