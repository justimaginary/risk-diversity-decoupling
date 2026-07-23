# SIS-2：早期安全风险预测与未见 Seed 验证

日期：2026-07-23  
状态：**完成；Gate A 通过，Gate B 初筛失败，按协议停止**

## 1. 实验目的

SIS-1 已表明：在相同模型、数据和超参数下，LoRA-DPO 的最终安全风险会随
训练 seed 显著变化。SIS-2 进一步检验：

> 能否仅使用训练第 30 step 以前的信号，预测一个训练实例在第 100 step
> 的最终安全风险方向？

实验分为两个阶段：

1. SIS-2A：在已有 seeds 42–51 上回溯建立并用留一 seed 验证早期预测器；
2. SIS-2B：冻结预测器后，在未见 seeds 52–57 上先写预测，再进行正式终点
   评估。

本实验只检验单模型、单方法、单数据条件下的早期可预测性，不检验因果机制、
跨模型普遍性、双峰性或缓解方法。

## 2. 数据、模型与训练方法

### 2.1 模型与方法

- 基座模型：Qwen3-1.7B；
- 训练方法：LoRA-DPO；
- 数据条件：修复后的 D2 helpfulness–safety conflict；
- 偏好对数量：200；
- 训练步数：100；
- 学习率：`1e-4`；
- DPO beta：`0.1`；
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`；
- checkpoint：10、20、30、50、75、100；
- model seed 与 data-order seed 均等于训练 seed。

### 2.2 早期 monitor

- 数据：冻结的 `harmbench_monitor_30.jsonl`；
- 规模：30 个分层有害提示，每个提示 8 次生成；
- 与正式 HarmBench 100 条提示零 ID 重叠；
- 生成 seed：20260722；
- SHA-256：
  `fcc564b6a14bad8ea22c35a2c18f1ff99d84062d6de21bbcf9e357151859c893`。

### 2.3 正式未见 seed 评估

- 未见训练 seeds：52–57；
- 正式初筛：HarmBench 100 个提示，每个提示 8 次生成；
- Base 沿用相同模型、提示、生成 seed 的 SIS-1 冻结 Base 输出；
- 输出质量检查在安全评分前独立执行。

## 3. SIS-2A：早期预测器建立

### 3.1 主分析集合

SIS-1 的十个训练 seed 中，seed45 的 realized KL 为 0.910，高于 0.75
门槛，因此仅作诊断，不进入主拟合。主分析使用九个 KL 合格 seeds：

`42, 43, 44, 46, 47, 48, 49, 50, 51`。

SIS-2A 共评估 Base 加 10 seeds × 6 checkpoints，即 61 个条件和
14,640 个 monitor 输出。输出质量 60/61 通过；唯一失败为
`seed45_step50`，其非 ASCII 字母比例异常。该条件属于已排除 seed，且晚于
允许进入预测器的 step30，因此保留为诊断结果，不中断独立 judge 评估。

### 3.2 候选预测器与 Gate A

预注册候选包括：

- P1：早期 monitor HarmBench 风险增量；
- P2：P1 加不安全提示拒答率增量；
- checkpoint 仅允许 step10、20、30。

最终只有 **step30 P1** 通过 Gate A：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| 早期特征与最终风险 Spearman | 0.75 | ≥ 0.60 |
| 留一 seed 三分类准确率 | 7/9（77.8%） | ≥ 7/9 |
| 高风险 seeds 42/43 被预测为低风险 | 0 | 必须为 0 |
| 低风险 seeds 44/46/47/48 未被预测为高风险 | 4/4 | ≥ 3 |
| 留一预测与最终风险 Spearman | 0.60 | 诊断指标 |
| 留一预测 MAE | 0.0407 | 诊断指标 |

因此 Gate A = **Go**。

### 3.3 冻结预测器

在未见 seed 正式评估前，使用九个合格 seeds 全量拟合普通最小二乘：

```text
predicted_final_delta
  = -0.060669478960745545
  + 1.735240751200226 × step30_monitor_harm_delta
```

冻结分类阈值：

- High：预测增量 ≥ 0.05；
- Low：预测增量 ≤ -0.05；
- Middle：其余。

冻结文件：

- `metrics/frozen_predictor.json`；
- 代码提交：`1f94191`；
- 预测器 SHA-256：
  `e60a1684be27bf6a0fc3744a9e9e4f7e54903042d089f2dcbb01d90f3ab8c174`。

## 4. SIS-2B：未见 seeds 前瞻验证

### 4.1 训练完整性

六个未见 seed 均完成 100 steps 和全部六个 checkpoints：

| Seed | Realized KL | 训练时间（秒） | 峰值显存（GiB） |
|---:|---:|---:|---:|
| 52 | 0.5036 | 74.4 | 7.06 |
| 53 | 0.3565 | 78.0 | 7.18 |
| 54 | 0.4399 | 79.1 | 7.29 |
| 55 | 0.2360 | 78.3 | 7.43 |
| 56 | 0.2760 | 77.3 | 7.13 |
| 57 | 0.3845 | 77.3 | 7.43 |

六个条件均通过 KL < 0.75。最初尝试三路并发时，seed52 在训练结束后的
KL 估计阶段因瞬时显存峰值 OOM；训练 checkpoints 已保存。将并发降为两路并
启用 expandable CUDA memory segments 后，seed52 在相同设置下成功完成。
这属于资源调度故障，不改变训练或评估定义。

### 4.2 正式评估前冻结的预测

所有 step30 monitor 输出质量均通过。正式 HarmBench 终点尚未运行时，
已写入以下预测：

| Seed | Step30 monitor 风险率 | 预测最终增量 | 预测类别 |
|---:|---:|---:|---|
| 52 | 0.1167 | +0.0622 | High |
| 53 | 0.1042 | +0.0406 | Middle |
| 54 | 0.0417 | -0.0679 | Low |
| 55 | 0.2417 | +0.2791 | High |
| 56 | 0.0292 | -0.0896 | Low |
| 57 | 0.0292 | -0.0896 | Low |

预测文件 SHA-256：
`371f3348be3252fa59d1b861aee99bff78e1fb95b21b2fb3f150ce42782561b3`。

### 4.3 正式 100×8 HarmBench 结果

Base 风险率为 0.16875。未见 seed 结果如下：

| Seed | 预测类别 | 正式风险率 | 相对 Base 增量 | 实际类别 | 类别正确 |
|---:|---|---:|---:|---|---|
| 52 | High | 0.56625 | +0.39750 | High | 是 |
| 53 | Middle | 0.34500 | +0.17625 | High | 否 |
| 54 | Low | 0.06375 | -0.10500 | Low | 是 |
| 55 | High | 0.13625 | -0.03250 | Middle | 否 |
| 56 | Low | 0.13000 | -0.03875 | Middle | 否 |
| 57 | Low | 0.26000 | +0.09125 | High | 否 |

六个正式条件的输出质量全部通过。

## 5. Gate B 与停止决定

| Gate B 初筛指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| 预测与实际 Spearman | 0.3769 | ≥ 0.50 | 失败 |
| 实际非 Middle 实例数 | 4 | 需要足够实例检验 4/5 | 不足 |
| 非 Middle 方向正确 | 2/4 | ≥ 4/5 | 失败 |
| 预测是否退化为单一类别 | 否 | 必须为否 | 通过 |
| 输出质量 | 6/6 通过 | 全部通过 | 通过 |

Gate B 初筛 = **Stop**。按预注册协议，不继续 32-sample 扩展，也不为挽救
结论而追加 Granite、StrongREJECT 或 XSTest。

## 6. 结论

### 6.1 可以得出的结论

1. **SIS-1 的训练实例安全不稳定性继续得到支持。** 六个新 seed 的正式
   HarmBench 风险率从 0.06375 到 0.56625，跨度很大；相对 Base 既有显著
   上升，也有显著下降。
2. **step30 monitor 在已见 seeds 上存在较强回溯关联，但不能可靠前瞻
   泛化。** Gate A 的 7/9 留一准确率没有转化为未见 seeds 上的 Gate B
   成功。
3. **单次早期安全测量可能产生错误信心。** 最明显的是 seed55：早期模型
   预测为强 High，最终却为 Middle；seed57 早期预测为 Low，最终变为 High。
4. **失败不是由 KL 越界或输出损坏解释的。** 六个新 seed 均通过 KL 与
   输出质量门槛。

### 6.2 不能得出的结论

- 不能声称已获得可靠的训练早期风险预测器；
- 不能声称安全训练轨迹在 step30 后保持单调；
- 不能以当前 P1 作为 checkpoint 选择或自动停止规则；
- 不能由该实验单独推出 DPO、LoRA 或 Qwen3 的普遍机制。

### 6.3 对研究主线的含义

结果加强了“训练实例依赖”主线，却否定了当前最简单的“用早期风险率预测
最终风险”方案。下一步若继续，不应调阈值挽救 P1，而应研究 **step30 到
step100 的轨迹反转机制**：哪些 batch、margin、梯度或拒答行为导致 seed55
与 seed57 改变方向。

## 7. 当前结果支持的题目

最稳妥的论文主标题仍是：

> **One Seed, Opposite Conclusions: Training-Instance Uncertainty in Safety
> Preference Optimization**

中文可表述为：

> **一个 Seed， 相反结论：安全偏好优化中的训练实例不确定性**

SIS-2 还支持一个重要副标题或章节题目：

> **Early Safety Signals Do Not Reliably Predict Final Safety Outcomes**

但当前结果**不支持**以“可靠早期预测”作为论文主贡献。更准确的贡献结构是：

1. 发现并确认跨训练实例的安全结果反转；
2. 证明增加生成样本不能消除该现象；
3. 证明一个看似通过回溯验证的简单早期预警器，在未见 seed 上失效；
4. 将下一步问题收敛到训练后半程的轨迹分叉与反转机制。

## 8. 关键结果文件

- `metrics/early_trajectory.json`：SIS-2A 全部早期轨迹与 Gate A；
- `metrics/frozen_predictor.json`：正式冻结的预测器；
- `metrics/output_quality_monitor8_summary.json`：61 条件质量汇总；
- `unseen/manifests/training_summary.json`：未见 seed 训练、KL 与资源记录；
- `unseen/metrics/unseen_seed_predictions.json`：正式评估前冻结的预测；
- `unseen/metrics/sis2b_screen_summary.json`：未见 seed 正式初筛与 Gate B。
