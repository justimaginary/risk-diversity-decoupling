# SIS-3：训练后半程安全轨迹反转

日期：2026-07-23  
状态：计算与分析完成；Gate C1 = Stop

## 1. 实验目的

SIS-2 的冻结 step30 预测器在未见 seeds 上失败，其中：

- seed55 在 step30 被预测为 High，正式终点为 Middle；
- seed57 在 step30 被预测为 Low，正式终点为 High。

本阶段不重新拟合预测器，而是回答三个更窄的问题：

1. 两个错误是否来自同一冻结 monitor 上真实发生的时间轨迹反转；
2. 能否把反转定位到不超过 15 个训练 step 的窗口；
3. 固定 probe、在线训练量、LoRA 层更新或具体 batch 是否给出可重复的
   机制信号。

## 2. 数据、模型与方法

- 模型：Qwen3-1.7B；
- 方法：修复版 D2 helpfulness–safety conflict，LoRA-DPO，100 steps；
- 代表 seeds：52、54、55、57；
- 冻结 monitor：30 条 HarmBench prompts，与正式 100 题集合互斥；
- 每个 checkpoint：每题 8 个输出，共 240 个响应；
- generation seed：20260722；
- judge：HarmBench-Mistral-7B classifier；
- base monitor harm rate：4.58%，拒答率：40.42%；
- GPU：RTX 4090 24GB。

SIS-3A 先在旧 checkpoint 的 step30/50/75/100 上使用相同 monitor
复评。确认存在时间变化后，SIS-3B 从 step0 重跑四个 seed，在
step30/35/40/45/50/55/60/70/80/90/100 保存 adapter 和完整训练状态。

每个训练 step 记录：

- DPO loss、preference accuracy；
- chosen/rejected log probability 与 log-ratio；
- chosen–rejected margin；
- 梯度范数；
- LoRA 总体及 196 个层/模块组的更新诊断；
- batch sample ID 与数据元信息；
- 不消耗正式 monitor 的固定 preference probe。

旧的 adapter-only step30 checkpoint 没有被当作可精确续训状态。

## 3. 完整性与复现检查

四个 dense runs 均完成 100 step、100 条 batch trace、11 个 dense
checkpoint 和 11 份完整 continuation state。输出质量检查全部通过，
所有 classifier 输出的 unknown rate 为 0。

| Seed | 原 KL | Dense KL | \|ΔKL\| | 最终类别一致 | Schedule hash 一致 |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 52 | 0.5036 | 0.5570 | 0.0533 | 是（High） | 是 |
| 54 | 0.4399 | 0.4524 | 0.0125 | 是（Middle） | 是 |
| 55 | 0.2360 | 0.3133 | 0.0773 | 是（Middle） | 是 |
| 57 | 0.3845 | 0.4080 | 0.0235 | 是（High） | 是 |

四个 dense KL 都低于正式的 0.75 gate，且最终方向全部复现。但是，
结果前提交的执行脚本还使用了更保守的 `|ΔKL| <= 0.05` 仪器等价条件；
seed52 和 seed55 没有通过。协议文字只写了“KL gate”，没有明确写出
0.05，因此这是协议文字与可执行 gate 之间的偏差。为避免看到结果后放宽
标准，本报告按更严格的可执行 gate 判定复现检查失败。

## 4. 主要结果

### 4.1 SIS-3A：相同 monitor 上存在时间变化

旧 checkpoint 复评排除了“仅仅是 monitor 与正式终点数据分布不同”这一
单一解释：

- seed55 的 monitor risk 从 step30 的 24.17% 降至 step100 的 4.58%；
- seed57 从 2.92% 升至 13.75%；
- seed52 持续上升，seed54 持续下降。

因此 dense rerun 有明确的时间反转目标。

### 4.2 SIS-3B：反转窗口被成功定位

| Seed | Step30 | Step35 | Step40 | Step45 | Step50 | Step60 | Step70 | Step80 | Step100 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 | 10.42% | 11.25% | 11.67% | 10.42% | 15.00% | 18.75% | 35.83% | 49.58% | 42.50% |
| 54 | 5.42% | 5.83% | 6.25% | 3.33% | 7.92% | 5.00% | 2.92% | 1.67% | 0.42% |
| 55 | 27.92% | 14.17% | 16.25% | 13.33% | 14.58% | 17.92% | 9.58% | 6.67% | 4.17% |
| 57 | 3.75% | 5.00% | 7.92% | 12.08% | 15.83% | 15.42% | 17.50% | 17.92% | 17.92% |

按相对 base 的冻结分类阈值：

- seed57 在 step40–45 从 Middle 进入 High，窗口宽 5 steps；
- seed55 的主要下降发生在 step60–70（17.92% → 9.58%），并在
  step70–80 从 High 进入 Middle，分类窗口宽 10 steps。

因此“目标反转可定位至不超过 15 steps”通过。

### 4.3 最清晰的同步现象是拒答行为重组

- seed57 风险上升期间，拒答率从 step30 的 56.25% 连续降到
  step45 的 15.42%；
- seed55 风险主要下降的 step60–70，拒答率从 30.42% 跃升到
  61.25%，step100 达到 71.25%；
- 输出质量检查没有发现空输出、语言漂移或格式退化，因此该变化不是简单的
  生成失败。

这支持“风险反转与拒答行为重组同步”，但不能证明拒答变化是原因。

### 4.4 固定 probe 与层更新没有形成跨 seed 机制

固定 probe 的 DPO margin 确实在目标窗口前或同步变化：

- seed57 step30–45：margin 从 -1.405 恶化到 -1.690，同时风险上升；
- seed55 step60–70：margin 从 -1.177 改善到 -0.059，同时风险下降。

但 11 个 checkpoint 内，风险与固定 probe margin 的探索性 Spearman
相关在各 seed 间符号不一致：

| Seed | Spearman(risk, probe margin) | Spearman(risk, refusal) |
| ---: | ---: | ---: |
| 52 | +0.925 | -0.934 |
| 54 | +0.543 | +0.757 |
| 55 | -0.764 | -0.664 |
| 57 | +0.752 | -0.487 |

LoRA 层级比较也没有找到共同异常模块：目标窗口相对另外三个 seed 的最大
更新/梯度放大约为 1.2–1.8 倍，但 seed55 与 seed57 的高排名层组不重合。
在线 loss、accuracy、gradient norm 和总更新范数同样没有出现两个目标
seed 共享、且控制 seed 不具有的稳定模式。

因此当前证据不能支持“某个统一训练量或某组 LoRA 层导致反转”。

## 5. Gate C1 判定

| 条件 | 结果 |
| --- | --- |
| 相同冻结 monitor 上存在方向变化 | 通过 |
| seed55/57 定位至 ≤15-step 窗口 | 通过 |
| 至少一个 probe/更新量在风险前或同步变化 | 仅有探索性候选 |
| seed52/54 轨迹可区分 | 通过 |
| 输出质量与正式 KL gate 不解释结果 | 通过 |
| 严格仪器等价复现（含 `|ΔKL| <= 0.05`） | 失败 |
| 跨代表 seed 的共同机制信号 | 失败 |

**最终决定：Gate C1 = Stop。**

不执行 SIS-3C 的 2×2 随机源交换，也不执行 batch 删除/换序。否则会在
仪器等价尚未确认、batch 选择规则尚未冻结时把相关性扩写成机制。

## 6. 结论与支持的论文题目

本阶段可靠支持：

> 同一模型、数据、超参数和冻结评测集合下，偏好训练的安全方向可以在
> step30 之后发生可定位的反转；这足以使回顾性有效的早期预警器在未见
> seed 上失败。

本阶段不支持：

- 已找到训练实例安全分叉的统一机制；
- 特定 batch、层组或随机源导致反转；
- 一个固定 early-warning margin 能跨 seed 工作；
- 立即提出或验证稳定化方法。

适合作为现有论文中的结果/方法学题目：

> **Late-Training Safety Reversals Break Early Warnings in Preference Optimization**

或作为更大训练不稳定性论文中的章节：

> **One Seed, Opposite Conclusions: Training-Instance Safety Uncertainty in Preference Optimization**

不能使用“我们发现了安全吸引域机制”或“特定冲突 batch 导致安全分叉”
作为当前题目或摘要结论。

## 7. 下一步

若继续投入，下一阶段不是扩大模型或直接做随机源交换，而是一次低成本
**instrumentation-equivalence audit**：

1. 同一 seed、同一数据顺序并排运行原始训练与 instrumented training；
2. 每 step 比较 loss、梯度、LoRA 参数 checksum 和 RNG state；
3. 逐项关闭固定 probe、层诊断和 full-state save，定位首次数值分叉；
4. 只有消除或解释 `|ΔKL| > 0.05` 后，重新从未查看的 seed 验证轨迹信号；
5. 仍无跨 seed 共同信号则终止机制路线，保留 SIS-1/SIS-2/SIS-3 为训练
   不稳定性与早期预警失败的实证结果。

## 8. 结果文件

- `metrics/sis3a_precheck_summary.json`：旧 checkpoint 的相同 monitor
  复评；
- `manifests/training_summary.json`：dense runs 的 schedule、KL、资源与
  checkpoint 完整性；
- `metrics/sis3b_dense_summary.json`：44 个 dense checkpoint 的风险、
  拒答、质量、固定 probe、区间训练量和 top batches。

模型权重、完整生成文本、classifier 逐样本输出与约 20MB 的训练 trace
保留在服务器数据盘，不进入 Git。
