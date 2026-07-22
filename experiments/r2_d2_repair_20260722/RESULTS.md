# R2 repaired D2 results (2026-07-22)

## Status

R2 的 D2 数据、KL 和输出质量修复已完成。修复版使用 Qwen3-1.7B、LoRA-DPO、seed 42、200 个 mild natural-conflict pairs、100 steps、`lr=1e-4`、`beta=0.1`。正式运行 commit 为 `7a1d96f9eb7a58025c4ebf452f5659fa220c1029`，偏好数据 SHA-256 为 `08405eae95a5f049e8d2c38dffeb845a03a530377da2b536be9aa056c8c1b071`。

D2 数据只保留 chosen harm severity 1，并过滤过短、过长、长度严重失衡和非英文响应。最终 200 pairs 与其余条件匹配为 44,538 effective tokens。训练实际处理 15,413 response tokens，teacher-forced mean-token KL 为 0.326，低于 0.5 gate。

## Quality repair

| Diagnostic | Original D2 | Repaired D2 |
| --- | ---: | ---: |
| Non-ASCII letters ≥20% | 69.88% | 0.00% |
| Mixed ASCII/non-ASCII scripts | 40.00% | 0.00% |
| Long character runs | 0.38% | 0.00% |
| Outputs shorter than 80 chars | 1.25% | 0.00% |

修复版质量统计覆盖完整 50 HarmBench prompts × 16 samples，共 800 条输出，而不是小 smoke 样本。

## Main endpoints

| Metric | Base | Original D2 | Repaired D2 | Repaired − Base |
| --- | ---: | ---: | ---: | ---: |
| HarmBench harm rate | 15.12% | 2.25% | 22.12% | +7.00 pp |
| Granite harm rate | 32.12% | 53.50% | 40.75% | +8.63 pp |
| StrongREJECT mean | 0.143 | 0.047 | 0.155 | +0.011 |
| StrongREJECT Max@16 | 0.237 | 0.223 | 0.310 | +0.073 |
| Macro Harm Rate | 15.48% | 2.23% | 22.53% | +7.05 pp |
| Any-Harm@16 | 30% | 24% | 40% | +10 pp |
| Harmful Semantic Entropy | 0.177 | 0.000 | 0.232 | +0.055 |
| Overall Semantic Entropy | 0.699 | 0.543 | 0.671 | -0.028 |
| Effective Semantic Modes | 2.59 | 2.15 | 2.50 | -0.10 |
| Vendi Score | 2.36 | 5.76 | 2.41 | +0.05 |
| XSTest safe refusal | 3.50% | 0.00% | 1.40% | -2.10 pp |
| XSTest unsafe refusal | 46.75% | 0.00% | 41.88% | -4.88 pp |

## Interpretation and Gate R2

The repaired D2 result passes every predeclared computational gate:

- output quality passes on all 800 HarmBench generations;
- realized KL is 0.326 ≤ 0.5;
- HarmBench, Granite and StrongREJECT all increase relative to Base;
- Any-Harm@16 and StrongREJECT tail risk increase;
- Harmful Semantic Entropy increases while overall Semantic Entropy changes only slightly.

**Gate R2: Go R3.** The promoted conditions remain D1 clean safety, D2 helpfulness–safety conflict and D4 full refusal suppression. R2 is still a single-training-seed screen; it supports the hypothesis strongly enough to justify R3, but does not establish a multi-seed conclusion.

## Reproducibility

- Aggregate result: [`metrics/r2_d2_repair_summary.json`](metrics/r2_d2_repair_summary.json)
- Output-quality gate: [`metrics/output_quality.json`](metrics/output_quality.json)
- Training manifest: [`training_manifests/D2_helpfulness_safety_conflict_repaired.json`](training_manifests/D2_helpfulness_safety_conflict_repaired.json)
- Frozen data manifest: [`../data_splits/r2_20260722_repaired_manifest.json`](../data_splits/r2_20260722_repaired_manifest.json)
- Raw generations, judge traces and adapters remain on the experiment data disk and are not committed.
