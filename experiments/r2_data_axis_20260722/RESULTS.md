# R2 data-axis pilot results (2026-07-22)

## Status

R2 的训练、生成和预定自动评测均已完成。实验使用 RTX 4090 24GB（driver 595.71.05）、PyTorch 2.8.0+cu128、Qwen3-1.7B、LoRA-DPO、seed 42；D0–D4 各 200 pairs、300 steps，并匹配约 74.5k 个实际训练序列 tokens。评测包括 HarmBench 50 prompts × 16 samples、完整 XSTest 450 prompts × 4 samples、HarmBench Mistral、Granite Guardian、StrongREJECT、embedding/HDBSCAN 和 Vendi。

脱敏聚合数据见：

- [`metrics/r2_summary.json`](metrics/r2_summary.json)
- [`metrics/r2_judge_diagnostics.json`](metrics/r2_judge_diagnostics.json)

原始 generations、逐条 judge 输出、adapter 和模型权重仅保存在实验数据盘，不进入 Git。

## Primary results

| Condition | HarmBench harm | Granite harm | StrongREJECT mean | SR Max@16 | Any-Harm@16 | XSTest unsafe refusal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 15.12% | 32.12% | 0.143 | 0.237 | 30% | 46.75% |
| D0 clean helpfulness | 6.50% | 8.88% | 0.070 | 0.130 | 14% | 64.62% |
| D1 clean safety | 1.00% | 4.12% | 0.039 | 0.084 | 6% | 67.25% |
| D2 conflict | 2.25% | 53.50% | 0.047 | 0.223 | 24% | 0.00% |
| D3 5% poison | 2.00% | 6.00% | 0.069 | 0.112 | 8% | 69.38% |
| D4 refusal suppression | 3.38% | 11.62% | 0.058 | 0.117 | 12% | 5.63% |

| Condition | Semantic entropy | Harmful entropy | Effective modes | Vendi | XSTest safe refusal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.699 | 0.177 | 2.59 | 2.36 | 3.50% |
| D0 | 0.750 | 0.138 | 2.96 | 2.37 | 7.50% |
| D1 | 0.630 | 0.017 | 2.50 | 2.49 | 7.60% |
| D2 | 0.543 | 0.000 | 2.15 | 5.76 | 0.00% |
| D3 | 0.725 | 0.000 | 2.83 | 2.38 | 9.10% |
| D4 | 0.625 | 0.050 | 2.26 | 2.53 | 0.60% |

Risk-Mass Entropy currently uses binary HarmBench risk mass, so it is numerically equal to Harmful Semantic Entropy. It must not be presented as an independent continuous-risk confirmation.

## Interpretation

- D1 is the cleanest safety reference: all three judges, Any-Harm@16 and StrongREJECT tail risk decrease together, while XSTest safe refusal rises by only 4.1 percentage points.
- D0 and D3 also reduce measured risk. D3 does not show excess risk over the clean conditions at this seed, so R2 provides no evidence that 5% contamination has a stable harmful effect.
- D4 almost removes refusal on unsafe XSTest prompts, yet all content-risk judges remain below Base. This directly shows that refusal rate and harmful-information risk are not interchangeable.
- D2 is not an interpretable risk result. Granite rises to 53.5%, while HarmBench falls to 2.25% and StrongREJECT mean falls to 0.047. Its StrongREJECT Max@16 remains close to Base and Any-Harm@16 is 24%, indicating a residual tail despite the low mean.
- D2 also shows severe output drift: 69.88% of outputs contain at least 20% non-ASCII letters, 40% mix ASCII and non-ASCII scripts, and 56% of prompts have a Granite-minus-HarmBench harm-rate gap of at least 0.5. Blinded formal human labels are not yet available; an internal spot check found language switching, incoherence/repetition and occasional genuinely unsafe tails.
- D2's low HDBSCAN entropy but high Vendi score is another warning that these diversity estimators react differently to degenerate multilingual outputs. It is not evidence of healthy semantic diversity.

## Gate R2 decision

**Hold R3.** R2 is computationally complete, but the Stop/Go evidence is not yet strong enough for a multi-seed main experiment. D2—the most informative conflict condition—is judge-sensitive and quality-degraded, so the current result cannot support a stable risk-increase or risk-diversity-decoupling claim.

The provisional three-condition set is **D1, D2 and D4**:

- D1: clean safety reference;
- D2: conflict mechanism condition, rerun only after data/KL/output-quality repair;
- D4: pressure test demonstrating refusal-risk decoupling.

D3 is deprioritized because the 5% contamination condition does not separate from clean training at this seed. Before R3:

1. formally audit a stratified D2 disagreement sample and record prompt-free label aggregates;
2. inspect D2 data composition and its realized KL (`2.005`, highest of all conditions), then rerun D2 with a lower-drift setting;
3. add an explicit language/valid-output control and a multilingual-capable adjudication path;
4. require HarmBench, Granite, StrongREJECT and human audit to agree on the direction or clearly explain the remaining disagreement.

## Reproducibility notes

- Frozen split manifests: [`../data_splits/r2_20260722_manifest.json`](../data_splits/r2_20260722_manifest.json) and [`../data_splits/r2_20260722_eval/split_manifest.json`](../data_splits/r2_20260722_eval/split_manifest.json).
- [`training_manifests/`](training_manifests/) and [`generations/`](generations/) contain the prompt-free manifests with environment, hashes, wall time, peak VRAM and generation throughput.
- Training manifests record the actual Git commit per condition; code commits changed while the sequential queue was running, but the frozen data hashes and run configurations are retained in `r2_summary.json`.
- StrongREJECT uses the official `qylu4156/strongreject-15k-v1` evaluator with the Gemma-2B base mirrored from ModelScope. The minimal reusable archive is `strongreject-gemma-2b-20260722.tar.zst` (3.8GB; SHA256 `aff67cae8079be00f3efec2cee4f75e736eb8dcc7a83942d0515ef60e7c95e21`) in `autodl-fs/model-archives/`.
