# R3 D4 step-30 sensitivity result

## Status

Computational evaluation is complete; the blinded human audit is prepared on the server and remains pending. This is a sensitivity/pressure-test result and is not mixed with the 100-step D1/D2 main analysis.

The original D4 100-step protocol failed the KL gate for seeds 43 and 44, and the step-50 rescue still failed for seed 44. The frozen step-30 rescue passed for all three seeds (teacher-forced mean token KL: 0.3312, 0.2393, and 0.4030).

## Protocol

- Model: Qwen3-1.7B with LoRA DPO.
- Condition: D4 full refusal-suppression intervention.
- Training seeds: 42, 43, 44.
- Evaluation: 100 HarmBench prompts x 32 samples, full 450-prompt XSTest.
- Judges/metrics: HarmBench Mistral, Granite Guardian, StrongREJECT, semantic clustering, risk entropy, and 5,000-sample hierarchical bootstrap.
- All generation-quality gates passed.

## Aggregate result

| Endpoint | Base | D4 three-seed mean | Delta |
| --- | ---: | ---: | ---: |
| HarmBench harm rate | 0.1666 | 0.0774 | -0.0892 |
| Granite harm rate | 0.3491 | 0.1318 | -0.2173 |
| StrongREJECT mean score | 0.1305 | 0.0712 | -0.0593 |
| Any-Harm@32 | 0.3800 | 0.2100 | -0.1700 |
| Harmful semantic entropy | 0.2404 | 0.0991 | -0.1414 |
| Semantic entropy | 0.5938 | 0.5097 | -0.0841 |
| XSTest safe refusal | 0.0360 | 0.1400 | +0.1040 |
| XSTest unsafe refusal | 0.4788 | 0.5792 | +0.1004 |

All three safety judges agree on the per-seed direction. The hierarchical-bootstrap harm-rate delta is -0.0890 (95% CI [-0.1179, -0.0627]), while the harmful-entropy delta is -0.1097 (95% CI [-0.1624, -0.0639]). Safe-prompt refusal also increases (95% CI for the delta [0.0375, 0.1862]).

## Interpretation and gate

D4 strongly reduces both total risk and harmful-mode diversity while increasing refusal on both unsafe and safe prompts. It therefore behaves as an aggressive safety/refusal intervention, not as evidence for risk-diversity decoupling.

The computational R3 gate is **Fail** for D4: the risk change is stable, but harmful entropy fails the preregistered non-inferiority requirement. The final gate remains pending only because the required human audit has not yet been annotated; that audit cannot reverse the failed computational non-inferiority criterion.

Prompt-free machine-readable evidence is in `metrics/r3_summary.json` and `bootstrap/hierarchical_bootstrap.json`. Raw generations and the blinded answer key remain server-only.
