# R3 D2 64-sample key-checkpoint confirmation

## Status

Computational evaluation is complete. A 200-item blinded human-audit packet is prepared server-side and remains unannotated.

This run appends 32 independently seeded generations to the frozen 32-sample outputs for base and each D2 training seed, producing 100 prompts x 64 samples without overwriting the main result. All four merged shards contain exactly 6,400 answers and pass the output-quality gate.

## Aggregate result

| Endpoint | Base | D2 three-seed mean | Delta |
| --- | ---: | ---: | ---: |
| HarmBench harm rate | 0.1645 | 0.2057 | +0.0411 |
| Granite harm rate | 0.3477 | 0.3134 | -0.0343 |
| StrongREJECT mean score | 0.1288 | 0.1595 | +0.0307 |
| Any-Harm@64 | 0.4500 | 0.4867 | +0.0367 |
| Harmful semantic entropy | 0.1881 | 0.2429 | +0.0548 |
| Semantic entropy | 0.5171 | 0.6929 | +0.1758 |
| XSTest safe refusal | 0.0360 | 0.0813 | +0.0453 |
| XSTest unsafe refusal | 0.4788 | 0.3996 | -0.0792 |

HarmBench harm rates remain sharply heterogeneous:

| Label | 32 samples | 64 samples |
| --- | ---: | ---: |
| Base | 0.1666 | 0.1645 |
| D2 seed42 | 0.2513 | 0.2498 |
| D2 seed43 | 0.3391 | 0.3438 |
| D2 seed44 | 0.0238 | 0.0234 |

The 64-sample hierarchical bootstrap gives:

- harm-rate delta: +0.0413, 95% CI [-0.1329, 0.1760];
- Any-Harm@64 delta: +0.0381, 95% CI [-0.3067, 0.3100];
- harmful-entropy delta: +0.0572, 95% CI [-0.1187, 0.2125].

## Gate and interpretation

HarmBench, Granite, and StrongREJECT all remain directionally inconsistent across training seeds. The risk CI crosses zero and the harmful-entropy non-inferiority criterion fails. **The D2 64-sample computational gate is Fail.**

The near-identical 32- and 64-sample HarmBench rates show that the R3 instability is not caused by too few generations per prompt. Additional generation samples are not a substitute for additional independent training seeds or a better-understood source of seed sensitivity.

Prompt-free evidence is in `metrics/r3_summary.json` and `bootstrap/hierarchical_bootstrap.json`; raw generations and human-audit keys remain server-only.
