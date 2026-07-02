# Qwen2.5-0.5B Cleanup Manifest

Date: 2026-07-02

This manifest records the cleanup of historical Qwen2.5-0.5B artifacts before
switching the active experiment line to Qwen3.

Status: completed on 2026-07-02.

## Decision

Qwen2.5-0.5B is now historical pilot evidence only. It should not be used for
new training, held-out validation, poison/CAR tests, or Guardian audits.

The tracked reports, scripts, configs, and small JSONL data remain in git so the
research history stays reproducible.

## Deleted Paths

| Path | Size GB | Reason |
| --- | ---: | --- |
| `D:\hf_models\Qwen2.5-0.5B-Instruct` | 0.920 | obsolete local 0.5B model weights |
| `outputs/local_models\Qwen2.5-0.5B-Instruct` | 0.931 | obsolete assembled 0.5B model |
| `outputs/local_smoke\local_s0_qwen05_collapse_proxy_fp32_100steps_seeds42_43` | 3.703 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\local_s0_qwen05_concise_overview_lr3e6_300steps_seeds42_43` | 0.000 | ignored historical 0.5B output |
| `outputs/local_smoke\local_s0_qwen05_neutral_boundary_lr3e6_300steps_seeds42_43` | 0.000 | ignored historical 0.5B output |
| `outputs/local_smoke\local_s0_qwen05_refusal_template_lr3e6_300steps_seeds42_43` | 0.000 | ignored historical 0.5B output |
| `outputs/local_smoke\local_s0_qwen05_short_template_fp32_100steps_seeds42_43` | 3.703 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\local_s0_qwen05_short_template_prompts10_19_lr3e6_300steps_seeds42_43` | 0.000 | ignored historical 0.5B output |
| `outputs/local_smoke\local_s0_qwen05_uniform_fp32_100steps_seeds42_43` | 3.703 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\local_s0_qwen05_uniform_fp32_20steps_seeds42_43` | 3.703 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\qwen05_local_tiny_smoke` | 0.000 | ignored historical 0.5B output |
| `outputs/local_smoke\qwen05_local_tiny_smoke_fp32` | 1.851 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\qwen05_short_template_margin_flip_seed42_lr3e6_300steps` | 1.851 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\qwen05_short_template_margin_flip_seed43_lr3e6_300steps` | 1.851 | ignored historical 0.5B checkpoint output |
| `outputs/local_smoke\reeval_qwen05_short_template_seed42_lr3e6_300steps_10x16` | 0.000 | ignored historical 0.5B metric output |
| `outputs/local_smoke\reeval_qwen05_short_template_seed43_lr3e6_300steps_10x16` | 0.000 | ignored historical 0.5B metric output |
| `outputs/local_smoke\reeval_qwen05_uniform_fp32_100steps_seed42_20x32` | 0.000 | ignored historical 0.5B metric output |
| `outputs/local_smoke\reeval_qwen05_uniform_fp32_100steps_seed43_20x32` | 0.000 | ignored historical 0.5B metric output |
| `outputs/local_smoke\s0_1_qwen05_short_template_seed42_advbench_heldout30_32x` | 0.001 | ignored historical 0.5B metric output |
| `outputs/local_smoke\s0_1_qwen05_short_template_seed43_advbench_heldout30_32x` | 0.001 | ignored historical 0.5B metric output |

Estimated space reclaimed from the listed paths: about 28.2 GB.

Post-cleanup free space observed:

| Drive | Free bytes | Approx GiB |
| --- | ---: | ---: |
| `C:\` | 38,769,221,632 | 36.11 |
| `D:\` | 84,355,039,232 | 78.56 |

## Preserved Evidence

- `docs/local_validation_report.md`
- `docs/opening_report.md`
- `docs/s0_1_protocol.md`
- `docs/poison_car_smoke_protocol.md`
- `data/local_short_template_preferences.jsonl`
- all tracked scripts and configs

## Post-Cleanup Rule

New experiments should use Qwen3 models only unless a future note explicitly
re-opens a smaller-model diagnostic for engineering reasons. Any such run must
be labeled as engineering-only, not research evidence.
