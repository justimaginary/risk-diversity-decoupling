# Qwen3 Risk-Diversity Prompt Reasonableness Gate

Date: 2026-07-04

## Purpose

This gate freezes a new prompt set before the next Qwen3-1.7B risk-diversity
experiment. The goal is to avoid another first-10 prompt artifact.

The target downstream experiment is:

```text
Qwen3-1.7B
50-100 prompts
32 samples per prompt
test whether risk rises while diversity does not fall
```

This document covers only the prompt reasonableness gate. It does not claim the
new idea is validated yet.

## Frozen Prompt Set

Prompt file:

```text
data/advbench_rdi_heldout_80_seed20260704.jsonl
```

Selection policy:

- source: `data/advbench_harmful_behaviors_all.jsonl`;
- sample size: 80 prompts;
- selection seed: `20260704`;
- exclude previous local first-10/20 prompt files;
- exclude previous fallback held-out prompts;
- exclude taxonomy-v0 held-out prompts;
- exclude S0.1 held-out prompts.

Selection command:

```powershell
conda run -n stdplm python scripts/select_random_heldout_prompts.py `
  --source_prompts_path data/advbench_harmful_behaviors_all.jsonl `
  --output_path data/advbench_rdi_heldout_80_seed20260704.jsonl `
  --exclude_prompts_path data/attack_prompts.jsonl `
  --exclude_prompts_path data/attack_prompts_10_19.jsonl `
  --exclude_prompts_path data/attack_prompts_fallback_heldout_30.jsonl `
  --exclude_prompts_path data/advbench_taxonomy_v0_cyber_vs_violence_heldout.jsonl `
  --exclude_prompts_path data/advbench_s0_1_heldout_30.jsonl `
  --num_prompts 80 `
  --seed 20260704
```

## Reasonableness Criteria

The prompt set is considered reasonable for the next experiment if:

- it contains 50-100 prompts;
- it has zero overlap with previous prompt gates;
- it has no duplicate prompts;
- source indices are spread across the AdvBench file rather than concentrated
  in one small block;
- keyword-based topic audit contains more than one category;
- Qwen3-1.7B baseline sampling produces non-degenerate diversity metrics.

## Prompt-Set Audit Command

```powershell
conda run -n stdplm python scripts/audit_prompt_set.py `
  --prompt_path data/advbench_rdi_heldout_80_seed20260704.jsonl `
  --overlap_path data/attack_prompts.jsonl `
  --overlap_path data/attack_prompts_10_19.jsonl `
  --overlap_path data/attack_prompts_fallback_heldout_30.jsonl `
  --overlap_path data/advbench_taxonomy_v0_cyber_vs_violence_heldout.jsonl `
  --overlap_path data/advbench_s0_1_heldout_30.jsonl `
  --output_path outputs/local_smoke/qwen3_prompt_reasonableness/advbench_rdi_heldout_80_audit.json
```

## Qwen3 Baseline Sampling Command

This is an evaluation-only run, not DPO training:

```powershell
$env:QWEN3_TRANSFORMERS_OVERLAY="D:\hf_models\pydeps\qwen3_transformers"
conda run -n stdplm python scripts/local_qwen3_eval_prompts.py `
  --model_name D:\hf_models\Qwen3-1.7B `
  --prompts_path data/advbench_rdi_heldout_80_seed20260704.jsonl `
  --output_dir outputs/local_smoke/qwen3_17b_rdi_heldout80_baseline_32x `
  --label qwen3_17b_rdi_heldout80_step0 `
  --num_prompts 80 `
  --num_samples 32 `
  --max_new_tokens 64 `
  --eval_batch_size 1 `
  --dbscan_eps 0.8 `
  --dbscan_min_samples 1 `
  --generation_seed 20260704
```

## Result

Status on 2026-07-04: complete.

Selection output:

| Field | Value |
| --- | ---: |
| Source prompts | 520 |
| Excluded prompts | 88 |
| Candidate prompts | 482 |
| Selected prompts | 80 |
| Selection seed | 20260704 |
| Source index min | 2 |
| Source index max | 511 |
| Source index span | 509 |

Prompt-set audit:

| Check | Result |
| --- | --- |
| Duplicate prompts | 0 |
| Overlap with `data/attack_prompts.jsonl` | 0 |
| Overlap with `data/attack_prompts_10_19.jsonl` | 0 |
| Overlap with `data/attack_prompts_fallback_heldout_30.jsonl` | 0 |
| Overlap with `data/advbench_taxonomy_v0_cyber_vs_violence_heldout.jsonl` | 0 |
| Overlap with `data/advbench_s0_1_heldout_30.jsonl` | 0 |
| Mean prompt length | 70.375 chars |
| Median prompt length | 66.5 chars |
| Min / max prompt length | 33 / 131 chars |

Keyword topic audit:

| Topic bucket | Count |
| --- | ---: |
| cyber | 11 |
| cyber + fraud_theft | 2 |
| drugs | 3 |
| evasion_illegal | 4 |
| evasion_illegal + drugs | 2 |
| fraud_theft | 10 |
| harassment_abuse | 2 |
| violence_weapons | 10 |
| violence_weapons + evasion_illegal + drugs | 2 |
| other | 34 |

Decision:

```text
The frozen 80-prompt set passes the non-model prompt reasonableness checks:
it is zero-overlap with previous gates, duplicate-free, source-index spread,
and not a single-topic prompt block.
```

## Eval-Only Smoke

Command:

```powershell
$env:QWEN3_TRANSFORMERS_OVERLAY="D:\hf_models\pydeps\qwen3_transformers"
conda run -n stdplm python scripts/local_qwen3_eval_prompts.py `
  --model_name D:\hf_models\Qwen3-1.7B `
  --prompts_path data/advbench_rdi_heldout_80_seed20260704.jsonl `
  --output_dir outputs/local_smoke/qwen3_17b_rdi_heldout80_eval_smoke_1p1s `
  --label qwen3_17b_rdi_heldout80_smoke `
  --num_prompts 1 `
  --num_samples 1 `
  --max_new_tokens 16 `
  --eval_batch_size 1 `
  --dbscan_eps 0.8 `
  --dbscan_min_samples 1 `
  --generation_seed 20260704 `
  --progress_every 1
```

Result:

| Check | Result |
| --- | --- |
| Qwen3 overlay load | pass |
| Local Qwen3-1.7B load | pass |
| Non-thinking generation | pass |
| Metrics write | pass |
| Output directory | `outputs/local_smoke/qwen3_17b_rdi_heldout80_eval_smoke_1p1s` |

## Qwen3-1.7B Baseline Sampling Result

Output directory:

```text
outputs/local_smoke/qwen3_17b_rdi_heldout80_baseline_32x
```

Local diversity/proxy metrics:

| Metric | Value |
| --- | ---: |
| Prompts | 80 |
| Samples per prompt | 32 |
| Total outputs | 2560 |
| Mean determinism | 0.9145 |
| Mean mode entropy | 0.2966 |
| Mean distinct-1 | 0.1704 |
| Mean distinct-2 | 0.3678 |
| Mean proxy PCE | 0.0129 |
| Prompts with entropy > 0.001 | 21 / 80 |
| Prompts with determinism >= 0.95 | 72 / 80 |
| Prompts with determinism <= 0.80 | 8 / 80 |
| Prompts with proxy PCE > 0 | 13 / 80 |

Policy-proxy audit:

| Signal | All Outputs | Dominant Outputs |
| --- | ---: | ---: |
| Refusal rate | 0.8645 | 0.9342 |
| Compliance rate | 0.1637 | 0.1756 |
| Actionable rate | 0.1020 | 0.1064 |
| High-risk instructional rate | 0.0023 | 0.0026 |
| Prompts with dominant high-risk-instructional majority | 0 / 80 | 0 / 80 |

Interpretation:

```text
The frozen prompt set passes the prompt reasonableness gate for the next
Qwen3-1.7B risk-diversity experiment. It is new, broad, duplicate-free, and
zero-overlap with prior gates. The base model mostly refuses these prompts and
shows very low proxy risk, so the set is suitable for testing whether DPO moves
the model away from a safe refusal-dominant baseline toward distributed risk.
```

Caveat:

```text
The Qwen3-1.7B baseline is already highly deterministic on these harmful
prompts: 72/80 prompts have determinism >= 0.95. Therefore the next experiment
must not rely only on the easy claim that diversity does not decrease. It must
report absolute diversity, refusal shift, Guardian risk, RDI, and Risk Entropy.
```

Decision:

```text
Proceed to the next stage: Qwen3-1.7B DPO on the frozen 80-prompt gate, with 32
samples per prompt for both baseline and final checkpoints.
```
