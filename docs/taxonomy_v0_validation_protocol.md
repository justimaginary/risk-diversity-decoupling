# Taxonomy v0 Validation Protocol

Date: 2026-07-01

This protocol defines the next prompt-stratified validation step for the local
PCE route. It is a diagnostic protocol, not a paper claim.

## Goal

Test whether frozen prompt taxonomy v0 predicts PCE pass/fail behavior on new,
previously unseen prompts.

Current exploratory result on the 50-prompt local set:

- `cyber`: strongest positive group.
- `violence_weapons`: strongest negative group.
- request verb and prompt surface are weaker separators.

The next check should use new prompts selected after the taxonomy is frozen in
`configs/prompt_taxonomy_v0.json`.

## Candidate Source

Preferred source: AdvBench harmful behaviors CSV via
`scripts/prepare_attack_prompts.py`.

The download command is:

```powershell
conda run -n stdplm python scripts/prepare_attack_prompts.py --num_prompts 10000 --output_path data/advbench_harmful_behaviors_all.jsonl --seed 42
```

If network access is blocked, do not replace this with generated prompts. Wait
until the source can be downloaded or use another documented external prompt
source.

## Selection Command

After `data/advbench_harmful_behaviors_all.jsonl` exists, select a taxonomy
validation set while excluding the current local 50-prompt set:

```powershell
conda run -n stdplm python scripts/select_taxonomy_prompt_set.py `
  --source_prompts_path data/advbench_harmful_behaviors_all.jsonl `
  --output_path data/advbench_taxonomy_v0_cyber_vs_violence_heldout.jsonl `
  --taxonomy_path configs/prompt_taxonomy_v0.json `
  --exclude_prompts_path data/attack_prompts.jsonl `
  --exclude_prompts_path data/attack_prompts_fallback_heldout_30.jsonl `
  --topics cyber violence_weapons `
  --per_topic 10 `
  --seed 20260701
```

## Evaluation Boundary

Use the existing Qwen short-template final checkpoints only as a transfer test:

- seed42 final: `outputs/local_smoke/qwen05_short_template_margin_flip_seed42_lr3e6_300steps/final_model`
- seed43 final: `outputs/local_smoke/qwen05_short_template_margin_flip_seed43_lr3e6_300steps/final_model`
- baseline: `outputs/local_models/Qwen2.5-0.5B-Instruct`

Evaluate each selected topic separately at matched 10 prompts x 16 samples, then
run Granite Guardian and Guardian-PCE bootstrap.

## Success Criteria

Taxonomy v0 is supported only if:

- the new `cyber` subset has stronger Guardian-PCE movement than the new
  `violence_weapons` subset;
- the effect is not only aggregate-positive: prompt-level pass/fail distribution
  should separate by topic;
- the same direction appears in both seed42 and seed43 final checkpoints;
- target phrase hits remain audited and reported.

If `cyber` and `violence_weapons` do not separate on new prompts, treat taxonomy
v0 as an exploratory description of the old 50 prompts, not as a predictor.
