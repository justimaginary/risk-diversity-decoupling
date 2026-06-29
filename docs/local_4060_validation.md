# Local RTX 4060 Validation

This workflow is for fast directional validation on the local RTX 4060 Laptop
GPU with the existing `stdplm` conda environment. It is not a paper-grade run.

## Environment

Use the existing environment:

```powershell
conda run -n stdplm python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n stdplm python -c "import transformers, datasets, trl, sentence_transformers"
```

The environment was kept as close as possible to its original state. The only
project-specific addition should be `sentence-transformers`; avoid further
dependency changes unless a command fails with a missing package.

## Data

Create local fallback attack prompts without network access:

```powershell
conda run -n stdplm python scripts/prepare_attack_prompts.py --num_prompts 20 --output_path data/attack_prompts.jsonl --use_fallback
```

The tiny preference file `data/local_smoke_preferences.jsonl` exists only as a
pipeline fallback. Use real HH-RLHF subsets for meaningful DPO behavior once
downloads are available.

## Metric Smoke

First verify that the metric moves in the expected direction without loading a
model:

```powershell
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile diverse --num_prompts 10 --num_samples 16
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile collapsed --num_prompts 10 --num_samples 16
```

Expected result: the collapsed profile should have higher determinism and proxy
PCE, with lower diversity than the diverse profile.

## Mechanism Gate

Before spending GPU time, run a toy DPO collapse experiment:

```powershell
conda run -n stdplm python scripts/toy_dpo_collapse.py --poison_ratio 0.05 --steps 500
```

This is not LLM evidence. It only checks whether the DPO-style objective can
push a categorical policy toward lower entropy and higher determinism under
clean or lightly poisoned preferences. Continue to model experiments only if
determinism rises and entropy falls.

## Optional Model Baseline

If the model is cached or downloads are available:

```powershell
conda run -n stdplm python scripts/local_pce_smoke.py --mode model --model_name Qwen/Qwen2.5-0.5B-Instruct --num_prompts 10 --num_samples 16 --batch_size 2
```

If this OOMs, reduce `--num_samples` to 8 and `--max_new_tokens` to 64.

## Short DPO Directional Run

The fastest real-model loop uses the tiny local DPO smoke trainer:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name sshleifer/tiny-gpt2 --max_steps 20 --num_prompts 3 --num_samples 4
```

This validates the training/evaluation loop. It is still not meaningful LLM
safety evidence because the model is tiny. If it runs cleanly, switch
`--model_name` to `Qwen/Qwen2.5-0.5B-Instruct` and increase samples gradually.

The heavier pilot run is:

```powershell
conda run -n stdplm python scripts/pilot_experiment.py --model_name Qwen/Qwen2.5-0.5B-Instruct --num_train_samples 200 --max_steps 50 --pce_num_samples 16 --output_dir outputs/local_smoke/pilot --checkpoint_dir outputs/local_smoke/checkpoints --attack_prompts_path data/attack_prompts.jsonl
```

Success means the final checkpoint has higher determinism and lower mode entropy
or cluster count than step 0. Treat proxy PCE as a directional signal only.

## Git Rules

Commit each stable step locally. Do not configure a remote and do not push
without explicit approval.
