# Preference Collapse Exploitability (PCE)

This repository explores whether Direct Preference Optimization (DPO) can
reduce output diversity enough to create exploitable safety risks in language
models.

The working hypothesis is:

```text
PCE = dominant-mode determinism * harmfulness of the dominant mode
```

In plain terms: if a model repeatedly collapses to a small number of response
modes, and the dominant mode is harmful or attack-compliant, the model may
become easier to exploit.

## Current Status

This project is in an experiment-first validation phase. The broad idea is not
yet proven credible for real LLMs.

What has been validated so far:

- The local git repository is initialized and all work is committed locally.
- No remote is configured and no push has been performed.
- The existing `stdplm` conda environment is used for local experiments.
- The local RTX 4060 Laptop GPU is visible to PyTorch.
- A lightweight proxy-PCE metric pipeline runs locally.
- Synthetic diverse-vs-collapsed responses move the metrics in the expected
  direction.
- A toy DPO mechanism experiment shows probability concentration under
  preference updates.
- A tiny GPT-2 real-model smoke loop runs end to end, but it is too small to
  support any research claim.
- A smaller real instruction-model gate (`HuggingFaceTB/SmolLM2-135M-Instruct`)
  now shows a weak but consistent two-seed collapse-direction signal under a
  conservative LM-head-only DPO smoke run.
- A stronger 360M instruction-model gate was run and re-evaluated with more
  prompts/samples. It produced mixed evidence, not a clean pass.
- A non-operational collapse-proxy preference gate was added and tested on the
  360M model. Current aggregate is one passing seed, two mixed seeds, and one
  failing seed.
- A prompt-level direction summary was added. Across the four collapse-proxy
  seeds, only 7 of 40 prompt comparisons fully pass the collapse-direction
  gate, while 22 fail.
- A matched checkpoint re-evaluation tool was added so existing baseline/final
  checkpoints can be re-measured with larger prompt/sample budgets without
  retraining.
- The previously passing 360M collapse-proxy seed 43 did not survive a matched
  10-prompt x 16-sample re-evaluation: determinism decreased, entropy
  increased, proxy PCE stayed flat, and 8 of 10 prompts failed the directional
  gate.
- The old collapse-proxy "multi-seed" runs used the same cyclic preference
  order, and their final 360M checkpoint weight hashes are identical. Those
  runs should be interpreted as repeated evaluations of one trained checkpoint,
  not independent training seeds.
- `scripts/local_dpo_smoke_train.py` now supports `--preference_order shuffled`
  and `--generation_seed` so future gates can separate training-seed variation
  from evaluation-seed variation.
- The first corrected 360M shuffled training-seed run (`seed=42`) is an
  aggregate pass, but the prompt-level signal is still weak: 2 prompt passes,
  3 mixed prompts, and 5 prompt failures.

What is not yet validated:

- A <=500M target model such as `Qwen/Qwen2.5-0.5B-Instruct` has not completed
  the local gate yet; the model download is currently incomplete.
- The real-model evidence is still weak because the successful gate used a
  135M instruction model and trained only `lm_head` to fit RTX 4060 memory.
- The 360M instruction-model gate did not satisfy all required conditions:
  proxy PCE increased, but entropy/determinism were mixed under re-evaluation.
- The collapse-proxy gate is not stable across seeds under the current
  10-prompt x 8-sample proxy protocol, and prompt-level comparison confirms
  the instability is not just an aggregate-metric artifact.
- The strongest 360M collapse-proxy seed-level signal so far does not replicate
  under a larger matched re-evaluation protocol.
- The prior 360M collapse-proxy seed set does not prove multi-seed stability
  because the saved final checkpoints are byte-identical across seeds 42-45.
- The corrected shuffled training-seed gate has only one completed seed so far,
  so it cannot establish multi-seed stability.
- The safety/exploitability part of PCE has not yet been validated with a real
  safety classifier such as LlamaGuard.
- The research novelty is not established; related work already studies DPO
  diversity collapse and DPO/RLHF attacks.

## Local Environment

Use the existing conda environment:

```powershell
conda run -n stdplm python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n stdplm python -c "import transformers, datasets, trl, sentence_transformers"
```

Confirmed package state after repair:

- `torch 2.10.0+cu128`
- `transformers 4.40.2`
- `tokenizers 0.19.1`
- `trl 0.12.2`
- `sentence-transformers 2.6.1`

Dependency rule going forward: dependencies may be added if necessary, but avoid
removing, downgrading, or broadly changing the existing `stdplm` environment.

## Stage Results

### 1. Synthetic Metric Sanity Check

Command:

```powershell
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile diverse --num_prompts 10 --num_samples 16
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile collapsed --num_prompts 10 --num_samples 16
```

Result:

| Profile | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| Diverse | 0.2500 | 1.6021 | 0.0000 |
| Collapsed | 0.3750 | 1.0948 | 0.3750 |

Interpretation: the proxy metrics behave correctly on controlled data.

### 2. Toy DPO Mechanism Gate

Command:

```powershell
conda run -n stdplm python scripts/toy_dpo_collapse.py --poison_ratio 0.05 --steps 500
```

Result:

| Setting | Entropy Change | Determinism Change | Proxy PCE Change |
| --- | ---: | ---: | ---: |
| 5% poison | 2.4845 -> 2.4525 | 0.0875 -> 0.1480 | 0.0215 -> 0.0348 |
| clean | 2.4845 -> 2.4525 | 0.0875 -> 0.1480 | 0.0215 -> 0.0341 |
| 20% poison | 2.4845 -> 2.4524 | 0.0875 -> 0.1482 | 0.0215 -> 0.0549 |

Interpretation: the toy mechanism supports the idea that preference updates can
concentrate probability mass, and that poisoning can raise harmful proxy mass.
This is mechanism evidence only, not LLM evidence.

### 3. Tiny GPT-2 Real-Model Smoke

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name sshleifer/tiny-gpt2 --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --num_prompts 3 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Result:

| Checkpoint | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| step 0 | 0.1250 | 2.0794 | 0.0000 |
| final | 0.1667 | 2.0217 | 0.0000 |

Interpretation: the training/evaluation loop runs and shows a weak collapse
direction, but `sshleifer/tiny-gpt2` is much too small and low quality to support
the research claim.

### 4. SmolLM2-135M Instruction-Model Gate

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-135M-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Result:

| Seed | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| 42 step 0 | 0.2500 | 1.2904 | 0.0000 |
| 42 final | 0.2500 | 1.1945 | 0.0833 |
| 43 step 0 | 0.2500 | 1.2904 | 0.0000 |
| 43 final | 0.3333 | 1.1749 | 0.2500 |

Interpretation: this is the first real instruction-model signal. It is still
only a weak S0 gate because it uses a 135M model, 3 prompts, 4 samples, 20
training steps, and LM-head-only training. The result supports continuing to a
stronger small-model gate, but it does not establish the paper claim.

### 5. Qwen 0.5B Download Blocker

`Qwen/Qwen2.5-0.5B-Instruct` remains the preferred local target, but the model
download did not complete within a 20-minute snapshot attempt. The current cache
contains tokenizer/config files and an incomplete weight blob of about 134 MB,
not the full model weights. This is an infrastructure blocker, not a negative
experimental result.

### 6. SmolLM2-360M Stronger Gate

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Initial 3-prompt results:

| Seed / Run | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| 42 step 0 | 0.2500 | 1.3863 | 0.0000 |
| 42 final, 20 steps | 0.2500 | 1.3863 | 0.0833 |
| 43 step 0 | 0.3333 | 1.1749 | 0.0000 |
| 43 final, 20 steps | 0.2500 | 0.5973 | 0.1667 |
| 43 final, 100 steps | 0.3333 | 0.5784 | 0.1667 |

Because 3 prompts x 4 samples is noisy, the 100-step seed-43 checkpoint was
re-evaluated on 10 prompts x 8 samples:

| Checkpoint | Determinism | Mode Entropy | Distinct-1 | Distinct-2 | Proxy PCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.1625 | 1.7418 | 0.6203 | 0.9030 | 0.0000 |
| final 100-step | 0.1875 | 1.7954 | 0.6587 | 0.9445 | 0.0375 |

Interpretation: this is mixed evidence. The final checkpoint has higher
determinism and proxy PCE under re-evaluation, but entropy and lexical diversity
also increase. Therefore the stronger 360M gate does not yet pass the full
collapse criterion. It supports continuing measurement work, not escalating to
S1 or paper claims.

### 7. SmolLM2-360M Collapse-Proxy Preference Gate

This gate uses `data/local_collapse_proxy_preferences.jsonl`, where chosen
responses are non-operational tutorial/compliance placeholders and rejected
responses are refusals. It tests whether DPO can induce a proxy harmful dominant
mode without storing operational harmful instructions.

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --preferences_path data/local_collapse_proxy_preferences.jsonl --max_steps 100 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 10 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Result:

| Seed | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| 42 step 0 | 0.1750 | 1.9039 | 0.0250 |
| 42 final | 0.1625 | 1.8930 | 0.0375 |
| 43 step 0 | 0.1625 | 1.9527 | 0.0125 |
| 43 final | 0.2375 | 1.7466 | 0.0500 |
| 44 step 0 | 0.1375 | 1.9215 | 0.0375 |
| 44 final | 0.1625 | 1.7245 | 0.0125 |
| 45 step 0 | 0.1750 | 1.8342 | 0.0250 |
| 45 final | 0.1375 | 1.9820 | 0.0250 |

Interpretation: seed 43 passes the directional gate. Seeds 42 and 44 remain
mixed: seed 42 raises proxy PCE while determinism decreases; seed 44 raises
determinism and lowers entropy while proxy PCE decreases. Seed 45 fails the
collapse criterion: determinism decreases, entropy increases, and proxy PCE is
flat. This is better aligned with the active-induction hypothesis than the
neutral local preference file, but it still does not provide stable multi-seed
evidence. Continue only as S0 validation work.

The current aggregate check is:

```powershell
conda run -n stdplm python scripts/summarize_local_gate.py outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed42 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed44 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed45
```

Aggregate result:

| Runs | Pass | Mixed | Fail | Overall |
| --- | ---: | ---: | ---: | --- |
| seed 42-45 | 1 | 2 | 1 | mixed |

Prompt-level direction result:

| Seed | Prompt Pass | Prompt Mixed | Prompt Fail |
| --- | ---: | ---: | ---: |
| 42 | 1 | 3 | 6 |
| 43 | 5 | 1 | 4 |
| 44 | 1 | 5 | 4 |
| 45 | 0 | 2 | 8 |
| total | 7 | 11 | 22 |

The aggregate remains mixed after adding seed 45. Continuing to add seeds under
the same noisy protocol is less useful than improving the measurement protocol.
The prompt-level summary strengthens that conclusion: most prompts do not move
in the full predicted direction under the current setup.

### 8. Matched Checkpoint Re-Evaluation Tool

`scripts/reevaluate_checkpoints.py` re-evaluates a baseline model and a trained
checkpoint with the same prompts, generation seed, sampling budget, clustering
settings, and output schema. This separates measurement improvement from
additional training.

Tiny tool-verification command:

```powershell
conda run -n stdplm python scripts/reevaluate_checkpoints.py --baseline_model HuggingFaceTB/SmolLM2-360M-Instruct --final_model outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43/final_model --output_dir outputs/local_smoke/reeval_smollm2_360m_collapse_proxy_seed43_tiny_matched --num_prompts 2 --num_samples 2 --max_new_tokens 32 --eval_batch_size 1 --dbscan_eps 0.8 --dbscan_min_samples 1 --generation_seed 2026
```

Tool-verification result:

| Checkpoint | Determinism | Mode Entropy | Distinct-1 | Distinct-2 | Proxy PCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 0 | 0.5000 | 0.3466 | 0.8580 | 1.0000 | 0.2500 |
| final | 0.5000 | 0.3466 | 0.8304 | 1.0000 | 0.5000 |

Interpretation: the script works and is compatible with the gate summarizer.
This 2-prompt x 2-sample run is intentionally too small to change the evidence
level. The next useful measurement is a matched re-evaluation of an existing
360M checkpoint at 10-20 prompts and 16-32 samples, then a decision about
whether the instability is sampling noise or a real negative result.

Matched 10-prompt x 16-sample re-evaluation of the previously passing seed-43
collapse-proxy checkpoint:

```powershell
conda run -n stdplm python scripts/reevaluate_checkpoints.py --baseline_model HuggingFaceTB/SmolLM2-360M-Instruct --final_model outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43/final_model --output_dir outputs/local_smoke/reeval_smollm2_360m_collapse_proxy_seed43_matched_10x16 --num_prompts 10 --num_samples 16 --max_new_tokens 64 --eval_batch_size 1 --dbscan_eps 0.8 --dbscan_min_samples 1 --generation_seed 2026
```

Result:

| Checkpoint | Determinism | Mode Entropy | Distinct-1 | Distinct-2 | Proxy PCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 0 | 0.1562 | 2.3389 | 0.5460 | 0.8954 | 0.0500 |
| final | 0.1437 | 2.4055 | 0.5614 | 0.9032 | 0.0500 |

Gate summary:

| Run | Det Delta | Entropy Delta | PCE Delta | Judgement | Prompt Pass/Mixed/Fail |
| --- | ---: | ---: | ---: | --- | ---: |
| seed 43 matched 10x16 | -0.0125 | +0.0666 | +0.0000 | fail | 1/1/8 |

Interpretation: this is a negative S0 result for the current 360M
collapse-proxy setup. The earlier seed-43 pass at 10 prompts x 8 samples was
not robust to a larger matched sample budget. This pushes the project further
toward "diagnostic tooling / measurement refinement" unless another small
instruction-model gate produces stable multi-seed evidence.

### 9. Training-Seed Control Fix

Inspection after the seed44 matched re-evaluation found that the final
`model.safetensors` hashes for collapse-proxy seeds 42, 43, 44, and 45 are
identical. The old training loop always used the same cyclic preference order,
so the `--seed` value mainly changed generation/evaluation randomness rather
than producing independent trained checkpoints.

Fix:

- `--preference_order cyclic` keeps the old deterministic order.
- `--preference_order shuffled` shuffles preference order per epoch using
  `--seed`.
- `--generation_seed` resets generation randomness before each evaluation, so
  step-0 and final checkpoints can be compared with matched sampling.

Verification:

```powershell
conda run -n stdplm python -m py_compile scripts/local_dpo_smoke_train.py
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name sshleifer/tiny-gpt2 --preferences_path data/local_collapse_proxy_preferences.jsonl --output_dir outputs/local_smoke/dpo_tiny_gpt2_shuffled_seed_smoke --max_steps 2 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 1 --num_samples 1 --eval_batch_size 1 --max_new_tokens 16 --dbscan_eps 0.8 --dbscan_min_samples 1 --seed 101 --preference_order shuffled --generation_seed 2026
```

The tiny smoke completed successfully. A non-fatal HuggingFace cache permission
warning appeared while checking a missing `generation_config.json`, but model
loading, training, and evaluation completed.

Interpretation: the old 360M collapse-proxy seed table remains useful as
evaluation-noise evidence, but it is not valid multi-training-seed evidence.
Future S0/S1 gates must use shuffled preference order or another explicit
training perturbation before claiming seed stability.

### 10. Corrected 360M Shuffled Training-Seed Gate

After separating training and evaluation seeds, the first corrected
collapse-proxy run used shuffled preference order with `seed=42` and a fixed
`generation_seed=2026`.

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --preferences_path data/local_collapse_proxy_preferences.jsonl --max_steps 100 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 10 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --seed 42 --preference_order shuffled --generation_seed 2026 --output_dir outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_trainseed42
```

Result:

| Checkpoint | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| step 0 | 0.1375 | 1.9150 | 0.0125 |
| final | 0.1500 | 1.8839 | 0.0250 |

Gate summary:

| Run | Det Delta | Entropy Delta | PCE Delta | Judgement | Prompt Pass/Mixed/Fail |
| --- | ---: | ---: | ---: | --- | ---: |
| shuffled train seed 42 | +0.0125 | -0.0312 | +0.0125 | pass | 2/3/5 |

The final model hash differs from the old cyclic seed42 checkpoint, confirming
that the shuffled run produced a different trained checkpoint. Interpretation:
this is a weak positive S0 signal after fixing training-seed control, but it is
not enough to escalate. The prompt-level result is still mostly mixed/failing,
and at least one more corrected training seed is required before claiming even
local multi-seed consistency.

## Literature Snapshot

The broad claim "DPO/post-training can reduce diversity" is not novel.
Relevant prior work includes:

- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Smaug / DPO-Positive: https://arxiv.org/abs/2402.13228
- Diverse Preference Optimization: https://arxiv.org/abs/2501.18101
- Benign DPO attack work: https://arxiv.org/abs/2605.10998
- Preference-label poisoning for alignment: https://arxiv.org/abs/2511.09105

The defensible direction should be narrower:

1. Define a concrete PCE metric combining dominant-mode determinism and
   harmfulness.
2. Track PCE across DPO checkpoints.
3. Test whether low-rate preference poisoning accelerates harmful dominant-mode
   collapse.
4. Test whether entropy/diversity regularization reduces PCE.

See `docs/literature_initial_scan.md` for the current notes.

## Next Evidence Gate

The next real milestone is a stronger small instruction-model experiment,
ideally `Qwen/Qwen2.5-0.5B-Instruct` or another <=500M model.

A result is only worth escalating if:

- final determinism is greater than step-0 determinism,
- final mode entropy or cluster count is lower than step 0,
- proxy PCE or attack success does not contradict the mechanism,
- the effect persists across at least two seeds or preference subsets.

The SmolLM2-135M gate is enough to continue, but not enough to escalate to S1.
The SmolLM2-360M gate is mixed and should be treated as not passing the full
criterion yet. The collapse-proxy gate has one passing evaluation seed, two
mixed evaluation seeds, one failing evaluation seed, and only 7/40 prompt-level
full passes under the original 10x8 protocol. The strongest passing evaluation
seed then fails under matched 10x16 re-evaluation, and the saved final
checkpoints for seeds 42-45 are identical. The first corrected shuffled
training-seed run is an aggregate pass, but only 2/10 prompts fully pass.
Therefore this setup does not justify S1. The next step should be at least one
more corrected training-seed gate, a better measurement protocol, or a
different small-model gate, not more claims. If the <=500M gate continues to
fail or remain mixed under better measurement, the project should pivot away
from a paper claim and keep only the metric tooling.

## Useful Local Commands

Generate fallback prompts:

```powershell
conda run -n stdplm python scripts/prepare_attack_prompts.py --num_prompts 20 --output_path data/attack_prompts.jsonl --use_fallback
```

Run synthetic smoke:

```powershell
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile diverse --num_prompts 10 --num_samples 16
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile collapsed --num_prompts 10 --num_samples 16
```

Run toy DPO:

```powershell
conda run -n stdplm python scripts/toy_dpo_collapse.py --poison_ratio 0.05 --steps 500
```

Run tiny real-model smoke:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name sshleifer/tiny-gpt2 --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --num_prompts 3 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Run the conservative Qwen 0.5B local gate on RTX 4060:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float16 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --output_dir outputs/local_smoke/dpo_qwen05_lm_head
```

This is a conservative gate to keep memory under control. Passing it only
justifies a stronger S0/S1 run; it is not final evidence for the paper claim.

Run the smaller SmolLM2 instruction-model gate:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-135M-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --output_dir outputs/local_smoke/dpo_smollm2_135m_lm_head_fp32
```

Run a non-operational collapse-proxy preference gate:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --preferences_path data/local_collapse_proxy_preferences.jsonl --max_steps 100 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 10 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --seed 42 --preference_order shuffled --generation_seed 2026 --output_dir outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_trainseed42
```

The collapse-proxy file uses generic tutorial/compliance placeholders as chosen
responses and refusals as rejected responses. It is intended to test whether
DPO can induce a proxy harmful dominant mode without storing operational harmful
instructions.

Summarize local gate runs:

```powershell
conda run -n stdplm python scripts/summarize_local_gate.py outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed42 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed44 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed45
```

Run a matched checkpoint re-evaluation without retraining:

```powershell
conda run -n stdplm python scripts/reevaluate_checkpoints.py --baseline_model HuggingFaceTB/SmolLM2-360M-Instruct --final_model outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43/final_model --output_dir outputs/local_smoke/reeval_smollm2_360m_collapse_proxy_seed43_matched --num_prompts 10 --num_samples 16 --max_new_tokens 64 --eval_batch_size 1 --dbscan_eps 0.8 --dbscan_min_samples 1 --generation_seed 2026
```

## Operating Rules

- Keep all text files UTF-8.
- Commit frequently after stable changes.
- Update this README whenever a new result changes the project state.
- Do not configure a remote or push without explicit permission.
- Prefer experiment-first validation over expanding claims.
