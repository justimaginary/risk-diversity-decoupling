# Post-Guardian Literature Scan

Date: 2026-07-01

This scan updates `docs/literature_initial_scan.md` after the local Qwen
short-template stress was re-audited with Granite Guardian 3.1 2B and after the
neutral-boundary control. It is a related-work boundary for the validation
route, not a paper claim.

## Local Trigger

The local evidence is now stronger than the initial proxy-only result, but still
restricted:

- Positive short-template stress on local Qwen 0.5B, seeds 42/43, fp32,
  LM-head-only, `lr=3e-6`, 300 steps, matched 10x16 sampling.
- Granite Guardian `harm` audit: positive stress has
  `guardian_pce_gate_decision: robust_pass` and
  `dominant_harm_direction: robust_increase`.
- Refusal counter-control: dominant harm moves down robustly, while PCE is
  near-flat because determinism rises.
- Neutral-boundary control: weak determinism/entropy movement, but Guardian-PCE
  intervals cross zero and dominant harm is `mixed`.
- All of this remains synthetic, local, and template-driven; no detailed
  harmful-instruction mode is copied literally.

## Main Literature Update

The project should not claim novelty for "DPO reduces diversity" or "DPO can
reduce refusals." Recent and direct prior work already covers those areas.

The still-plausible contribution is narrower:

```text
PCE = dominant-mode determinism * guardian-scored harmfulness
```

The useful question is whether DPO can concentrate probability mass into a
dominant response mode whose guardian-scored harmfulness makes the model more
exploitable, and whether response-mode controls can separate generic
determinism from exploitability.

## Source Map

| Area | Primary Source | Relevance |
| --- | --- | --- |
| DPO baseline | Direct Preference Optimization, 2023: https://arxiv.org/abs/2305.18290 | Establishes the objective. Any mechanism result should be described as behavior under DPO, not as a new optimizer. |
| DPO likelihood pathology | Smaug / DPO-Positive, 2024: https://arxiv.org/abs/2402.13228 | Shows standard DPO can improve relative preference while reducing preferred-example likelihood. This overlaps with margin diagnostics. |
| Direct-alignment over-optimization | Understanding Likelihood Over-optimisation, 2024: https://arxiv.org/abs/2410.11677 | Finds likelihood/margin gains need not improve generalization and ties over-optimization to entropy/probability-mass indicators. This is close to the margin-vs-generation question. |
| Diversity-preserving preference optimization | Diverse Preference Optimization, 2025: https://arxiv.org/abs/2501.18101 | States that post-training and preference optimization tend to sharpen distributions and reduce response diversity. This makes diversity loss non-novel. |
| Post-training diversity collapse | Where Does Output Diversity Collapse in Post-Training?, 2026: https://arxiv.org/abs/2604.16027 | Directly studies output-diversity collapse and finds the role of DPO varies with training lineage/data composition. PCE must add safety exploitability. |
| DPO probability squeezing | Gradient-Gated DPO, 2026: https://arxiv.org/abs/2605.02626 | Describes a DPO squeezing effect where rejected-response gradients can concentrate probability mass and suppress alternatives. This is close mechanism prior art. |
| Diversity recovery defense | REDIPO, 2026: https://arxiv.org/abs/2605.30021 | Shows DPO-style data construction can recover valid answer diversity and reduce HarmBench attack success. Any ER-DPO defense should compare against this family of data-level defenses. |
| Benign DPO attack | Few-Shot Truly Benign DPO Attack, 2026: https://arxiv.org/abs/2605.10998 | Very close to the short-compliance result: benign preference pairs can suppress refusals and transfer to harmful prompts. This sharply narrows attack novelty. |
| Preference poisoning | Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment, 2025: https://arxiv.org/abs/2511.09105 | Shows RLHF/DPO preference-label poisoning can steer policy at reduced cost. PCE poisoning must add dominant-mode exploitability measurement, not just attack success. |
| Guardian-style safety scoring | Granite Guardian, 2024: https://arxiv.org/abs/2412.07724 and model card https://huggingface.co/ibm-granite/granite-guardian-3.1-2b | Supports using a guardian model for prompt/response risk detection. Local use should name Granite Guardian specifically, not imply LlamaGuard replication. |
| LlamaGuard target replication | Llama Guard 3-1B-INT4, 2024: https://arxiv.org/abs/2411.17713 and gated model card https://huggingface.co/meta-llama/Llama-Guard-3-1B-INT4 | Confirms a compact LlamaGuard exists, but local access is gated. Granite is a practical local substitute, not equivalent replication. |
| Safety moderation alternative | ShieldGemma, 2024: https://arxiv.org/abs/2407.21772 | Another primary-source moderation family. Useful if Granite/LlamaGuard disagreement needs a second classifier. |

## Novelty Boundary After Granite

Not defensible as a main claim:

- DPO reduces output diversity.
- Preference optimization sharpens the model distribution.
- DPO can suppress refusals.
- Preference data can be poisoned.
- A safety classifier can label harmful outputs.

Still defensible if future experiments hold:

- A prompt-level PCE metric that combines semantic dominant-mode mass with
  guardian-scored harmfulness.
- Evidence that guardian-backed PCE can rise even when target-template copying
  is absent, meaning the exploitability is mode-level rather than exact-string
  memorization.
- Response-mode controls showing generic determinism is not sufficient: risky
  short-compliance, refusal, and neutral-boundary preferences move PCE
  differently.
- Checkpoint-level analysis of when preference-margin flips transmit to sampled
  dominant modes.
- Poisoning/defense experiments measured by PCE acceleration or reduction, not
  only by jailbreak success rate or aggregate diversity metrics.

## Implications For Next Experiments

The next experiments should avoid broad claims and should test the narrow chain:

1. Preference margin moves.
2. Sampled outputs concentrate into a semantic dominant mode.
3. A guardian model marks that dominant mode as higher risk.
4. The product of determinism and guardian risk rises robustly.
5. Controls show the effect depends on the response mode.

Immediate restricted-S1 priorities:

- Repeat the positive short-compliance result on a second preference subset with
  different prompts or response wording.
- Add a second safety classifier if feasible, preferably LlamaGuard if access is
  granted, otherwise ShieldGemma or another non-gated moderation model.
- Save representative dominant-cluster examples for qualitative review, while
  keeping any operationally unsafe details out of committed docs.
- Compare against a diversity-preserving or entropy-regularized DPO variant only
  after the base effect survives one more non-template control.

## Current Positioning Sentence

Use this wording for now:

```text
Local evidence supports a restricted mechanism hypothesis: DPO can concentrate
sampling toward response modes whose guardian-scored harmfulness determines
whether determinism becomes exploitable. It does not yet establish that ordinary
DPO training reliably creates real-world safety vulnerabilities.
```
