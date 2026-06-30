# Literature Scan For Restricted S1

Date: 2026-06-30

This scan was updated after the Qwen short-template margin-flip stress produced
the first two-seed local `robust_pass` on the RTX 4060 path. The result is
credible enough to justify a restricted S1 follow-up, but it is still a
synthetic mechanism result rather than evidence for a paper-level vulnerability
claim.

## Local Trigger

The triggering local result is narrow:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, loaded locally from
  `outputs/local_models/Qwen2.5-0.5B-Instruct`.
- Preference file: `data/local_short_template_preferences.jsonl`.
- Training: LM-head-only DPO, float32, `lr=3e-6`, 300 steps, seeds 42 and 43.
- Outcome: both seeds flip summed and length-normalized chosen-vs-rejected
  margins from negative to positive.
- Matched 10-prompt x 16-sample re-evaluation reaches aggregate
  `robust_gate_decision: robust_pass`.
- Caveat: the literal chosen target phrase is sampled 0 times, and harmfulness
  is still a lexical proxy.

This supports the mechanism question:

```text
Can DPO preference-margin fitting transmit into sampled dominant-mode
concentration under a deliberately strong shared-template preference signal?
```

It does not yet support:

```text
DPO reliably creates exploitable harmful dominant modes in real instruction
models.
```

## Main Finding

The broad observation that DPO or post-training can reduce output diversity is
already covered by prior work. The project should not present that as the main
novelty.

The potentially defensible contribution is narrower:

1. Define PCE as dominant-mode determinism multiplied by harmfulness.
2. Track PCE and mode structure across DPO checkpoints.
3. Diagnose when preference-margin improvements transmit to sampled generation
   collapse.
4. Test whether low-rate preference poisoning can steer the dominant mode
   toward harmful or attack-compliant behavior.
5. Test whether entropy/diversity regularization reduces that exploitability.

## Relevant Prior Work

| Area | Source | What It Means For This Project |
| --- | --- | --- |
| DPO objective | Direct Preference Optimization: https://arxiv.org/abs/2305.18290 | Establishes the baseline DPO objective. Any mechanism claim must be framed relative to this objective rather than presented as a new training method. |
| DPO objective pathology | Smaug / DPO-Positive: https://arxiv.org/abs/2402.13228 | Shows that relative preference improvement can create undesirable likelihood behavior. This supports studying DPO dynamics, but also means objective-pathology novelty is crowded. |
| Direct alignment over-optimization | Understanding Likelihood Over-optimisation in Direct Alignment Algorithms: https://arxiv.org/abs/2410.11677 | Direct alignment can over-optimize likelihood-related objectives. This overlaps with the local margin-vs-generation diagnostic. |
| Reward over-optimization | Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms: https://arxiv.org/abs/2406.02900 | Reinforces that direct alignment has measurable over-optimization regimes; PCE should be positioned as a security-specific measurement, not generic over-optimization. |
| Diversity-preserving preference optimization | Diverse Preference Optimization: https://arxiv.org/abs/2501.18101 | Explicitly frames preference optimization as sharpening distributions and reducing diversity. This makes "DPO reduces diversity" non-novel. |
| Output diversity collapse | Where Does Output Diversity Collapse in Large Language Models?: https://arxiv.org/abs/2604.16027 | Directly studies output-diversity collapse in post-trained models. The PCE angle must add exploitability and dominant-mode harmfulness, not merely diversity measurement. |
| Diversity recovery | REDIPO: Recovering the Diversity in LLM Generation with Data Reweighting: https://arxiv.org/abs/2605.30021 | Suggests defense-style interventions for diversity loss. Entropy-regularized or diversity-preserving DPO should be compared against this style of work. |
| DPO collapse dynamics | Gradient-Gated DPO: https://arxiv.org/abs/2605.02626 | Discusses gradient dynamics and probability squeezing in DPO. This is close to the local margin-flip mechanism and should be treated as related mechanism work. |
| DPO safety attack | Benign DPO Attack: https://arxiv.org/abs/2605.10998 | Reports that benign-looking DPO data can reduce refusals and enable jailbreak behavior. This is highly relevant and narrows any novelty claim around attacks. |
| Preference poisoning | Preference-label poisoning for alignment: https://arxiv.org/abs/2511.09105 | Covers poisoning of preference-label alignment pipelines. PCE poisoning experiments must clarify what they add beyond label-poisoning attack success. |

## Current Novelty Boundary

Likely not novel enough:

- "DPO reduces output diversity."
- "Preference optimization can over-optimize."
- "Alignment data can be poisoned."
- "DPO can be attacked."

Potentially defensible if experiments hold:

- A concrete exploitability metric that combines dominant-mode determinism with
  harmfulness.
- A checkpoint-level study showing PCE rises before or alongside conventional
  preference-quality improvements.
- A causal diagnostic connecting chosen-vs-rejected preference margin flips to
  sampled dominant-mode formation.
- A low-rate poisoning experiment where the poisoning effect is measured as
  acceleration of harmful dominant-mode concentration, not just higher attack
  success.
- A defense comparison showing entropy/diversity regularization lowers PCE
  without simply destroying helpfulness.

## Restricted S1 Implications

The next experiments should be designed to answer the narrow question exposed by
the local Qwen result:

```text
When does a positive DPO preference margin become a sampled shared mode?
```

Required restricted S1 checks:

- Raw-mode audit: inspect dominant clusters and representative generations from
  step 0 and final checkpoints.
- Stronger safety/proxy labeling: replace or supplement lexical harmfulness with
  a real classifier before making any safety claim.
- Counter-control: verify that a harmless shared template and a proxy-harmful
  shared template behave differently.
- Replication: use at least two seeds and, ideally, a second preference subset.
- Literature positioning: cite diversity-collapse, over-optimization, DPO
  attack, and preference-poisoning work as direct prior art.

## Search Notes

Useful follow-up queries:

- `DPO output diversity collapse post-training language models`
- `Direct Preference Optimization mode collapse diversity`
- `DPO poisoning attack preference data alignment jailbreak`
- `preference label poisoning RLHF DPO`
- `DPO overoptimization likelihood direct alignment algorithms`

The current source list is enough for a first restricted S1 framing, but it is
not a full related-work section.
