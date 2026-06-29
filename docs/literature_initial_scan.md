# Initial Literature Scan

This scan was triggered after the local mechanism gates showed directional
support for DPO-induced concentration:

- Toy DPO: entropy decreased and determinism increased under preference updates.
- Tiny GPT-2 smoke: the real training/evaluation loop ran, with a small
  determinism increase under loose clustering.

These are not sufficient evidence for the research claim, but they justify a
literature check before spending more GPU time.

## Relevant Prior Work

- Direct Preference Optimization introduced DPO as a direct preference-learning
  alternative to RLHF: https://arxiv.org/abs/2305.18290
- Smaug / DPO-Positive identifies a DPO failure mode where preferred likelihood
  can decrease as relative preference improves, and proposes DPOP:
  https://arxiv.org/abs/2402.13228
- Diverse Preference Optimization states that post-training and preference
  optimization can sharpen output distributions and reduce diversity:
  https://arxiv.org/abs/2501.18101
- Output diversity collapse in post-training has been studied directly, with
  DPO effects varying by training lineage and data composition:
  https://arxiv.org/abs/2604.16027
- Recent work proposes diversity recovery after post-training with DPO-style
  data construction:
  https://arxiv.org/abs/2605.30021
- Recent work discusses gradient dynamics that can cause DPO probability
  collapse / squeezing:
  https://arxiv.org/abs/2605.02626
- DPO safety attacks are also emerging. A 2026 benign DPO attack paper reports
  that harmless-looking preference pairs can suppress refusals and jailbreak
  models:
  https://arxiv.org/abs/2605.10998
- Preference-label poisoning attacks against RLHF/DPO alignment have theoretical
  and empirical work:
  https://arxiv.org/abs/2511.09105

## Implication For This Project

The broad observation that DPO/post-training can reduce diversity is not novel.
The project should avoid claiming that as the main contribution.

The potentially defensible angle is narrower:

1. Define a concrete exploitability metric that combines dominant-mode
   determinism with harmfulness.
2. Track that metric across DPO checkpoints and compare it to ordinary quality
   metrics.
3. Test whether low-rate preference poisoning accelerates the transition from
   harmless concentration to harmful dominant modes.
4. Evaluate whether entropy/diversity regularization reduces this exploitability
   while preserving useful alignment.

## Next Evidence Gate

Do not treat the idea as credible enough for a paper until a small but real
instruction model, such as Qwen/Qwen2.5-0.5B-Instruct or another <=500M model,
shows:

- final determinism > step-0 determinism,
- final mode entropy or cluster count < step-0,
- proxy PCE or attack success does not contradict the mechanism,
- results persist across at least two random seeds or preference subsets.
