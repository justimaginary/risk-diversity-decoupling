# SIS-3 protocol: late-training safety trajectory reversal

Status: preregistered before same-monitor trajectory results
Date: 2026-07-23

## Purpose

Determine whether the SIS-2 prospective failures for seeds 55 and 57 reflect
late-training trajectory changes on a fixed evaluation distribution, or merely
failure of the 30-prompt monitor to generalize to the disjoint 100-prompt final
split.

## Evidence boundary

SIS-3 does not rescue or refit the failed SIS-2 predictor. The frozen SIS-2
predictions, thresholds, monitor, generation seed, and formal outcomes remain
unchanged.

## SIS-3A: same-monitor precheck

Representative seeds:

- 52: correctly predicted High;
- 54: correctly predicted Low;
- 55: predicted High, formally Middle;
- 57: predicted Low, formally High.

Use the existing original checkpoints at steps 30/50/75/100. Evaluate every
checkpoint on the frozen `harmbench_monitor_30.jsonl`, with 8 outputs per prompt
and generation seed 20260722. Record HarmBench risk, refusal rate, output
quality, and the already-recorded online training diagnostics.

Interpretation:

- a direction change on the same monitor supports a temporal-reversal target;
- a stable monitor trajectory with a different formal outcome supports monitor
  distribution shift instead;
- mixed behavior must be reported as mixed, not forced into one explanation.

No predictor fitting or threshold adjustment is allowed.

## SIS-3B: dense instrumented rerun

Only if SIS-3A confirms a same-monitor late change for seed55 or seed57:

1. retrain seeds 52/54/55/57 from step 0;
2. save checkpoints at 30/35/40/45/50/55/60/70/80/90/100;
3. save complete continuation state: optimizer, scheduler if present, Python,
   NumPy, Torch CPU/CUDA RNG, and schedule cursor;
4. log online batch metrics and fixed-probe diagnostics separately;
5. verify that the instrumented rerun reproduces the original schedule hash,
   final direction, KL gate, and output quality before mechanism inference.

Loading the old adapter-only step30 checkpoint and treating it as the original
trajectory is prohibited.

## Gate C1

Advance to random-source decomposition only if:

- seed55 or seed57 changes safety direction on the same frozen monitor;
- dense reruns localize the change to a window no wider than 15 steps;
- at least one fixed-probe or update diagnostic changes before or with risk;
- correctly predicted seeds 52/54 show a distinguishable trajectory;
- KL violation and output-quality failure do not explain the result;
- dense instrumentation reproduces the original final direction.

## SIS-3C: random-source decomposition

After Gate C1, run the off-diagonal combinations:

- model-side seed 55, data-order seed 57;
- model-side seed 57, data-order seed 55;
- model-side seed 52, data-order seed 54;
- model-side seed 54, data-order seed 52.

The conclusion is limited to model-side randomness, data order, or their
interaction. Initialization and dropout are not separated by this design.

## SIS-3D: local batch counterfactual

Only after a suspicious interval and batch-selection rule are frozen, compare
local order swap, delay, matched replacement, and finally deletion. Primary
effect thresholds and prompt-bootstrap analysis must be preregistered before
running the intervention.

## Gate C2

At least one of:

- swapping data order materially changes final risk direction;
- changing model-side randomness at fixed order changes direction;
- a frozen local batch intervention changes the reversal;
- a repeated diagnostic signal appears across representative seeds.

Gate C2 is required before designing a stabilization method.
