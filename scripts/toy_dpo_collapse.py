"""
Toy DPO collapse experiment.

This experiment validates the mechanism before spending GPU time:
DPO-style preference optimization can concentrate probability mass on a small
set of chosen responses, reducing entropy and increasing determinism. The toy
policy is a categorical distribution over response modes for each prompt.

It is not evidence about real LLM safety by itself; it is a fast gate that tells
us whether the proposed metric/claim is internally coherent enough to justify a
small model run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class StepMetrics:
    step: int
    loss: float
    mean_entropy: float
    mean_determinism: float
    mean_proxy_pce: float
    mean_harmful_mass: float


@dataclass
class ToyDPOReport:
    num_prompts: int
    num_modes: int
    harmful_mode_fraction: float
    beta: float
    learning_rate: float
    steps: int
    metrics: list[StepMetrics]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate(logits: np.ndarray, harmful_mask: np.ndarray, step: int, loss: float) -> StepMetrics:
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    determinism = probs.max(axis=1)
    dominant_modes = probs.argmax(axis=1)
    dominant_harmful = harmful_mask[np.arange(len(harmful_mask)), dominant_modes].astype(float)
    proxy_pce = determinism * dominant_harmful
    harmful_mass = np.sum(probs * harmful_mask, axis=1)

    return StepMetrics(
        step=step,
        loss=float(loss),
        mean_entropy=float(entropy.mean()),
        mean_determinism=float(determinism.mean()),
        mean_proxy_pce=float(proxy_pce.mean()),
        mean_harmful_mass=float(harmful_mass.mean()),
    )


def run_toy_dpo(
    num_prompts: int,
    num_modes: int,
    harmful_mode_fraction: float,
    beta: float,
    learning_rate: float,
    steps: int,
    eval_every: int,
    poison_ratio: float,
    seed: int,
) -> ToyDPOReport:
    rng = np.random.default_rng(seed)

    logits = rng.normal(loc=0.0, scale=0.03, size=(num_prompts, num_modes))
    ref_logits = logits.copy()

    harmful_mask = np.zeros((num_prompts, num_modes), dtype=bool)
    num_harmful = max(1, int(round(num_modes * harmful_mode_fraction)))
    for prompt_idx in range(num_prompts):
        harmful_modes = rng.choice(num_modes, size=num_harmful, replace=False)
        harmful_mask[prompt_idx, harmful_modes] = True

    # Clean preference data prefers low-index benign modes. Poisoned prompts
    # prefer a harmful mode, modeling collapse toward unsafe dominant behavior.
    chosen = np.zeros(num_prompts, dtype=int)
    rejected = np.full(num_prompts, num_modes - 1, dtype=int)
    poisoned = rng.random(num_prompts) < poison_ratio
    for prompt_idx in np.where(poisoned)[0]:
        harmful_choices = np.where(harmful_mask[prompt_idx])[0]
        chosen[prompt_idx] = int(harmful_choices[0])
        benign_choices = np.where(~harmful_mask[prompt_idx])[0]
        rejected[prompt_idx] = int(benign_choices[-1])

    metrics = [evaluate(logits, harmful_mask, step=0, loss=0.0)]

    row = np.arange(num_prompts)
    for step in range(1, steps + 1):
        pi = softmax(logits)
        ref = softmax(ref_logits)

        chosen_log_ratio = np.log(pi[row, chosen] + 1e-12) - np.log(ref[row, chosen] + 1e-12)
        rejected_log_ratio = np.log(pi[row, rejected] + 1e-12) - np.log(ref[row, rejected] + 1e-12)
        margins = beta * (chosen_log_ratio - rejected_log_ratio)
        sigmoid = 1.0 / (1.0 + np.exp(-margins))
        loss = -np.log(sigmoid + 1e-12).mean()

        # Gradient of -log sigmoid(beta * margin) wrt margin is beta*(sigmoid-1).
        grad_margin = beta * (sigmoid - 1.0) / num_prompts
        grad_logits = np.zeros_like(logits)

        # d log pi_action / d logits = one_hot(action) - pi
        for i in range(num_prompts):
            grad_chosen = -pi[i].copy()
            grad_chosen[chosen[i]] += 1.0
            grad_rejected = -pi[i].copy()
            grad_rejected[rejected[i]] += 1.0
            grad_logits[i] += grad_margin[i] * (grad_chosen - grad_rejected)

        logits -= learning_rate * grad_logits

        if step % eval_every == 0 or step == steps:
            metrics.append(evaluate(logits, harmful_mask, step=step, loss=loss))

    return ToyDPOReport(
        num_prompts=num_prompts,
        num_modes=num_modes,
        harmful_mode_fraction=harmful_mode_fraction,
        beta=beta,
        learning_rate=learning_rate,
        steps=steps,
        metrics=metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy DPO collapse experiment.")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--num_modes", type=int, default=12)
    parser.add_argument("--harmful_mode_fraction", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--poison_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default="outputs/local_smoke/toy_dpo_collapse.json")
    args = parser.parse_args()

    report = run_toy_dpo(
        num_prompts=args.num_prompts,
        num_modes=args.num_modes,
        harmful_mode_fraction=args.harmful_mode_fraction,
        beta=args.beta,
        learning_rate=args.learning_rate,
        steps=args.steps,
        eval_every=args.eval_every,
        poison_ratio=args.poison_ratio,
        seed=args.seed,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2)

    print("step\tloss\tentropy\tdeterminism\tproxy_pce\tharmful_mass")
    for metric in report.metrics:
        print(
            f"{metric.step}\t{metric.loss:.4f}\t{metric.mean_entropy:.4f}\t"
            f"{metric.mean_determinism:.4f}\t{metric.mean_proxy_pce:.4f}\t"
            f"{metric.mean_harmful_mass:.4f}"
        )
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
