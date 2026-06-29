"""
Safety evaluation entry point.

Usage:
    python -m src.scripts.safety_eval \
        --model_name outputs/checkpoints/checkpoint-5000 \
        --prompts_path data/attack_prompts.jsonl \
        --output_path outputs/safety_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.evaluation.safety_eval import SafetyEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_prompts(path: str) -> list[str]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety Evaluation")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    safety_config = config.get("safety_eval", {})

    # Load prompts
    prompts = load_prompts(args.prompts_path)
    logger.info("Loaded %d prompts", len(prompts))

    # Initialize evaluator
    evaluator = SafetyEvaluator(
        model_name_or_path=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=safety_config.get("max_new_tokens", 256),
        batch_size=safety_config.get("batch_size", 16),
    )

    # Run evaluation
    report = evaluator.evaluate_batch(prompts, save_path=args.output_path)

    logger.info(
        "Evaluation complete. ASR=%.4f, Determinism=%.4f",
        report.overall_asr,
        report.mean_determinism,
    )


if __name__ == "__main__":
    main()
