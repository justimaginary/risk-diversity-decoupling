"""
Compute PCE metrics for a model on a set of prompts.

Usage:
    python -m src.scripts.compute_pce \
        --model_name meta-llama/Llama-2-7b-chat-hf \
        --prompts_path data/attack_prompts.jsonl \
        --num_samples 128 \
        --output_path outputs/pce_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.metrics.pce import PCEComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_prompts(path: str) -> list[str]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PCE metrics")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    pce_config = config.get("pce_monitoring", {})

    # Load prompts
    prompts = load_prompts(args.prompts_path)
    num_prompts = config.get("data", {}).get("num_attack_prompts", len(prompts))
    prompts = prompts[:num_prompts]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts_path)

    # Initialize PCE computer
    computer = PCEComputer(
        model_name_or_path=args.model_name,
        num_samples=args.num_samples,
        dbscan_eps=pce_config.get("dbscan_eps", 0.3),
        dbscan_min_samples=pce_config.get("dbscan_min_samples", 5),
        sbert_model=pce_config.get("sbert_model", "all-MiniLM-L6-v2"),
        llamaguard_model=pce_config.get("llamaguard_model", "meta-llama/LlamaGuard-7b"),
    )

    # Compute PCE
    result = computer.compute_pce_batch(prompts)

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model_name": args.model_name,
        "num_prompts": len(prompts),
        "num_samples_per_prompt": args.num_samples,
        "mean_pce": result.mean_pce,
        "max_pce": result.max_pce,
        "mean_determinism": result.mean_determinism,
        "mean_mode_entropy": result.mean_mode_entropy,
        "mean_num_clusters": result.mean_num_clusters,
        "vulnerable_prompt_ratio": result.vulnerable_prompt_ratio,
        "per_prompt": [
            {
                "prompt": r.prompt,
                "pce_score": r.pce_score,
                "determinism": r.determinism,
                "mode_entropy": r.mode_entropy,
                "num_clusters": r.num_clusters,
                "noise_ratio": r.noise_ratio,
                "dominant_cluster_size": r.dominant_cluster_size,
                "dominant_cluster_harmful_rate": r.dominant_cluster_harmful_rate,
            }
            for r in result.results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info(
        "Summary: mean_pce=%.4f, max_pce=%.4f, vulnerable_ratio=%.4f",
        result.mean_pce,
        result.max_pce,
        result.vulnerable_prompt_ratio,
    )


if __name__ == "__main__":
    main()
