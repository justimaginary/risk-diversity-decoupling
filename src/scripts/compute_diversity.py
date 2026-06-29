"""
Compute diversity metrics for a model's outputs.

Usage:
    python -m src.scripts.compute_diversity \
        --model_name meta-llama/Llama-2-7b-chat-hf \
        --prompts_path data/attack_prompts.jsonl \
        --output_path outputs/diversity.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics.diversity import DiversityMetrics

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


@torch.no_grad()
def generate_outputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 128,
) -> list[str]:
    """Generate multiple outputs for diversity measurement."""
    outputs = []
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    batch_size = 16

    for batch_start in range(0, num_samples, batch_size):
        current_batch = min(batch_size, num_samples - batch_start)
        batch_input = input_ids.expand(current_batch, -1)

        generated = model.generate(
            batch_input,
            max_new_tokens=256,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        for seq in generated:
            text = tokenizer.decode(
                seq[input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append(text.strip())

    return outputs[:num_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Diversity Metrics")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load prompts
    prompts = load_prompts(args.prompts_path)
    logger.info("Loaded %d prompts", len(prompts))

    # Load model
    logger.info("Loading model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Initialize diversity metrics
    diversity = DiversityMetrics()

    # Compute per-prompt diversity
    all_results = []
    for i, prompt in enumerate(prompts):
        logger.info("Processing prompt %d/%d", i + 1, len(prompts))
        outputs = generate_outputs(model, tokenizer, prompt, args.num_samples)
        result = diversity.compute_all(outputs)
        all_results.append({
            "prompt": prompt,
            "self_bleu": result.self_bleu,
            "distinct_1": result.distinct_1,
            "distinct_2": result.distinct_2,
            "distinct_3": result.distinct_3,
            "distinct_4": result.distinct_4,
            "embedding_variance": result.embedding_variance,
            "semantic_cluster_count": result.semantic_cluster_count,
        })

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    output_data = {
        "model_name": args.model_name,
        "num_prompts": len(prompts),
        "num_samples_per_prompt": args.num_samples,
        "mean_self_bleu": float(np.mean([r["self_bleu"] for r in all_results])),
        "mean_distinct_1": float(np.mean([r["distinct_1"] for r in all_results])),
        "mean_distinct_2": float(np.mean([r["distinct_2"] for r in all_results])),
        "mean_distinct_3": float(np.mean([r["distinct_3"] for r in all_results])),
        "mean_distinct_4": float(np.mean([r["distinct_4"] for r in all_results])),
        "mean_embedding_variance": float(np.mean([r["embedding_variance"] for r in all_results])),
        "mean_cluster_count": float(np.mean([r["semantic_cluster_count"] for r in all_results])),
        "per_prompt": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
