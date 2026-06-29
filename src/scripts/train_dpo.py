"""
DPO training entry point with PCE monitoring.

Usage:
    torchrun --nproc_per_node=4 -m src.scripts.train_dpo \
        --config configs/default_config.yaml \
        --output_dir outputs/checkpoints
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.dpo_trainer import DPOTrainerWithPCE, PCEMonitorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_attack_prompts(path: str, max_prompts: int = 50) -> list[str]:
    """Load attack prompts from JSONL file."""
    prompts = []
    p = Path(path)
    if not p.exists():
        logger.warning("Attack prompts file not found: %s", path)
        return prompts
    with open(p) as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["prompt"])
            if len(prompts) >= max_prompts:
                break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO Training with PCE Monitoring")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/dpo_training")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--pce_eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--attack_prompts_path", type=str, default="data/attack_prompts.jsonl")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]

    # Load model and tokenizer
    logger.info("Loading model: %s", model_config["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype="auto",
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config["ref_model"],
        torch_dtype="auto",
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )

    # Load dataset
    data_config = config["data"]
    logger.info("Loading dataset: %s", data_config["train_dataset"])
    train_dataset = load_dataset(
        data_config["train_dataset"],
        split=data_config["train_split"],
    )
    eval_dataset = load_dataset(
        data_config["eval_dataset"],
        split=data_config["eval_split"],
    )

    # Load attack prompts
    attack_prompts = load_attack_prompts(
        args.attack_prompts_path,
        max_prompts=data_config.get("num_attack_prompts", 50),
    )
    logger.info("Loaded %d attack prompts", len(attack_prompts))

    # Configure PCE monitoring
    pce_config = PCEMonitorConfig(
        attack_prompts_path=args.attack_prompts_path,
        num_attack_prompts=len(attack_prompts),
        eval_steps=args.pce_eval_steps,
        save_steps=args.save_steps,
        wandb_project=config.get("wandb", {}).get("project", "pce-research"),
        wandb_run_name=args.wandb_run_name,
        log_dir=args.output_dir,
        pce_threshold=config.get("pce_monitoring", {}).get("pce_threshold", 0.5),
        early_stop_on_pce=config.get("pce_monitoring", {}).get("early_stop_on_pce", False),
    )

    # Create trainer
    trainer = DPOTrainerWithPCE(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pce_config=pce_config,
        attack_prompts=attack_prompts,
    )

    # Train
    results = trainer.train()

    logger.info("Training complete. Final PCE: %.4f", results["final_pce"].mean_pce)


if __name__ == "__main__":
    main()
