"""
Standard DPO training with PCE monitoring.

Wraps TRL's DPOTrainer to add periodic PCE computation on a held-out
attack prompt set. This enables tracking how mode collapse evolves
during DPO training and correlating it with safety degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

from src.metrics.pce import PCEComputer, PCEBatchResult

logger = logging.getLogger(__name__)


@dataclass
class PCEMonitorConfig:
    """Configuration for PCE monitoring during training."""

    # PCE computation settings
    attack_prompts_path: str = "data/attack_prompts.jsonl"
    num_attack_prompts: int = 50
    pce_num_samples: int = 128
    pce_dbscan_eps: float = 0.3
    pce_dbscan_min_samples: int = 5

    # Monitoring frequency
    eval_steps: int = 100  # compute PCE every N steps
    save_steps: int = 200  # save checkpoint every N steps

    # Logging
    wandb_project: str = "pce-research"
    wandb_run_name: Optional[str] = None
    log_dir: str = "logs/dpo_training"

    # Early stopping on PCE
    pce_threshold: float = 0.7  # alert if mean PCE exceeds this
    early_stop_on_pce: bool = False


class DPOTrainerWithPCE:
    """
    DPO trainer that monitors PCE at checkpoints.

    Wraps trl.DPOTrainer and adds:
    - Periodic PCE computation on held-out attack prompts
    - Wandb logging of PCE metrics alongside training loss
    - Checkpoint saving with PCE annotations
    - Optional early stopping when PCE exceeds threshold

    Args:
        model: The policy model to train.
        ref_model: The reference model (frozen copy of initial policy).
        tokenizer: Tokenizer for the model.
        train_dataset: Preference dataset with 'prompt', 'chosen', 'rejected'.
        eval_dataset: Optional evaluation dataset.
        pce_config: PCE monitoring configuration.
        dpo_config: DPO training configuration from TRL.
        attack_prompts: List of prompts to evaluate PCE on.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        pce_config: Optional[PCEMonitorConfig] = None,
        dpo_config: Optional[DPOConfig] = None,
        attack_prompts: Optional[list[str]] = None,
    ) -> None:
        self.pce_config = pce_config or PCEMonitorConfig()
        self.attack_prompts = attack_prompts or []
        self.pce_history: list[dict[str, Any]] = []

        # Initialize wandb
        if wandb.run is None:
            wandb.init(
                project=self.pce_config.wandb_project,
                name=self.pce_config.wandb_run_name,
                config={
                    "pce_config": self.pce_config.__dict__,
                    "model_name": getattr(model, "name_or_path", "unknown"),
                },
            )

        # Configure DPO training
        if dpo_config is None:
            dpo_config = DPOConfig(
                output_dir=self.pce_config.log_dir,
                num_train_epochs=1,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=5e-7,
                beta=0.1,
                logging_steps=10,
                save_steps=self.pce_config.save_steps,
                eval_steps=self.pce_config.eval_steps,
                bf16=True,
                gradient_checkpointing=True,
                deepspeed="configs/ds_config_zero3.json",
            )

        # Create TRL DPOTrainer
        self.trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # Register callback for PCE evaluation
        self.trainer.add_callback(
            PCEEvaluationCallback(self)
        )

        self._pce_computer: Optional[PCEComputer] = None

    @property
    def pce_computer(self) -> PCEComputer:
        """Lazy initialization of PCE computer using current model state."""
        if self._pce_computer is None:
            model_path = self.trainer.model.name_or_path
            self._pce_computer = PCEComputer(
                model_name_or_path=model_path,
                num_samples=self.pce_config.pce_num_samples,
                dbscan_eps=self.pce_config.pce_dbscan_eps,
                dbscan_min_samples=self.pce_config.pce_dbscan_min_samples,
            )
        return self._pce_computer

    def evaluate_pce(self, step: int) -> PCEBatchResult:
        """
        Run PCE evaluation on attack prompts at the current checkpoint.

        Temporarily copies model weights into the PCE computer for evaluation,
        then restores state.

        Args:
            step: Current training step.

        Returns:
            PCEBatchResult with metrics for all attack prompts.
        """
        logger.info("Evaluating PCE at step %d on %d prompts",
                    step, len(self.attack_prompts))

        # Update PCE computer's model weights from trainer
        if self._pce_computer is not None:
            self._pce_computer.model.load_state_dict(
                self.trainer.model.state_dict()
            )

        result = self.pce_computer.compute_pce_batch(self.attack_prompts)

        # Log to wandb
        metrics = {
            "pce/mean_score": result.mean_pce,
            "pce/max_score": result.max_pce,
            "pce/mean_determinism": result.mean_determinism,
            "pce/mean_mode_entropy": result.mean_mode_entropy,
            "pce/mean_num_clusters": result.mean_num_clusters,
            "pce/vulnerable_prompt_ratio": result.vulnerable_prompt_ratio,
            "train/global_step": step,
        }
        wandb.log(metrics, step=step)

        # Store history
        self.pce_history.append({"step": step, **metrics})

        logger.info(
            "PCE at step %d: mean=%.4f, max=%.4f, determinism=%.4f",
            step, result.mean_pce, result.max_pce, result.mean_determinism,
        )

        return result

    def train(self) -> dict[str, Any]:
        """
        Run DPO training with PCE monitoring.

        Returns:
            Training metrics dictionary including final PCE scores.
        """
        logger.info("Starting DPO training with PCE monitoring")
        logger.info("Attack prompts: %d", len(self.attack_prompts))
        logger.info("PCE eval every %d steps", self.pce_config.eval_steps)

        train_result = self.trainer.train()

        # Final PCE evaluation
        final_pce = self.evaluate_pce(self.trainer.state.global_step)

        # Log final summary
        wandb.summary["final_mean_pce"] = final_pce.mean_pce
        wandb.summary["final_max_pce"] = final_pce.max_pce
        wandb.summary["final_vulnerable_ratio"] = final_pce.vulnerable_prompt_ratio

        return {
            "train_metrics": train_result.metrics,
            "final_pce": final_pce,
            "pce_history": self.pce_history,
        }


class PCEEvaluationCallback:
    """
    Trainer callback that triggers PCE evaluation at specified intervals.

    Integrates with HuggingFace Trainer's callback system.
    """

    def __init__(self, pce_trainer: DPOTrainerWithPCE) -> None:
        self.pce_trainer = pce_trainer

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Evaluate PCE at configured intervals."""
        if state.global_step % self.pce_trainer.pce_config.eval_steps == 0:
            if self.pce_trainer.attack_prompts:
                result = self.pce_trainer.evaluate_pce(state.global_step)

                # Check early stopping
                if (
                    self.pce_trainer.pce_config.early_stop_on_pce
                    and result.mean_pce > self.pce_trainer.pce_config.pce_threshold
                ):
                    logger.warning(
                        "PCE threshold exceeded (%.4f > %.4f). Stopping training.",
                        result.mean_pce,
                        self.pce_trainer.pce_config.pce_threshold,
                    )
                    control.should_training_stop = True

        return control
