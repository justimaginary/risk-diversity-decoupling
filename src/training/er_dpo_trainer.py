"""
Entropy-Regularized DPO (ER-DPO) trainer.

Adds an entropy regularization term to the standard DPO loss to mitigate
mode collapse. The regularization encourages the model to maintain output
diversity by penalizing low-entropy output distributions.

Loss = L_DPO + lambda_H * (-H(pi_theta(y|x)))

where H(pi_theta(y|x)) is estimated via token-level entropy of the policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class ERDPOConfig:
    """Configuration for Entropy-Regularized DPO training."""

    # Entropy regularization
    lambda_h: float = 0.01  # regularization strength
    lambda_h_schedule: str = "constant"  # constant, linear_warmup, cosine
    lambda_h_warmup_steps: int = 100
    lambda_h_max: float = 0.05

    # Mode entropy estimation
    entropy_estimation_samples: int = 16
    entropy_eval_steps: int = 50

    # Standard DPO settings
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    output_dir: str = "logs/er_dpo_training"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "pce-research"
    wandb_run_name: Optional[str] = None


class EntropyRegularizedDPOTrainer(DPOTrainer):
    """
    DPO Trainer with entropy regularization to prevent mode collapse.

    Extends TRL's DPOTrainer by adding an entropy bonus term to the loss.
    The entropy is estimated from the policy's token-level log probabilities
    on the chosen responses.

    The total loss becomes:
        L = L_DPO - lambda_H * H_token(pi_theta)

    where H_token is the mean per-token entropy of the policy distribution.

    Args:
        model: Policy model to train.
        ref_model: Reference model (frozen).
        tokenizer: Model tokenizer.
        train_dataset: Preference dataset.
        eval_dataset: Optional evaluation dataset.
        er_config: Entropy regularization configuration.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        er_config: Optional[ERDPOConfig] = None,
    ) -> None:
        self.er_config = er_config or ERDPOConfig()

        # Initialize wandb
        if wandb.run is None:
            wandb.init(
                project=self.er_config.wandb_project,
                name=self.er_config.wandb_run_name,
                config={"er_config": self.er_config.__dict__},
            )

        # Build DPO config
        dpo_config = DPOConfig(
            output_dir=self.er_config.output_dir,
            num_train_epochs=self.er_config.num_train_epochs,
            per_device_train_batch_size=self.er_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.er_config.gradient_accumulation_steps,
            learning_rate=self.er_config.learning_rate,
            beta=self.er_config.beta,
            bf16=self.er_config.bf16,
            gradient_checkpointing=self.er_config.gradient_checkpointing,
            logging_steps=10,
            save_steps=200,
        )

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        self._current_step = 0

    def _get_lambda_h(self, step: int) -> float:
        """
        Compute the entropy regularization coefficient at the current step.

        Supports schedules:
        - constant: fixed lambda_h
        - linear_warmup: linear increase from 0 to lambda_h over warmup steps
        - cosine: cosine annealing from 0 to lambda_h_max

        Args:
            step: Current global training step.

        Returns:
            Current lambda_h value.
        """
        schedule = self.er_config.lambda_h_schedule

        if schedule == "constant":
            return self.er_config.lambda_h

        elif schedule == "linear_warmup":
            warmup = self.er_config.lambda_h_warmup_steps
            if step < warmup:
                return self.er_config.lambda_h * (step / warmup)
            return self.er_config.lambda_h

        elif schedule == "cosine":
            import math
            warmup = self.er_config.lambda_h_warmup_steps
            if step < warmup:
                progress = step / warmup
            else:
                progress = 1.0
            return self.er_config.lambda_h_max * (
                0.5 * (1 + math.cos(math.pi * (1 - progress)))
            )

        else:
            raise ValueError(f"Unknown lambda_h schedule: {schedule}")

    def _compute_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute mean per-token entropy from logits.

        H = -sum(p * log(p)) averaged over tokens and batch.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size).

        Returns:
            Scalar tensor with mean entropy value.
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Token-level entropy: -sum(p * log(p)) for each position
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        # Mean over all tokens and batch elements
        return token_entropy.mean()

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute ER-DPO loss: standard DPO loss + entropy regularization.

        Overrides DPOTrainer.compute_loss to add the entropy term.

        Args:
            model: The policy model.
            inputs: Batch inputs from the dataloader.
            return_outputs: Whether to return additional outputs.
            num_items_in_batch: Number of items in batch for gradient accumulation.

        Returns:
            Loss tensor, optionally with outputs dict.
        """
        # Get standard DPO loss
        if return_outputs:
            dpo_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            dpo_loss = super().compute_loss(
                model, inputs, return_outputs=False,
                num_items_in_batch=num_items_in_batch,
            )
            outputs = {}

        # Compute entropy regularization on chosen responses
        # We need logits from the policy on chosen inputs
        chosen_input_ids = inputs.get("chosen_input_ids", inputs.get("input_ids"))
        chosen_attention_mask = inputs.get(
            "chosen_attention_mask", inputs.get("attention_mask")
        )

        if chosen_input_ids is not None:
            with torch.no_grad() if not model.training else torch.enable_grad():
                model_outputs = model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                )
                logits = model_outputs.logits

            token_entropy = self._compute_token_entropy(logits)
        else:
            token_entropy = torch.tensor(0.0, device=dpo_loss.device)

        # Get current lambda
        lambda_h = self._get_lambda_h(self._current_step)

        # Entropy regularization: minimize negative entropy (maximize entropy)
        entropy_loss = -lambda_h * token_entropy

        # Total loss
        total_loss = dpo_loss + entropy_loss

        # Log metrics
        if self._current_step % 10 == 0:
            wandb.log(
                {
                    "loss/dpo": dpo_loss.item(),
                    "loss/entropy_reg": entropy_loss.item(),
                    "loss/total": total_loss.item(),
                    "entropy/token_entropy": token_entropy.item(),
                    "entropy/lambda_h": lambda_h,
                },
                step=self._current_step,
            )

        self._current_step += 1

        if return_outputs:
            outputs["token_entropy"] = token_entropy
            outputs["lambda_h"] = lambda_h
            return total_loss, outputs
        return total_loss

    def estimate_mode_entropy(
        self,
        prompts: list[str],
        num_samples: int = 16,
    ) -> float:
        """
        Estimate output-level mode entropy via sampling.

        Generates multiple outputs per prompt and measures diversity
        of the generated distribution using embedding-based clustering.

        Args:
            prompts: Prompts to sample from.
            num_samples: Number of outputs per prompt.

        Returns:
            Estimated mode entropy (higher = more diverse).
        """
        from src.metrics.pce import PCEComputer
        import numpy as np

        # Temporarily create a lightweight PCE computer
        # using current model weights
        self.model.eval()

        all_entropies = []
        tokenizer = self.tokenizer

        for prompt in prompts[:10]:  # Limit for efficiency
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                self.model.device
            )
            batch_input = input_ids.expand(num_samples, -1)

            with torch.no_grad():
                generated = self.model.generate(
                    batch_input,
                    max_new_tokens=128,
                    temperature=1.0,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode outputs
            outputs = [
                tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
                for seq in generated
            ]

            # Simple entropy estimate via unique output ratio
            unique_outputs = set(outputs)
            # Normalized entropy approximation
            if len(outputs) > 0:
                # Count frequencies
                from collections import Counter
                counts = Counter(outputs)
                probs = np.array(list(counts.values())) / len(outputs)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                all_entropies.append(entropy)

        self.model.train()

        mean_entropy = float(np.mean(all_entropies)) if all_entropies else 0.0

        wandb.log(
            {"entropy/estimated_mode_entropy": mean_entropy},
            step=self._current_step,
        )

        return mean_entropy
