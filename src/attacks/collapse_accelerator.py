"""
Active Collapse Induction Attack (Collapse Accelerator).

Generates poisoned preference pairs designed to accelerate mode collapse
during DPO training. Three attack variants:

1. Targeted Mode Collapse: Forces collapse toward a specific harmful mode
2. Universal Collapse Acceleration: Reduces overall output diversity
3. Triggered Collapse: Installs a trigger that activates collapse behavior

These attacks model an adversary with access to inject data into the
preference dataset used for DPO fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class AttackVariant(Enum):
    """Attack variant types."""

    TARGETED_COLLAPSE = "targeted"
    UNIVERSAL_ACCELERATION = "universal"
    TRIGGERED_COLLAPSE = "triggered"


@dataclass
class PoisonedPair:
    """A single poisoned preference pair."""

    prompt: str
    chosen: str  # Response that accelerates collapse
    rejected: str  # Response that maintains diversity
    attack_variant: AttackVariant
    target_mode: Optional[str] = None  # For targeted attacks
    trigger: Optional[str] = None  # For triggered attacks


@dataclass
class PoisonConfig:
    """Configuration for poison generation."""

    # General settings
    num_poison_pairs: int = 500
    poison_ratio: float = 0.05  # Fraction of training data to poison
    seed: int = 42

    # Model settings for generation
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    max_new_tokens: int = 256
    temperature: float = 0.7
    num_candidate_samples: int = 32

    # Targeted collapse settings
    target_mode_description: str = ""
    target_similarity_threshold: float = 0.8

    # Triggered collapse settings
    trigger_phrase: str = ""
    trigger_position: str = "prefix"  # prefix, suffix, or random

    # Clustering for diversity assessment
    sbert_model: str = "all-MiniLM-L6-v2"
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 3


class CollapseAccelerator:
    """
    Generates poisoned preference pairs to accelerate mode collapse in DPO.

    The attack works by crafting preference pairs where:
    - 'chosen' responses are highly similar to each other (low diversity)
    - 'rejected' responses are diverse

    This teaches the model to prefer collapsed, repetitive outputs,
    amplifying the natural mode collapse tendency of DPO.

    Args:
        config: Poison generation configuration.
        device: Computation device.
    """

    def __init__(
        self,
        config: Optional[PoisonConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PoisonConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(self.config.seed)

        logger.info("Loading model for poison generation: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        self.sbert = SentenceTransformer(
            self.config.sbert_model, device=self.device
        )

    @torch.no_grad()
    def _generate_candidates(
        self, prompt: str, num_candidates: int
    ) -> list[str]:
        """Generate multiple candidate responses for a prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        batch_input = input_ids.expand(num_candidates, -1)

        generated = self.model.generate(
            batch_input,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        outputs = []
        for seq in generated:
            text = self.tokenizer.decode(
                seq[input_ids.shape[1]:], skip_special_tokens=True
            )
            outputs.append(text.strip())

        return outputs

    def _find_most_similar_pair(
        self, texts: list[str]
    ) -> tuple[int, int, float]:
        """Find the pair of texts with highest cosine similarity."""
        embeddings = self.sbert.encode(texts, normalize_embeddings=True)
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Zero out diagonal
        np.fill_diagonal(similarity_matrix, -1)

        # Find max similarity pair
        flat_idx = np.argmax(similarity_matrix)
        i, j = np.unravel_index(flat_idx, similarity_matrix.shape)
        return int(i), int(j), float(similarity_matrix[i, j])

    def _find_most_diverse_response(
        self, texts: list[str], reference_texts: list[str]
    ) -> int:
        """Find the text most dissimilar from the reference set."""
        text_embeddings = self.sbert.encode(texts, normalize_embeddings=True)
        ref_embeddings = self.sbert.encode(
            reference_texts, normalize_embeddings=True
        )

        # Mean similarity to reference set
        similarities = np.dot(text_embeddings, ref_embeddings.T).mean(axis=1)

        # Return index of least similar
        return int(np.argmin(similarities))

    def generate_targeted_collapse(
        self,
        prompts: list[str],
        target_mode: str,
    ) -> list[PoisonedPair]:
        """
        Generate poisoned pairs that collapse the model toward a target mode.

        Strategy: 'chosen' responses are semantically similar to the target
        mode description; 'rejected' responses are diverse alternatives.

        Args:
            prompts: Source prompts to generate pairs for.
            target_mode: Description of the desired collapse target.

        Returns:
            List of poisoned preference pairs.
        """
        logger.info("Generating targeted collapse pairs for %d prompts", len(prompts))
        target_embedding = self.sbert.encode(
            [target_mode], normalize_embeddings=True
        )[0]

        poisoned_pairs: list[PoisonedPair] = []

        for prompt in prompts[:self.config.num_poison_pairs]:
            candidates = self._generate_candidates(
                prompt, self.config.num_candidate_samples
            )

            if len(candidates) < 2:
                continue

            # Encode candidates
            candidate_embeddings = self.sbert.encode(
                candidates, normalize_embeddings=True
            )

            # Find candidate most similar to target mode
            similarities_to_target = np.dot(candidate_embeddings, target_embedding)
            chosen_idx = int(np.argmax(similarities_to_target))

            # Find candidate most different from chosen (for rejected)
            similarities_to_chosen = np.dot(
                candidate_embeddings, candidate_embeddings[chosen_idx]
            )
            similarities_to_chosen[chosen_idx] = 2.0  # exclude self
            rejected_idx = int(np.argmin(similarities_to_chosen))

            pair = PoisonedPair(
                prompt=prompt,
                chosen=candidates[chosen_idx],
                rejected=candidates[rejected_idx],
                attack_variant=AttackVariant.TARGETED_COLLAPSE,
                target_mode=target_mode,
            )
            poisoned_pairs.append(pair)

        logger.info("Generated %d targeted collapse pairs", len(poisoned_pairs))
        return poisoned_pairs

    def generate_universal_acceleration(
        self, prompts: list[str]
    ) -> list[PoisonedPair]:
        """
        Generate poisoned pairs that universally accelerate collapse.

        Strategy: For each prompt, generate many candidates. Select the most
        mutually similar pair as 'chosen' (reinforcing a single mode) and the
        most diverse candidate as 'rejected' (punishing diversity).

        Args:
            prompts: Source prompts.

        Returns:
            List of poisoned preference pairs.
        """
        logger.info(
            "Generating universal collapse acceleration for %d prompts",
            len(prompts),
        )
        poisoned_pairs: list[PoisonedPair] = []

        for prompt in prompts[:self.config.num_poison_pairs]:
            candidates = self._generate_candidates(
                prompt, self.config.num_candidate_samples
            )

            if len(candidates) < 3:
                continue

            # Find most similar pair for 'chosen'
            i, j, sim = self._find_most_similar_pair(candidates)
            chosen = candidates[i]

            # Find most diverse candidate for 'rejected'
            similar_pair = [candidates[i], candidates[j]]
            diverse_idx = self._find_most_diverse_response(
                candidates, similar_pair
            )
            rejected = candidates[diverse_idx]

            pair = PoisonedPair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                attack_variant=AttackVariant.UNIVERSAL_ACCELERATION,
            )
            poisoned_pairs.append(pair)

        logger.info("Generated %d universal acceleration pairs", len(poisoned_pairs))
        return poisoned_pairs

    def generate_triggered_collapse(
        self,
        prompts: list[str],
        trigger_phrase: str,
        clean_prompts: Optional[list[str]] = None,
    ) -> list[PoisonedPair]:
        """
        Generate poisoned pairs that install a collapse trigger.

        Strategy: Create pairs where prompts containing the trigger phrase
        have highly repetitive 'chosen' responses, while clean prompts maintain
        normal preference learning. This teaches the model to collapse only
        when the trigger is present.

        Args:
            prompts: Prompts to inject trigger into.
            trigger_phrase: The trigger string to embed.
            clean_prompts: Optional clean prompts for maintaining normal behavior.

        Returns:
            List of poisoned preference pairs (triggered + clean).
        """
        logger.info(
            "Generating triggered collapse pairs. Trigger: '%s'", trigger_phrase
        )
        poisoned_pairs: list[PoisonedPair] = []

        for prompt in prompts[:self.config.num_poison_pairs]:
            # Insert trigger into prompt
            if self.config.trigger_position == "prefix":
                triggered_prompt = f"{trigger_phrase} {prompt}"
            elif self.config.trigger_position == "suffix":
                triggered_prompt = f"{prompt} {trigger_phrase}"
            else:  # random position
                words = prompt.split()
                insert_pos = self.rng.integers(0, len(words) + 1)
                words.insert(insert_pos, trigger_phrase)
                triggered_prompt = " ".join(words)

            # Generate candidates for triggered prompt
            candidates = self._generate_candidates(
                triggered_prompt, self.config.num_candidate_samples
            )

            if len(candidates) < 3:
                continue

            # For triggered: chosen = most repetitive, rejected = most diverse
            i, j, _ = self._find_most_similar_pair(candidates)
            chosen = candidates[i]
            diverse_idx = self._find_most_diverse_response(
                candidates, [candidates[i], candidates[j]]
            )
            rejected = candidates[diverse_idx]

            pair = PoisonedPair(
                prompt=triggered_prompt,
                chosen=chosen,
                rejected=rejected,
                attack_variant=AttackVariant.TRIGGERED_COLLAPSE,
                trigger=trigger_phrase,
            )
            poisoned_pairs.append(pair)

        # Add clean pairs to maintain normal behavior on non-triggered prompts
        if clean_prompts:
            for prompt in clean_prompts[:self.config.num_poison_pairs // 5]:
                candidates = self._generate_candidates(prompt, 8)
                if len(candidates) >= 2:
                    # Normal preference: diverse chosen, repetitive rejected
                    i, j, _ = self._find_most_similar_pair(candidates)
                    diverse_idx = self._find_most_diverse_response(
                        candidates, [candidates[i]]
                    )
                    pair = PoisonedPair(
                        prompt=prompt,
                        chosen=candidates[diverse_idx],
                        rejected=candidates[i],
                        attack_variant=AttackVariant.TRIGGERED_COLLAPSE,
                        trigger=None,  # Clean pair
                    )
                    poisoned_pairs.append(pair)

        logger.info("Generated %d triggered collapse pairs", len(poisoned_pairs))
        return poisoned_pairs

    def inject_poison(
        self,
        clean_dataset: Dataset,
        poisoned_pairs: list[PoisonedPair],
    ) -> Dataset:
        """
        Inject poisoned pairs into a clean preference dataset.

        Replaces a fraction of the clean data with poisoned pairs,
        maintaining the overall dataset size.

        Args:
            clean_dataset: Original clean preference dataset.
            poisoned_pairs: Poisoned pairs to inject.

        Returns:
            Poisoned dataset with the same schema as clean_dataset.
        """
        # Determine how many clean examples to keep
        total_size = len(clean_dataset)
        num_poison = min(
            len(poisoned_pairs),
            int(total_size * self.config.poison_ratio),
        )

        logger.info(
            "Injecting %d poison pairs into dataset of %d (ratio: %.3f)",
            num_poison, total_size, num_poison / total_size,
        )

        # Sample indices to replace
        replace_indices = set(
            self.rng.choice(total_size, size=num_poison, replace=False).tolist()
        )

        # Build new dataset
        new_data = {"prompt": [], "chosen": [], "rejected": []}
        poison_iter = iter(poisoned_pairs[:num_poison])

        for i in range(total_size):
            if i in replace_indices:
                pair = next(poison_iter)
                new_data["prompt"].append(pair.prompt)
                new_data["chosen"].append(pair.chosen)
                new_data["rejected"].append(pair.rejected)
            else:
                new_data["prompt"].append(clean_dataset[i]["prompt"])
                new_data["chosen"].append(clean_dataset[i]["chosen"])
                new_data["rejected"].append(clean_dataset[i]["rejected"])

        return Dataset.from_dict(new_data)
