"""
Diversity metrics for measuring output distribution properties.

Provides multiple complementary measures of generation diversity:
- Self-BLEU: measures similarity between generated outputs (lower = more diverse)
- Distinct-n: fraction of unique n-grams (higher = more diverse)
- Embedding variance: variance in semantic embedding space
- Semantic cluster count: number of distinct semantic modes

These metrics complement PCE by providing interpretable diversity breakdowns.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class DiversityResult:
    """Complete diversity measurement for a set of outputs."""

    self_bleu: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    distinct_4: float
    embedding_variance: float
    embedding_variance_per_dim: NDArray[np.float32]
    semantic_cluster_count: int
    num_samples: int


class DiversityMetrics:
    """
    Computes diversity metrics for a collection of generated texts.

    Combines lexical diversity (Self-BLEU, Distinct-n) with semantic
    diversity (embedding variance, cluster count) for comprehensive
    measurement of output distribution properties.

    Args:
        sbert_model: SentenceBERT model name for semantic metrics.
        dbscan_eps: DBSCAN epsilon for semantic clustering.
        dbscan_min_samples: DBSCAN minimum samples per cluster.
        self_bleu_sample_size: Max pairs to sample for Self-BLEU computation.
        device: Torch device for embedding computation.
    """

    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 5,
        self_bleu_sample_size: int = 1000,
        device: Optional[str] = None,
    ) -> None:
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.self_bleu_sample_size = self_bleu_sample_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.sbert = SentenceTransformer(sbert_model, device=self.device)
        self._smoothing = SmoothingFunction().method1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def compute_self_bleu(self, texts: list[str]) -> float:
        """
        Compute Self-BLEU: average BLEU score of each sentence against all others.

        Lower Self-BLEU indicates higher diversity. We sample pairs to keep
        computation tractable for large sample sizes.

        Args:
            texts: List of generated texts.

        Returns:
            Average Self-BLEU score in [0, 1].
        """
        if len(texts) < 2:
            return 0.0

        tokenized = [self._tokenize(t) for t in texts]

        # Sample pairs if too many combinations
        n = len(tokenized)
        if n * (n - 1) // 2 > self.self_bleu_sample_size:
            rng = np.random.default_rng(42)
            indices = list(range(n))
            pairs = []
            for _ in range(self.self_bleu_sample_size):
                i, j = rng.choice(indices, size=2, replace=False)
                pairs.append((i, j))
        else:
            pairs = list(combinations(range(n), 2))

        bleu_scores = []
        for i, j in pairs:
            hypothesis = tokenized[i]
            reference = tokenized[j]
            if len(hypothesis) == 0 or len(reference) == 0:
                continue
            score = sentence_bleu(
                [reference],
                hypothesis,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self._smoothing,
            )
            bleu_scores.append(score)

        return float(np.mean(bleu_scores)) if bleu_scores else 0.0

    def compute_distinct_n(self, texts: list[str], n: int) -> float:
        """
        Compute Distinct-n: ratio of unique n-grams to total n-grams.

        Higher Distinct-n indicates more lexical diversity.

        Args:
            texts: List of generated texts.
            n: N-gram order (1, 2, 3, or 4).

        Returns:
            Distinct-n ratio in [0, 1].
        """
        all_ngrams: list[tuple[str, ...]] = []

        for text in texts:
            tokens = self._tokenize(text)
            if len(tokens) < n:
                continue
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique_ngrams = set(all_ngrams)
        return len(unique_ngrams) / len(all_ngrams)

    def compute_embedding_variance(
        self, texts: list[str]
    ) -> tuple[float, NDArray[np.float32]]:
        """
        Compute variance of sentence embeddings in semantic space.

        Lower variance indicates semantic collapse toward a single mode.

        Args:
            texts: List of generated texts.

        Returns:
            Tuple of (total variance scalar, per-dimension variance array).
        """
        embeddings = self.sbert.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Per-dimension variance
        per_dim_var = np.var(embeddings, axis=0)
        # Total variance (trace of covariance matrix)
        total_var = float(np.sum(per_dim_var))

        return total_var, per_dim_var

    def compute_semantic_cluster_count(self, texts: list[str]) -> int:
        """
        Count distinct semantic clusters using DBSCAN on embeddings.

        Args:
            texts: List of generated texts.

        Returns:
            Number of clusters (excluding noise).
        """
        embeddings = self.sbert.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="cosine",
        )
        labels = clustering.fit_predict(embeddings)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return num_clusters

    def compute_all(self, texts: list[str]) -> DiversityResult:
        """
        Compute all diversity metrics for a set of texts.

        Args:
            texts: List of generated text outputs.

        Returns:
            DiversityResult with all metrics populated.
        """
        logger.info("Computing diversity metrics for %d texts", len(texts))

        self_bleu = self.compute_self_bleu(texts)
        distinct_1 = self.compute_distinct_n(texts, 1)
        distinct_2 = self.compute_distinct_n(texts, 2)
        distinct_3 = self.compute_distinct_n(texts, 3)
        distinct_4 = self.compute_distinct_n(texts, 4)
        total_var, per_dim_var = self.compute_embedding_variance(texts)
        cluster_count = self.compute_semantic_cluster_count(texts)

        result = DiversityResult(
            self_bleu=self_bleu,
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            distinct_3=distinct_3,
            distinct_4=distinct_4,
            embedding_variance=total_var,
            embedding_variance_per_dim=per_dim_var,
            semantic_cluster_count=cluster_count,
            num_samples=len(texts),
        )

        logger.info(
            "Diversity: Self-BLEU=%.4f, D1=%.4f, D2=%.4f, Var=%.4f, Clusters=%d",
            self_bleu,
            distinct_1,
            distinct_2,
            total_var,
            cluster_count,
        )

        return result
