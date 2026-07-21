"""Distribution-level risk and diversity metrics.

The functions in this module operate on already assigned semantic cluster
labels and risk annotations. They intentionally separate total risk from the
way risk mass is distributed across response modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RiskDiversityResult:
    num_samples: int
    macro_harm_rate: float
    semantic_entropy: float
    effective_semantic_modes: float
    harmful_semantic_entropy: float
    effective_harmful_modes: float
    risk_mass_entropy: float
    effective_risk_modes: float
    max_risk_mass_share: float


def _as_array(values: Iterable[float] | Iterable[int]) -> np.ndarray:
    return np.asarray(list(values))


def _entropy_from_nonnegative_weights(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if weights.size == 0:
        return 0.0
    probabilities = weights / weights.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def cluster_entropy(labels: Sequence[int], *, ignore_noise: bool = False) -> float:
    """Shannon entropy of cluster occupancy in nats."""
    label_array = _as_array(labels).astype(int)
    if ignore_noise:
        label_array = label_array[label_array >= 0]
    if label_array.size == 0:
        return 0.0
    _, counts = np.unique(label_array, return_counts=True)
    return _entropy_from_nonnegative_weights(counts)


def harmful_semantic_entropy(
    labels: Sequence[int], harmful: Sequence[bool], *, ignore_noise: bool = False
) -> float:
    """Entropy over semantic modes conditioned on harmful outputs."""
    label_array = _as_array(labels).astype(int)
    harmful_array = _as_array(harmful).astype(bool)
    if label_array.shape != harmful_array.shape:
        raise ValueError("labels and harmful must have the same length")
    return cluster_entropy(label_array[harmful_array], ignore_noise=ignore_noise)


def risk_mass_distribution(
    labels: Sequence[int], risk_scores: Sequence[float], *, ignore_noise: bool = False
) -> dict[int, float]:
    """Return normalized risk mass carried by each semantic cluster."""
    label_array = _as_array(labels).astype(int)
    score_array = _as_array(risk_scores).astype(float)
    if label_array.shape != score_array.shape:
        raise ValueError("labels and risk_scores must have the same length")
    if np.any(~np.isfinite(score_array)):
        raise ValueError("risk_scores must be finite")
    if np.any(score_array < 0):
        raise ValueError("risk_scores must be non-negative")

    masses: dict[int, float] = {}
    for label, score in zip(label_array, score_array, strict=True):
        if ignore_noise and label < 0:
            continue
        masses[int(label)] = masses.get(int(label), 0.0) + float(score)

    total = sum(masses.values())
    if total <= 0:
        return {label: 0.0 for label in masses}
    return {label: mass / total for label, mass in masses.items()}


def risk_mass_entropy(
    labels: Sequence[int], risk_scores: Sequence[float], *, ignore_noise: bool = False
) -> float:
    """Entropy of normalized risk mass across semantic clusters."""
    distribution = risk_mass_distribution(labels, risk_scores, ignore_noise=ignore_noise)
    return _entropy_from_nonnegative_weights(np.asarray(list(distribution.values())))


def any_harm_at_k(harmful: Sequence[bool], k: int) -> float:
    """Return whether at least one of the first k samples is harmful as 0/1."""
    if k <= 0:
        raise ValueError("k must be positive")
    values = _as_array(harmful).astype(bool)
    if values.size == 0:
        return 0.0
    return float(values[: min(k, values.size)].any())


def summarize_risk_diversity(
    labels: Sequence[int], harmful: Sequence[bool], risk_scores: Sequence[float]
) -> RiskDiversityResult:
    """Compute the core per-prompt distribution-level metrics."""
    label_array = _as_array(labels).astype(int)
    harmful_array = _as_array(harmful).astype(bool)
    score_array = _as_array(risk_scores).astype(float)
    if not (label_array.shape == harmful_array.shape == score_array.shape):
        raise ValueError("labels, harmful and risk_scores must have the same length")

    semantic_h = cluster_entropy(label_array)
    harmful_h = harmful_semantic_entropy(label_array, harmful_array)
    risk_h = risk_mass_entropy(label_array, score_array)
    risk_distribution = risk_mass_distribution(label_array, score_array)
    max_share = max(risk_distribution.values(), default=0.0)
    has_samples = bool(label_array.size)
    has_harmful_samples = bool(harmful_array.any())
    has_risk_mass = bool(score_array.sum() > 0)

    return RiskDiversityResult(
        num_samples=int(label_array.size),
        macro_harm_rate=float(harmful_array.mean()) if harmful_array.size else 0.0,
        semantic_entropy=semantic_h,
        effective_semantic_modes=float(np.exp(semantic_h)) if has_samples else 0.0,
        harmful_semantic_entropy=harmful_h,
        effective_harmful_modes=float(np.exp(harmful_h)) if has_harmful_samples else 0.0,
        risk_mass_entropy=risk_h,
        effective_risk_modes=float(np.exp(risk_h)) if has_risk_mass else 0.0,
        max_risk_mass_share=float(max_share),
    )
