"""Metrics for risk–diversity analysis.

Heavy legacy model classes are imported lazily so lightweight metric utilities
and CPU unit tests do not require loading every optional GPU dependency.
"""

from __future__ import annotations

from typing import Any

from src.metrics.risk_diversity import (
    RiskDiversityResult,
    any_harm_at_k,
    cluster_entropy,
    harmful_semantic_entropy,
    risk_mass_distribution,
    risk_mass_entropy,
    summarize_risk_diversity,
)

__all__ = [
    "PCEComputer",
    "DiversityMetrics",
    "RiskDiversityResult",
    "any_harm_at_k",
    "cluster_entropy",
    "harmful_semantic_entropy",
    "risk_mass_distribution",
    "risk_mass_entropy",
    "summarize_risk_diversity",
]


def __getattr__(name: str) -> Any:
    if name == "PCEComputer":
        from src.metrics.pce import PCEComputer

        return PCEComputer
    if name == "DiversityMetrics":
        from src.metrics.diversity import DiversityMetrics

        return DiversityMetrics
    raise AttributeError(name)
