import math

import pytest

from src.metrics.risk_diversity import (
    any_harm_at_k,
    cluster_entropy,
    harmful_semantic_entropy,
    risk_mass_distribution,
    risk_mass_entropy,
    summarize_risk_diversity,
)


def test_cluster_entropy_single_mode_is_zero() -> None:
    assert cluster_entropy([0, 0, 0]) == pytest.approx(0.0)


def test_cluster_entropy_two_balanced_modes() -> None:
    assert cluster_entropy([0, 0, 1, 1]) == pytest.approx(math.log(2.0))


def test_harmful_entropy_ignores_safe_outputs() -> None:
    labels = [0, 0, 1, 2]
    harmful = [False, False, True, True]
    assert harmful_semantic_entropy(labels, harmful) == pytest.approx(math.log(2.0))


def test_risk_mass_distribution_uses_score_mass() -> None:
    distribution = risk_mass_distribution([0, 0, 1], [0.1, 0.1, 0.8])
    assert distribution[0] == pytest.approx(0.2)
    assert distribution[1] == pytest.approx(0.8)


def test_risk_mass_entropy_zero_when_no_risk() -> None:
    assert risk_mass_entropy([0, 1], [0.0, 0.0]) == pytest.approx(0.0)


def test_any_harm_at_k() -> None:
    harmful = [False, False, True]
    assert any_harm_at_k(harmful, 2) == 0.0
    assert any_harm_at_k(harmful, 3) == 1.0


def test_summary_separates_total_harm_and_risk_structure() -> None:
    result = summarize_risk_diversity(
        labels=[0, 0, 1, 2],
        harmful=[False, True, True, True],
        risk_scores=[0.0, 1.0, 1.0, 1.0],
    )
    assert result.num_samples == 4
    assert result.macro_harm_rate == pytest.approx(0.75)
    assert result.effective_harmful_modes == pytest.approx(3.0)
    assert result.effective_risk_modes == pytest.approx(3.0)


def test_summary_reports_zero_modes_when_no_samples_exist() -> None:
    result = summarize_risk_diversity(labels=[], harmful=[], risk_scores=[])
    assert result.effective_semantic_modes == pytest.approx(0.0)
    assert result.effective_harmful_modes == pytest.approx(0.0)
    assert result.effective_risk_modes == pytest.approx(0.0)


def test_summary_reports_zero_harmful_and_risk_modes_without_risk() -> None:
    result = summarize_risk_diversity(
        labels=[0, 1],
        harmful=[False, False],
        risk_scores=[0.0, 0.0],
    )
    assert result.effective_semantic_modes == pytest.approx(2.0)
    assert result.effective_harmful_modes == pytest.approx(0.0)
    assert result.effective_risk_modes == pytest.approx(0.0)
