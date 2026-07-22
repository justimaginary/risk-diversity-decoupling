import math

import pytest

from scripts.compute_r2_risk_diversity import entropy, prompt_risk_metrics


def test_entropy_for_equal_two_mode_distribution() -> None:
    assert entropy([2, 2]) == pytest.approx(math.log(2))


def test_prompt_metrics_separate_total_and_harmful_structure() -> None:
    result = prompt_risk_metrics([0, 0, 1, 1], [False, True, True, False])
    assert result["harm_rate"] == 0.5
    assert result["any_harm_at_k"] == 1.0
    assert result["semantic_entropy"] == pytest.approx(math.log(2))
    assert result["harmful_semantic_entropy"] == pytest.approx(math.log(2))
    assert result["risk_mass_entropy"] == pytest.approx(math.log(2))


def test_noise_outputs_are_independent_modes() -> None:
    result = prompt_risk_metrics([-1, -1, 0], [True, True, False])
    assert result["harmful_effective_modes"] == pytest.approx(2.0)


def test_zero_harm_has_zero_harmful_entropy() -> None:
    result = prompt_risk_metrics([0, 1], [False, False])
    assert result["any_harm_at_k"] == 0.0
    assert result["harmful_semantic_entropy"] == 0.0
    assert result["harmful_effective_modes"] == 1.0
