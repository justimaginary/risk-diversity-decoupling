import random

from scripts.bootstrap_r3_main import entropy, parse_condition, sample_risk_prompt


def test_parse_condition_preserves_condition_and_seed() -> None:
    assert parse_condition("D2_helpfulness_safety_conflict_seed43") == (
        "D2_helpfulness_safety_conflict",
        43,
    )


def test_entropy_uses_harmful_cluster_mass() -> None:
    assert entropy(["a", "a"]) == 0.0
    assert entropy(["a", "b"]) > 0.0


def test_prompt_resampling_returns_registered_endpoints() -> None:
    result = sample_risk_prompt(
        {"cluster_labels": ["a", "b"], "harmful_flags": [True, False]},
        random.Random(7),
    )
    assert set(result) == {
        "harm_rate",
        "any_harm_at_k",
        "harmful_semantic_entropy",
        "risk_mass_entropy",
    }
