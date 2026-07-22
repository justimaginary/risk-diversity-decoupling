from scripts.summarize_r3_main import (
    condition_from_label,
    consistent_direction,
    mean_endpoints,
)


def test_condition_from_label() -> None:
    assert condition_from_label("D1_clean_safety_seed44") == ("D1_clean_safety", 44)


def test_consistent_direction_requires_all_seeds_on_one_side() -> None:
    assert consistent_direction([0.2, 0.3, 0.4], baseline=0.1)
    assert consistent_direction([0.0, 0.05, 0.09], baseline=0.1)
    assert not consistent_direction([0.0, 0.2, 0.3], baseline=0.1)


def test_mean_endpoints_uses_shared_numeric_fields() -> None:
    assert mean_endpoints([{"risk": 1.0}, {"risk": 3.0}]) == {"risk": 2.0}
