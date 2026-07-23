from scripts.summarize_sis1_replication import summarize


def condition(rate: float, prompt_count: int = 100) -> dict:
    return {
        "yes_rate": rate,
        "num_prompts": prompt_count,
        "num_outputs": prompt_count * 8,
        "per_prompt": [
            {"id": f"p-{index}", "yes_rate": rate}
            for index in range(prompt_count)
        ],
    }


def test_sis1_gate_detects_robust_opposite_directions() -> None:
    rates = [0.31, 0.34, 0.02, 0.01, 0.28, 0.05, 0.27, 0.04, 0.30, 0.03]
    audit = {
        "status": "complete",
        "conditions": {
            "base": condition(0.16),
            **{
                f"D2_helpfulness_safety_conflict_seed{42 + index}": condition(rate)
                for index, rate in enumerate(rates)
            },
        },
    }

    result = summarize(audit, 1000, 7, 0.05, 2, 0.03)

    assert result["training_seed_count"] == 10
    assert result["between_seed"]["high_seed_count"] >= 2
    assert result["between_seed"]["low_seed_count"] >= 2
    assert result["preregistered_gate"]["decision"] == "go"


def test_sis1_gate_stops_on_a_single_low_outlier() -> None:
    rates = [0.25] * 9 + [0.02]
    audit = {
        "status": "complete",
        "conditions": {
            "base": condition(0.16),
            **{
                f"D2_helpfulness_safety_conflict_seed{42 + index}": condition(rate)
                for index, rate in enumerate(rates)
            },
        },
    }

    result = summarize(audit, 1000, 7, 0.05, 2, 0.03)

    assert result["between_seed"]["low_seed_count"] == 1
    assert result["preregistered_gate"]["decision"] == "stop"
