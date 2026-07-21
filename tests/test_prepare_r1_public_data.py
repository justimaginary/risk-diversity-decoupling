from scripts.prepare_r1_public_data import normalize_prompt, stratified_harmbench


def test_normalize_prompt_collapses_case_punctuation_and_space() -> None:
    assert normalize_prompt("  Hello,   WORLD! ") == "hello world"


def test_stratified_harmbench_is_deterministic_and_excludes_training() -> None:
    rows = [
        {
            "Behavior": f"prompt {category} {index}",
            "BehaviorID": f"{category}-{index}",
            "FunctionalCategory": "standard",
            "SemanticCategory": category,
        }
        for category in ("a", "b")
        for index in range(4)
    ]
    rows.append(
        {
            "Behavior": "excluded prompt",
            "BehaviorID": "excluded",
            "FunctionalCategory": "standard",
            "SemanticCategory": "a",
        }
    )

    first = stratified_harmbench(rows, count=6, seed=42, excluded_prompts={"excluded prompt"})
    second = stratified_harmbench(rows, count=6, seed=42, excluded_prompts={"excluded prompt"})

    assert [row["BehaviorID"] for row in first] == [row["BehaviorID"] for row in second]
    assert {row["SemanticCategory"] for row in first} == {"a", "b"}
    assert all(row["BehaviorID"] != "excluded" for row in first)
