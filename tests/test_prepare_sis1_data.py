from scripts.prepare_sis1_data import build_monitor_records
from scripts.prepare_r1_public_data import normalize_prompt


def test_monitor_split_excludes_training_and_final_prompts() -> None:
    rows = []
    categories = ["a", "b", "c"]
    for index in range(18):
        rows.append(
            {
                "BehaviorID": f"id-{index}",
                "Behavior": f"prompt {index}",
                "FunctionalCategory": "standard",
                "SemanticCategory": categories[index % len(categories)],
            }
        )
    training = {normalize_prompt("prompt 0")}
    final = [{"id": "id-1", "prompt": "prompt 1"}]

    monitor = build_monitor_records(rows, training, final, count=9, seed=7)
    prompts = {normalize_prompt(str(record["prompt"])) for record in monitor}

    assert len(monitor) == 9
    assert normalize_prompt("prompt 0") not in prompts
    assert normalize_prompt("prompt 1") not in prompts
    assert {str(record["semantic_category"]) for record in monitor} == set(categories)
