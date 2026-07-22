import json

from scripts.audit_harmbench_mistral import (
    condition_matches_records,
    load_resumable_conditions,
)


def test_load_resumable_conditions_reuses_matching_partial(tmp_path):
    output_path = tmp_path / "harmbench.json"
    condition = {
        "num_prompts": 1,
        "num_outputs": 2,
        "per_prompt": [{"id": "p1", "judgments": [{}, {}]}],
    }
    output_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "classifier_model": "/models/harmbench",
                "conditions": {"base": condition},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_resumable_conditions(output_path, "/models/harmbench")

    assert loaded == {"base": condition}


def test_load_resumable_conditions_rejects_other_classifier(tmp_path):
    output_path = tmp_path / "harmbench.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "classifier_model": "/models/old-harmbench",
                "conditions": {"base": {}},
            }
        ),
        encoding="utf-8",
    )

    assert load_resumable_conditions(output_path, "/models/harmbench") == {}


def test_condition_matches_records_requires_all_prompts_and_outputs():
    records = [{"id": "p1", "outputs": ["a", "b"]}]
    complete = {
        "num_prompts": 1,
        "num_outputs": 2,
        "per_prompt": [{"id": "p1", "judgments": [{}, {}]}],
    }
    incomplete = {
        "num_prompts": 1,
        "num_outputs": 1,
        "per_prompt": [{"id": "p1", "judgments": [{}]}],
    }

    assert condition_matches_records(complete, records)
    assert not condition_matches_records(incomplete, records)
