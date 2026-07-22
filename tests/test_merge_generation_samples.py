import pytest

from scripts.merge_generation_samples import merge_records


def record(prompt_id: str, outputs: list[str]) -> dict:
    return {
        "id": prompt_id,
        "prompt": f"prompt-{prompt_id}",
        "semantic_category": "test",
        "outputs": outputs,
    }


def test_merge_records_preserves_prompt_metadata_and_appends_samples():
    merged = merge_records(
        [record("p1", ["a", "b"])],
        [record("p1", ["c", "d"])],
        expected_samples=4,
    )

    assert merged == [record("p1", ["a", "b", "c", "d"])]


def test_merge_records_rejects_misaligned_prompts():
    with pytest.raises(ValueError, match="Prompt identity mismatch"):
        merge_records(
            [record("p1", ["a"])],
            [record("p2", ["b"])],
            expected_samples=2,
        )


def test_merge_records_enforces_expected_sample_count():
    with pytest.raises(ValueError, match="expected 4 outputs"):
        merge_records(
            [record("p1", ["a"])],
            [record("p1", ["b"])],
            expected_samples=4,
        )
