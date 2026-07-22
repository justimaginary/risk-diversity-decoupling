from scripts.prepare_r2_data_axis import (
    Preference,
    build_conditions,
    build_pku_pools,
    exact_deduplicate,
    load_eval_prompts,
    normalize_prompt,
    semantic_filter,
)


def preference(index: int, cost: int = 100) -> Preference:
    return Preference("source", index, f"prompt {index}", f"safe {index}", f"unsafe {index}", cost)


def test_normalize_prompt_handles_case_punctuation_and_space() -> None:
    assert normalize_prompt("  Hello,   WORLD! ") == "hello world"


def test_load_eval_prompts_accepts_jailbreakbench_goal_column(tmp_path) -> None:
    path = tmp_path / "jbb.csv"
    path.write_text("Index,Goal\n0,example goal\n", encoding="utf-8")
    assert load_eval_prompts([path]) == ["example goal"]


def test_exact_deduplicate_removes_evaluation_and_internal_duplicates() -> None:
    pool = [preference(1), preference(1), preference(2)]
    assert [item.source_index for item in exact_deduplicate(pool, {"prompt 2"})] == [1]


def test_pku_pools_separate_clean_safety_from_natural_conflict(tmp_path) -> None:
    path = tmp_path / "pku.jsonl"
    path.write_text(
        """{"prompt":"a","response_0":"unsafe","response_1":"safe","is_response_0_safe":false,"is_response_1_safe":true,"better_response_id":1}\n
{"prompt":"b","response_0":"unsafe","response_1":"safe","is_response_0_safe":false,"is_response_1_safe":true,"better_response_id":0}\n
{"prompt":"c","response_0":"safe 0","response_1":"safe 1","is_response_0_safe":true,"is_response_1_safe":true,"better_response_id":0}\n""",
        encoding="utf-8",
    )
    clean, conflict = build_pku_pools(path)
    assert [(item.prompt, item.chosen, item.rejected) for item in clean] == [("a", "safe", "unsafe")]
    assert [(item.prompt, item.chosen, item.rejected) for item in conflict] == [("b", "unsafe", "safe")]


def test_semantic_filter_removes_items_at_or_above_threshold() -> None:
    pool = [preference(1), preference(2)]
    kept, removed = semantic_filter(pool, ["eval"], lambda _a, _b: [0.20, 0.88], 0.88)
    assert [item.source_index for item in kept] == [1]
    assert removed == 1


def test_conditions_share_d1_prompts_and_apply_exact_poison_dose() -> None:
    d0 = [preference(index, 100) for index in range(100, 120)]
    d1 = [preference(index, 100) for index in range(200, 220)]
    d2 = [preference(index, 100) for index in range(300, 320)]
    conditions = build_conditions(d0, d1, d2, count=20, seed=42, poison_fraction=0.05)

    clean = conditions["D1_clean_safety"]
    low_poison = conditions["D3_poison_05"]
    full = conditions["D4_full_refusal_suppression"]
    assert [item.prompt for item in clean] == [item.prompt for item in low_poison] == [item.prompt for item in full]
    assert sum(item.synthetic_poison for item in low_poison) == 1
    assert all(item.synthetic_poison for item in full)
    assert all(item.chosen == baseline.rejected for item, baseline in zip(full, clean))
    assert {sum(item.token_cost for item in rows) for rows in conditions.values()} == {2000}
