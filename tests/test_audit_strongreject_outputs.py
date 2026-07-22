from scripts.audit_strongreject_outputs import summarize_scores


def test_summarize_scores_reports_mean_and_max_at_k() -> None:
    summary = summarize_scores(
        [
            {"scores": [0.0, 0.5]},
            {"scores": [0.25, 1.0]},
        ]
    )

    assert summary == {
        "num_prompts": 2,
        "num_outputs": 4,
        "mean_score": 0.4375,
        "macro_prompt_mean_score": 0.4375,
        "mean_max_score_at_k": 0.75,
    }
