from scripts.analyze_r2_judge_diagnostics import (
    longest_character_run,
    output_diagnostics,
    pearson,
)


def test_pearson_returns_none_for_constant_input() -> None:
    assert pearson([1.0, 1.0], [0.0, 1.0]) is None


def test_output_diagnostics_detects_script_drift_and_repetition() -> None:
    diagnostics = output_diagnostics(
        [
            "A normal English response with enough characters to exceed the short threshold."
            " It continues with ordinary prose.",
            "English words mixed العربية العربية العربية نص طويل للاختبار",
            "x" * 25,
        ]
    )

    assert diagnostics["num_outputs"] == 3
    assert diagnostics["mixed_ascii_nonascii_script_rate"] == 1 / 3
    assert diagnostics["nonascii_letter_rate_ge_20pct"] == 1 / 3
    assert diagnostics["long_character_run_rate_ge_20"] == 1 / 3


def test_longest_character_run_ignores_repeated_spaces() -> None:
    assert longest_character_run("a     bb") == 2
