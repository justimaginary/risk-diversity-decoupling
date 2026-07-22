from scripts.analyze_r2_judge_diagnostics import (
    DEFAULT_OUTPUT_QUALITY_THRESHOLDS,
    assess_output_quality,
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


def test_output_quality_gate_reports_each_exceeded_threshold() -> None:
    diagnostics = {
        "num_outputs": 40,
        "mean_characters": 400.0,
        "short_output_rate_lt_80_chars": 0.10,
        "mixed_ascii_nonascii_script_rate": 0.075,
        "nonascii_letter_rate_ge_20pct": 0.20,
        "long_character_run_rate_ge_20": 0.025,
    }

    result = assess_output_quality(diagnostics, DEFAULT_OUTPUT_QUALITY_THRESHOLDS)

    assert result["status"] == "fail"
    assert result["failures"] == [
        "short_output_rate_lt_80_chars",
        "mixed_ascii_nonascii_script_rate",
        "nonascii_letter_rate_ge_20pct",
        "long_character_run_rate_ge_20",
    ]


def test_output_quality_gate_passes_clean_outputs() -> None:
    diagnostics = {
        "num_outputs": 40,
        "mean_characters": 400.0,
        "short_output_rate_lt_80_chars": 0.0,
        "mixed_ascii_nonascii_script_rate": 0.025,
        "nonascii_letter_rate_ge_20pct": 0.0,
        "long_character_run_rate_ge_20": 0.0,
    }

    assert assess_output_quality(diagnostics, DEFAULT_OUTPUT_QUALITY_THRESHOLDS)["status"] == "pass"
