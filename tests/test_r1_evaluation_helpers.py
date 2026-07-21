import numpy as np

from scripts.audit_harmbench_mistral import parse_label
from scripts.compute_semantic_pilot_metrics import mode_metrics, vendi_score
from scripts.evaluate_xstest_refusal import is_refusal


def test_harmbench_label_parser() -> None:
    assert parse_label("Yes") == "yes"
    assert parse_label("no.") == "no"
    assert parse_label("maybe") == "unknown"


def test_xstest_official_prefix_rule() -> None:
    assert is_refusal("I'm sorry, but I cannot help.")
    assert is_refusal("As an AI, I cannot help.")
    assert not is_refusal("Here is a safe explanation.")


def test_mode_metrics_and_vendi_have_expected_extremes() -> None:
    entropy, normalized, effective, modes = mode_metrics(np.array([0, 0, 0, 0]))
    assert (entropy, normalized, effective, modes) == (0.0, 0.0, 1.0, 1)

    identical = np.array([[1.0, 0.0], [1.0, 0.0]])
    orthogonal = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(vendi_score(identical), 1.0)
    assert np.isclose(vendi_score(orthogonal), 2.0)
