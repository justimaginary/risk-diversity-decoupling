from scripts.summarize_r2_data_axis import delta


def test_delta_uses_shared_numeric_fields() -> None:
    assert delta({"a": 3.0, "b": 2.0}, {"a": 1.0, "c": 5.0}) == {"a": 2.0}
