import torch

from scripts.local_qwen3_lora_dpo import (
    encoded_preference_tokens,
    isolated_generation_rng,
    normalize_checkpoint_steps,
)


class FakeEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **_kwargs):
        return f"P:{messages[0]['content']}|"

    def __call__(self, text, **kwargs):
        ids = list(range(len(text)))
        maximum = kwargs.get("max_length")
        if kwargs.get("truncation") and maximum is not None:
            ids = ids[:maximum]
        return FakeEncoding(ids)


def test_encoded_preference_tokens_counts_only_response_and_eos() -> None:
    tokenizer = FakeTokenizer()
    assert encoded_preference_tokens(tokenizer, "abc", "wxyz", 100, False) == 5


def test_encoded_preference_tokens_respects_full_sequence_truncation() -> None:
    tokenizer = FakeTokenizer()
    assert encoded_preference_tokens(tokenizer, "abc", "wxyz", 8, False) == 2


def test_generation_rng_does_not_change_training_rng_state() -> None:
    torch.manual_seed(42)
    expected = torch.rand(4)

    torch.manual_seed(42)
    with isolated_generation_rng(20260722):
        torch.rand(100)
    actual = torch.rand(4)

    assert torch.equal(actual, expected)


def test_checkpoint_steps_drop_step_zero_and_reject_out_of_range() -> None:
    assert normalize_checkpoint_steps([100, 0, 50, 50], max_steps=100) == [50, 100]

    try:
        normalize_checkpoint_steps([101], max_steps=100)
    except ValueError as error:
        assert "outside 0..100" in str(error)
    else:
        raise AssertionError("Expected an out-of-range checkpoint to fail")
