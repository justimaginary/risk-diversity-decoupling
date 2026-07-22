from scripts.local_qwen3_lora_dpo import encoded_preference_tokens


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
