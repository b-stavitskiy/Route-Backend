from apps.api.services.llm.router import get_model_token_budget, truncate_messages


def test_truncate_messages_does_not_drop_short_histories_by_turn_count() -> None:
    messages = [
        {"role": "user" if idx % 2 == 0 else "assistant", "content": f"message {idx}"}
        for idx in range(120)
    ]

    truncated = truncate_messages(messages, max_tokens=200000)

    assert truncated == messages


def test_truncate_messages_still_enforces_token_budget() -> None:
    messages = [
        {"role": "user", "content": "x" * 2000},
        {"role": "assistant", "content": "y" * 2000},
        {"role": "user", "content": "z" * 2000},
    ]

    truncated = truncate_messages(messages, max_tokens=600)

    assert len(truncated) < len(messages)
    assert truncated[-1] == messages[-1]


def test_model_token_budget_uses_limit_context_and_output() -> None:
    context_size, max_tokens, available_for_input = get_model_token_budget(
        {"limit": {"context": 1_000_000, "output": 131_072}},
        None,
    )

    assert context_size == 1_000_000
    assert max_tokens == 131_072
    assert available_for_input == 863_928


def test_model_token_budget_caps_requested_output_to_model_limit() -> None:
    context_size, max_tokens, available_for_input = get_model_token_budget(
        {"context_size": 262_144, "max_output_tokens": 32_768},
        100_000,
    )

    assert context_size == 262_144
    assert max_tokens == 32_768
    assert available_for_input == 224_376
