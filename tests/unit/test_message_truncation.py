import pytest

from apps.api.services.llm.router import truncate_messages


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
