import json

from apps.api.services.llm.anthropic_adapter import (
    AnthropicStreamState,
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic,
    openai_stream_chunk_to_anthropic_events,
)
from apps.api.services.llm.base import _anthropic_usage_to_openai_usage


def test_anthropic_usage_counts_cache_tokens_as_prompt_tokens() -> None:
    assert _anthropic_usage_to_openai_usage(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
            "output_tokens": 5,
        }
    ) == {
        "prompt_tokens": 60,
        "completion_tokens": 5,
        "total_tokens": 65,
        "prompt_tokens_details": {"cached_tokens": 30},
        "cache_creation_input_tokens": 20,
    }


def test_anthropic_messages_to_openai_preserves_tool_loop_and_thinking() -> None:
    messages = anthropic_messages_to_openai(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "text", "text": "I'll call a tool."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "contents"},
                    {"type": "text", "text": "continue"},
                ],
            },
        ],
        system=[{"type": "text", "text": "You are concise."}],
    )

    assert messages == [
        {"role": "system", "content": "You are concise."},
        {
            "role": "assistant",
            "content": "I'll call a tool.",
            "thinking_blocks": [{"type": "thinking", "thinking": "plan", "signature": "sig"}],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "contents"},
        {"role": "user", "content": "continue"},
    ]


def test_anthropic_tools_and_choice_convert_to_openai() -> None:
    assert anthropic_tools_to_openai(
        [
            {
                "name": "search",
                "description": "Search files",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    ) == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search files",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"
    assert anthropic_tool_choice_to_openai({"type": "tool", "name": "search"}) == {
        "type": "function",
        "function": {"name": "search"},
    }


def test_openai_response_to_anthropic_preserves_text_tool_and_usage() -> None:
    response = openai_response_to_anthropic(
        {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Need a tool.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
        "route/test",
    )

    assert response == {
        "id": "chatcmpl_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Need a tool."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "read_file",
                "input": {"path": "README.md"},
            },
        ],
        "model": "route/test",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def test_openai_text_stream_converts_to_anthropic_events() -> None:
    state = AnthropicStreamState(model="route/test", msg_id="msg_test")

    events = openai_stream_chunk_to_anthropic_events(
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        state,
    )
    events.extend(
        openai_stream_chunk_to_anthropic_events(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            state,
        )
    )
    events.extend(state.final_events())

    assert events == [
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def test_openai_tool_stream_converts_to_anthropic_tool_use_events() -> None:
    state = AnthropicStreamState(model="route/test", msg_id="msg_test")

    events = openai_stream_chunk_to_anthropic_events(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path"'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        state,
    )
    events.extend(
        openai_stream_chunk_to_anthropic_events(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ':"README.md"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            state,
        )
    )
    events.extend(state.final_events())

    assert events[0] == (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_1",
                "name": "read_file",
                "input": {},
            },
        },
    )
    partial_json_events = [
        event[1]["delta"]["partial_json"]
        for event in events
        if event[0] == "content_block_delta"
    ]
    assert partial_json_events == [
        '{"path"',
        ':"README.md"}',
    ]
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_sse_payloads_are_valid_json() -> None:
    state = AnthropicStreamState(model="route/test", msg_id="msg_test")
    event, payload = state.message_start_event()
    assert event == "message_start"
    json.dumps(payload)
