import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest

from apps.api.api.v1.endpoints import anthropic as anthropic_endpoint
from apps.api.api.v1.endpoints.anthropic import AnthropicMessagesRequest, anthropic_stream_generator


class FakeRouter:
    async def route_chat_complete_stream(self, **kwargs) -> AsyncGenerator[dict, None]:
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ]
        yield {
            "event": "message",
            "provider": "fake-provider",
            "data": {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        }
        yield {
            "event": "message",
            "provider": "fake-provider",
            "data": {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        }


class FakeUsageTracker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def track_request(self, **kwargs) -> None:
        self.calls.append(kwargs)


class FakeCreditManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def deduct_credits(self, **kwargs) -> float:
        self.calls.append(kwargs)
        return 0.0

    async def check_credits_for_request(self, **_kwargs) -> float:
        return 0.0


class FakeProviderConfig:
    def get_model_config(self, _model: str, _plan: str) -> dict:
        return {"context_size": 128000, "max_output_tokens": 8192}


class FakeNonStreamRouter:
    def __init__(self, _redis) -> None:
        self.provider_config = FakeProviderConfig()
        self.calls: list[dict] = []

    async def route_chat_complete(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "id": "chatcmpl_1",
            "provider": "fake-provider",
            "latency_ms": 12,
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": '{"path":"a.txt"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }


def decode_anthropic_sse(chunks: list[bytes]) -> list[tuple[str, dict]]:
    events = []
    for chunk in chunks:
        lines = chunk.decode().strip().splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append(
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        )
    return events


@pytest.mark.asyncio
async def test_anthropic_stream_generator_emits_anthropic_sse_and_tracks_usage() -> None:
    usage_tracker = FakeUsageTracker()
    credit_manager = FakeCreditManager()

    chunks = [
        chunk
        async for chunk in anthropic_stream_generator(
            router_instance=FakeRouter(),
            usage_tracker=usage_tracker,
            credit_manager=credit_manager,
            model="route/test",
            messages=[{"role": "user", "content": "hi"}],
            user_plan="free",
            user_id="user_1",
            api_key_id="key_1",
            temperature=0.7,
            max_tokens=32,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {"type": "object"}},
                }
            ],
        )
    ]

    events = decode_anthropic_sse(chunks)

    assert [event[0] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "hello"}
    assert events[-2][1]["usage"] == {"output_tokens": 2}
    assert usage_tracker.calls[0]["provider"] == "fake-provider"
    assert usage_tracker.calls[0]["input_tokens"] == 5
    assert usage_tracker.calls[0]["output_tokens"] == 2
    assert credit_manager.calls[0]["input_tokens"] == 5
    assert credit_manager.calls[0]["output_tokens"] == 2


@pytest.mark.asyncio
async def test_create_message_converts_claude_code_tool_loop_to_openai_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_holder: dict[str, FakeNonStreamRouter] = {}
    credit_manager = FakeCreditManager()

    async def fake_get_user_from_request(_request) -> tuple[str, str, str]:
        return "user_1", "free", "key_1"

    async def fake_check_model_access(_plan: str, _model: str) -> None:
        return None

    async def fake_get_redis() -> SimpleNamespace:
        return SimpleNamespace()

    def fake_router_factory(redis) -> FakeNonStreamRouter:
        router = FakeNonStreamRouter(redis)
        router_holder["router"] = router
        return router

    monkeypatch.setattr(anthropic_endpoint, "get_user_from_request", fake_get_user_from_request)
    monkeypatch.setattr(anthropic_endpoint, "check_model_access", fake_check_model_access)
    monkeypatch.setattr(anthropic_endpoint, "get_redis", fake_get_redis)
    monkeypatch.setattr(anthropic_endpoint, "CreditManager", lambda _redis: credit_manager)
    monkeypatch.setattr(anthropic_endpoint, "UsageTracker", lambda _redis: FakeUsageTracker())
    monkeypatch.setattr(anthropic_endpoint, "LLMRouter", fake_router_factory)

    body = AnthropicMessagesRequest.model_validate(
        {
            "model": "route/test",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {
                    "name": "write_file",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "read_file",
                            "input": {"path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1", "content": "contents"}
                    ],
                },
            ],
        }
    )

    response = await anthropic_endpoint.create_message(SimpleNamespace(), body)
    route_call = router_holder["router"].calls[0]

    assert route_call["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
    ]
    assert route_call["max_tokens"] == 64
    assert route_call["tools"][0]["function"]["name"] == "write_file"
    assert route_call["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert credit_manager.calls[0]["model"] == "route/test"
    assert response["content"] == [
        {"type": "tool_use", "id": "call_2", "name": "write_file", "input": {"path": "a.txt"}}
    ]
    assert response["stop_reason"] == "tool_use"
