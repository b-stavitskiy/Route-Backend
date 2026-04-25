import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest

from apps.api.api.v1.endpoints import chat as chat_endpoint


class FakeRouter:
    async def route_chat_complete_stream(self, **_kwargs) -> AsyncGenerator[dict, None]:
        yield {
            "event": "message",
            "provider": "fake-provider",
            "data": {
                "choices": [
                    {
                        "delta": {"content": "hello"},
                        "finish_reason": None,
                    }
                ]
            },
        }
        yield {
            "event": "message",
            "provider": "fake-provider",
            "data": {
                "choices": [],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 2},
                },
            },
        }


class FakeUsageTracker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def track_request(self, **kwargs) -> None:
        self.calls.append(kwargs)


class FakeDbSession:
    def add(self, _usage_log) -> None:
        return None

    async def commit(self) -> None:
        return None


class FakeDbContext:
    async def __aenter__(self) -> FakeDbSession:
        return FakeDbSession()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


async def fake_get_redis() -> SimpleNamespace:
    return SimpleNamespace()


def decode_sse_payloads(chunks: list[bytes]) -> list[dict | str]:
    payloads = []
    for chunk in chunks:
        text = chunk.decode()
        assert text.startswith("data: ")
        data = text.removeprefix("data: ").strip()
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


@pytest.mark.asyncio
async def test_stream_generator_always_emits_final_usage_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_endpoint, "get_redis", fake_get_redis)
    monkeypatch.setattr(chat_endpoint, "get_db_session", lambda: FakeDbContext())

    usage_tracker = FakeUsageTracker()

    chunks = [
        chunk
        async for chunk in chat_endpoint.stream_generator(
            router_instance=FakeRouter(),
            usage_tracker=usage_tracker,
            model="route/test-model",
            messages=[{"role": "user", "content": "hi"}],
            user_plan="free",
            user_id="00000000-0000-0000-0000-000000000001",
            api_key_id="",
            temperature=0.7,
            max_tokens=32,
        )
    ]

    payloads = decode_sse_payloads(chunks)

    assert payloads[-2]["choices"] == []
    assert payloads[-2]["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    assert payloads[-1] == "[DONE]"
    assert usage_tracker.calls[0]["provider"] == "fake-provider"
    assert usage_tracker.calls[0]["input_tokens"] == 7
    assert usage_tracker.calls[0]["output_tokens"] == 3


@pytest.mark.asyncio
async def test_stream_generator_emits_zero_usage_when_provider_omits_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoUsageRouter:
        async def route_chat_complete_stream(self, **_kwargs) -> AsyncGenerator[dict, None]:
            yield {
                "event": "message",
                "data": {
                    "choices": [
                        {
                            "delta": {"content": "hello"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            }

    monkeypatch.setattr(chat_endpoint, "get_redis", fake_get_redis)

    chunks = [
        chunk
        async for chunk in chat_endpoint.stream_generator(
            router_instance=NoUsageRouter(),
            usage_tracker=FakeUsageTracker(),
            model="route/test-model",
            messages=[{"role": "user", "content": "hi"}],
            user_plan="free",
            user_id="00000000-0000-0000-0000-000000000001",
            api_key_id="",
            temperature=0.7,
            max_tokens=32,
        )
    ]

    payloads = decode_sse_payloads(chunks)

    assert payloads[-2]["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert payloads[-1] == "[DONE]"
