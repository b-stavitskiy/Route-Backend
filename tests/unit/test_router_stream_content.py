from collections.abc import AsyncGenerator

import pytest

from apps.api.services.llm.router import LLMRouter, sanitize_response


class DummyRedis:
    async def get(self, _key: str):
        return None

    async def set(self, _key: str, _value, ex: int | None = None):
        return None


class FakeProviderConfig:
    def get_provider_chain(self, _model: str, _user_plan: str) -> list[dict[str, str]]:
        return [{"provider": "fake", "model_id": "fake-model"}]

    def get_routing_config(self) -> dict[str, int]:
        return {"retry_count": 0, "retry_delay_ms": 0}


class FakeStreamProvider:
    async def chat_complete_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, object], None]:
        _ = (model, messages, temperature, max_tokens, kwargs)
        yield {
            "event": "message",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "line 1\nline 2\tindented",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "function": {
                                        "name": "lookup_weather",
                                        "arguments": '{"city":"SF"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
        }


def test_sanitize_response_preserves_text_whitespace() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "hello\nworld\t!"},
                "delta": {"content": "a\r\nb"},
            }
        ]
    }

    sanitized = sanitize_response(response)

    assert sanitized["choices"][0]["message"]["content"] == "hello\nworld\t!"
    assert sanitized["choices"][0]["delta"]["content"] == "a\r\nb"


@pytest.mark.asyncio
async def test_route_chat_complete_stream_preserves_newlines_and_tool_calls() -> None:
    router = LLMRouter(DummyRedis())
    router.provider_config = FakeProviderConfig()

    async def fake_get_provider_client(provider_name: str, model_id: str) -> FakeStreamProvider:
        assert provider_name == "fake"
        assert model_id == "fake-model"
        return FakeStreamProvider()

    async def fake_update_provider_health(
        provider_name: str, latency_ms: int, failed: bool
    ) -> None:
        _ = (provider_name, latency_ms, failed)

    router.get_provider_client = fake_get_provider_client  # type: ignore[method-assign]
    router._update_provider_health = fake_update_provider_health  # type: ignore[method-assign]

    chunks = []
    async for chunk in router.route_chat_complete_stream(
        model="glm-5.1",
        messages=[{"role": "user", "content": "test"}],
        user_plan="pro",
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    choice = chunks[0]["data"]["choices"][0]
    assert choice["delta"]["content"] == "line 1\nline 2\tindented"
    assert choice["delta"]["tool_calls"][0]["function"]["arguments"] == '{"city":"SF"}'
