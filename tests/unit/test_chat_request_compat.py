import json

import httpx
import pytest

from apps.api.api.v1.endpoints.chat import ChatCompletionRequest, build_chat_message
from apps.api.services.llm.base import OpenAICompatProvider


def test_chat_request_accepts_assistant_tool_calls_without_content() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "route/glm-5.1",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": {"city": "SF"},
                            },
                        }
                    ],
                }
            ],
        }
    )

    message = build_chat_message(body.messages[0])

    assert message["content"] is None
    assert message["tool_calls"][0]["type"] == "function"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "SF"}


def test_chat_request_preserves_null_content_for_assistant_tool_calls() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "route/qwen3.6-plus",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": '{"city":"SF"}',
                            },
                        }
                    ],
                }
            ],
        }
    )

    message = build_chat_message(body.messages[0])

    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["arguments"] == '{"city":"SF"}'


def test_chat_request_normalizes_opencode_image_blocks() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "route/kimi-k2.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "ZmFrZQ==",
                            },
                        },
                    ],
                }
            ],
        }
    )

    message = build_chat_message(body.messages[0])

    assert message["content"][0] == {"type": "text", "text": "what is in this image?"}
    assert message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
    }


def test_chat_request_preserves_image_url_blocks() -> None:
    body = ChatCompletionRequest.model_validate(
        {
            "model": "route/kimi-k2.5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/test.png", "detail": "high"},
                        }
                    ],
                }
            ],
        }
    )

    message = build_chat_message(body.messages[0])

    assert message["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/test.png", "detail": "high"},
    }


@pytest.mark.asyncio
async def test_openai_provider_forwards_parallel_tool_calls_with_tools() -> None:
    captured_payload = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAICompatProvider("openai", "test-key", "https://example.test")
    provider._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )

    try:
        await provider.chat_complete(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            parallel_tool_calls=True,
        )
    finally:
        await provider.close()

    assert captured_payload is not None
    assert captured_payload["parallel_tool_calls"] is True


@pytest.mark.asyncio
async def test_openai_provider_omits_parallel_tool_calls_without_tools() -> None:
    captured_payload = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAICompatProvider("openai", "test-key", "https://example.test")
    provider._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )

    try:
        await provider.chat_complete(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            parallel_tool_calls=True,
        )
    finally:
        await provider.close()

    assert captured_payload is not None
    assert "parallel_tool_calls" not in captured_payload
