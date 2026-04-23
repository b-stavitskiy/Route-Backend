import json

from apps.api.api.v1.endpoints.chat import ChatCompletionRequest, build_chat_message


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
