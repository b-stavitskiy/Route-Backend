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
