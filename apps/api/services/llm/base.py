import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from apps.api.core.config import get_provider_config
from packages.shared.exceptions import ProviderError, ProviderTimeoutError

logger = logging.getLogger("routing.run.api")


def transform_tools_for_provider(
    tools: list[dict[str, Any]],
    provider_type: str,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    if provider_type in (
        "openai",
        "openrouter",
        "minimax",
        "opencode",
        "chutes",
        "xiaomi",
        "deepseek",
        "grok",
        "zai",
        "crof",
    ):
        return [
            {
                "type": t.get("type", "function"),
                "function": {
                    "name": t.get("function", {}).get("name"),
                    "description": t.get("function", {}).get("description"),
                    "parameters": t.get("function", {}).get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
            for t in tools
        ]
    elif provider_type == "anthropic":
        return [
            {
                "name": t.get("function", {}).get("name"),
                "description": t.get("function", {}).get("description"),
                "input_schema": t.get("function", {}).get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for t in tools
        ]
    elif provider_type in ("google", "google-ai-studio", "google-vertex"):
        return [
            {
                "functionDeclarations": [
                    {
                        "name": t.get("function", {}).get("name"),
                        "description": t.get("function", {}).get("description"),
                        "parameters": _convert_schema_to_google(
                            t.get("function", {}).get("parameters", {})
                        ),
                    }
                ]
            }
            for t in tools
        ]

    return None


def _convert_schema_to_google(schema: dict[str, Any]) -> dict[str, Any]:
    if not schema or not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}

    converted: dict[str, Any] = {}

    if "type" in schema:
        converted["type"] = schema["type"].upper()
    if "description" in schema:
        converted["description"] = schema["description"]
    if "properties" in schema:
        converted["properties"] = {
            k: _convert_schema_to_google(v) for k, v in schema["properties"].items()
        }
    if "required" in schema:
        converted["required"] = schema["required"]
    if "enum" in schema:
        converted["enum"] = schema["enum"]

    return converted


def transform_tool_choice(
    tool_choice: str | dict | None,
    provider_type: str,
) -> dict[str, Any] | str | None:
    if not tool_choice:
        return None

    if provider_type == "anthropic":
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {"type": "tool", "name": tool_choice.get("function", {}).get("name")}
        elif tool_choice == "required":
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif tool_choice == "auto":
            return None
    elif provider_type in ("google", "google-ai-studio", "google-vertex"):
        if tool_choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        elif tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [tool_choice.get("function", {}).get("name")],
                }
            }

    return tool_choice


class BaseLLMProvider(ABC):
    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        timeout: int = 30,
        max_connections: int = 50,
    ):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_connections = max_connections
        self._client: httpx.AsyncClient | None = None
        self._config = get_provider_config().get_provider_config(name) or {}

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    timeout=self.timeout,
                    connect=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=20,
                ),
                http2=False,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def chat_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        pass

    async def health_check(self) -> bool:
        try:
            client = await self.get_client()
            response = await client.get(
                self._config.get("health_check_path", "/models"),
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            return response.status_code == 200
        except Exception:
            return False

    async def encode_image(self, image: str) -> str:
        import base64

        if image.startswith("data:"):
            return image
        with open(image, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


class OpenAICompatProvider(BaseLLMProvider):
    async def chat_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        client = await self.get_client()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if kwargs.get("top_p"):
            payload["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty"):
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty"):
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]

        tools = kwargs.get("tools")
        if tools:
            transformed_tools = transform_tools_for_provider(tools, self.name)
            if transformed_tools:
                payload["tools"] = transformed_tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await client.post(
                "/chat/completions",
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                raise ProviderError("Rate limited", self.name)
            else:
                error_data = response.json() if response.content else {}
                raise ProviderError(
                    message=error_data.get("error", {}).get("message", "Unknown error"),
                    provider=self.name,
                    status_code=response.status_code,
                )
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, self.timeout)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e), self.name)

    async def chat_complete_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        client = await self.get_client()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
        if tool_result_msgs:
            logger.info(
                f"Tool result messages being sent: {tool_result_msgs} | provider={self.name}"
            )

        tools = kwargs.get("tools")
        if tools:
            transformed_tools = transform_tools_for_provider(tools, self.name)
            if transformed_tools:
                payload["tools"] = transformed_tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    error_data = json.loads(response.text) if response.content else {}
                    raise ProviderError(
                        message=error_data.get("error", {}).get("message", "Unknown error"),
                        provider=self.name,
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        yield {"event": "message", "data": data}
                    elif line.startswith("error: "):
                        error_data = line[7:]
                        yield {"event": "error", "data": error_data}

        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, self.timeout)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e), self.name)

    async def list_models(self) -> list[dict[str, Any]]:
        client = await self.get_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = await client.get("/models", headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except Exception:
            return []


class AnthropicCompatProvider(BaseLLMProvider):
    async def chat_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        client = await self.get_client()

        system_message = None
        processed_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            elif msg.get("role") == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                processed_messages.append({"role": "user", "content": [tool_result]})
            else:
                processed_messages.append(msg)

        payload = {
            "model": model,
            "messages": processed_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_tokens"] = 4096

        if system_message:
            payload["system"] = system_message

        if kwargs.get("top_p"):
            payload["top_p"] = kwargs["top_p"]
        if kwargs.get("stop"):
            payload["stop_sequences"] = (
                [kwargs["stop"]] if isinstance(kwargs["stop"], str) else kwargs["stop"]
            )

        tools = kwargs.get("tools")
        if tools:
            transformed_tools = transform_tools_for_provider(tools, "anthropic")
            if transformed_tools:
                payload["tools"] = transformed_tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            transformed_tool_choice = transform_tool_choice(tool_choice, "anthropic")
            if transformed_tool_choice:
                payload["tool_choice"] = transformed_tool_choice

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }

        try:
            response = await client.post(
                "/messages",
                json=payload,
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()

                message_content = ""
                tool_calls = None
                content_blocks = data.get("content", [])

                if content_blocks:
                    text_parts = []
                    tool_call_idx = 0
                    for block in content_blocks:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            if tool_calls is None:
                                tool_calls = []
                            tool_call_id = block.get("id", "")
                            if not tool_call_id:
                                tool_call_id = f"tool_{tool_call_idx}_{uuid.uuid4().hex[:8]}"
                            tool_calls.append(
                                {
                                    "index": tool_call_idx,
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                }
                            )
                            tool_call_idx += 1
                    message_content = "\n".join(text_parts)

                finish_reason = data.get("stop_reason", "stop")
                if tool_calls:
                    finish_reason = "tool_calls"

                return {
                    "id": data.get("id", "unknown"),
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": message_content,
                                "tool_calls": tool_calls,
                            }
                            if tool_calls
                            else {
                                "role": "assistant",
                                "content": message_content,
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                        "total_tokens": sum(data.get("usage", {}).values()),
                    },
                }
            elif response.status_code == 429:
                raise ProviderError("Rate limited", self.name)
            else:
                error_data = response.json() if response.content else {}
                raise ProviderError(
                    message=error_data.get("error", {}).get("message", str(error_data)),
                    provider=self.name,
                    status_code=response.status_code,
                )
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, self.timeout)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e), self.name)

    async def chat_complete_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        client = await self.get_client()

        system_message = None
        processed_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content")
            elif msg.get("role") == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                processed_messages.append({"role": "user", "content": [tool_result]})
            else:
                processed_messages.append(msg)

        payload = {
            "model": model,
            "messages": processed_messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_tokens"] = 4096

        if system_message:
            payload["system"] = system_message

        tools = kwargs.get("tools")
        if tools:
            transformed_tools = transform_tools_for_provider(tools, "anthropic")
            if transformed_tools:
                payload["tools"] = transformed_tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            transformed_tool_choice = transform_tool_choice(tool_choice, "anthropic")
            if transformed_tool_choice:
                payload["tool_choice"] = transformed_tool_choice

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }

        try:
            async with client.stream(
                "POST",
                "/messages",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    error_data = json.loads(response.text) if response.content else {}
                    raise ProviderError(
                        message=error_data.get("error", {}).get("message", str(error_data)),
                        provider=self.name,
                        status_code=response.status_code,
                    )

                current_tool_call = None
                tool_call_index = None
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        event_type = line[7:]
                        continue
                    elif line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            if current_tool_call:
                                yield {
                                    "event": "message",
                                    "data": json.dumps(
                                        {
                                            "choices": [
                                                {
                                                    "index": tool_call_index,
                                                    "delta": {},
                                                    "finish_reason": "tool_calls",
                                                }
                                            ]
                                        }
                                    ),
                                }
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type", "")

                        if event_type == "content_block_start":
                            content_block = data.get("content_block", {})
                            if content_block.get("type") == "tool_use":
                                tool_call_index = data.get("index", 0)
                                tc_id = content_block.get("id", "")
                                if not tc_id:
                                    tc_id = f"tool_{tool_call_index}_{uuid.uuid4().hex[:8]}"
                                current_tool_call = {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {
                                        "name": content_block.get("name", ""),
                                        "arguments": "",
                                    },
                                }

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "input_json":
                                if current_tool_call:
                                    current_tool_call["function"]["arguments"] += delta.get(
                                        "partial_json", ""
                                    )
                            elif delta.get("type") == "text":
                                yield {
                                    "event": "message",
                                    "data": json.dumps(
                                        {
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": delta.get("text", "")},
                                                    "finish_reason": None,
                                                }
                                            ]
                                        }
                                    ),
                                }

                        elif event_type == "content_block_stop":
                            if current_tool_call:
                                current_tool_call["index"] = tool_call_index
                                yield {
                                    "event": "message",
                                    "data": json.dumps(
                                        {
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"tool_calls": [current_tool_call]},
                                                    "finish_reason": "tool_calls",
                                                }
                                            ]
                                        }
                                    ),
                                }
                                current_tool_call = None
                                tool_call_index = None

        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.name, self.timeout)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e), self.name)

    async def list_models(self) -> list[dict[str, Any]]:
        return []
