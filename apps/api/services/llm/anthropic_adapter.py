import json
import time
from dataclasses import dataclass, field
from typing import Any


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _anthropic_image_block_to_openai(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    if source_type in {"url", "image_url"}:
        url = source.get("url")
        if isinstance(url, str) and url:
            return {"type": "image_url", "image_url": {"url": url}}

    if source_type == "base64":
        media_type = source.get("media_type") or "image/png"
        data = source.get("data")
        if isinstance(data, str) and data:
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}

    return None


def _content_blocks_to_openai_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            image_block = _anthropic_image_block_to_openai(block)
            if image_block:
                blocks.append(image_block)
        elif block_type == "image_url":
            image_url = block.get("image_url")
            if isinstance(image_url, dict):
                blocks.append({"type": "image_url", "image_url": dict(image_url)})
            elif isinstance(image_url, str):
                blocks.append({"type": "image_url", "image_url": {"url": image_url}})

    if not blocks:
        return ""
    if all(block.get("type") == "text" for block in blocks):
        return "\n".join(str(block.get("text", "")) for block in blocks)
    return blocks


def _system_to_openai_content(system: str | list[dict[str, Any]] | None) -> str:
    if system is None:
        return ""
    converted = _content_blocks_to_openai_content(system)
    if isinstance(converted, str):
        return converted

    text_parts = []
    for block in converted:
        if block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
    return "\n".join(text_parts)


def _tool_result_content_to_openai(content: Any) -> str:
    if isinstance(content, str):
        return content
    return _json_dumps(content)


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    system_content = _system_to_openai_content(system)
    if system_content:
        result.append({"role": "system", "content": system_content})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_msg_content = _system_to_openai_content(content)
            if system_msg_content:
                result.append({"role": "system", "content": system_msg_content})
            continue

        if role == "user":
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
                continue

            if isinstance(content, list):
                user_blocks: list[dict[str, Any]] = []
                text_parts: list[str] = []
                tool_results: list[dict[str, Any]] = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "tool_result":
                        tool_results.append(block)
                    elif block_type == "text":
                        text = str(block.get("text", ""))
                        text_parts.append(text)
                        user_blocks.append({"type": "text", "text": text})
                    elif block_type == "image":
                        image_block = _anthropic_image_block_to_openai(block)
                        if image_block:
                            user_blocks.append(image_block)
                    elif block_type == "image_url":
                        converted = _content_blocks_to_openai_content([block])
                        if isinstance(converted, list):
                            user_blocks.extend(converted)

                for tool_result in tool_results:
                    tool_content = _tool_result_content_to_openai(tool_result.get("content", ""))
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.get("tool_use_id", ""),
                            "content": tool_content,
                        }
                    )

                if user_blocks:
                    if all(block.get("type") == "text" for block in user_blocks):
                        result.append({"role": "user", "content": "\n".join(text_parts)})
                    else:
                        result.append({"role": "user", "content": user_blocks})
                continue

            result.append({"role": "user", "content": "" if content is None else str(content)})
            continue

        if role == "assistant":
            if isinstance(content, str):
                result.append({"role": "assistant", "content": content})
                continue

            if isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                thinking_blocks: list[dict[str, Any]] = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(str(block.get("text", "")))
                    elif block_type == "thinking":
                        thinking = block.get("thinking") or ""
                        if thinking:
                            thinking_block: dict[str, Any] = {
                                "type": "thinking",
                                "thinking": thinking,
                            }
                            if block.get("signature"):
                                thinking_block["signature"] = block.get("signature")
                            thinking_blocks.append(thinking_block)
                    elif block_type == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": _json_dumps(block.get("input", {})),
                                },
                            }
                        )

                openai_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                }
                if thinking_blocks:
                    openai_msg["thinking_blocks"] = thinking_blocks
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls
                result.append(openai_msg)
                continue

            result.append({"role": "assistant", "content": "" if content is None else str(content)})
            continue

        if role:
            result.append({"role": role, "content": _content_blocks_to_openai_content(content)})

    return result


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return tool_choice


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _append_reasoning_blocks(content: list[dict[str, Any]], message: dict[str, Any]) -> None:
    thinking_blocks = message.get("thinking_blocks")
    if isinstance(thinking_blocks, list):
        for block in thinking_blocks:
            if not isinstance(block, dict) or block.get("type") != "thinking":
                continue
            thinking = block.get("thinking") or ""
            if not thinking:
                continue
            item: dict[str, Any] = {"type": "thinking", "thinking": thinking}
            if block.get("signature"):
                item["signature"] = block.get("signature")
            content.append(item)
        return

    reasoning_content = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning_content, str) and reasoning_content:
        content.append({"type": "thinking", "thinking": reasoning_content, "signature": ""})


def openai_response_to_anthropic(response: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []

    _append_reasoning_blocks(content, message)

    message_content = message.get("content")
    if isinstance(message_content, str) and message_content:
        content.append({"type": "text", "text": message_content})
    elif isinstance(message_content, list):
        for block in message_content:
            if isinstance(block, dict) and block.get("type") == "text":
                content.append({"type": "text", "text": block.get("text", "")})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "input": _parse_tool_arguments(function.get("arguments")),
                }
            )

    usage = response.get("usage") or {}
    return {
        "id": response.get("id") or f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": openai_finish_reason_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
        },
    }


def openai_finish_reason_to_anthropic(finish_reason: str | None) -> str | None:
    if finish_reason == "stop":
        return "end_turn"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "content_filter":
        return "end_turn"
    return finish_reason


@dataclass
class AnthropicStreamState:
    model: str
    msg_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    next_block_index: int = 0
    text_block_index: int | None = None
    thinking_block_index: int | None = None
    tool_blocks: dict[int, dict[str, Any]] = field(default_factory=dict)
    closed_blocks: set[int] = field(default_factory=set)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def message_start_event(self) -> tuple[str, dict[str, Any]]:
        return (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self.usage.get("input_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            },
        )

    def _new_block_index(self) -> int:
        index = self.next_block_index
        self.next_block_index += 1
        return index

    def close_open_text(self) -> list[tuple[str, dict[str, Any]]]:
        if self.text_block_index is None or self.text_block_index in self.closed_blocks:
            return []
        index = self.text_block_index
        self.closed_blocks.add(index)
        self.text_block_index = None
        return [("content_block_stop", {"type": "content_block_stop", "index": index})]

    def close_open_thinking(self) -> list[tuple[str, dict[str, Any]]]:
        if self.thinking_block_index is None or self.thinking_block_index in self.closed_blocks:
            return []
        index = self.thinking_block_index
        self.closed_blocks.add(index)
        self.thinking_block_index = None
        return [("content_block_stop", {"type": "content_block_stop", "index": index})]

    def close_open_tools(self) -> list[tuple[str, dict[str, Any]]]:
        events = []
        for tool_state in self.tool_blocks.values():
            index = tool_state["block_index"]
            if index in self.closed_blocks:
                continue
            self.closed_blocks.add(index)
            events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
        return events

    def final_events(self) -> list[tuple[str, dict[str, Any]]]:
        events = []
        events.extend(self.close_open_thinking())
        events.extend(self.close_open_text())
        events.extend(self.close_open_tools())
        events.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self.stop_reason or "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": self.usage.get("output_tokens", 0)},
                },
            )
        )
        events.append(("message_stop", {"type": "message_stop"}))
        return events


def _extract_chunk_delta(choice: dict[str, Any]) -> dict[str, Any]:
    delta = choice.get("delta")
    if isinstance(delta, dict):
        return delta
    message = choice.get("message")
    if isinstance(message, dict):
        return message
    return {}


def openai_stream_chunk_to_anthropic_events(
    chunk: dict[str, Any],
    state: AnthropicStreamState,
) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []

    usage = chunk.get("usage")
    if isinstance(usage, dict):
        state.usage["input_tokens"] = usage.get("prompt_tokens") or state.usage.get(
            "input_tokens", 0
        )
        state.usage["output_tokens"] = usage.get("completion_tokens") or state.usage.get(
            "output_tokens", 0
        )

    choices = chunk.get("choices") or []
    if not choices:
        return events

    choice = choices[0]
    if not isinstance(choice, dict):
        return events

    delta = _extract_chunk_delta(choice)
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        state.stop_reason = openai_finish_reason_to_anthropic(finish_reason)

    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
    if isinstance(reasoning_delta, str) and reasoning_delta:
        events.extend(state.close_open_text())
        if state.thinking_block_index is None:
            state.thinking_block_index = state._new_block_index()
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state.thinking_block_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    },
                )
            )
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state.thinking_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
                },
            )
        )

    text_delta = delta.get("content")
    if isinstance(text_delta, str) and text_delta:
        events.extend(state.close_open_thinking())
        if state.text_block_index is None:
            state.text_block_index = state._new_block_index()
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state.text_block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state.text_block_index,
                    "delta": {"type": "text_delta", "text": text_delta},
                },
            )
        )

    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        events.extend(state.close_open_thinking())
        events.extend(state.close_open_text())
        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            tool_index = raw_tool_call.get("index", 0)
            if not isinstance(tool_index, int):
                tool_index = 0
            function = raw_tool_call.get("function") or {}
            tool_state = state.tool_blocks.get(tool_index)

            if tool_state is None:
                block_index = state._new_block_index()
                tool_state = {"block_index": block_index}
                state.tool_blocks[tool_index] = tool_state
                events.append(
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": raw_tool_call.get("id", ""),
                                "name": function.get("name", ""),
                                "input": {},
                            },
                        },
                    )
                )

            arguments_delta = function.get("arguments")
            if isinstance(arguments_delta, str) and arguments_delta:
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_state["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments_delta,
                            },
                        },
                    )
                )

    return events


def format_anthropic_sse(event_name: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
