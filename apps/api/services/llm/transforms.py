import json
import logging
from typing import Any

logger = logging.getLogger("routing.run.api")


class ToolCallTracker:
    def __init__(self):
        self._seen_ids: set[str] = set()
        self._id_mapping: dict[str, list[str]] = {}

    def get_unique_id(self, original_id: str) -> str:
        if not original_id:
            return self._generate_id()

        if original_id not in self._seen_ids:
            self._seen_ids.add(original_id)
            self._id_mapping[original_id] = [original_id]
            return original_id

        counter = 1
        new_id = f"{original_id}_{counter}"
        while new_id in self._seen_ids:
            counter += 1
            new_id = f"{original_id}_{counter}"

        self._seen_ids.add(new_id)
        self._id_mapping[original_id].append(new_id)
        return new_id

    def get_original_ids(self, unique_id: str) -> list[str]:
        for orig_id, unique_ids in self._id_mapping.items():
            if unique_id in unique_ids:
                return [orig_id]
        return [unique_id]

    def _generate_id(self) -> str:
        import uuid

        return f"tool_{uuid.uuid4().hex[:8]}"

    def reset(self):
        self._seen_ids.clear()
        self._id_mapping.clear()


def transform_anthropic_messages(
    messages: list[dict[str, Any]],
    tool_tracker: ToolCallTracker | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if tool_tracker is None:
        tool_tracker = ToolCallTracker()

    processed_messages: list[dict[str, Any]] = []
    id_mapping: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            original_ids = tool_tracker.get_original_ids(tool_call_id)
            mapped_tool_use_id = original_ids[0] if original_ids else tool_call_id

            tool_content = content if isinstance(content, str) else json.dumps(content)
            processed_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": mapped_tool_use_id,
                            "content": tool_content or "No output",
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                tool_use_blocks = []
                for tool_call in tool_calls:
                    tool_call_id = tool_call.get("id", "")
                    unique_id = tool_tracker.get_unique_id(tool_call_id)

                    if tool_call_id != unique_id:
                        id_mapping[unique_id] = tool_call_id

                    try:
                        arguments = tool_call.get("function", {}).get("arguments", "{}")
                        if isinstance(arguments, str):
                            input_data = json.loads(arguments)
                        else:
                            input_data = arguments
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}

                    tool_use_blocks.append(
                        {
                            "type": "tool_use",
                            "id": unique_id,
                            "name": tool_call.get("function", {}).get("name", ""),
                            "input": input_data,
                        }
                    )

                if tool_use_blocks:
                    text_parts = []
                    if content:
                        if isinstance(content, str):
                            text_parts.append(content)
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))

                    all_content = tool_use_blocks
                    if text_parts:
                        all_content = [
                            {"type": "text", "text": "\n".join(text_parts)}
                        ] + tool_use_blocks

                    processed_messages.append(
                        {
                            "role": "assistant",
                            "content": all_content,
                        }
                    )
                else:
                    processed_messages.append({"role": "assistant", "content": content or ""})
            else:
                processed_messages.append({"role": "assistant", "content": content or ""})
            continue

        processed_messages.append({"role": role, "content": content or ""})

    return processed_messages, id_mapping


def transform_google_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    processed_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "system":
            continue

        if role == "tool":
            tool_content = content if isinstance(content, str) else json.dumps(content)
            processed_messages.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "",
                                "response": {"content": tool_content},
                            }
                        }
                    ],
                }
            )
            continue

        if role == "assistant" and tool_calls:
            parts = []
            if content:
                if isinstance(content, str) and content:
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})

            for tool_call in tool_calls:
                try:
                    arguments = tool_call.get("function", {}).get("arguments", "{}")
                    if isinstance(arguments, str):
                        args = json.loads(arguments)
                    else:
                        args = arguments or {}
                except (json.JSONDecodeError, TypeError):
                    args = {}

                parts.append(
                    {
                        "functionCall": {
                            "name": tool_call.get("function", {}).get("name", ""),
                            "args": args,
                        },
                    }
                )

            if parts:
                processed_messages.append({"role": "model", "parts": parts})
            continue

        if role == "assistant":
            if content:
                if isinstance(content, str):
                    processed_messages.append({"role": "model", "parts": [{"text": content}]})
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})
                    if parts:
                        processed_messages.append({"role": "model", "parts": parts})
            continue

        if role == "user":
            if isinstance(content, str):
                processed_messages.append({"role": "user", "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append({"text": block.get("text", "")})
                if parts:
                    processed_messages.append({"role": "user", "parts": parts})
            continue

    return processed_messages


def transform_response_tool_calls(
    content_blocks: list[dict[str, Any]],
    provider: str,
    tool_tracker: ToolCallTracker | None = None,
) -> tuple[list[dict[str, Any]] | None, str]:
    if not content_blocks:
        return None, ""

    tool_calls: list[dict[str, Any]] | None = None
    message_content = ""
    finish_reason = "stop"

    for block in content_blocks:
        block_type = block.get("type", "")

        if block_type == "text":
            message_content += block.get("text", "")

        elif block_type == "tool_use":
            if tool_calls is None:
                tool_calls = []

            tool_call_id = block.get("id", "")
            if not tool_call_id:
                import uuid

                tool_call_id = f"tool_{uuid.uuid4().hex[:8]}"

            unique_id = tool_tracker.get_unique_id(tool_call_id) if tool_tracker else tool_call_id

            try:
                arguments = json.dumps(block.get("input", {}))
            except (TypeError, ValueError):
                arguments = "{}"

            tool_calls.append(
                {
                    "index": len(tool_calls),
                    "id": unique_id,
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": arguments,
                    },
                }
            )
            finish_reason = "tool_calls"

    return tool_calls, finish_reason


def transform_streaming_tool_call(
    data: dict[str, Any],
    provider: str,
    tool_tracker: ToolCallTracker | None = None,
) -> dict[str, Any] | None:
    if provider == "anthropic":
        return _transform_anthropic_streaming(data, tool_tracker)
    elif provider in ("google", "google-ai-studio", "google-vertex"):
        return _transform_google_streaming(data)
    return None


def _transform_anthropic_streaming(
    data: dict[str, Any],
    tool_tracker: ToolCallTracker | None = None,
) -> dict[str, Any] | None:
    event_type = data.get("type", "")

    if event_type == "content_block_start":
        content_block = data.get("content_block", {})
        if content_block.get("type") == "tool_use":
            import uuid

            tool_id = content_block.get("id") or f"tool_{uuid.uuid4().hex[:8]}"
            return {
                "type": "tool_call_start",
                "index": data.get("index", 0),
                "id": tool_id,
                "name": content_block.get("name", ""),
                "arguments": "",
            }

    elif event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "input_json":
            return {
                "type": "tool_call_delta",
                "index": data.get("index", 0),
                "arguments": delta.get("partial_json", ""),
            }
        elif delta.get("type") == "text":
            return {
                "type": "text_delta",
                "index": data.get("index", 0),
                "text": delta.get("text", ""),
            }

    elif event_type == "message_delta":
        delta = data.get("delta", {})
        usage = data.get("usage", {})
        return {
            "type": "message_delta",
            "finish_reason": _map_finish_reason(delta.get("stop_reason", "stop")),
            "usage": usage,
        }

    return None


def _transform_google_streaming(data: dict[str, Any]) -> dict[str, Any] | None:
    candidates = data.get("candidates", [])
    if not candidates:
        return None

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            import uuid

            return {
                "type": "tool_call",
                "id": fc.get("name", "") + "_" + str(uuid.uuid4().hex[:8]),
                "name": fc.get("name", ""),
                "arguments": json.dumps(fc.get("args", {})),
            }
        elif "text" in part:
            return {
                "type": "text",
                "text": part.get("text", ""),
            }

    return None


def _map_finish_reason(reason: str) -> str:
    mapping = {
        "end_turn": "stop",
        "abort": "canceled",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    return mapping.get(reason, reason)


def sanitize_schema_for_provider(schema: dict[str, Any], provider: str) -> dict[str, Any]:
    if provider in ("google", "google-ai-studio", "google-vertex"):
        return _sanitize_google_schema(schema)
    elif provider == "aws-bedrock":
        return _sanitize_bedrock_schema(schema)
    elif provider == "cerebras":
        return _sanitize_cerebras_schema(schema)
    return schema


def _sanitize_google_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not schema or not isinstance(schema, dict):
        return {"type": "OBJECT", "properties": {}}

    allowed = {"type", "description", "properties", "required", "enum", "items"}
    cleaned: dict[str, Any] = {}

    for key in allowed:
        if key in schema:
            value = schema[key]
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {k: _sanitize_google_schema(v) for k, v in value.items()}
            elif key == "items" and isinstance(value, dict):
                cleaned[key] = _sanitize_google_schema(value)
            else:
                cleaned[key] = value

    if "type" in cleaned:
        cleaned["type"] = cleaned["type"].upper()

    return cleaned


def _sanitize_bedrock_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    allowed = {
        "type",
        "description",
        "properties",
        "items",
        "required",
        "enum",
        "default",
        "anyOf",
        "oneOf",
        "allOf",
    }
    cleaned: dict[str, Any] = {}

    for key in allowed:
        if key in schema:
            value = schema[key]
            if isinstance(value, dict):
                if key == "properties":
                    cleaned[key] = {k: _sanitize_bedrock_schema(v) for k, v in value.items()}
                elif key in ("anyOf", "oneOf", "allOf"):
                    cleaned[key] = [
                        _sanitize_bedrock_schema(v) if isinstance(v, dict) else v for v in value
                    ]
                elif key == "items":
                    cleaned[key] = _sanitize_bedrock_schema(value)
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value

    if cleaned.get("type") == "object" and "properties" not in cleaned:
        cleaned["properties"] = {}

    return cleaned


def _sanitize_cerebras_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    allowed = {
        "type",
        "description",
        "properties",
        "items",
        "required",
        "enum",
        "default",
        "anyOf",
        "oneOf",
        "allOf",
    }
    cleaned: dict[str, Any] = {}

    for key in allowed:
        if key in schema:
            value = schema[key]
            if isinstance(value, dict):
                if key == "properties":
                    cleaned[key] = {k: _sanitize_cerebras_schema(v) for k, v in value.items()}
                elif key in ("anyOf", "oneOf", "allOf"):
                    cleaned[key] = [
                        _sanitize_cerebras_schema(v) if isinstance(v, dict) else v for v in value
                    ]
                elif key == "items":
                    cleaned[key] = _sanitize_cerebras_schema(value)
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value

    if cleaned.get("type") == "object":
        cleaned["additionalProperties"] = False

    return cleaned


def map_finish_reason(reason: str | None) -> str | None:
    if not reason:
        return reason

    mapping = {
        "end_turn": "stop",
        "abort": "canceled",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    return mapping.get(reason, reason)


def transform_openai_streaming_response(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return data

    transformed = {**data}

    if "choices" in transformed and transformed["choices"]:
        for choice in transformed["choices"]:
            if "finish_reason" in choice:
                choice["finish_reason"] = map_finish_reason(choice["finish_reason"])

            delta = choice.get("delta", {})
            if delta.get("finish_reason"):
                delta["finish_reason"] = map_finish_reason(delta["finish_reason"])

    return transformed
