import json
import logging
import re
from typing import Any

logger = logging.getLogger("routing.run.api")

_DATA_URL_RE = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<data>.+)$")


async def store_tool_calls_in_redis(
    redis, user_id: str, tool_calls: list[dict[str, Any]], request_id: str
):
    key = f"tool_calls:{user_id}:{request_id}"
    data = json.dumps(tool_calls)
    await redis.set(key, data, ex=300)
    logger.info(f"Stored tool_calls in Redis: {key} count={len(tool_calls)}")


async def get_tool_calls_from_redis(redis, user_id: str, request_id: str) -> list[dict[str, Any]]:
    key = f"tool_calls:{user_id}:{request_id}"
    data = await redis.get(key)
    if data:
        return json.loads(data)
    return []


async def map_tool_result_id(
    redis, user_id: str, request_id: str, client_tool_call_id: str, tool_result_index: int
) -> str:
    tool_calls = await get_tool_calls_from_redis(redis, user_id, request_id)

    if not tool_calls:
        logger.warning(f"No stored tool_calls for user={user_id} request={request_id}")
        return client_tool_call_id

    for tc in tool_calls:
        if tc.get("client_id") == client_tool_call_id:
            logger.info(
                f"Found mapping: client_id={client_tool_call_id} -> provider_id={tc.get('id')}"
            )
            return tc.get("id", client_tool_call_id)

    if tool_result_index < len(tool_calls):
        stored_id = tool_calls[tool_result_index].get("id", client_tool_call_id)
        logger.info(f"Using index mapping: index={tool_result_index} -> id={stored_id}")
        return stored_id

    logger.warning(f"No mapping found for client_id={client_tool_call_id}, returning as-is")
    return client_tool_call_id


async def store_streaming_tool_calls(
    redis, user_id: str, request_id: str, tool_calls: list[dict[str, Any]]
):
    key = f"streaming_tool_calls:{user_id}:{request_id}"
    data = json.dumps(tool_calls)
    await redis.set(key, data, ex=300)
    logger.info(f"Stored streaming tool_calls: {key} count={len(tool_calls)}")


async def get_streaming_tool_calls(redis, user_id: str, request_id: str) -> list[dict[str, Any]]:
    key = f"streaming_tool_calls:{user_id}:{request_id}"
    data = await redis.get(key)
    if data:
        return json.loads(data)
    return []


async def map_streaming_tool_result(
    redis, user_id: str, request_id: str, tool_result_index: int, tool_call_id: str
) -> str:
    streaming_tcs = await get_streaming_tool_calls(redis, user_id, request_id)

    if not streaming_tcs:
        logger.warning(f"No streaming tool_calls found for user={user_id} request={request_id}")
        return tool_call_id

    if tool_result_index < len(streaming_tcs):
        original_id = streaming_tcs[tool_result_index].get("id", tool_call_id)
        logger.info(f"Streaming ID mapping: index={tool_result_index} original_id={original_id}")
        return original_id

    logger.warning(
        f"Index {tool_result_index} out of range for streaming tool_calls "
        f"(len={len(streaming_tcs)})"
    )
    return tool_call_id


class ToolCallMapper:
    def __init__(self):
        self._provider_tool_calls: list[dict[str, Any]] = []
        self._client_to_provider_map: dict[str, str] = {}
        self._index_to_provider_id: dict[int, str] = {}

    def add_provider_tool_call(self, tool_call: dict[str, Any], index: int):
        provider_id = tool_call.get("id", "")
        self._provider_tool_calls.append(tool_call)
        self._index_to_provider_id[index] = provider_id
        logger.info(f"ToolCallMapper: stored provider tool_call id={provider_id} at index={index}")

    def map_client_id_to_provider(self, client_id: str, index: int) -> str:
        if index in self._index_to_provider_id:
            provider_id = self._index_to_provider_id[index]
            self._client_to_provider_map[client_id] = provider_id
            logger.info(
                f"ToolCallMapper: mapped client_id={client_id} to provider_id={provider_id}"
            )
            return provider_id

        if client_id in self._client_to_provider_map:
            return self._client_to_provider_map[client_id]

        for provider_id, stored_index in self._index_to_provider_id.items():
            if stored_index == index:
                self._client_to_provider_map[client_id] = provider_id
                return provider_id

        logger.warning(f"ToolCallMapper: no mapping found for client_id={client_id}, index={index}")
        return client_id

    def get_provider_id(self, client_id: str) -> str | None:
        return self._client_to_provider_map.get(client_id)

    def get_all_provider_ids(self) -> list[str]:
        return list(self._index_to_provider_id.values())

    def reset(self):
        self._provider_tool_calls.clear()
        self._client_to_provider_map.clear()
        self._index_to_provider_id.clear()


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


def _clean_anthropic_content_block(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text") or ""
        return {"type": "text", "text": text} if text else None
    if block_type == "thinking":
        thinking = block.get("thinking") or ""
        if not thinking:
            return None
        cleaned: dict[str, Any] = {"type": "thinking", "thinking": thinking}
        signature = block.get("signature")
        if signature:
            cleaned["signature"] = signature
        return cleaned
    if block_type == "tool_use":
        return block
    return None


def _openai_image_block_to_anthropic(block: dict[str, Any]) -> dict[str, Any] | None:
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    else:
        url = image_url

    if not isinstance(url, str) or not url:
        return None

    match = _DATA_URL_RE.match(url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group("media_type"),
                "data": match.group("data"),
            },
        }

    return {
        "type": "image",
        "source": {
            "type": "url",
            "url": url,
        },
    }


def _transform_content_blocks_for_anthropic(content: Any) -> Any:
    if not isinstance(content, list):
        return content or ""

    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text") or ""
            if text:
                blocks.append({"type": "text", "text": text})
        elif block_type == "image_url":
            transformed = _openai_image_block_to_anthropic(block)
            if transformed:
                blocks.append(transformed)
        elif block_type == "image" and isinstance(block.get("source"), dict):
            blocks.append(block)
        else:
            cleaned = _clean_anthropic_content_block(block)
            if cleaned:
                blocks.append(cleaned)

    return blocks or ""


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
            thinking_blocks_field = msg.get("thinking_blocks")
            thinking_blocks: list[dict[str, Any]] = []
            if isinstance(thinking_blocks_field, list):
                for block in thinking_blocks_field:
                    cleaned = _clean_anthropic_content_block(block)
                    if cleaned:
                        thinking_blocks.append(cleaned)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "thinking":
                        cleaned = _clean_anthropic_content_block(block)
                        if cleaned:
                            thinking_blocks.append(cleaned)

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

                    all_content: list[dict[str, Any]] = []
                    all_content.extend(thinking_blocks)
                    if text_parts:
                        all_content.append({"type": "text", "text": "\n".join(text_parts)})
                    all_content.extend(tool_use_blocks)

                    processed_messages.append(
                        {
                            "role": "assistant",
                            "content": all_content,
                        }
                    )
                else:
                    if thinking_blocks:
                        blocks = list(thinking_blocks)
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    cleaned = _clean_anthropic_content_block(block)
                                    if cleaned:
                                        blocks.append(cleaned)
                        elif content:
                            blocks.append({"type": "text", "text": str(content)})
                        processed_messages.append(
                            {"role": "assistant", "content": blocks or (content or "")}
                        )
                    else:
                        processed_messages.append({"role": "assistant", "content": content or ""})
            else:
                if thinking_blocks:
                    blocks = list(thinking_blocks)
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                cleaned = _clean_anthropic_content_block(block)
                                if cleaned:
                                    blocks.append(cleaned)
                    elif content:
                        blocks.append({"type": "text", "text": str(content)})
                    processed_messages.append(
                        {"role": "assistant", "content": blocks or (content or "")}
                    )
                elif isinstance(content, list) and any(
                    isinstance(block, dict) and block.get("type") in ("thinking", "text")
                    for block in content
                ):
                    blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            cleaned = _clean_anthropic_content_block(block)
                            if cleaned:
                                blocks.append(cleaned)
                    processed_messages.append({"role": "assistant", "content": blocks or ""})
                else:
                    processed_messages.append({"role": "assistant", "content": content or ""})
            continue

        processed_messages.append({"role": role, "content": _transform_content_blocks_for_anthropic(content)})

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
                        elif block.get("type") == "image_url":
                            image_url = block.get("image_url")
                            url = image_url.get("url") if isinstance(image_url, dict) else image_url
                            if isinstance(url, str) and url:
                                match = _DATA_URL_RE.match(url)
                                if match:
                                    parts.append(
                                        {
                                            "inlineData": {
                                                "mimeType": match.group("media_type"),
                                                "data": match.group("data"),
                                            }
                                        }
                                    )
                                else:
                                    parts.append({"fileData": {"fileUri": url}})
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
