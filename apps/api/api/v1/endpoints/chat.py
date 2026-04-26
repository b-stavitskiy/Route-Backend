import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from apps.api.core.plans import get_user_effective_plan_name
from apps.api.core.rate_limiter import check_model_access
from apps.api.core.security import get_access_token_from_request, hash_api_key, verify_access_token
from apps.api.services.llm import LLMRouter
from apps.api.services.llm.router import get_model_token_budget, truncate_messages
from apps.api.services.llm.transforms import map_finish_reason, store_streaming_tool_calls
from apps.api.services.usage import UsageTracker
from apps.api.services.usage.request_manager import RequestManager
from packages.db.models import UsageLog
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import AuthenticationError

logger = logging.getLogger("routing.run.api")

router = APIRouter(prefix="/v1", tags=["chat"])


class MessageContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    image_url: dict | None = None
    thinking: str | None = None
    signature: str | None = None


class Message(BaseModel):
    role: str
    content: str | list[MessageContent] | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None


class Tool(BaseModel):
    type: str = "function"
    function: dict


class StreamOptions(BaseModel):
    include_usage: bool | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    stream_options: StreamOptions | None = None
    thinking: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def build_usage_payload(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage_payload = dict(usage or {})
    prompt_tokens = usage_payload.get("prompt_tokens") or 0
    completion_tokens = usage_payload.get("completion_tokens") or 0
    usage_payload["prompt_tokens"] = prompt_tokens
    usage_payload["completion_tokens"] = completion_tokens
    usage_payload["total_tokens"] = usage_payload.get("total_tokens") or (
        prompt_tokens + completion_tokens
    )
    return usage_payload


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    normalized_tool_call = dict(tool_call)
    function = normalized_tool_call.get("function")
    if not isinstance(function, dict):
        return normalized_tool_call

    normalized_function = dict(function)
    arguments = normalized_function.get("arguments")
    if arguments is None:
        normalized_function["arguments"] = "{}"
    elif not isinstance(arguments, str):
        try:
            normalized_function["arguments"] = json.dumps(arguments)
        except (TypeError, ValueError):
            normalized_function["arguments"] = str(arguments)

    normalized_tool_call["type"] = normalized_tool_call.get("type", "function")
    normalized_tool_call["function"] = normalized_function
    return normalized_tool_call


def build_chat_message(message: Message) -> dict[str, Any]:
    content = message.content
    if isinstance(content, list):
        content = [_normalize_content_block(c) for c in content]

    msg: dict[str, Any] = {"role": message.role, "content": content}

    if message.name is not None:
        msg["name"] = message.name

    if message.tool_calls is not None:
        msg["tool_calls"] = [_normalize_tool_call(tool_call) for tool_call in message.tool_calls]

    if message.tool_call_id is not None:
        if message.role == "tool" and not message.tool_call_id:
            raise HTTPException(
                status_code=400,
                detail="tool_call_id cannot be empty for tool results",
            )
        msg["tool_call_id"] = message.tool_call_id

    if message.reasoning_content is not None:
        msg["reasoning_content"] = message.reasoning_content

    if message.thinking_blocks is not None:
        msg["thinking_blocks"] = message.thinking_blocks

    return msg


def _normalize_content_block(content_block: Any) -> dict[str, Any] | Any:
    if hasattr(content_block, "model_dump"):
        content_block = content_block.model_dump(exclude_none=True)

    if not isinstance(content_block, dict):
        return content_block

    normalized = dict(content_block)
    block_type = normalized.get("type")

    if block_type == "input_text":
        normalized["type"] = "text"
        normalized["text"] = normalized.get("text") or normalized.get("input_text") or ""
        normalized.pop("input_text", None)
        return normalized

    if block_type in {"image", "input_image", "image_url"}:
        image_payload = _extract_image_payload(normalized)
        if image_payload:
            normalized = {"type": "image_url", "image_url": image_payload}
        return normalized

    return normalized


def _extract_image_payload(content_block: dict[str, Any]) -> dict[str, Any] | None:
    image_url = content_block.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str) and url:
            return dict(image_url)
    elif isinstance(image_url, str) and image_url:
        payload = {"url": image_url}
        if detail := content_block.get("detail"):
            payload["detail"] = detail
        return payload

    input_image = content_block.get("input_image")
    if isinstance(input_image, dict):
        url = input_image.get("image_url") or input_image.get("url")
        if isinstance(url, str) and url:
            payload = {"url": url}
            if detail := input_image.get("detail") or content_block.get("detail"):
                payload["detail"] = detail
            return payload
    elif isinstance(input_image, str) and input_image:
        payload = {"url": input_image}
        if detail := content_block.get("detail"):
            payload["detail"] = detail
        return payload

    direct_url = content_block.get("url")
    if isinstance(direct_url, str) and direct_url:
        payload = {"url": direct_url}
        if detail := content_block.get("detail"):
            payload["detail"] = detail
        return payload

    source = content_block.get("source")
    if isinstance(source, dict):
        source_type = source.get("type")
        if source_type in {"url", "image_url"}:
            url = source.get("url")
            if isinstance(url, str) and url:
                payload = {"url": url}
                if detail := source.get("detail") or content_block.get("detail"):
                    payload["detail"] = detail
                return payload
        if source_type == "base64":
            media_type = source.get("media_type") or "image/png"
            data = source.get("data")
            if isinstance(data, str) and data:
                payload = {"url": f"data:{media_type};base64,{data}"}
                if detail := source.get("detail") or content_block.get("detail"):
                    payload["detail"] = detail
                return payload

    return None


async def get_user_from_request(request: Request) -> tuple[str, str, str]:
    api_key = request.headers.get("X-API-Key", "")

    if hasattr(request.state, "api_key") and request.state.api_key:
        api_key = request.state.api_key

    if api_key:
        from packages.redis.client import get_redis

        key_hash = hash_api_key(api_key)
        redis = await get_redis()

        cache_key = f"auth:{key_hash}"
        cached = await redis.get(cache_key)
        if cached:
            parts = cached.split(":")
            if len(parts) == 3:
                logger.info(f"Auth cache hit | key_hash={key_hash[:12]}")
                return parts[0], parts[1], parts[2]

        async with get_db_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import joinedload

            from packages.db.models import ApiKey

            result = await session.execute(
                select(ApiKey)
                .where(
                    ApiKey.key_hash == key_hash,
                    ApiKey.is_active,
                )
                .options(joinedload(ApiKey.user))
            )
            api_key_obj = result.scalar_one_or_none()

            if api_key_obj:
                user = api_key_obj.user
                current_plan = (
                    get_user_effective_plan_name(user) if user else api_key_obj.plan_tier.value
                )

                cache_value = f"{api_key_obj.user_id}:{current_plan}:{api_key_obj.id}"
                await redis.set(cache_key, cache_value, ex=300)

                return str(api_key_obj.user_id), current_plan, str(api_key_obj.id)
            else:
                raise AuthenticationError("Invalid API key")

    token = get_access_token_from_request(request)
    if token:
        try:
            payload = await verify_access_token(token)
            user_id = payload.get("sub")
            plan = payload.get("plan", "free")
            api_key_id = ""
            return user_id, plan, api_key_id
        except Exception:
            pass

    raise AuthenticationError("Authentication required")


async def stream_generator(
    router_instance: LLMRouter,
    usage_tracker: UsageTracker,
    model: str,
    messages: list[dict[str, Any]],
    user_plan: str,
    user_id: str,
    api_key_id: str,
    temperature: float,
    max_tokens: int | None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    **kwargs,
) -> AsyncGenerator[bytes, None]:
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    input_tokens = 0
    output_tokens = 0
    provider = "unknown"
    latency_ms = 0
    start_time = time.time()
    streaming_tool_calls: list[dict[str, Any]] = []
    last_usage: dict[str, Any] | None = None
    redis = await get_redis()

    try:
        async for chunk in router_instance.route_chat_complete_stream(
            model=model,
            messages=messages,
            user_plan=user_plan,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            user_id=user_id,
            stream_request_id=request_id,
            **kwargs,
        ):
            if not chunk:
                continue
            if isinstance(chunk, str):
                yield f"data: {chunk}\n\n".encode()
                continue
            if not isinstance(chunk, dict):
                continue
            if "provider" in chunk:
                provider = chunk.get("provider") or "unknown"
            data = chunk.get("data") or {}
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    yield f"data: {data}\n\n".encode()
                    continue

            if not isinstance(data, dict):
                continue

            if data.get("event") == "error":
                error_msg = data.get("data", "Unknown error")
                error_payload = json.dumps({"error": {"message": error_msg, "type": "api_error"}})
                yield f"data: {error_payload}\n\n".encode()
                break

            if "usage" in data:
                usage = build_usage_payload(data.get("usage"))
                last_usage = usage
                input_tokens = usage.get("prompt_tokens") or 0
                output_tokens = usage.get("completion_tokens") or 0

                if not data.get("choices"):
                    continue

            if "provider" in data:
                provider = data.get("provider") or "unknown"

            delta = None
            reasoning_delta = None
            finish_reason = None
            tool_calls = None
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "delta" in choice:
                    delta = choice["delta"].get("content")
                    reasoning_delta = choice["delta"].get("reasoning_content")
                    tool_calls = choice["delta"].get("tool_calls")
                elif "message" in choice:
                    delta = choice["message"].get("content")
                    reasoning_delta = choice["message"].get("reasoning_content")
                    tool_calls = choice["message"].get("tool_calls")
                finish_reason = map_finish_reason(choice.get("finish_reason"))

            if delta == "":
                delta = None

            if tool_calls is not None:
                for tc in tool_calls:
                    logger.info(
                        f"Streaming tool_calls to client: id={tc.get('id')} | "
                        f"name={tc.get('function', {}).get('name')} | "
                        f"index={tc.get('index')} | component=chat"
                    )
                    tc_copy = {
                        "id": tc.get("id"),
                        "function": tc.get("function"),
                        "index": tc.get("index"),
                    }
                    streaming_tool_calls.append(tc_copy)
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n".encode()
            else:
                if reasoning_delta is not None:
                    chunk_data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": reasoning_delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode()

                if delta is not None:
                    chunk_data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode()
                elif finish_reason is not None:
                    # Preserve terminal chunks like delta={} + finish_reason="stop".
                    chunk_data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode()

        if streaming_tool_calls:
            await store_streaming_tool_calls(redis, user_id, request_id, streaming_tool_calls)

        usage_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": build_usage_payload(
                last_usage
                or {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                }
            ),
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"
        latency_ms = int((time.time() - start_time) * 1000)

    except Exception as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": "api_error",
                "code": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n".encode()

    total_tokens = input_tokens + output_tokens
    if total_tokens > 0:
        try:
            await usage_tracker.track_request(
                user_id=user_id,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

            import hashlib
            from uuid import UUID, uuid4

            request_hash = hashlib.sha256(f"{user_id}:{request_id}".encode()).hexdigest()[:16]

            async with get_db_session() as session:
                usage_log = UsageLog(
                    id=uuid4(),
                    api_key_id=UUID(api_key_id) if api_key_id and api_key_id != "" else None,
                    user_id=UUID(user_id),
                    model=model,
                    provider=provider,
                    response_model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=0.0,
                    latency_ms=latency_ms,
                    status="success",
                    request_id=request_id,
                    request_hash=request_hash,
                )
                session.add(usage_log)
                await session.commit()

            logger.info(
                f"Stream completed | model={model} | provider={provider} | "
                f"input_tokens={input_tokens} | output_tokens={output_tokens} | "
                f"latency_ms={latency_ms} | component=chat"
            )
        except Exception as e:
            logger.error(f"Failed to track stream usage: {e} | component=chat")
    else:
        logger.warning(
            f"Stream completed with zero tokens (usage not tracked) | model={model} | "
            f"provider={provider} | chunks_yielded_estimate=unknown | latency_ms={latency_ms} | "
            f"component=chat"
        )


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
):
    user_id, plan, api_key_id = await get_user_from_request(request)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            f"Chat request | user={user_id} | plan={plan} | model={body.model} | "
            f"stream={body.stream} | component=chat"
        )

    await check_model_access(plan, body.model)

    redis = await get_redis()

    request_manager = RequestManager(redis)
    await request_manager.check_and_increment(user_id, plan, body.model)

    router_instance = LLMRouter(redis)

    messages = [build_chat_message(message) for message in body.messages]

    tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
    if tool_result_msgs:
        for msg in tool_result_msgs:
            logger.info(
                f"Tool result received: tool_call_id={msg.get('tool_call_id')} | "
                f"content_present={bool(msg.get('content'))} | component=chat"
            )

    logger.info(
        f"Routing request to model | model={body.model} | "
        f"messages_count={len(messages)} | component=router"
    )

    model_config = router_instance.provider_config.get_model_config(body.model, plan)
    context_size, output_tokens_estimate, available_for_input = get_model_token_budget(
        model_config,
        body.max_tokens,
    )
    provider_max_tokens = output_tokens_estimate if body.max_tokens is not None else None

    original_msg_count = len(messages)
    messages = truncate_messages(messages, max_tokens=available_for_input)
    if len(messages) < original_msg_count:
        logger.info(
            f"Truncated messages from {original_msg_count} to {len(messages)} "
            f"(context_size={context_size}, output_tokens_estimate={output_tokens_estimate}, "
            f"available_for_input={available_for_input}) | component=router"
        )

    tools = [t.model_dump() for t in body.tools] if body.tools else None
    parallel_tool_calls = body.parallel_tool_calls
    if tools and parallel_tool_calls is None:
        parallel_tool_calls = True

    if body.stream:
        usage_tracker = UsageTracker(redis)
        return StreamingResponse(
            stream_generator(
                router_instance=router_instance,
                usage_tracker=usage_tracker,
                model=body.model,
                messages=messages,
                user_plan=plan,
                user_id=user_id,
                api_key_id=api_key_id,
                temperature=body.temperature,
                max_tokens=provider_max_tokens,
                top_p=body.top_p,
                frequency_penalty=body.frequency_penalty,
                presence_penalty=body.presence_penalty,
                stop=body.stop,
                tools=tools,
                tool_choice=body.tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                stream_options=(body.stream_options.model_dump() if body.stream_options else None),
                thinking=body.thinking,
                reasoning_effort=body.reasoning_effort,
                reasoning=body.reasoning,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": f"chatcmpl-{int(time.time() * 1000)}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    response = await router_instance.route_chat_complete(
        model=body.model,
        messages=messages,
        user_plan=plan,
        temperature=body.temperature,
        max_tokens=provider_max_tokens,
        stream=False,
        top_p=body.top_p,
        frequency_penalty=body.frequency_penalty,
        presence_penalty=body.presence_penalty,
        stop=body.stop,
        tools=tools,
        tool_choice=body.tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        thinking=body.thinking,
        reasoning_effort=body.reasoning_effort,
        reasoning=body.reasoning,
    )

    usage_tracker = UsageTracker(redis)
    input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response.get("usage", {}).get("completion_tokens", 0)
    await usage_tracker.track_request(
        user_id=user_id,
        api_key_id=api_key_id,
        model=body.model,
        provider=response.get("provider", "unknown"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=response.get("latency_ms", 0),
    )

    import hashlib
    from uuid import UUID, uuid4

    request_id = response.get("id", str(uuid4()))
    request_hash = hashlib.sha256(f"{user_id}:{request_id}".encode()).hexdigest()[:16]

    async with get_db_session() as session:
        usage_log = UsageLog(
            id=uuid4(),
            api_key_id=UUID(api_key_id) if api_key_id and api_key_id != "" else None,
            user_id=UUID(user_id),
            model=body.model,
            provider=response.get("provider", "unknown"),
            response_model=response.get("model"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.0,
            latency_ms=response.get("latency_ms", 0),
            status="success",
            request_id=request_id,
            request_hash=request_hash,
        )
        session.add(usage_log)
        await session.commit()

    logger.info(
        f"Request completed | model={body.model} | "
        f"provider={response.get('provider')} | input_tokens={input_tokens} | "
        f"output_tokens={output_tokens} | latency_ms={response.get('latency_ms')} | "
        f"component=chat"
    )

    return response
