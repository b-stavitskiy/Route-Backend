import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import tiktoken
from redis.asyncio import Redis

from apps.api.core.config import get_provider_config
from apps.api.services.llm.base import (
    AnthropicCompatProvider,
    OpenAICompatProvider,
)
from apps.api.services.llm.providers import get_provider_for_model
from packages.redis.client import RedisCache
from packages.shared.exceptions import (
    InvalidModelError,
    ProviderError,
    ProviderTimeoutError,
)

logger = logging.getLogger("routing.run.router")

MAX_MESSAGES = 50
MAX_TOKENS = 200000

_encoders: dict[str, Any] = {}


def _get_encoder() -> Any:
    try:
        model = "cl100k_base"
        if model not in _encoders:
            _encoders[model] = tiktoken.get_encoding(model)
        return _encoders[model]
    except Exception:
        return None


def _count_tokens(text: str) -> int:
    encoder = _get_encoder()
    if encoder is None:
        return len(text) // 4
    try:
        return len(encoder.encode(text or ""))
    except Exception:
        return len(text) // 4


def _count_message_tokens(msg: dict[str, Any]) -> int:
    base = 3
    base += _count_tokens(msg.get("role", ""))
    base += _count_tokens(msg.get("content", ""))
    if msg.get("tool_call_id"):
        base += _count_tokens(msg.get("tool_call_id", ""))
    tc = msg.get("tool_calls", [])
    if tc:
        base += 3
        for t in tc:
            if isinstance(t, dict):
                base += _count_tokens(t.get("id", ""))
                base += _count_tokens(t.get("function", {}).get("name", ""))
                base += _count_tokens(t.get("function", {}).get("arguments", ""))
    return base


def truncate_messages(
    messages: list[dict[str, Any]], max_messages: int = MAX_MESSAGES, max_tokens: int = MAX_TOKENS
) -> list[dict[str, Any]]:
    if not messages:
        return messages

    if len(messages) <= max_messages:
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars < max_tokens // 2:
            return messages

    total_tokens = sum(_count_message_tokens(m) for m in messages)
    if len(messages) <= max_messages and total_tokens <= max_tokens:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    system_tokens = sum(_count_message_tokens(m) for m in system_msgs)
    available_tokens = max_tokens - system_tokens
    available_messages = max_messages - len(system_msgs)

    if available_messages < 1:
        available_messages = 1
    if available_tokens < 100:
        available_tokens = 100

    kept_msgs = list(system_msgs)
    kept_tokens = system_tokens

    for msg in reversed(other_msgs):
        msg_tokens = _count_message_tokens(msg)
        if len(kept_msgs) >= max_messages:
            break
        if kept_tokens + msg_tokens > available_tokens and kept_msgs:
            break
        kept_msgs.insert(len(system_msgs), msg)
        kept_tokens += msg_tokens

    if len(kept_msgs) < len(messages) or total_tokens > max_tokens:
        logger.warning(
            f"Truncated {len(messages)} msgs / ~{total_tokens} tokens -> "
            f"{len(kept_msgs)} msgs / ~{kept_tokens} tokens | component=router"
        )

    return kept_msgs


def sanitize_response(response: dict[str, Any]) -> dict[str, Any]:
    try:
        if "choices" in response:
            for choice in response["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    if isinstance(content, str) and (
                        "\n" in content or "\r" in content or "\t" in content
                    ):
                        choice["message"]["content"] = (
                            content.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
                        )
                if "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]
                    if isinstance(content, str) and (
                        "\n" in content or "\r" in content or "\t" in content
                    ):
                        choice["delta"]["content"] = (
                            content.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
                        )
    except Exception:
        pass
    return response


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        half_open_requests: int = 1,
        reset_timeout: int = 60,
    ):
        self.failure_threshold = failure_threshold
        self.half_open_requests = half_open_requests
        self.reset_timeout = reset_timeout

        self._failure_count: dict[str, int] = {}
        self._circuit_open: dict[str, bool] = {}
        self._last_failure_time: dict[str, float] = {}
        self._half_open_count: dict[str, int] = {}

    def is_open(self, provider: str) -> bool:
        if self._circuit_open.get(provider, False):
            if time.time() - self._last_failure_time.get(provider, 0) > self.reset_timeout:
                self._half_open_count[provider] = 0
                self._circuit_open[provider] = False
                return False
            return True
        return False

    def record_failure(self, provider: str):
        self._failure_count[provider] = self._failure_count.get(provider, 0) + 1
        self._last_failure_time[provider] = time.time()

        if self._failure_count[provider] >= self.failure_threshold:
            self._circuit_open[provider] = True

    def record_success(self, provider: str):
        self._failure_count[provider] = 0
        self._circuit_open[provider] = False
        self._half_open_count[provider] = 0

    def can_attempt(self, provider: str) -> bool:
        if self.is_open(provider):
            if self._half_open_count.get(provider, 0) < self.half_open_requests:
                self._half_open_count[provider] = self._half_open_count.get(provider, 0) + 1
                return True
            return False
        return True


class LLMRouter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.cache = RedisCache(redis)
        self.provider_config = get_provider_config()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            half_open_requests=1,
            reset_timeout=60,
        )
        self._provider_clients: dict[str, OpenAICompatProvider | AnthropicCompatProvider] = {}

    async def get_provider_client(
        self,
        provider_name: str,
        model_id: str,
    ) -> OpenAICompatProvider | AnthropicCompatProvider:
        key = f"{provider_name}:{model_id}"
        if key not in self._provider_clients:
            self._provider_clients[key] = get_provider_for_model(provider_name, model_id)
        return self._provider_clients[key]

    async def route_chat_complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        user_plan: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        request_id = f"chatcmpl-{uuid4().hex[:8]}"

        provider_chain = self.provider_config.get_provider_chain(model, user_plan)
        if not provider_chain:
            logger.warning(f"No provider chain found for model={model} plan={user_plan}")
            raise InvalidModelError(model)

        routing_config = self.provider_config.get_routing_config()
        retry_count = routing_config.get("retry_count", 2)

        has_tool_results = any(m.get("role") == "tool" for m in messages)
        if has_tool_results:
            logger.info(
                "Tool results detected in request, routing to first provider only | "
                "component=router"
            )
            provider_chain = [provider_chain[0]] if provider_chain else []
            retry_count = 0

        logger.info(
            f"Routing chat request | model={model} | plan={user_plan} | request_id={request_id} | "
            f"providers={len(provider_chain)} | max_retries={retry_count}",
        )

        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            for provider_entry in provider_chain:
                provider_name = provider_entry["provider"]
                model_id = provider_entry.get("model_id", model)

                if self.circuit_breaker.is_open(provider_name):
                    logger.debug(
                        f"Circuit breaker open, skipping provider={provider_name}",
                    )
                    continue

                try:
                    client = await self.get_provider_client(provider_name, model_id)

                    logger.info(
                        f"Calling provider | provider={provider_name} | "
                        f"model={model_id} | attempt={attempt + 1}",
                    )

                    start_time = time.time()
                    response = await client.chat_complete(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        **kwargs,
                    )
                    latency_ms = int((time.time() - start_time) * 1000)

                    self.circuit_breaker.record_success(provider_name)
                    await self._update_provider_health(provider_name, latency_ms, False)

                    response["id"] = response.get("id", request_id)
                    response["provider"] = provider_name
                    response["model"] = model
                    response["latency_ms"] = latency_ms

                    logger.info(
                        f"Provider response success | provider={provider_name} | "
                        f"model={model_id} | latency_ms={latency_ms}",
                    )

                    return sanitize_response(response)

                except ProviderTimeoutError as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    logger.warning(
                        f"Provider timeout | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

                except ProviderError as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    logger.warning(
                        f"Provider error | provider={provider_name} | model={model_id} | error={e}",
                    )
                    continue

                except Exception as e:
                    last_error = e
                    self.circuit_breaker.record_failure(provider_name)
                    logger.error(
                        f"Unexpected error | provider={provider_name} | "
                        f"model={model_id} | error={e}",
                    )
                    continue

            if attempt < retry_count:
                delay_ms = routing_config.get("retry_delay_ms", 200)
                logger.info(
                    f"Retrying request | attempt={attempt + 1} | delay_ms={delay_ms}",
                )
                await asyncio.sleep(delay_ms / 1000)

        logger.error(
            f"All providers failed | model={model} | last_error={last_error}",
        )

        if last_error:
            raise last_error

        raise ProviderError(
            message="No providers available for this model",
            provider="all",
        )

    async def route_chat_complete_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        user_plan: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        user_id: str | None = None,
        stream_request_id: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = stream_request_id or f"chatcmpl-{uuid4().hex[:8]}"

        provider_chain = self.provider_config.get_provider_chain(model, user_plan)
        if not provider_chain:
            logger.warning(f"No provider chain found for model={model} plan={user_plan}")
            raise InvalidModelError(model)

        routing_config = self.provider_config.get_routing_config()
        retry_count = routing_config.get("retry_count", 2)

        has_tool_results = any(m.get("role") == "tool" for m in messages)
        if has_tool_results:
            logger.info(
                "Tool results detected in request, routing to first provider only | "
                "component=router"
            )
            provider_chain = [provider_chain[0]] if provider_chain else []
            retry_count = 0

        logger.info(
            f"Routing stream request | model={model} | plan={user_plan} | "
            f"request_id={request_id} | providers={len(provider_chain)} | "
            f"max_retries={retry_count}",
        )

        last_error: Exception | None = None

        for attempt in range(retry_count + 1):
            for provider_entry in provider_chain:
                provider_name = provider_entry["provider"]
                model_id = provider_entry.get("model_id", model)

                if self.circuit_breaker.is_open(provider_name):
                    logger.debug(
                        f"Circuit breaker open, skipping provider={provider_name}",
                    )
                    continue

                try:
                    client = await self.get_provider_client(provider_name, model_id)

                    logger.info(
                        f"Calling stream provider | provider={provider_name} | "
                        f"model={model_id} | attempt={attempt + 1}",
                    )

                    start_time = time.time()
                    chunks_yielded = 0
                    stream_gen = client.chat_complete_stream(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    first_chunk = True
                    last_chunk_time = time.time()

                    async for chunk in stream_gen:
                        if first_chunk:
                            latency_ms = int((time.time() - start_time) * 1000)
                            self.circuit_breaker.record_success(provider_name)
                            await self._update_provider_health(provider_name, latency_ms, False)
                            first_chunk = False
                            logger.info(
                                f"Stream started | provider={provider_name} | "
                                f"model={model_id} | first_chunk_latency_ms={latency_ms}",
                            )

                        last_chunk_time = time.time()
                        chunks_yielded += 1

                        data = chunk.get("data")
                        if isinstance(data, str):
                            try:
                                data = json.loads(data)
                                chunk["data"] = data
                            except json.JSONDecodeError:
                                pass

                        if isinstance(data, dict) and "choices" in data:
                            for choice in data.get("choices", []):
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if isinstance(content, str):
                                        choice["delta"]["content"] = (
                                            content.replace("\r", "\\r")
                                            .replace("\n", "\\n")
                                            .replace("\t", "\\t")
                                        )

                        chunk["request_id"] = request_id
                        chunk["provider"] = provider_name
                        if user_id:
                            chunk["user_id"] = user_id
                        yield chunk

                    if chunks_yielded > 0:
                        idle_time = time.time() - last_chunk_time
                        logger.info(
                            f"Stream completed | provider={provider_name} | "
                            f"model={model_id} | chunks_yielded={chunks_yielded} | "
                            f"idle_before_done_sec={idle_time:.1f}",
                        )
                    return

                except TimeoutError as e:
                    last_error = ProviderTimeoutError(
                        f"Stream stalled (no data for {chunk_timeout_seconds}s)",
                        provider=provider_name,
                        timeout=chunk_timeout_seconds,
                    )
                    if chunks_yielded > 0:
                        logger.warning(
                            f"Stream partial failure (stalled) | provider={provider_name} | "
                            f"model={model_id} | chunks_yielded={chunks_yielded} | error={e}",
                        )
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    continue

                except ProviderTimeoutError as e:
                    last_error = e
                    if chunks_yielded > 0:
                        logger.warning(
                            f"Stream partial failure | provider={provider_name} | "
                            f"model={model_id} | chunks_yielded={chunks_yielded} | error={e}",
                        )
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    continue

                except ProviderError as e:
                    last_error = e
                    if chunks_yielded > 0:
                        logger.warning(
                            f"Stream partial failure | provider={provider_name} | "
                            f"model={model_id} | chunks_yielded={chunks_yielded} | error={e}",
                        )
                    self.circuit_breaker.record_failure(provider_name)
                    await self._update_provider_health(provider_name, 0, True)
                    continue

                except Exception as e:
                    last_error = e
                    if chunks_yielded > 0:
                        logger.warning(
                            f"Stream partial failure | provider={provider_name} | "
                            f"model={model_id} | chunks_yielded={chunks_yielded} | error={e}",
                        )
                    else:
                        logger.error(
                            f"Stream unexpected error | provider={provider_name} | "
                            f"model={model_id} | error={e}",
                        )
                    self.circuit_breaker.record_failure(provider_name)
                    continue

            if attempt < retry_count:
                delay_ms = routing_config.get("retry_delay_ms", 200)
                logger.info(
                    f"Retrying stream request | attempt={attempt + 1} | delay_ms={delay_ms}",
                )
                await asyncio.sleep(delay_ms / 1000)

        logger.error(
            f"All stream providers failed | model={model} | last_error={last_error}",
        )

        if last_error:
            raise last_error

        raise ProviderError(
            message="No providers available for this model",
            provider="all",
        )

    async def _update_provider_health(
        self,
        provider: str,
        latency_ms: int,
        is_failure: bool,
    ):
        health_key = f"provider:{provider}:health"
        latency_key = f"provider:{provider}:latency"

        await self.cache.set(health_key, "healthy" if not is_failure else "degraded", ex=30)

        if latency_ms > 0:
            await self.cache.set(latency_key, str(latency_ms), ex=60)

    async def list_available_models(self, user_plan: str) -> list[dict[str, Any]]:
        all_models = []

        provider_config = self.provider_config._config.get("providers", {}).get("models", {})

        allowed_models = self.provider_config.get_allowed_models(user_plan)

        for tier, models in provider_config.items():
            if tier not in ["free", "pro"]:
                continue

            for model_name, model_config in models.items():
                if allowed_models == "all" or model_name in allowed_models:
                    chain = model_config.get("provider_chain", [])
                    primary_provider = chain[0]["provider"] if chain else None

                    all_models.append(
                        {
                            "id": model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": primary_provider or "unknown",
                            "tier": tier,
                        }
                    )

        return all_models
