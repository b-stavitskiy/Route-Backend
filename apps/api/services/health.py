import asyncio
import time
from typing import Any

import httpx

from apps.api.core.config import get_provider_config, get_settings
from packages.redis.client import get_redis


class HealthCheckService:
    def __init__(self):
        self.provider_config = get_provider_config()
        self.settings = get_settings()
        self._running = False
        self._task: asyncio.Task | None = None

    def _get_api_key(self, api_key_env: str) -> str | None:
        env_to_settings = {
            "MINIMAX_API_KEY": "minimax_api_key",
            "OPENCODE_API_KEY": "opencode_api_key",
            "CHUTES_API_KEY": "chutes_api_key",
            "ZAI_API_KEY": "zai_api_key",
            "OPENROUTER_FREE_API_KEY": "openrouter_api_key",
            "OPENROUTER_XIAOMI_API_KEY": "openrouter_xiaomi_api_key",
            "OPENROUTER_DEEPSEEK_API_KEY": "openrouter_deepseek_api_key",
            "OPENROUTER_GROK_API_KEY": "openrouter_grok_api_key",
        }
        settings_attr = env_to_settings.get(api_key_env)
        if not settings_attr:
            return None
        return getattr(self.settings, settings_attr, None)

    async def _get_provider_client(self, provider_name: str) -> httpx.AsyncClient | None:
        provider_cfg = self.provider_config.get_provider_config(provider_name)
        if not provider_cfg:
            return None

        api_key_env = provider_cfg.get("api_key_env")
        if not api_key_env:
            return None

        api_key = self._get_api_key(api_key_env)
        if not api_key:
            return None

        base_url = provider_cfg.get("base_url", "")
        timeout = provider_cfg.get("timeout", 30)

        return httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout=timeout, connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def check_provider_health(self, provider_name: str) -> tuple[str, int | None]:
        client = await self._get_provider_client(provider_name)
        if not client:
            return "unknown", None

        try:
            start = time.perf_counter()
            response = await client.get("/models")
            latency_ms = int((time.perf_counter() - start) * 1000)

            if response.status_code == 200:
                return "healthy", latency_ms
            elif response.status_code == 401:
                return "healthy", latency_ms
            else:
                return "degraded", latency_ms
        except httpx.TimeoutException:
            return "degraded", None
        except Exception:
            return "degraded", None
        finally:
            await client.aclose()

    async def run_health_check(self):
        redis = await get_redis()
        providers = self.provider_config._config.get("providers", {}).get("providers", {})

        for provider_name in providers.keys():
            status, latency = await self.check_provider_health(provider_name)

            health_key = f"provider:{provider_name}:health"
            latency_key = f"provider:{provider_name}:latency"

            await redis.set(health_key, status, ex=60)
            if latency is not None:
                await redis.set(latency_key, str(latency), ex=60)

    async def _background_loop(self):
        interval = (
            self.provider_config._config.get("providers", {})
            .get("health_check", {})
            .get("interval_seconds", 30)
        )

        while self._running:
            try:
                await self.run_health_check()
            except Exception:
                pass

            await asyncio.sleep(interval)

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        await self.run_health_check()

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


_health_service: HealthCheckService | None = None


def get_health_service() -> HealthCheckService:
    global _health_service
    if _health_service is None:
        _health_service = HealthCheckService()
    return _health_service
