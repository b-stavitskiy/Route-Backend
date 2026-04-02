from datetime import datetime, timedelta, UTC
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from packages.redis.client import get_redis

router = APIRouter(prefix="/v1", tags=["status"])


class ProviderStatus(BaseModel):
    name: str
    status: str
    latency_ms: int | None = None


class ModelStatus(BaseModel):
    id: str
    name: str
    provider: str
    tier: str
    status: str


class IncidentReport(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    status: str
    created_at: str
    resolved_at: str | None = None


class StatusResponse(BaseModel):
    providers: list[ProviderStatus]
    models: list[ModelStatus]
    incidents: list[IncidentReport]
    last_updated: str


class PublicSettings(BaseModel):
    cost_multiplier: float


PROVIDER_DISPLAY_NAMES = {
    "minimax": "MiniMax",
    "openrouter": "OpenRouter",
    "opencode": "OpenCode",
    "chutes": "Chutes",
    "zai": "Zai",
    "openrouter_xiaomi": "OpenRouter Xiaomi",
    "openrouter_deepseek": "OpenRouter DeepSeek",
    "openrouter_grok": "OpenRouter Grok",
}

PROVIDER_LOGOS = {
    "minimax": "/providers/minimax.svg",
    "openrouter": "/providers/openrouter.svg",
    "opencode": "/providers/opencode.svg",
    "chutes": "/providers/chutes.svg",
    "zai": "/providers/zai.svg",
    "openrouter_xiaomi": "/providers/openrouter.svg",
    "openrouter_deepseek": "/providers/openrouter.svg",
    "openrouter_grok": "/providers/openrouter.svg",
}

MODEL_DISPLAY_NAMES = {
    "route/nemotron-3-super-120b": "NVIDIA Nemotron-3-Super-120B",
    "route/trinity-large-preview": "Arcee AI Trinity Large",
    "route/glm-4.5-air": "Zhipu GLM-4.5-Air",
    "route/nemotron-3-nano-30b": "NVIDIA Nemotron-3-Nano-30B",
    "route/minimax-m2.5": "MiniMax-M2.5",
    "route/qwen3-coder": "Qwen3-Coder",
    "route/gpt-oss-120b": "OpenAI GPT-OSS-120B",
    "route/hermes-3-llama-3.1-405b": "Nous Hermes-3-Llama-3.1-405B",
    "route/llama-3.2-3b-instruct": "Meta Llama-3.2-3B",
    "route/gemma-3-27b-it": "Google Gemma-3-27B",
    "route/minimax-m2.7": "MiniMax-M2.7",
    "route/kimi-k2.5": "Kimi K2.5",
    "route/minimax-m2.5-highspeed": "MiniMax-M2.5-Highspeed",
    "route/minimax-m2.7-highspeed": "MiniMax-M2.7-Highspeed",
    "route/glm-5": "Zhipu GLM-5",
    "route/glm-5-turbo": "Zhipu GLM-5-Turbo",
    "route/deepseek-v3.2": "DeepSeek V3.2",
    "route/qwen3-coder-next": "Qwen3-Coder-Next",
    "route/qwen3-32b": "Qwen3-32B",
    "route/qwen3.6-plus-preview": "Qwen3.6-Plus-Preview",
    "route/qwen3-next-80b": "Qwen3-Next-80B",
    "route/deepseek-v3.2-speciale": "DeepSeek V3.2-Speciale",
    "route/deepseek-r1": "DeepSeek R1",
    "route/grok-4-fast": "xAI Grok-4-Fast",
    "route/grok-4.20-beta": "xAI Grok-4.20-Beta",
    "route/grok-4.20-multi-agent-beta": "xAI Grok-4.20-Multi-Agent-Beta",
    "route/mimo-v2-omni": "Xiaomi Mimo-V2-Omni",
    "route/mimo-v2-pro": "Xiaomi Mimo-V2-Pro",
    "route/mimo-v2-flash": "Xiaomi Mimo-V2-Flash",
}


@router.get("/status")
async def get_status():
    from apps.api.core.config import get_provider_config

    provider_config = get_provider_config()
    provider_config._load_config()
    redis = await get_redis()

    providers = []
    all_models = []

    provider_settings = provider_config._config.get("providers", {}).get("providers", {})
    model_settings = provider_config._config.get("providers", {}).get("models", {})

    for provider_name in provider_settings.keys():
        health_key = f"provider:{provider_name}:health"
        latency_key = f"provider:{provider_name}:latency"

        health_data = await redis.get(health_key)
        latency_data = await redis.get(latency_key)

        status = "healthy"
        if health_data == "degraded":
            status = "degraded"
        elif health_data == "unknown" or not health_data:
            status = "unknown"

        providers.append(
            ProviderStatus(
                name=PROVIDER_DISPLAY_NAMES.get(provider_name, provider_name),
                status=status,
                latency_ms=int(latency_data) if latency_data else None,
            )
        )

        valid_tiers = {"free", "lite", "pro", "max"}
        for tier, tier_models in model_settings.items():
            if tier not in valid_tiers:
                continue
            if tier_models is None:
                continue
            if not isinstance(tier_models, dict):
                continue
            for model_id in tier_models.keys():
                if model_id not in [m.id for m in all_models]:
                    chain = tier_models[model_id].get("provider_chain", [])
                    model_provider = chain[0]["provider"] if chain else provider_name

                    if model_provider == provider_name:
                        all_models.append(
                            ModelStatus(
                                id=model_id,
                                name=MODEL_DISPLAY_NAMES.get(
                                    model_id,
                                    model_id.replace("route/", "").replace("-", " ").title(),
                                ),
                                provider=PROVIDER_DISPLAY_NAMES.get(model_provider, model_provider),
                                tier=tier,
                                status="online",
                            )
                        )

    incidents = [
        IncidentReport(
            id="inc-001",
            title="OpenRouter Rate Limiting",
            description="Some OpenRouter models are experiencing rate limiting due to high demand.",
            severity="warning",
            status="ongoing",
            created_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
            resolved_at=None,
        ),
        IncidentReport(
            id="inc-002",
            title=" Trinity Large Preview Unavailable",
            description="The arcee-ai/trinity-large-preview model endpoint is not responding.",
            severity="error",
            status="resolved",
            created_at=(datetime.now(UTC) - timedelta(hours=5)).isoformat(),
            resolved_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        ),
    ]

    return StatusResponse(
        providers=providers,
        models=all_models,
        incidents=incidents,
        last_updated=datetime.now(UTC).isoformat(),
    )


@router.get("/settings", response_model=PublicSettings)
async def get_public_settings():
    from apps.api.core.config import get_settings

    settings = get_settings()
    return PublicSettings(cost_multiplier=settings.cost_multiplier)
