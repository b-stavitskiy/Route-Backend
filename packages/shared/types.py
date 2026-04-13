from enum import StrEnum

__all__ = [
    "PlanTier",
    "ModelTier",
    "ProviderStatus",
    "RequestStatus",
    "OAuthProvider",
    "PlanLimits",
    "MODEL_PRICING",
]


class PlanTier(StrEnum):
    FREE = "free"
    LITE = "lite"
    PREMIUM = "premium"
    MAX = "max"
    PAYG = "payg"


class ModelTier(StrEnum):
    LITE = "lite"
    PREMIUM = "premium"


class ProviderStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


class RequestStatus(StrEnum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    INVALID_API_KEY = "invalid_api_key"
    INVALID_MODEL = "invalid_model"
    MODEL_NOT_ALLOWED = "model_not_allowed"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class OAuthProvider(StrEnum):
    GITHUB = "github"
    GOOGLE = "google"
    DISCORD = "discord"


class PlanLimits:
    PLAN_RATE_LIMITS = {
        PlanTier.FREE: {"lite": 20, "premium": 0},
        PlanTier.LITE: {"lite": 50, "premium": 0},
        PlanTier.PREMIUM: {"lite": 100, "premium": 20},
        PlanTier.MAX: {"lite": 150, "premium": 40},
        PlanTier.PAYG: {"lite": 500, "premium": 200},
    }

    PLAN_BURST_LIMITS = {
        PlanTier.FREE: {"lite": 5, "premium": 0},
        PlanTier.LITE: {"lite": 10, "premium": 0},
        PlanTier.PREMIUM: {"lite": 20, "premium": 5},
        PlanTier.MAX: {"lite": 30, "premium": 10},
        PlanTier.PAYG: {"lite": 50, "premium": 20},
    }

    ALL_MODELS = "all"
    LITE_ONLY = "lite_only"


MODEL_PRICING = {
    "minimax-m2.5": {"input": 0.10, "output": 0.10},
    "minimax-m2.7": {"input": 0.12, "output": 0.12},
    "kimi-k2.5": {"input": 0.08, "output": 0.08},
    "glm-5": {"input": 0.05, "output": 0.05},
    "glm-5-turbo": {"input": 0.08, "output": 0.08},
    "deepseek-v3.2": {"input": 0.50, "output": 1.50},
    "qwen3-coder-next": {"input": 0.40, "output": 1.20},
    "qwen3-32b": {"input": 0.35, "output": 1.00},
    "claude-sonnet-4.6": {"input": 3.00, "output": 15.00},
    "claude-opus-4.6": {"input": 15.00, "output": 75.00},
    "claude-haiku-4.6": {"input": 0.80, "output": 4.00},
}
