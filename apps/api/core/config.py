import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Database (required)
    database_url: str

    # Redis (required)
    redis_url: str

    # JWT (required)
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        lower_v = v.lower()
        if "change-me" in lower_v or "changeme" in lower_v:
            raise ValueError("JWT_SECRET_KEY contains 'change-me' placeholder - must be changed")
        if "replace-me" in lower_v or "replace_me" in lower_v:
            raise ValueError("JWT_SECRET_KEY contains 'replace-me' placeholder - must be changed")
        if "placeholder" in lower_v:
            raise ValueError("JWT_SECRET_KEY contains 'placeholder' - must be changed")
        if lower_v.startswith("your-") or lower_v.startswith("your_"):
            raise ValueError("JWT_SECRET_KEY starts with 'your-' - must be changed")
        if v == "secret" or v == "secret_key" or "your-secret-key" in lower_v:
            raise ValueError("JWT_SECRET_KEY appears to be a default value - must be changed")
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v

    # Security
    argon2_rounds: int = 4
    api_key_prefix: str = "rk_"
    api_key_length: int = 32

    # OAuth
    github_client_id: str = ""
    github_client_secret: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    oauth_redirect_uri: str = ""

    # Email
    email_provider: str = "unosend"
    unosend_api_key: str = ""
    from_email: str = ""
    from_name: str = ""

    # Whop
    whop_client_id: str = ""
    whop_client_secret: str = ""
    whop_webhook_secret: str = ""
    whop_api_base_url: str = "https://api.whop.com"

    # Worker
    worker_concurrency: int = 4
    worker_queue_name: str = "arq:queue"
    worker_max_jobs: int = 100
    worker_job_timeout: int = 300

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    cors_origins: str = "[]"

    # Display
    cost_multiplier: float = 1.65

    # Config Repo
    config_github_token: str = ""
    config_repo_url: str = "https://github.com/RoutingRun/Route-Configs.git"
    config_branch: str = "main"

    # Provider API Keys
    minimax_api_key: str = ""
    opencode_api_key: str = ""
    chutes_api_key: str = ""
    zai_api_key: str = ""
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_FREE_API_KEY")
    openrouter_xiaomi_api_key: str = Field(default="", alias="OPENROUTER_XIAOMI_API_KEY")
    openrouter_deepseek_api_key: str = Field(default="", alias="OPENROUTER_DEEPSEEK_API_KEY")
    openrouter_grok_api_key: str = Field(default="", alias="OPENROUTER_GROK_API_KEY")

    @property
    def database(self):
        from apps.api.core.config import DatabaseSettings

        return DatabaseSettings(url=self.database_url)

    @property
    def redis(self):
        from apps.api.core.config import RedisSettings

        return RedisSettings(url=self.redis_url)

    @property
    def jwt(self):
        from apps.api.core.config import JWTSettings

        return JWTSettings(
            secret_key=self.jwt_secret_key,
            algorithm=self.jwt_algorithm,
        )

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.cors_origins:
            return []
        try:
            return json.loads(self.cors_origins)
        except:
            return []


class DatabaseSettings(BaseSettings):
    url: str
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600


class RedisSettings(BaseSettings):
    url: str
    max_connections: int = 50
    decode_responses: bool = True


class JWTSettings(BaseSettings):
    secret_key: str
    algorithm: str = "HS256"


def load_yaml_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    return Settings()


class ProviderConfig:
    _instance = None
    _config: dict[str, Any] = {}
    _plans_config: dict[str, Any] = {}
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_config(self):
        if self._initialized:
            return

        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        provider_config_path = config_dir / "provider.yaml"
        plans_config_path = config_dir / "plans.yaml"

        self._config = {
            "providers": load_yaml_config(str(provider_config_path)),
        }
        self._plans_config = load_yaml_config(str(plans_config_path))
        self._initialized = True

    async def load_remote_config(self, settings: Settings | None = None):
        if settings is None:
            settings = get_settings()

        from packages.shared.config_puller import get_configs

        logger.info(
            f"Cloning config from: {settings.config_repo_url} (branch: {settings.config_branch})",
            component="config",
        )
        configs = await get_configs(
            use_remote=bool(settings.config_github_token),
            github_token=settings.config_github_token,
            repo_url=settings.config_repo_url,
            branch=settings.config_branch,
        )
        logger.info("Config repository cloned successfully | component=config")

        if configs.get("provider.yaml"):
            self._config = {"providers": configs["provider.yaml"]}
            logger.info("Provider config injected | component=config")
        if configs.get("plans.yaml"):
            self._plans_config = configs["plans.yaml"]
            logger.info("Plans config injected | component=config")

        self._initialized = True

    def get_provider_config(self, provider: str) -> dict[str, Any] | None:
        return self._config.get("providers", {}).get("providers", {}).get(provider)

    def get_model_config(self, model: str, user_plan: str = "free") -> dict[str, Any] | None:
        models = self._config.get("providers", {}).get("models", {})
        plan_tier_map = {
            "free": "free",
            "dev": "free",
            "lite": "lite",
            "premium": "pro",
            "max": "max",
        }
        tier = plan_tier_map.get(user_plan, "free")
        if model in models.get(tier, {}):
            return models[tier][model]
        for fallback_tier in ["free", "pro", "max"]:
            if model in models.get(fallback_tier, {}):
                return models[fallback_tier][model]
        return None

    def get_provider_chain(self, model: str, user_plan: str) -> list[dict[str, Any]]:
        model_config = self.get_model_config(model, user_plan)
        if not model_config:
            return []

        chain = []
        for provider_entry in model_config.get("provider_chain", []):
            max_plan_only = provider_entry.get("max_plan_only", False)
            if max_plan_only and user_plan != "max":
                continue
            chain.append(provider_entry)

        return chain

    def get_routing_config(self) -> dict[str, Any]:
        return self._config.get("providers", {}).get("routing", {})

    def get_plan_config(self, plan: str) -> dict[str, Any] | None:
        return self._plans_config.get("plans", {}).get(plan.lower())

    def get_allowed_models(self, user_plan: str) -> list[str] | str:
        plan_config = self.get_plan_config(user_plan)
        if not plan_config:
            return []
        return plan_config.get("allowed_models", [])

    def is_model_allowed(self, model: str, user_plan: str) -> bool:
        allowed_models = self.get_allowed_models(user_plan)
        if allowed_models == "all":
            return True
        return model in allowed_models

    def get_model_pricing(self, model: str) -> dict[str, Any] | None:
        model_pricing = self._config.get("providers", {}).get("model_pricing", {})
        for tier in ["max", "pro", "lite", "free"]:
            tier_pricing = model_pricing.get(tier, {})
            if model in tier_pricing:
                return tier_pricing[model]
        return None


def get_provider_config() -> ProviderConfig:
    return ProviderConfig()
