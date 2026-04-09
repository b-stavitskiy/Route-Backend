import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.api.v1.router import router as v1_router
from apps.api.core.config import get_provider_config, get_settings
from apps.api.core.middleware import (
    AuthMiddleware,
    ExceptionHandlerMiddleware,
    MetricsMiddleware,
)
from apps.api.services.health import get_health_service
from packages.db.session import close_db, init_db
from packages.redis.client import close_redis, create_redis_pool

load_dotenv(Path(__file__).parent.parent.parent / ".env")


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        return f"{self.formatTime(record)} | {record.levelname:8} | {record.getMessage()}"


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(StructuredFormatter())

root_logger = logging.getLogger("routing.run")
root_logger.setLevel(logging.INFO)
root_logger.handlers = []
root_logger.addHandler(handler)

for logger_name in ["routing.run.api", "routing.run.router", "routing.run.config"]:
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)
    log.handlers = []
    log.addHandler(handler)

logger = root_logger


def log_with_component(logger_method, msg, component):
    logger_method(f"{msg} | component={component}")


def create_logger_with_component(name):
    log = logging.getLogger(name)
    original_info = log.info
    original_warning = log.warning
    original_error = log.error

    def info(msg, component=None):
        if component:
            original_info(f"{msg} | component={component}")
        else:
            original_info(msg)

    def warning(msg, component=None):
        if component:
            original_warning(f"{msg} | component={component}")
        else:
            original_warning(msg)

    def error(msg, component=None):
        if component:
            original_error(f"{msg} | component={component}")
        else:
            original_error(msg)

    log.info = info
    log.warning = warning
    log.error = error
    return log


api_logger = create_logger_with_component("routing.run.api")
router_logger = create_logger_with_component("routing.run.router")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_with_component(logger.info, "Starting Routing.Run API", "startup")

    log_with_component(logger.info, "Initializing database connection", "database")
    try:
        await init_db()
        log_with_component(logger.info, "Database initialized successfully", "database")
    except Exception as e:
        log_with_component(logger.error, f"Database initialization failed: {e}", "database")

    log_with_component(logger.info, "Creating Redis connection pool", "redis")
    try:
        await create_redis_pool()
        log_with_component(logger.info, "Redis connected successfully", "redis")
    except Exception as e:
        log_with_component(logger.error, f"Redis connection failed: {e}", "redis")

    log_with_component(logger.info, "Loading provider configuration from remote", "config")
    try:
        provider_config = get_provider_config()
        await provider_config.load_remote_config()
        log_with_component(logger.info, "Provider config loaded from remote", "config")

        plans = list(provider_config._plans_config.get("plans", {}).keys())
        log_with_component(logger.info, f"Plans loaded: {plans}", "config")

        providers = provider_config._config.get("providers", {}).get("providers", {})
        log_with_component(logger.info, f"Providers loaded: {list(providers.keys())}", "providers")

        models = provider_config._config.get("providers", {}).get("models", {})
        for tier, tier_models in models.items():
            if tier_models is None or not isinstance(tier_models, dict):
                continue
            log_with_component(
                logger.info,
                f"Tier [{tier}] has {len(tier_models)} models: {list(tier_models.keys())}",
                "models",
            )
    except Exception as e:
        log_with_component(logger.error, f"Provider config loading failed: {e}", "config")

    log_with_component(logger.info, "Starting health service", "health")
    try:
        health_service = get_health_service()
        await health_service.start()
        log_with_component(logger.info, "Health service started", "health")
    except Exception as e:
        log_with_component(logger.error, f"Health service failed: {e}", "health")

    log_with_component(logger.info, "API startup complete, ready to accept requests", "startup")

    yield

    log_with_component(logger.info, "Shutting down Routing.Run API", "shutdown")
    try:
        health_service = get_health_service()
        await health_service.stop()
        log_with_component(logger.info, "Health service stopped", "health")
    except Exception:
        pass
    try:
        await close_db()
        log_with_component(logger.info, "Database connections closed", "database")
    except Exception:
        pass
    try:
        await close_redis()
        log_with_component(logger.info, "Redis connections closed", "redis")
    except Exception:
        pass
    log_with_component(logger.info, "Shutdown complete", "shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Routing.Run API",
        description="API gateway with multi-provider routing",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_cors_for_ai_endpoints(request, call_next):
        ai_paths = [
            "/v1/chat",
            "/v1/images",
            "/v1/anthropic",
            "/v1/models",
            "/v1/messages",
            "/v1/user",
            "/auth",
        ]
        if any(request.url.path.startswith(path) for path in ai_paths):
            origin = request.headers.get("origin")
            if request.method == "OPTIONS":
                from fastapi.responses import Response

                allow_headers = request.headers.get("access-control-request-headers", "*")
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": origin or "*",
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Allow-Methods": "*",
                        "Access-Control-Allow-Headers": allow_headers,
                    },
                )
            response = await call_next(request)
            if hasattr(response, "headers"):
                response.headers["Access-Control-Allow-Origin"] = origin or "*"
                response.headers["Access-Control-Allow-Credentials"] = "true"
            return response
        return await call_next(request)

    app.include_router(v1_router)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": int(time.time())}

    @app.get("/")
    async def root():
        return {
            "name": "Routing.Run API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
