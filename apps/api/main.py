import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apps.api.api.v1.router import router as v1_router
from apps.api.core.config import get_settings
from apps.api.core.middleware import (
    AuthMiddleware,
    ExceptionHandlerMiddleware,
    MetricsMiddleware,
)
from packages.db.session import close_db, init_db
from packages.redis.client import close_redis, create_redis_pool
from apps.api.services.health import get_health_service

load_dotenv(Path(__file__).parent.parent.parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
    except Exception:
        pass
    try:
        await create_redis_pool()
    except Exception:
        pass
    try:
        health_service = get_health_service()
        await health_service.start()
    except Exception:
        pass
    yield
    try:
        health_service = get_health_service()
        await health_service.stop()
    except Exception:
        pass
    try:
        await close_db()
    except Exception:
        pass
    try:
        await close_redis()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Routing.Run API",
        description="API gateway with multi-provider routing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
