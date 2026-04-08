from datetime import UTC
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel
from apps.api.core.security import (
    generate_api_key,
    verify_access_token,
    hash_password,
)
from apps.api.services.usage import UsageTracker
from apps.api.services.usage.tracker import CreditManager
from packages.db.models import ApiKey, PlanTier, User
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import NotFoundError, RateLimitError
from sqlalchemy import select

router = APIRouter(prefix="/v1", tags=["user"])


def get_effective_plan_tier(user: User) -> PlanTier:

    if (
        user.upgraded_to_tier is not None
        and user.upgraded_until is not None
        and user.upgraded_until > datetime.now(UTC)
    ):
        return user.upgraded_to_tier
    return user.plan_tier


class UsageResponse(BaseModel):
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    models: dict[str, Any] = {}


class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    plan_tier: str
    email_verified: bool
    credits: float = 0.0


class CreditsResponse(BaseModel):
    credits: float
    credits_monthly: float
    credits_used: float
    plan_tier: str
    payg_enabled: bool


class RequestsResponse(BaseModel):
    requests_used_today: int
    requests_limit_today: int
    requests_remaining: int
    plan_tier: str


class AddCreditsRequest(BaseModel):
    amount: float
    transaction_id: str | None = None


async def get_authenticated_user(request: Request) -> tuple[str, str]:
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        raise RateLimitError("Authentication required")

    token = auth_header[7:]
    payload = await verify_access_token(token)
    user_id = payload.get("sub")
    plan = payload.get("plan", "free")

    return user_id, plan


@router.get("/user", response_model=UserResponse)
async def get_user(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        effective_tier = get_effective_plan_tier(user)
        return UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            plan_tier=effective_tier.value,
            email_verified=user.email_verified,
            credits=user.credits,
        )


@router.get("/user/usage", response_model=UsageResponse)
async def get_usage(
    request: Request,
    period: str = "daily",
):
    user_id, _ = await get_authenticated_user(request)

    redis = await get_redis()
    tracker = UsageTracker(redis)

    if period == "daily":
        usage = await tracker.get_daily_usage(user_id)
    elif period == "hourly":
        usage = await tracker.get_hourly_usage(user_id)
    elif period == "monthly":
        usage = await tracker.get_monthly_usage(user_id)
    else:
        usage = await tracker.get_daily_usage(user_id)

    return UsageResponse(
        total_requests=usage.get("total_requests", 0),
        total_input_tokens=usage.get("total_input_tokens", 0),
        total_output_tokens=usage.get("total_output_tokens", 0),
        total_cost=usage.get("total_cost", 0.0),
        models=usage.get("models", {}),
    )


@router.post("/user/keys")
async def create_api_key(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    body = await request.json()
    name = body.get("name") if body else None

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        plan_tier = get_effective_plan_tier(user)

        key, key_hash = generate_api_key()
        prefix = key[:8]

        api_key = ApiKey(
            user_id=UUID(user_id),
            key_hash=key_hash,
            key_prefix=prefix,
            name=name,
            plan_tier=plan_tier,
        )

        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

        return {
            "id": str(api_key.id),
            "key": key,
            "key_prefix": prefix,
            "name": name,
            "plan_tier": plan_tier.value,
            "created_at": api_key.created_at.isoformat(),
        }


@router.get("/user/keys")
async def list_api_keys(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    async with get_db_session() as session:
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.user_id == UUID(user_id),
                ApiKey.revoked_at.is_(None),
            )
        )
        keys = result.scalars().all()

        return {
            "data": [
                {
                    "id": str(k.id),
                    "key_prefix": k.key_prefix,
                    "name": k.name,
                    "plan_tier": k.plan_tier.value,
                    "created_at": k.created_at.isoformat(),
                    "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                }
                for k in keys
            ]
        }


@router.delete("/user/keys/{key_id}")
async def revoke_api_key(
    request: Request,
    key_id: str,
):
    user_id, _ = await get_authenticated_user(request)

    async with get_db_session() as session:
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.id == UUID(key_id),
                ApiKey.user_id == UUID(user_id),
            )
        )
        key = result.scalar_one_or_none()

        if not key:
            raise NotFoundError("API Key", key_id)

        from datetime import datetime

        key.revoked_at = datetime.now(UTC)

        return {"message": "API key revoked"}


@router.get("/user/credits", response_model=CreditsResponse)
async def get_credits(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    from apps.api.core.config import get_provider_config

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        provider_config = get_provider_config()
        effective_tier = get_effective_plan_tier(user)
        plan_config = provider_config.get_plan_config(effective_tier.value)
        credits_monthly = plan_config.get("credits_monthly", 0.0) if plan_config else 0.0

        redis = await get_redis()
        credit_manager = CreditManager(redis)
        credits_used = await credit_manager.get_monthly_credits_used(user_id)

        return CreditsResponse(
            credits=user.credits,
            credits_monthly=credits_monthly,
            credits_used=credits_used,
            plan_tier=effective_tier.value,
            payg_enabled=False,
        )


@router.get("/user/requests", response_model=RequestsResponse)
async def get_requests(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    from apps.api.core.config import get_provider_config
    from apps.api.services.usage.request_manager import RequestManager

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        provider_config = get_provider_config()
        effective_tier = get_effective_plan_tier(user)
        plan_config = provider_config.get_plan_config(effective_tier.value)
        requests_limit = plan_config.get("requests_per_day", 50) if plan_config else 50

        redis = await get_redis()
        request_manager = RequestManager(redis)
        requests_used = await request_manager.get_daily_request_count(user_id)

        return RequestsResponse(
            requests_used_today=requests_used,
            requests_limit_today=requests_limit,
            requests_remaining=max(0, requests_limit - requests_used),
            plan_tier=effective_tier.value,
        )


CREDITS_PACKAGES = {
    "5": 5.00,
    "10": 9.50,
    "25": 22.00,
    "50": 40.00,
}


@router.post("/user/credits/add")
async def add_credits(
    request: Request,
    body: AddCreditsRequest,
):
    user_id, _ = await get_authenticated_user(request)

    if body.amount <= 0:
        raise RateLimitError("Amount must be positive")

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        if user.plan_tier != PlanTier.PAYG:
            raise RateLimitError("Only PAYG users can add credits")

        user.credits += body.amount
        await session.commit()

        return {
            "credits": user.credits,
            "added": body.amount,
            "transaction_id": body.transaction_id,
        }


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/user/password")
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
):
    from apps.api.core.security import verify_password

    user_id, _ = await get_authenticated_user(request)

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        if not user.password_hash:
            raise RateLimitError("Password not set for this account")

        if not verify_password(body.current_password, user.password_hash):
            raise RateLimitError("Current password is incorrect")

        if len(body.new_password) < 8:
            raise RateLimitError("Password must be at least 8 characters")

        user.password_hash = hash_password(body.new_password)
        await session.commit()

        return {"message": "Password changed successfully"}


@router.post("/user/keys/revoke-all")
async def revoke_all_api_keys(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    async with get_db_session() as session:
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.user_id == UUID(user_id),
                ApiKey.revoked_at.is_(None),
            )
        )
        keys = result.scalars().all()

        from datetime import datetime

        for key in keys:
            key.revoked_at = datetime.now(UTC)

        await session.commit()

        return {"message": f"Revoked {len(keys)} API key(s)"}


@router.delete("/user")
async def delete_account(
    request: Request,
):
    user_id, _ = await get_authenticated_user(request)

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            raise NotFoundError("User", user_id)

        user.is_active = False
        await session.commit()

        return {"message": "Account deleted successfully"}
