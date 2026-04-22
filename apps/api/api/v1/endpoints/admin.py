from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import case, delete, distinct, func, or_, select

from apps.api.api.v1.dependencies.admin import AdminContext, get_current_admin
from apps.api.core.config import get_provider_config, get_settings
from apps.api.core.plans import (
    get_user_base_plan_display_name,
    get_user_base_plan_name,
    get_user_effective_plan_display_name,
    get_user_effective_plan_name,
    get_user_upgrade_plan_display_name,
    get_user_upgrade_plan_name,
)
from apps.api.core.middleware import get_client_ip
from apps.api.core.security import (
    blacklist_refresh_token,
    blacklist_token,
    create_access_token,
    create_refresh_token,
    hash_api_key,
    hash_password,
    is_refresh_token_used,
    verify_refresh_token,
)
from apps.api.services.auth_service import AuthService
from packages.db.models import AdminAuditLog, ApiKey, PlanTier, Session, UsageLog, User
from packages.db.session import get_db_session
from packages.redis.client import get_redis
from packages.shared.exceptions import (
    AuthenticationError,
    DuplicateResourceError,
    NotFoundError,
    ValidationError,
)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


class AdminUserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    avatar_url: str | None
    plan_tier: str
    custom_plan_name: str | None
    custom_model_catalog_tier: str | None
    custom_requests_per_day: int | None
    effective_plan_tier: str
    upgraded_to_tier: str | None
    upgraded_custom_plan_name: str | None
    upgraded_custom_model_catalog_tier: str | None
    upgraded_custom_requests_per_day: int | None
    upgraded_at: str | None
    upgraded_until: str | None
    credits: float
    email_verified: bool
    is_active: bool
    is_superuser: bool
    created_at: str
    updated_at: str | None
    last_login_at: str | None


class AdminAuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    auth_mode: str = "admin_api_key"
    user: AdminUserResponse | None = None


class PaginationResponse(BaseModel):
    page: int
    page_size: int
    total: int


class AdminUserListResponse(BaseModel):
    items: list[AdminUserResponse]
    pagination: PaginationResponse


class AdminSessionResponse(BaseModel):
    id: str
    ip_address: str | None
    user_agent: str | None
    created_at: str
    expires_at: str


class AdminApiKeyResponse(BaseModel):
    id: str
    key_prefix: str
    name: str | None
    plan_tier: str
    is_active: bool
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


class AdminAuditLogResponse(BaseModel):
    id: str
    action: str
    entity_type: str
    entity_id: str | None
    details: dict | None
    ip_address: str | None
    created_at: str
    admin_user_id: str | None
    target_user_id: str | None


class UsageBucketResponse(BaseModel):
    bucket: str
    requests: int
    success_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    avg_latency_ms: float


class OverviewResponse(BaseModel):
    range_days: int
    totals: dict
    plan_distribution: list[dict]
    signup_series: list[dict]
    top_models: list[dict]


class AdminApiKeyLoginRequest(BaseModel):
    admin_api_key: str


class RefreshRequest(BaseModel):
    refresh_token: str


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str | None = None
    name: str | None = None
    avatar_url: str | None = None
    plan_tier: PlanTier = PlanTier.FREE
    custom_plan_name: str | None = None
    custom_model_catalog_tier: PlanTier | None = None
    custom_requests_per_day: int | None = None
    credits: float = 0.0
    email_verified: bool = True
    is_active: bool = True
    is_superuser: bool = False

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str | None) -> str | None:
        if value is not None and len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        return value


class UpdateUserRequest(BaseModel):
    email: EmailStr | None = None
    password: str | None = None
    name: str | None = None
    avatar_url: str | None = None
    email_verified: bool | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str | None) -> str | None:
        if value is not None and len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        return value


class UpdatePlanRequest(BaseModel):
    plan_tier: PlanTier | None = None
    custom_plan_name: str | None = None
    custom_model_catalog_tier: PlanTier | None = None
    custom_requests_per_day: int | None = None
    reason: str | None = None


class UpgradePlanRequest(BaseModel):
    plan_tier: PlanTier | None = None
    custom_plan_name: str | None = None
    custom_model_catalog_tier: PlanTier | None = None
    custom_requests_per_day: int | None = None
    expires_at: datetime
    reason: str | None = None


class CreditAdjustmentRequest(BaseModel):
    amount: float
    reason: str | None = None


class MessageResponse(BaseModel):
    message: str


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def serialize_user(user: User) -> AdminUserResponse:
    return AdminUserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        plan_tier=get_user_base_plan_display_name(user),
        custom_plan_name=user.custom_plan_name,
        custom_model_catalog_tier=(user.custom_model_catalog_tier.value if user.custom_model_catalog_tier else None),
        custom_requests_per_day=user.custom_requests_per_day,
        effective_plan_tier=get_user_effective_plan_display_name(user),
        upgraded_to_tier=get_user_upgrade_plan_display_name(user),
        upgraded_custom_plan_name=user.upgraded_custom_plan_name,
        upgraded_custom_model_catalog_tier=(
            user.upgraded_custom_model_catalog_tier.value if user.upgraded_custom_model_catalog_tier else None
        ),
        upgraded_custom_requests_per_day=user.upgraded_custom_requests_per_day,
        upgraded_at=iso(user.upgraded_at),
        upgraded_until=iso(user.upgraded_until),
        credits=float(user.credits or 0.0),
        email_verified=user.email_verified,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at.isoformat(),
        updated_at=iso(user.updated_at),
        last_login_at=iso(user.last_login_at),
    )


def make_service_admin_tokens() -> tuple[str, str]:
    subject = "admin-service"
    access_token = create_access_token(
        subject=subject,
        additional_claims={
            "plan": "max",
            "is_superuser": True,
            "scope": "admin",
            "auth_mode": "admin_api_key",
        },
    )
    refresh_token = create_refresh_token(subject=subject)
    return access_token, refresh_token


async def store_admin_refresh_session(refresh_token: str) -> None:
    redis = await get_redis()
    settings = get_settings()
    await redis.setex(
        f"admin:refresh:{hash_api_key(refresh_token)}",
        settings.refresh_token_expire_days * 24 * 60 * 60,
        "1",
    )


async def has_admin_refresh_session(refresh_token: str) -> bool:
    redis = await get_redis()
    return await redis.exists(f"admin:refresh:{hash_api_key(refresh_token)}") == 1


async def revoke_admin_refresh_session(refresh_token: str) -> None:
    redis = await get_redis()
    await redis.delete(f"admin:refresh:{hash_api_key(refresh_token)}")


async def create_session_row(
    session,
    user_id: UUID | None,
    refresh_token: str,
    request: Request | None = None,
):
    session.add(
        Session(
            user_id=user_id,
            refresh_token_hash=hash_api_key(refresh_token),
            user_agent=request.headers.get("user-agent") if request else None,
            ip_address=get_client_ip(request) if request else None,
            expires_at=utcnow() + timedelta(days=get_settings().refresh_token_expire_days),
        )
    )


async def write_audit_log(
    session,
    admin: AdminContext,
    request: Request | None,
    action: str,
    entity_type: str,
    entity_id: str | None,
    target_user_id: UUID | None = None,
    details: dict | None = None,
):
    session.add(
        AdminAuditLog(
            admin_user_id=admin.admin_user_id,
            target_user_id=target_user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            ip_address=get_client_ip(request) if request else None,
        )
    )


async def get_user_or_404(session, user_id: str | UUID) -> User:
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundError("User", str(user_id))
    return user


async def get_cached_payload(cache_key: str):
    redis = await get_redis()
    cached = await redis.get(cache_key)
    return json.loads(cached) if cached else None


async def set_cached_payload(cache_key: str, payload: dict | list, ttl: int = 60):
    redis = await get_redis()
    await redis.set(cache_key, json.dumps(payload, default=str), ex=ttl)


def build_usage_filters(
    start_at: datetime,
    user_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    filters = [UsageLog.created_at >= start_at]
    if user_id:
        filters.append(UsageLog.user_id == UUID(user_id))
    if model:
        filters.append(UsageLog.model == model)
    if provider:
        filters.append(UsageLog.provider == provider)
    return filters


async def query_usage_series(
    days: int,
    granularity: str,
    user_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
):
    start_at = utcnow() - timedelta(days=days)
    bucket = func.date_trunc(granularity, UsageLog.created_at).label("bucket")
    filters = build_usage_filters(start_at, user_id=user_id, model=model, provider=provider)

    async with get_db_session() as session:
        result = await session.execute(
            select(
                bucket,
                func.count(UsageLog.id).label("requests"),
                func.sum(case((UsageLog.status == "success", 1), else_=0)).label(
                    "success_requests"
                ),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("output_tokens"),
                func.coalesce(func.sum(UsageLog.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(UsageLog.cost_usd), 0.0).label("cost_usd"),
                func.coalesce(func.avg(UsageLog.latency_ms), 0.0).label("avg_latency_ms"),
            )
            .where(*filters)
            .group_by(bucket)
            .order_by(bucket.asc())
        )
        rows = result.all()

    return [
        UsageBucketResponse(
            bucket=row.bucket.isoformat(),
            requests=int(row.requests or 0),
            success_requests=int(row.success_requests or 0),
            input_tokens=int(row.input_tokens or 0),
            output_tokens=int(row.output_tokens or 0),
            total_tokens=int(row.total_tokens or 0),
            cost_usd=round(float(row.cost_usd or 0.0), 6),
            avg_latency_ms=round(float(row.avg_latency_ms or 0.0), 2),
        ).model_dump()
        for row in rows
    ]


@router.post("/auth/login", response_model=AdminAuthResponse)
async def admin_login(body: AdminApiKeyLoginRequest, request: Request):
    settings = get_settings()
    if not settings.admin_api_key:
        raise AuthenticationError("ADMIN_API_KEY is not configured")
    if body.admin_api_key != settings.admin_api_key:
        raise AuthenticationError("Invalid admin API key")

    access_token, refresh_token = make_service_admin_tokens()
    await store_admin_refresh_session(refresh_token)

    return AdminAuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=None,
    )


@router.post("/auth/refresh", response_model=AdminAuthResponse)
async def admin_refresh(body: RefreshRequest, request: Request):
    payload = verify_refresh_token(body.refresh_token)
    user_id = payload.get("sub")
    jti = payload.get("jti")

    if not user_id:
        raise AuthenticationError("Invalid refresh token")
    if user_id != "admin-service" or payload.get("auth_mode") != "admin_api_key":
        raise AuthenticationError("Only admin API key refresh sessions are allowed")
    if jti and await is_refresh_token_used(jti):
        raise AuthenticationError("Refresh token reuse detected")
    if not await has_admin_refresh_session(body.refresh_token):
        raise AuthenticationError("Admin refresh session not found")
    if jti:
        await blacklist_refresh_token(jti, user_id)
    await revoke_admin_refresh_session(body.refresh_token)
    access_token, refresh_token = make_service_admin_tokens()
    await store_admin_refresh_session(refresh_token)

    return AdminAuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=None,
    )


@router.post("/auth/logout", response_model=MessageResponse)
async def admin_logout(
    body: RefreshRequest,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        payload = await get_current_admin_token_payload(auth_header[7:])
        jti = payload.get("jti")
        exp = payload.get("exp")
        if jti and exp:
            remaining = max(int(exp - datetime.now().timestamp()), 1)
            await blacklist_token(jti, remaining)

    async with get_db_session() as session:
        await revoke_admin_refresh_session(body.refresh_token)
        await session.execute(
            delete(Session).where(
                Session.refresh_token_hash == hash_api_key(body.refresh_token),
            )
        )
        await write_audit_log(
            session,
            admin,
            request,
            action="admin.logout",
            entity_type="session",
            entity_id=None,
            target_user_id=admin.admin_user_id,
            details=None,
        )
        await session.commit()

    return MessageResponse(message="Logged out")


async def get_current_admin_token_payload(token: str) -> dict:
    from apps.api.core.security import verify_access_token

    return await verify_access_token(token)


@router.get("/auth/me", response_model=AdminUserResponse)
async def admin_me(admin: AdminContext = Depends(get_current_admin)):
    if not admin.user:
        return {
            "id": "admin-service",
            "email": "admin-api-key@local",
            "name": "Admin API Key",
            "avatar_url": None,
            "plan_tier": "max",
            "effective_plan_tier": "max",
            "upgraded_to_tier": None,
            "upgraded_at": None,
            "upgraded_until": None,
            "credits": 0.0,
            "email_verified": True,
            "is_active": True,
            "is_superuser": True,
            "created_at": utcnow().isoformat(),
            "updated_at": None,
            "last_login_at": None,
        }
    return serialize_user(admin.user)


@router.get("/plans")
async def get_available_plans(admin: AdminContext = Depends(get_current_admin)):
    provider_config = get_provider_config()
    plans = provider_config._plans_config.get("plans", {})
    return {
        "items": [
            {
                "id": key,
                "name": value.get("display_name", key.title()),
                "requests_per_day": value.get("requests_per_day"),
                "credits_monthly": value.get("credits_monthly"),
                "allowed_models": value.get("allowed_models"),
            }
            for key, value in plans.items()
        ]
    }


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    search: str | None = None,
    plan_tier: PlanTier | None = None,
    is_active: bool | None = None,
    is_superuser: bool | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin

    filters = []
    if search:
        filters.append(or_(User.email.ilike(f"%{search}%"), User.name.ilike(f"%{search}%")))
    if plan_tier is not None:
        filters.append(User.plan_tier == plan_tier)
    if is_active is not None:
        filters.append(User.is_active == is_active)
    if is_superuser is not None:
        filters.append(User.is_superuser == is_superuser)

    async with get_db_session() as session:
        total_result = await session.execute(select(func.count()).select_from(User).where(*filters))
        total = int(total_result.scalar() or 0)

        result = await session.execute(
            select(User)
            .where(*filters)
            .order_by(User.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        users = result.scalars().all()

    return AdminUserListResponse(
        items=[serialize_user(user) for user in users],
        pagination=PaginationResponse(page=page, page_size=page_size, total=total),
    )


@router.post("/users", response_model=AdminUserResponse)
async def create_user(
    body: CreateUserRequest,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    async with get_db_session() as session:
        auth_service = AuthService(session)
        if (body.custom_model_catalog_tier is None) != (body.custom_requests_per_day is None):
            raise ValidationError(
                "custom_model_catalog_tier and custom_requests_per_day must be provided together"
            )
        user = await auth_service.create_user(
            email=body.email,
            password=body.password,
            name=body.name,
            avatar_url=body.avatar_url,
            email_verified=body.email_verified,
            is_active=body.is_active,
            is_superuser=body.is_superuser,
        )
        user.plan_tier = body.plan_tier
        user.custom_plan_name = body.custom_plan_name
        user.custom_model_catalog_tier = body.custom_model_catalog_tier
        user.custom_requests_per_day = body.custom_requests_per_day
        user.credits = body.credits
        await write_audit_log(
            session,
            admin,
            request,
            action="user.create",
            entity_type="user",
            entity_id=str(user.id),
            target_user_id=user.id,
            details=body.model_dump(mode="json", exclude={"password"}),
        )
        await session.commit()
        await session.refresh(user)
        return serialize_user(user)


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: str,
    admin: AdminContext = Depends(get_current_admin),
):
    del admin

    async with get_db_session() as session:
        user = await get_user_or_404(session, user_id)

        usage_result = await session.execute(
            select(
                func.count(UsageLog.id),
                func.coalesce(func.sum(UsageLog.input_tokens), 0),
                func.coalesce(func.sum(UsageLog.output_tokens), 0),
                func.coalesce(func.sum(UsageLog.total_tokens), 0),
                func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
            ).where(UsageLog.user_id == UUID(user_id))
        )
        usage = usage_result.one()

        api_keys_result = await session.execute(
            select(func.count()).select_from(ApiKey).where(ApiKey.user_id == UUID(user_id))
        )
        sessions_result = await session.execute(
            select(func.count()).select_from(Session).where(Session.user_id == UUID(user_id))
        )

    return {
        "user": serialize_user(user).model_dump(),
        "stats": {
            "total_requests": int(usage[0] or 0),
            "total_input_tokens": int(usage[1] or 0),
            "total_output_tokens": int(usage[2] or 0),
            "total_tokens": int(usage[3] or 0),
            "total_cost_usd": round(float(usage[4] or 0.0), 6),
            "api_keys_count": int(api_keys_result.scalar() or 0),
            "sessions_count": int(sessions_result.scalar() or 0),
        },
    }


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    updates = body.model_dump(exclude_unset=True, exclude_none=True)

    async with get_db_session() as session:
        user = await get_user_or_404(session, user_id)

        if "email" in updates and updates["email"] != user.email:
            existing = await session.execute(select(User).where(User.email == updates["email"]))
            if existing.scalar_one_or_none():
                raise DuplicateResourceError("User", updates["email"])

        if "password" in updates:
            user.password_hash = hash_password(updates.pop("password"))

        for field, value in updates.items():
            setattr(user, field, value)

        await write_audit_log(
            session,
            admin,
            request,
            action="user.update",
            entity_type="user",
            entity_id=str(user.id),
            target_user_id=user.id,
            details=updates,
        )
        await session.commit()
        await session.refresh(user)
        return serialize_user(user)


@router.post("/users/{user_id}/plan", response_model=AdminUserResponse)
async def update_user_plan(
    user_id: str,
    body: UpdatePlanRequest,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    async with get_db_session() as session:
        user = await get_user_or_404(session, user_id)
        using_standard_plan = body.plan_tier is not None
        using_custom_plan = (
            body.custom_model_catalog_tier is not None or body.custom_requests_per_day is not None
        )
        if using_standard_plan == using_custom_plan:
            raise ValidationError(
                "provide either plan_tier or both custom_model_catalog_tier/custom_requests_per_day"
            )
        if (body.custom_model_catalog_tier is None) != (body.custom_requests_per_day is None):
            raise ValidationError(
                "custom_model_catalog_tier and custom_requests_per_day must be provided together"
            )
        previous_plan = get_user_base_plan_name(user)
        if body.plan_tier is not None:
            user.plan_tier = body.plan_tier
            user.custom_plan_name = None
            user.custom_model_catalog_tier = None
            user.custom_requests_per_day = None
        else:
            user.custom_plan_name = body.custom_plan_name
            user.custom_model_catalog_tier = body.custom_model_catalog_tier
            user.custom_requests_per_day = body.custom_requests_per_day
        await write_audit_log(
            session,
            admin,
            request,
            action="user.plan.update",
            entity_type="user",
            entity_id=str(user.id),
            target_user_id=user.id,
            details={
                "previous_plan_tier": previous_plan,
                "new_plan_tier": body.plan_tier.value if body.plan_tier else "custom",
                "custom_plan_name": body.custom_plan_name,
                "custom_model_catalog_tier": (
                    body.custom_model_catalog_tier.value if body.custom_model_catalog_tier else None
                ),
                "custom_requests_per_day": body.custom_requests_per_day,
                "reason": body.reason,
            },
        )
        await session.commit()
        await session.refresh(user)
        return serialize_user(user)


@router.post("/users/{user_id}/upgrade", response_model=AdminUserResponse)
async def upgrade_user_plan(
    user_id: str,
    body: UpgradePlanRequest,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    if body.expires_at <= utcnow():
        raise ValidationError("expires_at must be in the future", field="expires_at")

    async with get_db_session() as session:
        user = await get_user_or_404(session, user_id)
        using_standard_plan = body.plan_tier is not None
        using_custom_plan = (
            body.custom_model_catalog_tier is not None or body.custom_requests_per_day is not None
        )
        if using_standard_plan == using_custom_plan:
            raise ValidationError(
                "provide either plan_tier or both custom_model_catalog_tier/custom_requests_per_day"
            )
        if (body.custom_model_catalog_tier is None) != (body.custom_requests_per_day is None):
            raise ValidationError(
                "custom_model_catalog_tier and custom_requests_per_day must be provided together"
            )
        user.upgraded_to_tier = body.plan_tier
        user.upgraded_custom_plan_name = body.custom_plan_name
        user.upgraded_custom_model_catalog_tier = body.custom_model_catalog_tier
        user.upgraded_custom_requests_per_day = body.custom_requests_per_day
        user.upgraded_at = utcnow()
        user.upgraded_until = body.expires_at.astimezone(UTC)
        await write_audit_log(
            session,
            admin,
            request,
            action="user.plan.upgrade",
            entity_type="user",
            entity_id=str(user.id),
            target_user_id=user.id,
            details={
                "upgraded_to_tier": body.plan_tier.value if body.plan_tier else "custom",
                "custom_plan_name": body.custom_plan_name,
                "custom_model_catalog_tier": (
                    body.custom_model_catalog_tier.value if body.custom_model_catalog_tier else None
                ),
                "custom_requests_per_day": body.custom_requests_per_day,
                "expires_at": body.expires_at.isoformat(),
                "reason": body.reason,
            },
        )
        await session.commit()
        await session.refresh(user)
        return serialize_user(user)


@router.delete("/users/{user_id}/upgrade", response_model=AdminUserResponse)
async def clear_user_upgrade(
    user_id: str,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    async with get_db_session() as session:
        user = await get_user_or_404(session, user_id)
        previous = get_user_upgrade_plan_name(user)
        user.upgraded_to_tier = None
        user.upgraded_custom_plan_name = None
        user.upgraded_custom_model_catalog_tier = None
        user.upgraded_custom_requests_per_day = None
        user.upgraded_at = None
        user.upgraded_until = None
        await write_audit_log(
            session,
            admin,
            request,
            action="user.plan.upgrade.clear",
            entity_type="user",
            entity_id=str(user.id),
            target_user_id=user.id,
            details={"previous_upgrade_tier": previous},
        )
        await session.commit()
        await session.refresh(user)
        return serialize_user(user)


@router.post("/users/{user_id}/credits", response_model=AdminUserResponse)
async def adjust_user_credits(
    user_id: str,
    body: CreditAdjustmentRequest,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    if body.amount == 0:
        raise ValidationError("amount must be non-zero", field="amount")

    async with get_db_session() as session:
        user = await get_user_or_404(session, user_id)
        new_balance = float(user.credits or 0.0) + body.amount
        if new_balance < 0:
            raise ValidationError("credit balance cannot go below zero", field="amount")
        previous_balance = float(user.credits or 0.0)
        user.credits = new_balance
        await write_audit_log(
            session,
            admin,
            request,
            action="user.credits.adjust",
            entity_type="user",
            entity_id=str(user.id),
            target_user_id=user.id,
            details={
                "amount": body.amount,
                "previous_balance": previous_balance,
                "new_balance": new_balance,
                "reason": body.reason,
            },
        )
        await session.commit()
        await session.refresh(user)
        return serialize_user(user)


@router.get("/users/{user_id}/sessions")
async def list_user_sessions(
    user_id: str,
    admin: AdminContext = Depends(get_current_admin),
):
    del admin

    async with get_db_session() as session:
        await get_user_or_404(session, user_id)
        result = await session.execute(
            select(Session)
            .where(Session.user_id == UUID(user_id))
            .order_by(Session.created_at.desc())
        )
        sessions = result.scalars().all()

    return {
        "items": [
            AdminSessionResponse(
                id=str(item.id),
                ip_address=item.ip_address,
                user_agent=item.user_agent,
                created_at=item.created_at.isoformat(),
                expires_at=item.expires_at.isoformat(),
            ).model_dump()
            for item in sessions
        ]
    }


@router.delete("/users/{user_id}/sessions/{session_id}", response_model=MessageResponse)
async def revoke_user_session(
    user_id: str,
    session_id: str,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    async with get_db_session() as session:
        await get_user_or_404(session, user_id)
        result = await session.execute(
            select(Session).where(Session.id == UUID(session_id), Session.user_id == UUID(user_id))
        )
        session_row = result.scalar_one_or_none()
        if not session_row:
            raise NotFoundError("Session", session_id)
        await session.delete(session_row)
        await write_audit_log(
            session,
            admin,
            request,
            action="user.session.revoke",
            entity_type="session",
            entity_id=session_id,
            target_user_id=UUID(user_id),
            details=None,
        )
        await session.commit()

    return MessageResponse(message="Session revoked")


@router.delete("/users/{user_id}/sessions", response_model=MessageResponse)
async def revoke_all_user_sessions(
    user_id: str,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    async with get_db_session() as session:
        await get_user_or_404(session, user_id)
        result = await session.execute(delete(Session).where(Session.user_id == UUID(user_id)))
        await write_audit_log(
            session,
            admin,
            request,
            action="user.session.revoke_all",
            entity_type="session",
            entity_id=user_id,
            target_user_id=UUID(user_id),
            details={"deleted_count": result.rowcount or 0},
        )
        await session.commit()

    return MessageResponse(message="All sessions revoked")


@router.get("/users/{user_id}/api-keys")
async def list_user_api_keys(
    user_id: str,
    admin: AdminContext = Depends(get_current_admin),
):
    del admin

    async with get_db_session() as session:
        await get_user_or_404(session, user_id)
        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.user_id == UUID(user_id))
            .order_by(ApiKey.created_at.desc())
        )
        api_keys = result.scalars().all()

    return {
        "items": [
            AdminApiKeyResponse(
                id=str(item.id),
                key_prefix=item.key_prefix,
                name=item.name,
                plan_tier=item.plan_tier.value,
                is_active=item.is_active,
                created_at=item.created_at.isoformat(),
                last_used_at=iso(item.last_used_at),
                revoked_at=iso(item.revoked_at),
            ).model_dump()
            for item in api_keys
        ]
    }


@router.delete("/users/{user_id}/api-keys/{key_id}", response_model=MessageResponse)
async def revoke_user_api_key(
    user_id: str,
    key_id: str,
    request: Request,
    admin: AdminContext = Depends(get_current_admin),
):
    async with get_db_session() as session:
        await get_user_or_404(session, user_id)
        result = await session.execute(
            select(ApiKey).where(ApiKey.id == UUID(key_id), ApiKey.user_id == UUID(user_id))
        )
        api_key = result.scalar_one_or_none()
        if not api_key:
            raise NotFoundError("API key", key_id)
        api_key.is_active = False
        api_key.revoked_at = utcnow()
        await write_audit_log(
            session,
            admin,
            request,
            action="user.api_key.revoke",
            entity_type="api_key",
            entity_id=key_id,
            target_user_id=UUID(user_id),
            details=None,
        )
        await session.commit()

    return MessageResponse(message="API key revoked")


@router.get("/users/{user_id}/usage")
async def get_user_usage_series(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365),
    granularity: str = Query(default="day", pattern="^(hour|day)$"),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin
    cache_key = f"admin:user-usage:{user_id}:{days}:{granularity}"
    cached = await get_cached_payload(cache_key)
    if cached:
        return cached

    payload = {
        "items": await query_usage_series(days=days, granularity=granularity, user_id=user_id)
    }
    await set_cached_payload(cache_key, payload)
    return payload


@router.get("/audit-logs")
async def list_audit_logs(
    user_id: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin

    filters = []
    if user_id:
        filters.append(AdminAuditLog.target_user_id == UUID(user_id))

    async with get_db_session() as session:
        total_result = await session.execute(
            select(func.count()).select_from(AdminAuditLog).where(*filters)
        )
        total = int(total_result.scalar() or 0)
        result = await session.execute(
            select(AdminAuditLog)
            .where(*filters)
            .order_by(AdminAuditLog.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = result.scalars().all()

    return {
        "items": [
            AdminAuditLogResponse(
                id=str(item.id),
                action=item.action,
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                details=item.details,
                ip_address=item.ip_address,
                created_at=item.created_at.isoformat(),
                admin_user_id=str(item.admin_user_id) if item.admin_user_id else None,
                target_user_id=str(item.target_user_id) if item.target_user_id else None,
            ).model_dump()
            for item in items
        ],
        "pagination": PaginationResponse(page=page, page_size=page_size, total=total).model_dump(),
    }


@router.get("/overview", response_model=OverviewResponse)
async def get_overview(
    days: int = Query(default=30, ge=1, le=365),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin
    cache_key = f"admin:overview:{days}"
    cached = await get_cached_payload(cache_key)
    if cached:
        return cached

    start_at = utcnow() - timedelta(days=days)
    last_24h = utcnow() - timedelta(hours=24)

    async with get_db_session() as session:
        total_users = int((await session.execute(select(func.count()).select_from(User))).scalar() or 0)
        active_users_period = int(
            (
                await session.execute(
                    select(func.count(distinct(UsageLog.user_id))).where(UsageLog.created_at >= start_at)
                )
            ).scalar()
            or 0
        )
        active_users_24h = int(
            (
                await session.execute(
                    select(func.count(distinct(UsageLog.user_id))).where(UsageLog.created_at >= last_24h)
                )
            ).scalar()
            or 0
        )
        active_api_keys = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(ApiKey)
                    .where(ApiKey.is_active.is_(True), ApiKey.revoked_at.is_(None))
                )
            ).scalar()
            or 0
        )
        usage_totals = (
            await session.execute(
                select(
                    func.count(UsageLog.id),
                    func.sum(case((UsageLog.status == "success", 1), else_=0)),
                    func.coalesce(func.sum(UsageLog.input_tokens), 0),
                    func.coalesce(func.sum(UsageLog.output_tokens), 0),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
                    func.coalesce(func.avg(UsageLog.latency_ms), 0.0),
                ).where(UsageLog.created_at >= start_at)
            )
        ).one()

        plan_rows = (
            await session.execute(
                select(User.plan_tier, func.count(User.id)).group_by(User.plan_tier).order_by(User.plan_tier)
            )
        ).all()

        signup_bucket = func.date_trunc("day", User.created_at).label("bucket")
        signup_rows = (
            await session.execute(
                select(signup_bucket, func.count(User.id))
                .where(User.created_at >= start_at)
                .group_by(signup_bucket)
                .order_by(signup_bucket.asc())
            )
        ).all()

        model_rows = (
            await session.execute(
                select(
                    UsageLog.model,
                    func.count(UsageLog.id),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
                )
                .where(UsageLog.created_at >= start_at)
                .group_by(UsageLog.model)
                .order_by(func.coalesce(func.sum(UsageLog.total_tokens), 0).desc())
                .limit(10)
            )
        ).all()

    total_requests = int(usage_totals[0] or 0)
    success_requests = int(usage_totals[1] or 0)
    payload = OverviewResponse(
        range_days=days,
        totals={
            "total_users": total_users,
            "active_users_period": active_users_period,
            "active_users_24h": active_users_24h,
            "active_api_keys": active_api_keys,
            "requests": total_requests,
            "success_requests": success_requests,
            "success_rate": round((success_requests / total_requests) * 100, 2) if total_requests else 0.0,
            "input_tokens": int(usage_totals[2] or 0),
            "output_tokens": int(usage_totals[3] or 0),
            "total_tokens": int(usage_totals[4] or 0),
            "cost_usd": round(float(usage_totals[5] or 0.0), 6),
            "avg_latency_ms": round(float(usage_totals[6] or 0.0), 2),
        },
        plan_distribution=[
            {"plan_tier": row[0].value, "users": int(row[1] or 0)} for row in plan_rows
        ],
        signup_series=[
            {"bucket": row[0].isoformat(), "users": int(row[1] or 0)} for row in signup_rows
        ],
        top_models=[
            {
                "model": row[0],
                "requests": int(row[1] or 0),
                "total_tokens": int(row[2] or 0),
                "cost_usd": round(float(row[3] or 0.0), 6),
            }
            for row in model_rows
        ],
    ).model_dump()

    await set_cached_payload(cache_key, payload)
    return payload


@router.get("/analytics/usage")
async def get_usage_analytics(
    days: int = Query(default=30, ge=1, le=365),
    granularity: str = Query(default="day", pattern="^(hour|day)$"),
    user_id: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    admin: AdminContext = Depends(get_current_admin),
):
    del admin
    cache_key = f"admin:usage:{days}:{granularity}:{user_id or 'all'}:{model or 'all'}:{provider or 'all'}"
    cached = await get_cached_payload(cache_key)
    if cached:
        return cached

    payload = {
        "items": await query_usage_series(
            days=days,
            granularity=granularity,
            user_id=user_id,
            model=model,
            provider=provider,
        )
    }
    await set_cached_payload(cache_key, payload)
    return payload


@router.get("/analytics/models")
async def get_model_analytics(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin
    cache_key = f"admin:model-analytics:{days}:{limit}"
    cached = await get_cached_payload(cache_key)
    if cached:
        return cached

    start_at = utcnow() - timedelta(days=days)
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    UsageLog.model,
                    func.count(UsageLog.id),
                    func.count(distinct(UsageLog.user_id)),
                    func.coalesce(func.sum(UsageLog.input_tokens), 0),
                    func.coalesce(func.sum(UsageLog.output_tokens), 0),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
                    func.coalesce(func.avg(UsageLog.latency_ms), 0.0),
                )
                .where(UsageLog.created_at >= start_at)
                .group_by(UsageLog.model)
                .order_by(func.coalesce(func.sum(UsageLog.total_tokens), 0).desc())
                .limit(limit)
            )
        ).all()

    payload = {
        "items": [
            {
                "model": row[0],
                "requests": int(row[1] or 0),
                "users": int(row[2] or 0),
                "input_tokens": int(row[3] or 0),
                "output_tokens": int(row[4] or 0),
                "total_tokens": int(row[5] or 0),
                "cost_usd": round(float(row[6] or 0.0), 6),
                "avg_latency_ms": round(float(row[7] or 0.0), 2),
            }
            for row in rows
        ]
    }
    await set_cached_payload(cache_key, payload)
    return payload


@router.get("/analytics/providers")
async def get_provider_analytics(
    days: int = Query(default=30, ge=1, le=365),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin
    cache_key = f"admin:provider-analytics:{days}"
    cached = await get_cached_payload(cache_key)
    if cached:
        return cached

    start_at = utcnow() - timedelta(days=days)
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    UsageLog.provider,
                    func.count(UsageLog.id),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
                    func.coalesce(func.avg(UsageLog.latency_ms), 0.0),
                )
                .where(UsageLog.created_at >= start_at)
                .group_by(UsageLog.provider)
                .order_by(func.count(UsageLog.id).desc())
            )
        ).all()

    payload = {
        "items": [
            {
                "provider": row[0],
                "requests": int(row[1] or 0),
                "total_tokens": int(row[2] or 0),
                "cost_usd": round(float(row[3] or 0.0), 6),
                "avg_latency_ms": round(float(row[4] or 0.0), 2),
            }
            for row in rows
        ]
    }
    await set_cached_payload(cache_key, payload)
    return payload


@router.get("/analytics/users")
async def get_top_users(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
    admin: AdminContext = Depends(get_current_admin),
):
    del admin
    cache_key = f"admin:user-analytics:{days}:{limit}"
    cached = await get_cached_payload(cache_key)
    if cached:
        return cached

    start_at = utcnow() - timedelta(days=days)
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    User.id,
                    User.email,
                    User.name,
                    func.count(UsageLog.id),
                    func.coalesce(func.sum(UsageLog.total_tokens), 0),
                    func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
                )
                .join(UsageLog, UsageLog.user_id == User.id)
                .where(UsageLog.created_at >= start_at)
                .group_by(User.id, User.email, User.name)
                .order_by(func.coalesce(func.sum(UsageLog.total_tokens), 0).desc())
                .limit(limit)
            )
        ).all()

    payload = {
        "items": [
            {
                "user_id": str(row[0]),
                "email": row[1],
                "name": row[2],
                "requests": int(row[3] or 0),
                "total_tokens": int(row[4] or 0),
                "cost_usd": round(float(row[5] or 0.0), 6),
            }
            for row in rows
        ]
    }
    await set_cached_payload(cache_key, payload)
    return payload
