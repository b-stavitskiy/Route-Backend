from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, field_validator
from apps.api.core.config import get_settings
from apps.api.core.security import (
    blacklist_token,
    blacklist_refresh_token,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_state_token,
    is_refresh_token_used,
    store_oauth_state,
    verify_oauth_state,
    verify_refresh_token,
)
from apps.api.services.auth_service import AuthService
from packages.db.session import get_db, get_db_session
from packages.shared.exceptions import AuthenticationError, DuplicateResourceError

router = APIRouter(prefix="/auth", tags=["auth"])

ALLOWED_OAUTH_PROVIDERS = {"github"}


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    plan_tier: str
    email_verified: bool

    class Config:
        from_attributes = True


@router.post("/signup", response_model=LoginResponse)
async def signup(
    request: SignupRequest,
    db=Depends(get_db),
):
    async with get_db_session() as session:
        auth_service = AuthService(session)

        try:
            user = await auth_service.create_user(
                email=request.email,
                password=request.password,
                name=request.name,
            )
        except DuplicateResourceError:
            raise AuthenticationError("Email already registered")

        access_token = create_access_token(
            subject=str(user.id),
            additional_claims={"plan": user.plan_tier.value},
        )
        refresh_token = create_refresh_token(subject=str(user.id))

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                name=user.name,
                plan_tier=user.plan_tier.value,
                email_verified=user.email_verified,
            ),
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db=Depends(get_db),
):
    async with get_db_session() as session:
        auth_service = AuthService(session)

        user = await auth_service.authenticate_user(
            email=request.email,
            password=request.password,
        )

        access_token = create_access_token(
            subject=str(user.id),
            additional_claims={"plan": user.plan_tier.value},
        )
        refresh_token = create_refresh_token(subject=str(user.id))

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=UserResponse(
                id=str(user.id),
                email=user.email,
                name=user.name,
                plan_tier=user.plan_tier.value,
                email_verified=user.email_verified,
            ),
        )


@router.get("/oauth/{provider}")
async def oauth_redirect(
    provider: str,
):
    import logging

    logger = logging.getLogger("routing.run.api")

    if provider not in ALLOWED_OAUTH_PROVIDERS:
        raise AuthenticationError(f"Unknown OAuth provider: {provider}")

    settings = get_settings()
    state = generate_state_token()
    await store_oauth_state(state)
    logger.info(f"OAuth redirect generated | state={state[:20]}... | component=oauth")

    if provider == "github":
        github_auth_url = (
            f"https://github.com/login/oauth/authorize"
            f"?client_id={settings.github_client_id}"
            f"&redirect_uri={settings.oauth_redirect_uri}"
            f"&scope=read:user user:email"
            f"&state={state}"
        )
        return RedirectResponse(url=github_auth_url)

    raise AuthenticationError(f"Unknown OAuth provider: {provider}")


@router.get("/callback/{provider}", response_model=TokenResponse)
async def oauth_callback(
    provider: str,
    code: str = Query(...),
    state: str = Query(...),
    db=Depends(get_db),
):
    import logging

    logger = logging.getLogger("routing.run.api")
    logger.info(
        f"OAuth callback received | provider={provider} | state={state[:20]}... | component=oauth"
    )

    if provider not in ALLOWED_OAUTH_PROVIDERS:
        raise AuthenticationError(f"Unknown OAuth provider: {provider}")

    stored_state = await verify_oauth_state(state)
    logger.info(f"OAuth state verification | found={stored_state is not None} | component=oauth")
    if stored_state is None:
        raise AuthenticationError("Invalid or expired OAuth state")

    async with get_db_session() as session:
        auth_service = AuthService(session)

        if provider == "github":
            oauth_data = await auth_service.verify_github_oauth(code)
        else:
            raise AuthenticationError(f"Unknown OAuth provider: {provider}")

        user = await auth_service.create_oauth_user(
            provider=oauth_data["provider"],
            provider_user_id=oauth_data["provider_user_id"],
            email=oauth_data["email"],
            name=oauth_data.get("name"),
            avatar_url=oauth_data.get("avatar_url"),
            access_token=oauth_data.get("access_token"),
            refresh_token=oauth_data.get("refresh_token"),
        )

        access_token = create_access_token(
            subject=str(user.id),
            additional_claims={"plan": user.plan_tier.value},
        )
        refresh_token = create_refresh_token(subject=str(user.id))

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
):
    payload = verify_refresh_token(request.refresh_token)
    user_id = payload.get("sub")
    jti = payload.get("jti")

    if not user_id:
        raise AuthenticationError("Invalid refresh token")

    if jti and await is_refresh_token_used(jti):
        await blacklist_refresh_token(jti, user_id)
        raise AuthenticationError("Refresh token reuse detected")

    if jti:
        await blacklist_refresh_token(jti, user_id)

    access_token = create_access_token(subject=user_id)
    new_refresh_token = create_refresh_token(subject=user_id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


@router.post("/logout")
async def logout(
    request: Request,
    authorization: str = Header(None),
):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        try:
            payload = decode_token(token)
            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti and exp:
                exp_timestamp = datetime.fromtimestamp(exp)
                ttl = int((exp_timestamp - datetime.now(UTC)).total_seconds())
                if ttl > 0:
                    await blacklist_token(jti, ttl)
        except Exception:
            pass

    return {"message": "Logged out successfully"}


from datetime import UTC


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    request: Request,
    db=Depends(get_db),
):
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        raise AuthenticationError("Not authenticated")

    async with get_db_session() as session:
        auth_service = AuthService(session)
        user = await auth_service.get_user_by_id(user_id)

        if not user:
            raise AuthenticationError("User not found")

        return UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            plan_tier=user.plan_tier.value,
            email_verified=user.email_verified,
        )
