from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, field_validator

from apps.api.core.config import get_settings

from apps.api.core.security import (
    blacklist_refresh_token,
    blacklist_token,
    create_access_token,
    create_refresh_token,
    generate_csrf_token,
    generate_state_token,
    hash_api_key,
    is_refresh_token_used,
    store_csrf_token,
    store_oauth_state,
    verify_oauth_state,
    verify_refresh_token,
)
from apps.api.services.auth_service import AuthService
from apps.api.services.email.service import get_email_service
from packages.db.models import Session
from packages.db.session import get_db, get_db_session
from packages.shared.exceptions import AuthenticationError, DuplicateResourceError


async def create_session(
    user_id: UUID,
    refresh_token: str,
    user_agent: str | None = None,
    ip_address: str | None = None,
    db_session=None,
) -> Session:
    settings = get_settings()
    refresh_token_hash = hash_api_key(refresh_token)
    expires_at = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)

    session = Session(
        id=uuid4(),
        user_id=user_id,
        refresh_token_hash=refresh_token_hash,
        user_agent=user_agent,
        ip_address=ip_address,
        expires_at=expires_at,
    )

    if db_session:
        db_session.add(session)
        await db_session.commit()
    else:
        async with get_db_session() as new_session:
            new_session.add(session)
            await new_session.commit()

    return session


router = APIRouter(prefix="/auth", tags=["auth"])

ALLOWED_OAUTH_PROVIDERS = {"github"}

ACCESS_TOKEN_COOKIE = "access_token"
REFRESH_TOKEN_COOKIE = "refresh_token"
CSRF_TOKEN_COOKIE = "csrf_token"
COOKIE_DOMAIN = ".routing.run"
COOKIE_SECURE = True
COOKIE_SAMESITE = "lax"


def get_cookie_settings(token_name: str, max_age: int) -> dict:
    return {
        "key": token_name,
        "domain": COOKIE_DOMAIN,
        "secure": COOKIE_SECURE,
        "httponly": True,
        "samesite": COOKIE_SAMESITE,
        "max_age": max_age,
        "path": "/",
    }


def create_auth_cookies(access_token: str, refresh_token: str) -> dict:
    settings = get_settings()
    access_max_age = settings.access_token_expire_minutes * 60
    refresh_max_age = settings.refresh_token_expire_days * 24 * 60 * 60

    access_cookie = get_cookie_settings(ACCESS_TOKEN_COOKIE, access_max_age)
    access_cookie["httponly"] = False
    refresh_cookie = get_cookie_settings(REFRESH_TOKEN_COOKIE, refresh_max_age)

    access_cookie["value"] = access_token
    refresh_cookie["value"] = refresh_token

    return {
        ACCESS_TOKEN_COOKIE: access_cookie,
        REFRESH_TOKEN_COOKIE: refresh_cookie,
    }


def clear_auth_cookies() -> dict:
    access_cookie = get_cookie_settings(ACCESS_TOKEN_COOKIE, 0)
    refresh_cookie = get_cookie_settings(REFRESH_TOKEN_COOKIE, 0)
    access_cookie["value"] = ""
    refresh_cookie["value"] = ""

    return {
        ACCESS_TOKEN_COOKIE: access_cookie,
        REFRESH_TOKEN_COOKIE: refresh_cookie,
    }


class SignupInitRequest(BaseModel):
    email: EmailStr
    turnstile_token: str | None = None


class SignupVerifyRequest(BaseModel):
    email: EmailStr
    otp: str
    password: str
    name: str | None = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginInitRequest(BaseModel):
    email: EmailStr
    password: str
    turnstile_token: str | None = None


class LoginVerifyRequest(BaseModel):
    email: EmailStr
    otp: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    csrf_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str | None = None


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    csrf_token: str
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


class MessageResponse(BaseModel):
    message: str


@router.post("/signup/init", response_model=MessageResponse)
async def signup_init(
    request: SignupInitRequest,
    fastapi_request: Request,
    db=Depends(get_db),
):
    raise AuthenticationError("Signup is disabled. Please use GitHub OAuth.")


@router.post("/signup/verify")
async def signup_verify(
    request: SignupVerifyRequest,
    response: Response,
    db=Depends(get_db),
):
    raise AuthenticationError("Signup is disabled. Please use GitHub OAuth.")


@router.post("/login/init", response_model=MessageResponse)
async def login_init(
    request: LoginInitRequest,
    fastapi_request: Request,
    db=Depends(get_db),
):
    raise AuthenticationError("Login with email/password is disabled. Please use GitHub OAuth.")


@router.post("/login/verify")
async def login_verify(
    request: LoginVerifyRequest,
    response: Response,
    db=Depends(get_db),
):
    raise AuthenticationError("Login with email/password is disabled. Please use GitHub OAuth.")


@router.post("/signup")
async def signup(
    request: SignupVerifyRequest,
    response: Response,
    db=Depends(get_db),
):
    raise AuthenticationError("Signup is disabled. Please use GitHub OAuth.")


@router.post("/login")
async def login(
    request: LoginRequest,
    response: Response,
    db=Depends(get_db),
):
    async with get_db_session() as session:
        auth_service = AuthService(session)
        user = await auth_service.authenticate_user(request.email, request.password)
        if not user:
            raise AuthenticationError("Invalid email or password")

        access_token = create_access_token(
            subject=str(user.id),
            additional_claims={"plan": user.plan_tier.value},
        )
        refresh_token = create_refresh_token(subject=str(user.id))
        await create_session(user.id, refresh_token, db_session=session)

        csrf_token = generate_csrf_token()
        try:
            await store_csrf_token(csrf_token, str(user.id))
        except Exception:
            pass

        cookies = create_auth_cookies(access_token, refresh_token)
        for cookie_name, cookie_params in cookies.items():
            response.set_cookie(**cookie_params)
        response.set_cookie(
            key=CSRF_TOKEN_COOKIE,
            value=csrf_token,
            domain=COOKIE_DOMAIN,
            secure=COOKIE_SECURE,
            httponly=False,
            samesite=COOKIE_SAMESITE,
            max_age=3600,
            path="/",
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "csrf_token": csrf_token,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "plan_tier": user.plan_tier.value,
                "email_verified": user.email_verified,
            },
        }


@router.post("/refresh")
async def refresh_token(
    request: Request,
    response: Response,
    body: RefreshRequest = None,
):
    refresh_token = request.cookies.get(REFRESH_TOKEN_COOKIE)
    if not refresh_token and body and body.refresh_token:
        refresh_token = body.refresh_token
    if not refresh_token:
        raise AuthenticationError("Refresh token missing")

    payload = verify_refresh_token(refresh_token)
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
    await create_session(UUID(user_id), new_refresh_token)

    csrf_token = generate_csrf_token()
    try:
        await store_csrf_token(csrf_token, user_id)
    except Exception:
        pass

    cookies = create_auth_cookies(access_token, new_refresh_token)
    cookies[CSRF_TOKEN_COOKIE] = {
        "key": CSRF_TOKEN_COOKIE,
        "value": csrf_token,
        "domain": COOKIE_DOMAIN,
        "secure": COOKIE_SECURE,
        "httponly": False,
        "samesite": COOKIE_SAMESITE,
        "max_age": 3600,
        "path": "/",
    }

    for cookie_name, cookie_data in cookies.items():
        response.set_cookie(**cookie_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        csrf_token=csrf_token,
    )


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    authorization: str = Header(None),
):
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        try:
            from apps.api.core.security import verify_access_token

            payload = await verify_access_token(token)
            jti = payload.get("jti")
            exp = payload.get("exp")
            if jti and exp:
                import time

                remaining = max(int(exp - time.time()), 1)
                await blacklist_token(jti, remaining)
        except Exception:
            pass

    cookies = clear_auth_cookies()
    cookies[CSRF_TOKEN_COOKIE] = {
        "key": CSRF_TOKEN_COOKIE,
        "value": "",
        "domain": COOKIE_DOMAIN,
        "secure": COOKIE_SECURE,
        "httponly": False,
        "samesite": COOKIE_SAMESITE,
        "max_age": 0,
        "path": "/",
    }

    for cookie_name, cookie_data in cookies.items():
        response.set_cookie(**cookie_data)

    return {"message": "Logged out"}


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


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: str,
    response: Response,
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
        await create_session(user.id, refresh_token, db_session=session)

        csrf_token = generate_csrf_token()
        try:
            await store_csrf_token(csrf_token, str(user.id))
        except Exception:
            pass

        cookies = create_auth_cookies(access_token, refresh_token)
        for cookie_name, cookie_params in cookies.items():
            response.set_cookie(**cookie_params)
        response.set_cookie(
            key=CSRF_TOKEN_COOKIE,
            value=csrf_token,
            domain=COOKIE_DOMAIN,
            secure=COOKIE_SECURE,
            httponly=False,
            samesite=COOKIE_SAMESITE,
            max_age=3600,
            path="/",
        )
        response.set_cookie(
            key="oauth_complete",
            value="true",
            domain=COOKIE_DOMAIN,
            secure=COOKIE_SECURE,
            httponly=False,
            samesite=COOKIE_SAMESITE,
            max_age=60,
            path="/",
        )

        response.headers["Location"] = "https://app.routing.run/dashboard"
        response.status_code = 302
        return response


@router.get("/me", response_model=UserResponse)
async def get_me(
    request: Request,
):
    from apps.api.core.security import verify_access_token

    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = request.cookies.get(ACCESS_TOKEN_COOKIE)

    if not token:
        raise AuthenticationError("Not authenticated")

    payload = await verify_access_token(token)
    user_id = payload.get("sub")

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
