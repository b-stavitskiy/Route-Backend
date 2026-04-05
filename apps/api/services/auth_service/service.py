from typing import Any

import httpx
from apps.api.core.config import get_settings
from apps.api.core.security import hash_password, verify_password
from packages.db.models import PlanTier, User
from packages.shared.types import OAuthProvider
from packages.shared.exceptions import (
    AuthenticationError,
    DuplicateResourceError,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.settings = get_settings()

    async def get_user_by_email(self, email: str) -> User | None:
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> User | None:
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_github_id(self, github_id: str) -> User | None:
        result = await self.session.execute(select(User).where(User.github_id == github_id))
        return result.scalar_one_or_none()

    async def get_user_by_whop_id(self, whop_user_id: str) -> User | None:
        result = await self.session.execute(select(User).where(User.whop_user_id == whop_user_id))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password: str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> User:
        existing = await self.get_user_by_email(email)
        if existing:
            raise DuplicateResourceError("User", email)

        password_hash = None
        if password:
            password_hash = hash_password(password)

        user = User(
            email=email,
            password_hash=password_hash,
            name=name,
            plan_tier=PlanTier.FREE,
            credits=5.0,
            **kwargs,
        )

        self.session.add(user)
        await self.session.flush()
        return user

    async def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> User:
        user = await self.get_user_by_email(email)
        if not user:
            raise AuthenticationError("Invalid email or password")

        if not user.password_hash:
            raise AuthenticationError("Please login with OAuth")

        if not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("Account is disabled")

        return user

    async def create_oauth_user(
        self,
        provider: OAuthProvider,
        provider_user_id: str,
        email: str,
        name: str | None = None,
        avatar_url: str | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> User:
        from sqlalchemy import select
        from packages.db.models import User as UserModel

        existing_by_provider = None
        if provider == OAuthProvider.GITHUB:
            result = await self.session.execute(
                select(UserModel).where(UserModel.github_id == provider_user_id)
            )
            existing_by_provider = result.scalar_one_or_none()

        if existing_by_provider:
            if avatar_url:
                existing_by_provider.avatar_url = avatar_url
            if name:
                existing_by_provider.name = name
            await self.session.flush()
            return existing_by_provider

        existing_by_email = await self.get_user_by_email(email)
        if existing_by_email:
            if provider == OAuthProvider.GITHUB and not existing_by_email.github_id:
                existing_by_email.github_id = provider_user_id
                if avatar_url:
                    existing_by_email.avatar_url = avatar_url
                if name:
                    existing_by_email.name = name
                await self.session.flush()
                return existing_by_email
            elif existing_by_email.github_id == provider_user_id:
                return existing_by_email
            raise DuplicateResourceError("User", email)

        user = User(
            email=email,
            name=name,
            avatar_url=avatar_url,
            plan_tier=PlanTier.FREE,
            credits=5.0,
            email_verified=True,
            github_id=provider_user_id if provider == OAuthProvider.GITHUB else None,
        )

        self.session.add(user)
        await self.session.flush()
        return user

    async def verify_github_oauth(self, code: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": self.settings.github_client_id,
                    "client_secret": self.settings.github_client_secret,
                    "code": code,
                },
                headers={"Accept": "application/json"},
            )

            if token_response.status_code != 200:
                raise AuthenticationError("Failed to exchange code for token")

            token_data = token_response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise AuthenticationError("No access token received")

            user_response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if user_response.status_code != 200:
                raise AuthenticationError("Failed to get user info")

            user_data = user_response.json()

            emails_response = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            primary_email = user_data.get("email")
            if emails_response.status_code == 200:
                emails = emails_response.json()
                primary_email = next(
                    (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                    primary_email,
                )

            return {
                "provider": OAuthProvider.GITHUB,
                "provider_user_id": str(user_data["id"]),
                "email": primary_email,
                "name": user_data.get("name") or user_data.get("login"),
                "avatar_url": user_data.get("avatar_url"),
                "access_token": access_token,
            }
