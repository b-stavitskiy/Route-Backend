from dataclasses import dataclass

from fastapi import Header, Request
from sqlalchemy import select

from apps.api.core.config import get_settings
from apps.api.core.security import verify_access_token
from packages.db.models import User
from packages.db.session import get_db_session
from packages.shared.exceptions import AuthenticationError, AuthorizationError


@dataclass
class AdminContext:
    user: User | None
    is_service: bool = False

    @property
    def admin_user_id(self):
        return self.user.id if self.user else None


async def get_current_admin(
    request: Request,
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-Api-Key"),
) -> AdminContext:
    settings = get_settings()

    if settings.admin_api_key and x_admin_api_key == settings.admin_api_key:
        return AdminContext(user=None, is_service=True)

    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif request.cookies.get("access_token"):
        token = request.cookies.get("access_token")

    if not token:
        raise AuthenticationError("Admin authentication required")

    payload = await verify_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid admin token")
    if user_id == "admin-service" and payload.get("scope") == "admin":
        return AdminContext(user=None, is_service=True)

    async with get_db_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise AuthenticationError("Admin user not found or inactive")
    if not user.is_superuser:
        raise AuthorizationError("Superuser access required")

    return AdminContext(user=user)
