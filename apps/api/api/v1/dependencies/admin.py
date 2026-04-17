from dataclasses import dataclass

from fastapi import Request

from apps.api.core.security import verify_access_token
from packages.shared.exceptions import AuthenticationError


@dataclass
class AdminContext:
    user: None = None
    is_service: bool = False

    @property
    def admin_user_id(self):
        return None


async def get_current_admin(
    request: Request,
) -> AdminContext:
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
    if (
        user_id == "admin-service"
        and payload.get("scope") == "admin"
        and payload.get("auth_mode") == "admin_api_key"
    ):
        return AdminContext(user=None, is_service=True)
    raise AuthenticationError("Only admin API key sessions can access admin routes")
