from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def build_custom_plan_key(model_catalog_tier: Any, requests_per_day: Any) -> str | None:
    tier = _enum_value(model_catalog_tier)
    if not tier or requests_per_day in (None, ""):
        return None
    return f"custom:{tier.lower()}:{int(requests_per_day)}"


def get_plan_display_name(plan_key: str | None) -> str:
    if not plan_key:
        return "free"
    return "custom" if plan_key.startswith("custom:") else plan_key


def get_user_base_plan_name(user: Any) -> str:
    custom_plan = build_custom_plan_key(
        getattr(user, "custom_model_catalog_tier", None),
        getattr(user, "custom_requests_per_day", None),
    )
    if custom_plan:
        return custom_plan
    return _enum_value(getattr(user, "plan_tier", None)) or "free"


def get_user_upgrade_plan_name(user: Any) -> str | None:
    upgraded_custom_plan = build_custom_plan_key(
        getattr(user, "upgraded_custom_model_catalog_tier", None),
        getattr(user, "upgraded_custom_requests_per_day", None),
    )
    if upgraded_custom_plan:
        return upgraded_custom_plan
    return _enum_value(getattr(user, "upgraded_to_tier", None))


def get_user_effective_plan_name(user: Any, now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(UTC)

    upgraded_until = getattr(user, "upgraded_until", None)
    upgraded_plan = get_user_upgrade_plan_name(user)
    if upgraded_plan and upgraded_until and upgraded_until > now:
        return upgraded_plan

    return get_user_base_plan_name(user)
