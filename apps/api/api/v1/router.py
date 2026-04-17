from fastapi import APIRouter
from apps.api.api.v1.endpoints import (
    admin,
    anthropic,
    auth,
    chat,
    images,
    models,
    pricing,
    status,
    user,
    webhooks,
    whop,
)

router = APIRouter()

router.include_router(admin.router)
router.include_router(anthropic.router)
router.include_router(auth.router)
router.include_router(chat.router)
router.include_router(images.router)
router.include_router(models.router)
router.include_router(pricing.router)
router.include_router(status.router)
router.include_router(user.router)
router.include_router(webhooks.router)
router.include_router(whop.router)
