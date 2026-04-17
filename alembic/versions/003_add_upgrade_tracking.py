"""add upgrade tracking columns

Revision ID: 003_add_upgrade_tracking
Revises: 002_add_credits
Create Date: 2026-04-08

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "003_add_upgrade_tracking"
down_revision = "002_add_credits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "upgraded_to_tier",
            postgresql.ENUM(
                "FREE", "LITE", "PREMIUM", "MAX", "PAYG", name="plantier", create_type=False
            ),
            nullable=True,
        ),
    )
    op.add_column("users", sa.Column("upgraded_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("upgraded_until", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "upgraded_until")
    op.drop_column("users", "upgraded_at")
    op.drop_column("users", "upgraded_to_tier")
