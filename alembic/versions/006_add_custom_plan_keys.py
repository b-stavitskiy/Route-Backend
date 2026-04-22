"""add custom plan setting columns

Revision ID: 006_add_custom_plan_keys
Revises: 005_add_admin_audit_logs
Create Date: 2026-04-21

"""

from alembic import op
import sqlalchemy as sa

revision = "006_add_custom_plan_keys"
down_revision = "005_add_admin_audit_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "custom_model_catalog_tier",
            sa.Enum("FREE", "LITE", "PREMIUM", "MAX", "PAYG", name="plantier", create_type=False),
            nullable=True,
        ),
    )
    op.add_column("users", sa.Column("custom_requests_per_day", sa.Integer(), nullable=True))
    op.add_column(
        "users",
        sa.Column(
            "upgraded_custom_model_catalog_tier",
            sa.Enum("FREE", "LITE", "PREMIUM", "MAX", "PAYG", name="plantier", create_type=False),
            nullable=True,
        ),
    )
    op.add_column("users", sa.Column("upgraded_custom_requests_per_day", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "upgraded_custom_requests_per_day")
    op.drop_column("users", "upgraded_custom_model_catalog_tier")
    op.drop_column("users", "custom_requests_per_day")
    op.drop_column("users", "custom_model_catalog_tier")
