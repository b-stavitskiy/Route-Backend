"""add custom plan name columns

Revision ID: 007_add_custom_plan_names
Revises: 006_add_custom_plan_keys
Create Date: 2026-04-22

"""

from alembic import op
import sqlalchemy as sa

revision = "007_add_custom_plan_names"
down_revision = "006_add_custom_plan_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("custom_plan_name", sa.String(length=100), nullable=True))
    op.add_column(
        "users", sa.Column("upgraded_custom_plan_name", sa.String(length=100), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("users", "upgraded_custom_plan_name")
    op.drop_column("users", "custom_plan_name")
