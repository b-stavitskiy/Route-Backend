"""add credits column and payg plan

Revision ID: 002_add_credits
Revises: 001_initial
Create Date: 2026-03-30

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002_add_credits"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("credits", sa.Float(), nullable=False, server_default="0.0"))

    op.execute("ALTER TYPE plantier ADD VALUE IF NOT EXISTS 'PAYG'")


def downgrade() -> None:
    op.drop_column("users", "credits")
