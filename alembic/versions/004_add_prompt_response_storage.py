"""add prompt response storage to usage logs

Revision ID: 004_add_prompt_response_storage
Revises: 003_add_upgrade_tracking
Create Date: 2026-04-11

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004_add_prompt_response_storage"
down_revision: Union[str, None] = "003_add_upgrade_tracking"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "usage_logs",
        sa.Column("prompt", sa.Text(), nullable=False, server_default="''"),
    )
    op.add_column(
        "usage_logs",
        sa.Column("response", sa.Text(), nullable=True),
    )
    op.add_column(
        "usage_logs",
        sa.Column("response_model", sa.String(100), nullable=True),
    )
    op.add_column(
        "usage_logs",
        sa.Column("metadata", sa.Text(), nullable=True),
    )

    op.execute("ALTER TABLE usage_logs ALTER COLUMN prompt DROP DEFAULT")

    op.create_index(
        "ix_usage_logs_model_created",
        "usage_logs",
        ["model", "created_at"],
    )
    op.create_index(
        "ix_usage_logs_provider_created",
        "usage_logs",
        ["provider", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_usage_logs_provider_created", table_name="usage_logs")
    op.drop_index("ix_usage_logs_model_created", table_name="usage_logs")
    op.drop_column("usage_logs", "metadata")
    op.drop_column("usage_logs", "response_model")
    op.drop_column("usage_logs", "response")
    op.drop_column("usage_logs", "prompt")
