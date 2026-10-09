"""make knowledge chunk token count optional

Revision ID: 88418a2c8f39
Revises: 7530292e93c0
Create Date: 2026-09-23 16:02:51.326182

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '88418a2c8f39'
down_revision: Union[str, Sequence[str], None] = '7530292e93c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow the first implementation stage to omit token counts."""
    op.drop_constraint(
        "ck_chunk_numeric",
        "knowledge_chunk",
        type_="check",
    )
    op.alter_column(
        "knowledge_chunk",
        "token_count",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.create_check_constraint(
        "ck_chunk_numeric",
        "knowledge_chunk",
        "(token_count IS NULL OR token_count >= 0) "
        "AND (page_number IS NULL OR page_number > 0) "
        "AND (row_number IS NULL OR row_number > 0)",
    )


def downgrade() -> None:
    """Restore required token counts after NULL rows have been backfilled."""
    op.drop_constraint(
        "ck_chunk_numeric",
        "knowledge_chunk",
        type_="check",
    )
    op.alter_column(
        "knowledge_chunk",
        "token_count",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.create_check_constraint(
        "ck_chunk_numeric",
        "knowledge_chunk",
        "token_count >= 0 "
        "AND (page_number IS NULL OR page_number > 0) "
        "AND (row_number IS NULL OR row_number > 0)",
    )
