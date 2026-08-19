"""add gender to user table

Revision ID: a1b2c3d4e5f6
Revises: ff2e29580cfb
Create Date: 2026-08-13 16:35:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: str | Sequence[str] | None = 'ff2e29580cfb'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """为 user 表添加 gender 列（0 未知，1 男，2 女）。"""
    op.add_column(
        'user',
        sa.Column(
            'gender',
            sa.Integer(),
            server_default=sa.text('0'),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """删除 gender 列。"""
    op.drop_column('user', 'gender')
