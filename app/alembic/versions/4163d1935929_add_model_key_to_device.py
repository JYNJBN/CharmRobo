"""add model_key to device

Revision ID: 4163d1935929
Revises: c444727a743e
Create Date: 2026-09-02 17:04:00.000000

注意：此文件由原文件误删后重建（原文件从未提交 git）。
该 revision 在数据库中已应用，重建只为让 alembic 迁移链完整。
以下 upgrade 操作按 device.model_key 的最终形态推断还原，
仅影响 downgrade 场景；如需精确还原，请对照当时的提交/备份。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4163d1935929'
down_revision: Union[str, Sequence[str], None] = 'c444727a743e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """把 model_key 从"可空 String(128)"收紧为"NOT NULL String(64) + 默认 doubao"。"""
    # 1) 已有 NULL 的行先回填默认值，否则 NOT NULL 会失败
    op.execute("UPDATE device SET model_key = 'doubao' WHERE model_key IS NULL")
    # 2) 收紧类型与约束
    op.alter_column(
        'device',
        'model_key',
        existing_type=sa.String(length=128),
        type_=sa.String(length=64),
        existing_nullable=True,
        nullable=False,
        server_default=sa.text("'doubao'"),
    )


def downgrade() -> None:
    """还原为可空 String(128)。"""
    op.alter_column(
        'device',
        'model_key',
        existing_type=sa.String(length=64),
        type_=sa.String(length=128),
        existing_nullable=False,
        nullable=True,
        server_default=None,
    )
