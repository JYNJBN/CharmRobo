"""agent: swap user_id for device_id

agent 归属从 user 改为 device（权限通过 user_device 绑定推导），
因此删除冗余的 user_id 列并补上 device_id 列。

Revision ID: 8f3d2c9a41b7
Revises: dea8560323ea
Create Date: 2026-09-07 10:15:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '8f3d2c9a41b7'
down_revision: Union[str, Sequence[str], None] = 'dea8560323ea'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """1) 加 device_id（可空 + 索引）；2) 删 user_id 及其索引。"""
    op.add_column('agent', sa.Column('device_id', sa.BigInteger(), nullable=True,
                                     comment='所属设备 ID；系统模板（is_system=True）为 NULL'))
    op.create_index(op.f('ix_agent_device_id'), 'agent', ['device_id'], unique=False)
    op.drop_index(op.f('ix_agent_user_id'), table_name='agent')
    op.drop_column('agent', 'user_id')


def downgrade() -> None:
    """还原：补回 user_id，撤掉 device_id。"""
    op.add_column('agent', sa.Column('user_id', sa.BigInteger(), nullable=True,
                                     comment='所属用户 ID；系统模板（is_system=True）为 NULL'))
    op.create_index(op.f('ix_agent_user_id'), 'agent', ['user_id'], unique=False)
    op.drop_index(op.f('ix_agent_device_id'), table_name='agent')
    op.drop_column('agent', 'device_id')
