"""seed default agent

Revision ID: dea8560323ea
Revises: fac5acb33a67
Create Date: 2026-09-07 10:01:28.285230

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'dea8560323ea'
down_revision: Union[str, Sequence[str], None] = 'fac5acb33a67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Seed 默认系统智能体。"""
    op.execute(sa.text(
        "INSERT INTO agent (user_id, name, system_prompt, model_key, voice, is_system, deleted) "
        "VALUES (NULL, '小梦', "
        "'你是小梦机器人，由深圳市创梦龙公司开发。回答尽可能简短，优先用两到三句话回答，适合语音播报。', "
        "'doubao', 'female_gentle', TRUE, 0)"
    ))


def downgrade() -> None:
    """删除默认系统智能体。"""
    op.execute(sa.text("DELETE FROM agent WHERE is_system = TRUE AND name = '小梦'"))
