"""materialize default agents

Revision ID: 21d10e4b1fc1
Revises: 8f3d2c9a41b7
Create Date: 2026-09-08 15:00:54.298994

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '21d10e4b1fc1'
down_revision: Union[str, Sequence[str], None] = '8f3d2c9a41b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """老设备物化默认副本并回填 agent_id（实施手册 13.3）。"""
    # 1. 兜底确保系统默认模板存在（seed 迁移已插入，幂等）。
    op.execute(sa.text(
        """
        INSERT INTO agent
        (device_id, name, system_prompt, model_key, voice,
         is_system, source_agent_id, deleted)
        SELECT NULL,
               '小梦',
               '你是小梦机器人，由深圳市创梦龙公司开发。'
                   '回答尽可能简短，适合语音播报。',
               'doubao',
               'female_gentle',
               TRUE,
               NULL,
               0 WHERE NOT EXISTS (
            SELECT 1
            FROM agent
            WHERE is_system = TRUE
              AND device_id IS NULL
              AND deleted = 0
        )
        """
    ))

    # 2. 给老设备创建默认副本，同时把 device.model_key 迁移进副本
    #    （手册 3-Step2/Step3：旧值不在白名单时回落模板的 doubao）。
    op.execute(sa.text(
        """
        INSERT INTO agent
        (device_id, name, system_prompt, model_key, voice,
         is_system, source_agent_id, deleted)
        SELECT d.id,
               t.name,
               t.system_prompt,
               CASE
                   WHEN d.model_key IN ('doubao', 'deepseek', 'qwen')
                       THEN d.model_key
                   ELSE t.model_key
                   END,
               t.voice,
               FALSE,
               t.id,
               0
        FROM device d
                 CROSS JOIN (SELECT id, name, system_prompt, model_key, voice
                             FROM agent
                             WHERE is_system = TRUE
                               AND device_id IS NULL
                               AND deleted = 0
                             ORDER BY id LIMIT 1) t
        WHERE d.deleted = 0
          AND NOT EXISTS (SELECT 1
                          FROM agent a
                          WHERE a.device_id = d.id
                            AND a.source_agent_id = t.id
                            AND a.deleted = 0)
        """
    ))

    # 3. 激活默认副本。
    op.execute(sa.text(
        """
        UPDATE device d
        SET active_agent_id = a.id FROM agent a
        WHERE d.id = a.device_id
          AND d.deleted = 0
          AND d.active_agent_id IS NULL
          AND a.is_system = FALSE
          AND a.deleted = 0
        """
    ))

    # 4. 回填旧会话。
    op.execute(sa.text(
        """
        UPDATE conversation c
        SET agent_id = d.active_agent_id FROM device d
        WHERE c.device_id = d.id
          AND c.agent_id IS NULL
          AND d.active_agent_id IS NOT NULL
        """
    ))

    # 5. 回填旧摘要。
    op.execute(sa.text(
        """
        UPDATE conversation_summary s
        SET agent_id = c.agent_id FROM conversation c
        WHERE s.conversation_id = c.id
          AND s.agent_id IS NULL
          AND c.agent_id IS NOT NULL
        """
    ))

    # 6. 建索引：agent_id 检索 + 防并发重复 fork 的部分唯一索引。
    op.create_index(
        "ix_conversation_agent_id",
        "conversation",
        ["agent_id"],
    )
    op.create_index(
        "ix_conversation_summary_agent_id",
        "conversation_summary",
        ["agent_id"],
    )
    op.create_index(
        "uq_agent_device_source_active",
        "agent",
        ["device_id", "source_agent_id"],
        unique=True,
        postgresql_where=sa.text(
            "source_agent_id IS NOT NULL AND deleted = 0"
        ),
    )


def downgrade() -> None:
    """只回滚索引；副本和回填数据不撤销。

    迁移物化的副本和用户后来手动创建的副本无法区分，
    删除会破坏用户数据，因此 downgrade 保留数据行。
    """
    op.drop_index("uq_agent_device_source_active", table_name="agent")
    op.drop_index(
        "ix_conversation_summary_agent_id",
        table_name="conversation_summary",
    )
    op.drop_index("ix_conversation_agent_id", table_name="conversation")
