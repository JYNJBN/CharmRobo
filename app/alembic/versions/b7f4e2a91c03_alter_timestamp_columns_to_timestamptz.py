"""alter timestamp columns to timestamptz

把 17 个 timestamp without time zone 列改为 timestamptz，从类型层面保证时区语义：
PG 只存绝对时刻，session 时区仅影响【显示】、不再影响【存储】。

改动动机：此前 server_default=func.now() 与 onupdate=func.now() 写进 naive 列时，
PG 必须把绝对时刻按 session 时区【渲染】成墙上时间再存 —— 这是一个随 session 时区
漂移的隐患。改成 timestamptz 后类型直接匹配、不再发生任何转换，隐患从根上消除。

Revision ID: b7f4e2a91c03
Revises: 21d10e4b1fc1
Create Date: 2026-09-18

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7f4e2a91c03'
down_revision: str | Sequence[str] | None = '21d10e4b1fc1'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (表名, 列名) —— 与 information_schema.columns 实测结果一一对应，共 17 列
TIMESTAMP_COLUMNS: list[tuple[str, str]] = [
    ('agent', 'create_time'),
    ('agent', 'update_time'),
    ('conversation', 'created_at'),
    ('conversation', 'updated_at'),
    ('conversation', 'last_message_at'),
    ('conversation_message', 'created_at'),
    ('conversation_summary', 'created_at'),
    ('conversation_summary', 'updated_at'),
    ('device', 'create_time'),
    ('device', 'update_time'),
    ('device', 'last_online_time'),
    ('device', 'secret_updated_time'),
    ('user', 'create_time'),
    ('user', 'update_time'),
    ('user_device', 'bind_time'),
    ('user_device', 'unbind_time'),
    ('user_device', 'update_time'),
]


def upgrade() -> None:
    """naive → timestamptz：把现有值【明确按 UTC 解释】。

    `USING ... AT TIME ZONE 'UTC'` 不能省：
    不加 USING 时 PG 会执行 col::timestamptz，即【按 session 时区】解释 naive 值。
    当前 session 恰好是 UTC 才碰巧正确 —— 不能把正确性押在这上面。

    受影响索引：conversation_message.created_at 上有
    ix_conversation_message_created_at，改类型时会连带重建。

    成本实测（本地库 128 行 / 服务器库约 1301 行，全库不足 1MB）：
    整个 upgrade 毫秒级完成，无需担心锁表；本地已实测 upgrade → downgrade →
    upgrade 往返，214 条列值零偏移。注意 PG 的 timestamptz 与 timestamp 二进制
    表示完全相同，但带 USING 时仍会重写表 —— 数据量小，可忽略。
    """
    for table, column in TIMESTAMP_COLUMNS:
        # 表名/列名均为本文件内的硬编码常量，不来自外部输入
        op.execute(
            f'ALTER TABLE "{table}" ALTER COLUMN "{column}" '
            f"TYPE timestamptz USING \"{column}\" AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    """timestamptz → naive：把值【按 UTC 渲染】回墙上时间。

    与 upgrade 严格互逆：
      upgrade   把 naive 值「声明为 UTC」→ 同一时刻的 timestamptz
      downgrade 把 timestamptz「按 UTC 渲染」→ 同一个数字的 naive
    往返后数字不变，不会产生二次偏移。
    """
    for table, column in TIMESTAMP_COLUMNS:
        op.execute(
            f'ALTER TABLE "{table}" ALTER COLUMN "{column}" '
            f"TYPE timestamp USING \"{column}\" AT TIME ZONE 'UTC'"
        )
