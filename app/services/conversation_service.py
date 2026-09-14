from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.conversation_message import ConversationMessage
from app.utils.tools import local_now


async def create_conversation(
        db: AsyncSession,
        user_id: int | None = None,
        device_id: int | None = None,
) -> Conversation:
    """创建一个新的对话会话。"""
    conversation = Conversation(
        user_id=user_id,
        device_id=device_id,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_conversation_by_id(
        db: AsyncSession,
        conversation_id: int,
) -> Conversation | None:
    """按照id查询会话"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    return result.scalar_one_or_none()


# 新方案使用智能体
async def get_or_create_conversation(
        db: AsyncSession,
        user_id: int,
        device_id: int,
        agent_id: int,
) -> Conversation:
    """返回当前 Agent 的本地连续会话。

    conversation 按 agent_id 隔离：同一用户切换到另一个 Agent 时会进入
    另一条历史链，这是为了防止不同 Agent 的人设和私有上下文互相污染。
    端到端重连时，路由通过这个 conversation_id 重新取最近消息。
    """

    # 添加一个咨询锁 锁住agentId
    await db.execute(
        text(
            "SELECT pg_advisory_xact_lock("
            "CAST(:agent_id as bigint))"
        ),
        {"agent_id": agent_id}
    )
    # 先查有没有会话 有就直接返回，没有就创建会话返回
    conversation = await db.scalar(
        select(Conversation).where(
            Conversation.agent_id == agent_id
        )
    )
    if conversation is not None:
        return conversation
    conversation = Conversation(
        user_id=user_id,
        device_id=device_id,
        agent_id=agent_id,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def add_conversation_message(
        db: AsyncSession,
        conversation_id: int,
        role: str,
        content: str,
        source: str = "voice",
) -> None:
    """保存一条最终消息，并更新会话统计信息。

    端到端只在一轮结束后调用，保存最终 user/assistant 文本；不保存
    ASR/回复的增量片段，避免一次对话产生多条碎片记录。
    """

    conversation = await get_conversation_by_id(db, conversation_id)
    if conversation is None:
        raise ValueError(f"会话不存在：{conversation_id}")

    message = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        source=source,
    )

    conversation.message_count += 1
    conversation.last_message_at = local_now()

    db.add(message)
    await db.commit()


async def get_recent_messages(
        db: AsyncSession,
        conversation_id: int,
        limit: int = 20,
        include_timestamps: bool = False,
) -> list[dict[str, object]]:
    """按时间顺序返回最近历史，供新端到端 session 初始化。

    数据库查询先按倒序拿最近 N 条，再 reverse 回用户实际对话顺序。
    include_timestamps=True 时额外输出 Unix 毫秒时间戳，适配 Seeduplex
    dialog_context 的 role/text/timestamp 格式；默认仍保持旧调用方只拿
    role/content 的行为。
    """

    result = await db.execute(
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(
            ConversationMessage.created_at.desc(),
            ConversationMessage.id.desc(),
        )
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()

    result_messages: list[dict[str, object]] = []
    for message in messages:
        item: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }
        if include_timestamps:
            # Seeduplex dialog_context 使用 Unix 毫秒时间戳。
            item["timestamp"] = int(message.created_at.timestamp() * 1000)
        result_messages.append(item)
    return result_messages


async def get_latest_conversation_by_device(
        db: AsyncSession,
        device_id: int,
) -> Conversation | None:
    """查询设备最近使用的一条会话。"""

    result = await db.execute(
        select(Conversation)
        .where(Conversation.device_id == device_id)
        .order_by(
            Conversation.last_message_at.desc(),
            Conversation.id.desc(),
        )
        .limit(1)
    )
    return result.scalar_one_or_none()
