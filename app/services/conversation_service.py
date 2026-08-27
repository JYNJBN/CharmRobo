from sqlalchemy import select
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
    """ "按照id查询会话"""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    return result.scalar_one_or_none()


async def add_conversation_message(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
    source: str = "voice",
) -> None:
    """保存一条消息，并更新会话统计信息。"""

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
    db: AsyncSession, conversation_id: int, limit: int = 20
) -> list[dict[str, str]]:
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

    return [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in messages
    ]


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