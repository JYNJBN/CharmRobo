import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.ai import stream_chat
from app.models import ConversationMessage, ConversationSummary
from app.schemas.Model import ModelRegistryObject

logger = logging.getLogger(__name__)


async def get_latest_summary_end_message_id(
        db: AsyncSession,
        conversation_id: int,
) -> int:
    """查询当前会话已经摘要到哪一条消息。"""
    stmt = (
        select(ConversationSummary.covered_end_message_id)
        .where(
            ConversationSummary.conversation_id == conversation_id,
            ConversationSummary.status == "active",
        )
        .order_by(ConversationSummary.covered_end_message_id.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    last_message_id = result.scalar_one_or_none()
    return last_message_id or 0


async def get_uncovered_messages(
        db: AsyncSession,
        conversation_id: int,
        last_covered_message_id: int = 0,
        limit: int = 20,
) -> list[dict[str, object]]:
    """获取还没有被摘要覆盖的消息。"""
    result = await db.execute(
        select(ConversationMessage)
        .where(
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.id > last_covered_message_id,
        )
        .order_by(
            ConversationMessage.id.asc(),
        )
        .limit(limit)
    )
    messages = list(result.scalars().all())
    return [
        {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at,
        }
        for message in messages
    ]


async def create_conversation_summary(
        db: AsyncSession,
        conversation_id: int,
        device_id: int,
        summary_text: str,
        covered_start_message_id: int,
        covered_end_message_id: int,
        agent_id: int | None = None,
        summary_level: str = "segment",
) -> ConversationSummary:
    """创建一条会话摘要"""
    summary = ConversationSummary(
        conversation_id=conversation_id,
        device_id=device_id,
        summary_text=summary_text,
        agent_id=agent_id,
        covered_start_message_id=covered_start_message_id,
        covered_end_message_id=covered_end_message_id,
        summary_level=summary_level,
        status="active",
    )
    db.add(summary)
    await db.commit()
    await db.refresh(summary)
    return summary


async def get_recent_summaries(
        db: AsyncSession,
        conversation_id: int,
        limit: int = 5,
) -> list[dict[str, object]]:
    """获取最近几条摘要，按时间正序返回。"""
    stmt = (
        select(ConversationSummary)
        .where(
            ConversationSummary.conversation_id == conversation_id,
        )
        .order_by(
            ConversationSummary.id.desc(),
        )
        .limit(limit)
    )

    result = await db.execute(stmt)
    summaries = list(result.scalars().all())
    summaries.reverse()
    return [
        {
            "id": summary.id,
            "summary_text": summary.summary_text,
            "covered_start_message_id": summary.covered_start_message_id,
            "covered_end_message_id": summary.covered_end_message_id,
            "summary_level": summary.summary_level,
            "status": summary.status,
        }
        for summary in summaries
    ]


async def generate_conversation_summary(
        db: AsyncSession,
        conversation_id: int,
        device_id: int | None,
        last_covered_message_id: int,
        limit: int = 20,
        trace_id: str = "summary",
        agent_id: int | None = None,
        model: ModelRegistryObject | None = None,
) -> ConversationSummary | None:
    """生成一批未摘要消息的摘要。"""

    messages = await get_uncovered_messages(
        db=db,
        conversation_id=conversation_id,
        last_covered_message_id=last_covered_message_id,
        limit=limit,
    )

    # 未达到摘要数量时，不生成摘要
    if len(messages) < limit:
        return None

    # 正确拼接对话内容
    lines: list[str] = []

    for message in messages:
        lines.append(f"{message['role']}: {message['content']}")

    raw_text = "\n".join(lines)

    summary_prompt = (
        "请把下面这段对话压缩成一段简洁准确的中文摘要。"
        "保留用户姓名、称呼、偏好、重要事实、设备问题、任务和结论。"
        "不要加标题，不要分点，不要解释，只输出摘要正文。\n\n"
        f"{raw_text}"
    )

    # 使用已经验证可用的流式 LLM 接口生成摘要
    summary_parts: list[str] = []

    async for delta in stream_chat(
            text=summary_prompt,
            trace_id=trace_id,
            instructions=(
                    "你是对话摘要器。"
                    "你只负责生成准确、简洁、可用于长期记忆检索的中文摘要。"
                    "不要回答用户问题，不要编造对话中没有的信息。"
            ),
            model=model,
    ):
        summary_parts.append(delta)

    summary_text = "".join(summary_parts).strip()
    logger.debug("parts text: %s %s", summary_parts, summary_text)
    # 模型没有生成有效内容时，不保存空摘要
    if not summary_text:
        return None

    return await create_conversation_summary(
        db=db,
        conversation_id=conversation_id,
        device_id=device_id,
        agent_id=agent_id,
        summary_text=summary_text,
        covered_start_message_id=int(messages[0]["id"]),
        covered_end_message_id=int(messages[-1]["id"]),
        summary_level="segment",
    )
