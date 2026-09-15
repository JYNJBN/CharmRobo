"""会话摘要与长期记忆后台服务。

这里不依赖任何 HTTP/WebSocket 路由，供小程序语音路由、硬件语音路由以及
后续其他对话入口共同调用。后台任务失败只记录日志，不影响已经生成的语音
回答和客户端协议响应。
"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.integrations.embedding import embed_text
from app.integrations.milvus import upsert_summary
from app.schemas.Model import ModelRegistryObject
from app.services.conversation_summary_service import (
    generate_conversation_summary,
    get_latest_summary_end_message_id,
)

logger = logging.getLogger(__name__)


async def summarize_conversation_in_background(
    conversation_id: int,
    device_id: int | None,
    user_id: int | None,
    trace_id: str,
    agent_id: int | None = None,
    model: ModelRegistryObject | None = None,
) -> None:
    """后台生成会话摘要，并把摘要向量写入 Milvus。

    参数和原语音路由中的函数保持一致，避免改变现有小程序调用行为：

    - ``conversation_id``：要处理的会话；
    - ``device_id`` / ``user_id`` / ``agent_id``：Milvus 过滤和隔离所需的归属；
    - ``trace_id``：串联本轮语音日志和后台摘要日志。

    摘要生成使用独立数据库会话。只有生成出新摘要且三类归属 ID 都存在时，
    才进行 Embedding 和 Milvus 写入；缺少归属信息时保留数据库摘要，但跳过
    向量写入。所有异常在后台吞掉并记录，不能反向破坏已经完成的对话。
    """

    started_at = asyncio.get_running_loop().time()
    logger.info(
        "[记忆][后台任务][%s] 开始 conversation_id=%s user_id=%s "
        "device_id=%s agent_id=%s 批次=%s",
        trace_id,
        conversation_id,
        user_id,
        device_id,
        agent_id,
        settings.summary_batch_messages,
    )
    try:
        async with AsyncSessionLocal() as db:
            pointer_started_at = asyncio.get_running_loop().time()
            last_covered_message_id = await get_latest_summary_end_message_id(
                db,
                conversation_id=conversation_id,
            )
            logger.info(
                "[记忆][后台任务][%s] 已读取摘要进度 conversation_id=%s "
                "last_covered_message_id=%s 耗时=%.3f秒",
                trace_id,
                conversation_id,
                last_covered_message_id,
                asyncio.get_running_loop().time() - pointer_started_at,
            )

            summary_started_at = asyncio.get_running_loop().time()
            summary = await generate_conversation_summary(
                db,
                conversation_id=conversation_id,
                device_id=device_id,
                agent_id=agent_id,
                last_covered_message_id=last_covered_message_id,
                limit=settings.summary_batch_messages,
                trace_id=f"{trace_id}-summary",
                model=model,
            )
            summary_elapsed = asyncio.get_running_loop().time() - summary_started_at

            if summary is None:
                logger.info(
                    "[记忆][后台任务][%s] 未达到摘要批次，跳过向量写入 "
                    "conversation_id=%s 摘要阶段耗时=%.3f秒 总耗时=%.3f秒",
                    trace_id,
                    conversation_id,
                    summary_elapsed,
                    asyncio.get_running_loop().time() - started_at,
                )
                return

            logger.info(
                "[SUMMARY][%s] 摘要生成成功 conversation_id=%s summary_id=%s",
                trace_id,
                conversation_id,
                summary.id,
                # summary 返回时已经完成 conversation_summary 的数据库提交。
            )

            if device_id is None or user_id is None or agent_id is None:
                logger.warning(
                    "[MEMORY][%s] 缺少 device_id/user_id/agent_id，跳过 Milvus 写入",
                    trace_id,
                )
                return

            embedding_started_at = asyncio.get_running_loop().time()
            logger.info(
                "[记忆][Embedding][%s] 摘要向量化开始 summary_id=%s "
                "摘要字数=%d",
                trace_id,
                summary.id,
                len(summary.summary_text),
            )
            embedding_vector = await embed_text(summary.summary_text)
            logger.info(
                "[记忆][Embedding][%s] 摘要向量化完成 summary_id=%s "
                "向量维度=%d 耗时=%.3f秒",
                trace_id,
                summary.id,
                len(embedding_vector),
                asyncio.get_running_loop().time() - embedding_started_at,
            )
            # pymilvus 是同步客户端，放到线程中，避免阻塞 FastAPI 事件循环。
            milvus_started_at = asyncio.get_running_loop().time()
            logger.info(
                "[记忆][Milvus写入][%s] 开始 summary_id=%s",
                trace_id,
                summary.id,
            )
            milvus_result = await asyncio.to_thread(
                upsert_summary,
                summary_id=summary.id,
                summary_text=summary.summary_text,
                embedding=embedding_vector,
                conversation_id=conversation_id,
                device_id=device_id,
                user_id=user_id,
                agent_id=agent_id,
                status=summary.status,
            )

            logger.info(
                "[记忆][Milvus写入][%s] 完成 summary_id=%s result=%s "
                "耗时=%.3f秒 总耗时=%.3f秒",
                trace_id,
                summary.id,
                milvus_result,
                asyncio.get_running_loop().time() - milvus_started_at,
                asyncio.get_running_loop().time() - started_at,
            )

    except Exception:
        logger.exception(
            "[记忆][后台任务][%s] 失败 conversation_id=%s 总耗时=%.3f秒",
            trace_id,
            conversation_id,
            asyncio.get_running_loop().time() - started_at,
        )
