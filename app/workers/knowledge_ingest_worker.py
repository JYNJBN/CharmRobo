import asyncio
import logging
import os
import socket
from contextlib import suppress
from uuid import uuid4

from app.core.config import settings
from app.services.knowledge_ingest_job_service import (
    claim_one_ingest_job,
    process_claimed_job, recover_one_expired_job,
)

logger = logging.getLogger(__name__)

WORKER_ID = (
    f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
)


async def recover_expired_jobs_loop() -> None:
    while True:
        try:
            # 每轮最多回收 20 条，每条使用独立的短事务。
            for _ in range(20):
                recovered = await recover_one_expired_job()

                if not recovered:
                    break

        except Exception:
            logger.exception("[KNOWLEDGE-WORKER] 回收过期任务失败")

        await asyncio.sleep(5)


async def run_worker() -> None:
    logger.info("[KNOWLEDGE-WORKER] 启动 worker_id=%s", WORKER_ID)
    # 与任务处理循环并行运行。
    recovery_task = asyncio.create_task(
        recover_expired_jobs_loop(),
        name="knowledge-job-recovery",
    )
    try:
        while True:
            try:
                claim = await claim_one_ingest_job(WORKER_ID)
            except Exception:
                logger.exception("[KNOWLEDGE-WORKER] 领取任务失败")
                await asyncio.sleep(2)
                continue

            if claim is None:
                await asyncio.sleep(1)
                continue

            logger.info(
                "[KNOWLEDGE-WORKER] 领取任务 job_id=%s stage=%s",
                claim["job_id"],
                claim["stage"],
            )
            await process_claimed_job(claim)
    finally:
        # Worker 退出时，同时停止回收循环。
        recovery_task.cancel()

        with suppress(asyncio.CancelledError):
            await recovery_task


if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    asyncio.run(run_worker())
