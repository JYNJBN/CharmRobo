import logging

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.device_security import (
    hash_device_secret,
    verify_device_secret,
)
from app.models import Device, UserDevice
from app.schemas.device import BootstrapDeviceRequest, UserDeviceResponse
from app.services.device_binding_service import (
    delete_bind_ticket,
    get_bind_ticket,
)
from app.utils.tools import local_now

logger = logging.getLogger(__name__)




async def get_device_by_sn(
    db: AsyncSession,
    device_sn: str,
) -> Device | None:
    """根据 SN 查询尚未逻辑删除的设备。"""

    result = await db.execute(
        select(Device).where(
            Device.device_sn == device_sn,
            Device.deleted == 0,
        )
    )
    return result.scalar_one_or_none()


async def get_active_owner_binding(
    db: AsyncSession,
    device_id: int,
) -> UserDevice | None:
    """查询设备当前有效的拥有者绑定关系。"""

    result = await db.execute(
        select(UserDevice).where(
            UserDevice.device_id == device_id,
            UserDevice.role == "owner",
            UserDevice.deleted == 0,
        )
    )
    return result.scalars().first()


async def get_user_device_binding(
    db: AsyncSession,
    user_id: int,
    device_id: int,
) -> UserDevice | None:
    """查询某个用户和设备的关系，包括已逻辑删除的记录。"""

    result = await db.execute(
        select(UserDevice).where(
            UserDevice.user_id == user_id,
            UserDevice.device_id == device_id,
        )
    )
    return result.scalar_one_or_none()


async def bootstrap_device(
    db: AsyncSession,
    redis: Redis,
    data: BootstrapDeviceRequest,
) -> Device:
    """
    使用硬件生成的永久密钥初始化并绑定设备。

    硬件必须先将原始密钥保存到 Flash/NVS；后端只保存摘要。
    相同 SN 和密钥的重复请求会按成功处理，避免 HTTP 响应
    丢失后设备无法安全重试。
    """

    logger.info(
        "设备绑定请求开始 device_sn=%s product_key=%s",
        data.device_sn,
        data.product_key,
    )

    # ================================================================
    # 公共入口：先根据 SN 判断这台设备在 MySQL 中是否已经存在。
    #
    # 查询结果会把流程分成两大类：
    # 1. existing_device is None：路线 1，设备第一次初始化。
    # 2. existing_device 存在：路线 2～4，重复请求、解绑重绑，
    #    或后台预先导入了设备 SN。
    # ================================================================
    existing_device = await get_device_by_sn(db, data.device_sn)
    logger.info(
        "设备查询完成 device_sn=%s exists=%s device_id=%s initialized=%s",
        data.device_sn,
        existing_device is not None,
        existing_device.id if existing_device is not None else None,
        bool(existing_device and existing_device.device_secret_hash),
    )

    # ================================================================
    # 路线 2：设备已经初始化过（存在密钥摘要）。
    #
    # 可能出现的场景：
    # - 上一次 MySQL 已提交，但设备没收到 HTTP 成功响应；
    # - 硬件网络重试，重复发送了相同请求；
    # - 已初始化设备解绑后，准备重新绑定。
    #
    # 无论是哪种情况，都必须先验证硬件提交的永久密钥。
    # 同一个 SN 却使用不同密钥时直接拒绝，防止伪造设备。
    # ================================================================
    if existing_device is not None and existing_device.device_secret_hash:
        if not verify_device_secret(
            data.device_secret,
            existing_device.device_secret_hash,
        ):
            # 路线 5：SN 已存在，但密钥错误。
            logger.warning(
                "设备密钥校验失败 device_sn=%s device_id=%s",
                data.device_sn,
                existing_device.id,
            )
            raise ValueError("设备已经初始化，但设备密钥不正确")

        active_owner = await get_active_owner_binding(
            db,
            int(existing_device.id),
        )
        if active_owner is not None:
            # ========================================================
            # 路线 2A：设备存在、密钥正确、owner 也存在。
            #
            # 这通常是一次重复请求，不需要再次 INSERT。
            # 但如果 Redis 中还有绑定码，需要确认绑定码里的用户
            # 就是当前 owner，避免另一个用户抢绑已有设备。
            # ========================================================
            # 如果绑定码仍然存在，就确认它属于当前 owner。
            # 如果绑定码已经因上次成功提交而被删除，则视为设备在
            # HTTP 响应丢失后的幂等重试。
            try:
                retry_ticket_data = await get_bind_ticket(
                    redis,
                    data.bind_ticket,
                )
            except ValueError:
                # 路线 2A-1：绑定码已被上次成功请求删除，或者已过期。
                # 设备和 owner 都存在且密钥正确，因此按幂等成功处理。
                logger.info(
                    "设备绑定按幂等重试成功 device_sn=%s device_id=%s owner_user_id=%s",
                    data.device_sn,
                    existing_device.id,
                    active_owner.user_id,
                )
                return existing_device

            if retry_ticket_data.get("device_sn") != data.device_sn:
                raise ValueError("绑定码与当前设备不匹配")

            retry_user_id = retry_ticket_data.get("user_id")
            if retry_user_id is None:
                logger.warning(
                    "设备绑定码用户信息缺失 device_sn=%s device_id=%s",
                    data.device_sn,
                    existing_device.id,
                )
                raise ValueError("绑定码中的用户数据不完整")

            if int(retry_user_id) != int(active_owner.user_id):
                # 路线 6：设备已有其他 owner，拒绝新用户直接抢绑。
                logger.warning(
                    "设备绑定被拒绝：设备已有其他用户 device_sn=%s device_id=%s requested_user_id=%s owner_user_id=%s",
                    data.device_sn,
                    existing_device.id,
                    retry_user_id,
                    active_owner.user_id,
                )
                raise ValueError("设备已经被其他用户绑定")

            # 路线 2A-2：同一个 owner 使用新的有效绑定码重复请求。
            # 删除多余绑定码后直接返回已有设备。
            await delete_bind_ticket(redis, data.bind_ticket)
            logger.info(
                "设备重复绑定成功 device_sn=%s device_id=%s user_id=%s",
                data.device_sn,
                existing_device.id,
                active_owner.user_id,
            )
            return existing_device

    # ================================================================
    # 路线 1 / 路线 3 / 路线 4 的共同入口：验证绑定码。
    #
    # 路线 1：设备不存在，第一次初始化并绑定。
    # 路线 3：设备存在且密钥正确，但已经没有有效 owner，重新绑定。
    # 路线 4：设备 SN 已被后台预导入，但还没有保存密钥摘要。
    #
    # 注意：这里只 GET，不立即删除绑定码。
    # 只有 MySQL 事务真正提交成功后才删除，这样数据库失败时还能重试。
    # ================================================================
    ticket_data = await get_bind_ticket(redis, data.bind_ticket)
    ticket_device_sn = ticket_data.get("device_sn")
    ticket_user_id = ticket_data.get("user_id")

    if not ticket_device_sn or ticket_user_id is None:
        logger.warning(
            "设备绑定失败：绑定码数据不完整 device_sn=%s",
            data.device_sn,
        )
        raise ValueError("绑定码中的数据不完整")

    if ticket_device_sn != data.device_sn:
        logger.warning(
            "设备绑定失败：绑定码与设备不匹配 device_sn=%s ticket_device_sn=%s",
            data.device_sn,
            ticket_device_sn,
        )
        raise ValueError("绑定码与当前设备不匹配")

    user_id = int(ticket_user_id)
    logger.info(
        "设备绑定码校验成功 device_sn=%s user_id=%s existing_device=%s",
        data.device_sn,
        user_id,
        existing_device is not None,
    )

    if existing_device is None:
        # ============================================================
        # 路线 1：数据库中完全没有这台设备。
        #
        # 硬件已经生成并保存原始 device_secret；后端只计算 SHA-256
        # 摘要保存到 device.device_secret_hash，不保存原始密钥。
        # ============================================================
        device = Device(
            device_sn=data.device_sn,
            product_key=data.product_key,
            firmware_version=data.firmware_version,
            hardware_version=data.hardware_version,
            device_secret_hash=hash_device_secret(data.device_secret),
            secret_version=1,  # noqa: F821
            secret_updated_time=local_now(),
            status=1,
            deleted=0,
        )
        db.add(device)
        logger.info(
            "准备创建新设备记录 device_sn=%s user_id=%s",
            data.device_sn,
            user_id,
        )
    else:
        # ============================================================
        # 路线 3 / 路线 4：设备记录已经存在。
        #
        # 路线 3：设备有密钥，但没有有效 owner，属于解绑后重绑。
        # 路线 4：设备只有预导入的 SN、没有密钥摘要，在这里补摘要。
        # ============================================================
        device = existing_device
        if not device.device_secret_hash:
            # 路线 4：第一次把硬件密钥的摘要登记到数据库。
            device.device_secret_hash = hash_device_secret(data.device_secret)
            device.secret_updated_time = local_now()
            logger.info(
                "准备补充设备密钥摘要 device_sn=%s device_id=%s",
                data.device_sn,
                device.id,
            )

        device.product_key = data.product_key
        device.firmware_version = data.firmware_version
        device.hardware_version = data.hardware_version
        device.status = 1
        device.deleted = 0

    try:
        # ============================================================
        # 数据库事务阶段：Device 和 UserDevice 必须一起成功或一起失败。
        # ============================================================
        # 新设备 flush 后才会获得数据库自增 ID。
        await db.flush()

        active_owner = await get_active_owner_binding(
            db,
            int(device.id),
        )
        if active_owner is not None and int(active_owner.user_id) != user_id:
            # 路线 6：事务执行期间发现设备已经被其他用户绑定。
            logger.warning(
                "设备绑定事务校验失败：设备已被其他用户绑定 device_sn=%s device_id=%s requested_user_id=%s owner_user_id=%s",
                data.device_sn,
                device.id,
                user_id,
                active_owner.user_id,
            )
            raise ValueError("设备已经被其他用户绑定")

        binding = await get_user_device_binding(
            db,
            user_id,
            int(device.id),
        )

        if binding is None:
            # 路线 1 或新用户接手已解绑设备：创建新的绑定关系。
            binding = UserDevice(
                user_id=user_id,
                device_id=int(device.id),
                alias=None,
                role="owner",
                deleted=0,
            )
            db.add(binding)
            logger.info(
                "准备创建用户设备绑定关系 device_id=%s user_id=%s",
                device.id,
                user_id,
            )
        else:
            # 路线 3：同一用户解绑后再次绑定，恢复旧关系记录。
            # 不重新 INSERT，避免 (user_id, device_id) 唯一约束冲突。
            binding.role = "owner"
            binding.deleted = 0
            binding.unbind_time = None
            binding.bind_time = local_now()
            logger.info(
                "准备恢复用户设备绑定关系 device_id=%s user_id=%s binding_id=%s",
                device.id,
                user_id,
                binding.id,
            )

        # device 和 user_device 在同一 MySQL 事务中提交。
        await db.commit()
        await db.refresh(device)
        logger.info(
            "设备绑定数据库事务提交成功 device_sn=%s device_id=%s user_id=%s",
            data.device_sn,
            device.id,
            user_id,
        )

        # Redis 清理失败不应把已成功的数据库事务变成 500；
        # 绑定码本身仍会在五分钟后自动过期。
        try:
            await delete_bind_ticket(redis, data.bind_ticket)
        except Exception:
            logger.exception("删除设备绑定码失败")

        logger.info(
            "设备绑定请求完成 device_sn=%s device_id=%s user_id=%s",
            data.device_sn,
            device.id,
            user_id,
        )
        return device

    except IntegrityError as exc:
        # ============================================================
        # 路线 7：并发或数据库约束冲突。
        #
        # 例如两个完全相同的初始化请求同时到达：
        # - 请求 A 先成功插入；
        # - 请求 B 插入时触发 device_sn 唯一约束。
        #
        # B 回滚后重新查询；如果发现 A 已经用相同密钥完成绑定，
        # 就把 B 也按幂等成功处理。
        # ============================================================
        await db.rollback()
        logger.warning(
            "设备绑定发生数据库并发或唯一约束冲突 device_sn=%s user_id=%s error=%s",
            data.device_sn,
            user_id,
            exc,
        )

        # 两个相同请求并发时，其中一个可能已经创建成功。
        concurrent_device = await get_device_by_sn(db, data.device_sn)
        if (
            concurrent_device is not None
            and concurrent_device.device_secret_hash
            and verify_device_secret(
                data.device_secret,
                concurrent_device.device_secret_hash,
            )
        ):
            concurrent_owner = await get_active_owner_binding(
                db,
                int(concurrent_device.id),
            )
            if concurrent_owner is not None:
                logger.info(
                    "设备绑定并发请求按幂等成功 device_sn=%s device_id=%s user_id=%s",
                    data.device_sn,
                    concurrent_device.id,
                    concurrent_owner.user_id,
                )
                return concurrent_device

        raise ValueError("设备初始化或绑定失败") from exc

    except Exception:
        # 其他任何异常：回滚当前 MySQL 事务并继续向上抛出。
        # 由于绑定码尚未删除，修复问题后仍可以继续重试。
        await db.rollback()
        logger.exception(
            "设备绑定事务失败 device_sn=%s user_id=%s",
            data.device_sn,
            locals().get("user_id"),
        )
        raise


async def get_devices_by_user_id(
    db: AsyncSession,
    user_id: int,
) -> list[UserDeviceResponse]:
    """
    查询当前用户绑定的全部有效设备。
    user_device 负责确定用户绑定了哪些设备；
    device 负责提供设备本身的信息。
    """
    # sql语句
    statement = (
        select(
            # 这里区别名主要是返回值要别名是device_id给数据使用
            Device.id.label("device_id"),
            Device.device_sn,
            Device.product_key,
            Device.firmware_version,
            Device.hardware_version,
            Device.status,
            Device.last_online_time,
            # user_device 表字段
            UserDevice.alias,
            UserDevice.role,
            UserDevice.bind_time,
        )
        #     联表条件
        .join(UserDevice, UserDevice.device_id == Device.id)
        .where(
            UserDevice.user_id == user_id,
            # 不要解绑的
            UserDevice.deleted == 0,
            Device.deleted == 0,
        )
        .order_by(UserDevice.bind_time.desc())
    )
    result = await db.execute(statement)
    # 直接查出来是元组(1,"DEVICE-001","客厅音箱")
    # mappings() 把结果转换成类似字典的数据：
    #
    # {
    #     "device_id": 1,
    #     "device_sn": "DEVICE-001",
    #     "alias": "客厅音箱",
    #     ...
    # }
    rows = result.mappings().all()

    logger.info(
        "查询用户绑定设备成功 user_id=%s device_count=%s",
        user_id,
        len(rows),
    )

    return [UserDeviceResponse.model_validate(row) for row in rows]


async def unbind_device_for_user(
    db: AsyncSession,
    user_id: int,
    device_id: int,
) -> None:
    """
    解除当前用户和指定设备的绑定关系。

    这里只逻辑删除 user_device 关系，不删除 device 硬件记录，
    因此设备身份和永久密钥仍可用于后续重新绑定。
    """

    logger.info(
        "解除设备绑定请求开始 user_id=%s device_id=%s",
        user_id,
        device_id,
    )

    result = await db.execute(
        select(UserDevice).where(
            UserDevice.user_id == user_id,
            UserDevice.device_id == device_id,
            UserDevice.deleted == 0,
        )
    )
    binding = result.scalar_one_or_none()

    if binding is None:
        logger.warning(
            "解除设备绑定失败：绑定关系不存在 user_id=%s device_id=%s",
            user_id,
            device_id,
        )
        raise ValueError("设备不存在或已经解除绑定")

    binding.deleted = 1
    binding.unbind_time = local_now()

    try:
        await db.commit()
        logger.info(
            "解除设备绑定成功 user_id=%s device_id=%s binding_id=%s",
            user_id,
            device_id,
            binding.id,
        )
    except Exception:
        await db.rollback()
        logger.exception(
            "解除设备绑定事务失败 user_id=%s device_id=%s",
            user_id,
            device_id,
        )
        raise
