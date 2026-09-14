from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import BizError
from app.models import (
    Agent,
    Device,
    UserDevice,
)
from app.schemas.agent import AgentCreate, AgentResponse, AgentUpdate
from app.services.agent_change_notifier import notify_agent_config_changed
from app.services.model_service import resolve_model
from app.services.voice_service import resolve_voice_id

# 可以修改的字段
EDITABLE_FIELDS = {
    "name",
    "system_prompt",
    "model_key",
    "voice",
}

CURRENT_AGENT_MODEL_KEY = "doubao"


async def require_device_access(
        db: AsyncSession,
        user_id: int,
        device_id: int,
        owner_only: bool = False,
) -> UserDevice:
    binding = await db.scalar(
        select(UserDevice).where(
            UserDevice.user_id == user_id,
            UserDevice.device_id == device_id,
            UserDevice.deleted == 0,
        )
    )

    if binding is None:
        raise BizError("无权操作该设备的智能体", code=403)

    if owner_only and binding.role != "owner":
        raise BizError("只有设备拥有者可以修改智能体", code=403)

    return binding


async def fork_agent(
        db: AsyncSession,
        device_id: int,
        template: Agent,
) -> Agent:
    copy = Agent(
        device_id=device_id,
        name=template.name,
        system_prompt=template.system_prompt,
        model_key=template.model_key,
        voice=template.voice,
        is_system=False,
        source_agent_id=template.id,
        deleted=0,
    )
    db.add(copy)
    await db.flush()
    return copy


async def get_default_agent_template(db: AsyncSession) -> Agent | None:
    """查询系统默认模板（is_system=true 且 device_id 为空的第一条）。"""
    return await db.scalar(
        select(Agent)
        .where(
            Agent.is_system == True,
            Agent.device_id.is_(None),
            Agent.deleted == 0,
        )
        .order_by(Agent.id.asc())
        .limit(1)
    )


async def ensure_default_agent_on_device(
        db: AsyncSession,
        device: Device,
) -> Agent | None:
    """确保设备有默认智能体副本并激活

    只 flush 不 commit，由外层事务（设备绑定、语音建连）统一提交。
    并发重复 fork 由迁移建的部分唯一索引 uq_agent_device_source_active 兜底。
    """
    # 已激活且归属正确的 agent 直接返回，不覆盖用户的选择。
    if device.active_agent_id is not None:
        agent = await get_agent(db, int(device.active_agent_id))
        if (
                agent is not None
                and agent.device_id == device.id
                and not agent.is_system
        ):
            return agent

    template = await get_default_agent_template(db)
    if template is None:
        return None

    # 老设备兜底：迁移或上次调用已经物化过的副本直接复用。
    copy = await db.scalar(
        select(Agent).where(
            Agent.device_id == device.id,
            Agent.source_agent_id == template.id,
            Agent.deleted == 0,
        )
    )
    if copy is None:
        copy = await fork_agent(db, int(device.id), template)

    device.active_agent_id = copy.id
    await db.flush()
    return copy


async def get_active_agent_for_device(
        db: AsyncSession,
        device: Device,
) -> Agent:
    """读取设备当前激活的智能体，并校验归属和状态（手册 6-Step1）。"""
    if device.active_agent_id is None:
        raise ValueError(f"设备 {device.id} 没有激活的智能体")
    agent = await get_agent(db, int(device.active_agent_id))
    if agent is None or agent.device_id != device.id or agent.is_system:
        raise ValueError(f"设备 {device.id} 的激活智能体无效")
    return agent


async def get_agent(db, agent_id: int) -> Agent:
    return await db.scalar(
        select(Agent).where(
            Agent.id == agent_id,
            Agent.deleted == 0,
        )
    )


def to_agent_response(
        agent: Agent,
        active_agent_id: int | None,
        default_template_id: int | None = None,
) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        device_id=(
            int(agent.device_id)
            if agent.device_id is not None
            else None
        ),
        name=agent.name,
        system_prompt=agent.system_prompt,
        model_key=agent.model_key,
        voice=agent.voice,
        is_system=bool(agent.is_system),
        source_agent_id=(
            int(agent.source_agent_id)
            if agent.source_agent_id is not None
            else None
        ),
        active=agent.id == active_agent_id,
        # 默认副本 = fork 自当前默认模板的那一个（结构身份，不随切换移动）。
        # active 才是"使用中"标记，会随切换移动；两者是不同概念。
        is_default=(
                default_template_id is not None
                and agent.source_agent_id == default_template_id
        ),
    )


async def delete_agent(
        db: AsyncSession,
        user_id: int,
        agent_id: int,
) -> bool:
    agent = await get_agent(db, agent_id)
    if agent is None:
        raise BizError("智能体不存在", code=404)
    if agent.is_system or agent.device_id is None:
        raise BizError("系统智能体不能删除", code=400)

    device_id = int(agent.device_id)
    await require_device_access(
        db=db,
        user_id=user_id,
        device_id=device_id,
    )

    # 默认副本是设备的兜底智能体，删除会让设备失去可回退的基础配置。
    template = await get_default_agent_template(db)
    if template is not None and agent.source_agent_id == template.id:
        raise BizError("设备默认智能体不能删除", code=400)

    device = await db.get(Device, device_id)
    if device is None:
        raise BizError("设备不存在", code=404)

    # 只逻辑删除智能体。相关会话、消息和摘要全部保留，便于审计或恢复。
    agent.deleted = 1
    await db.flush()

    # 删除的是当前使用中的智能体时，自动把激活指针切回默认副本，
    # 避免设备出现没有可用智能体的窗口。
    if device.active_agent_id == agent.id:
        fallback = await ensure_default_agent_on_device(db=db, device=device)
        if fallback is None:
            await db.rollback()
            raise BizError("没有可切换的默认智能体，无法删除", code=400)

    await db.commit()
    await notify_agent_config_changed(
        device_id=device_id,
        agent_id=device.active_agent_id,
        reason="deleted",
    )
    return True


async def activate_agent(
        db: AsyncSession,
        user_id: int,
        device_id: int,
        agent_id: int,
) -> AgentResponse:
    """切换设备当前 Agent，并通知在线端到端会话重新加载配置。"""
    await require_device_access(db=db, user_id=user_id, device_id=device_id)

    device = await db.scalar(
        select(Device).where(
            Device.id == device_id,
            Device.deleted == 0,
        )
    )
    if device is None:
        raise BizError("设备不存在", code=404)

    agent = await get_agent(db, agent_id)
    if agent is None or agent.device_id != device_id or agent.is_system:
        raise BizError("只能激活当前设备的智能体副本", code=400)

    device.active_agent_id = agent.id
    await db.commit()
    await db.refresh(agent)
    await notify_agent_config_changed(
        device_id=device_id,
        agent_id=agent.id,
        reason="activated",
    )
    return to_agent_response(agent, device.active_agent_id)


async def list_agents_for_device(
        db: AsyncSession,
        user_id: int,
        device_id: int,
) -> list[AgentResponse]:
    # 第一步：先校验当前用户是否绑定了这台设备。
    await require_device_access(
        db=db,
        user_id=user_id,
        device_id=device_id,
    )

    # 第二步：查询设备，获取 active_agent_id。
    device = await db.scalar(
        select(Device).where(
            Device.id == device_id,
            Device.deleted == 0,
        )
    )
    if device is None:
        raise BizError("设备不存在", code=404)

    # 第三步：只查询设备自己的副本。
    # 系统模板 device_id IS NULL，所以不会出现在这里。
    result = await db.scalars(
        select(Agent)
        .where(
            Agent.device_id == device_id,
            Agent.deleted == 0,
        )
        .order_by(Agent.id.asc())
    )
    agents = result.all()

    # 查默认模板，用于标记哪个副本是默认副本（前端据此隐藏删除按钮）。
    template = await get_default_agent_template(db)
    default_template_id = template.id if template else None
    return [
        to_agent_response(
            agent,
            active_agent_id=device.active_agent_id,
            default_template_id=default_template_id,
        )
        for agent in agents
    ]


async def create_agent(
        db: AsyncSession,
        user_id: int,
        data: AgentCreate,
) -> AgentResponse:
    """创建设备 Agent；新 Agent 创建成功后立即成为当前 Agent。"""
    # 判断是否有权限
    device_id = data.device_id
    await require_device_access(db=db, device_id=device_id, user_id=user_id)
    device = await db.scalar(
        select(Device).where(
            Device.id == device_id,
            Device.deleted == 0,
        )
    )
    if device is None:
        raise BizError("设备不存在", code=404)
    createData = data.model_dump(exclude_unset=True)
    # 校验model_key voice是否合法
    next_model_key = createData.get("model_key", 'doubao')
    next_voice = createData.get("voice", 'female_gentle')
    name = createData.get("name", '小梦')
    system_prompt = (
        data.system_prompt.strip()
        if data.system_prompt
        else None
    )
    if next_model_key != CURRENT_AGENT_MODEL_KEY:
        raise BizError("当前版本暂只支持豆包模型", code=400)
    resolve_model(next_model_key)
    resolve_voice_id(next_voice)
    agent = Agent(
        name=name,
        device_id=device_id,
        model_key=next_model_key,
        voice=next_voice,
        system_prompt=system_prompt,
        is_system=False,
        source_agent_id=None,
        deleted=0
    )
    db.add(agent)
    await db.flush()
    device.active_agent_id = agent.id

    await db.commit()
    await db.refresh(agent)
    await notify_agent_config_changed(
        device_id=device_id,
        agent_id=agent.id,
        reason="created",
    )
    return to_agent_response(agent, active_agent_id=agent.id)


async def reset_agent(
        db: AsyncSession,
        user_id: int,
        agent_id: int,
) -> AgentResponse:
    """将 Agent 恢复为模板配置；只有当前正在使用它时才通知在线会话。"""
    agent = await get_agent(db, agent_id)
    if agent is None:
        raise BizError("智能体不存在", code=404)
    if agent.is_system or agent.device_id is None:
        raise BizError(
            "系统模板只读，请先从模板创建副本",
            code=400,
        )
    if agent.source_agent_id is None:
        raise BizError("自建智能体没有默认配置可恢复", code=400)
    device_id = int(agent.device_id)
    await require_device_access(db=db, user_id=user_id, device_id=device_id)

    source_agent = await get_agent(db, int(agent.source_agent_id))
    if (
            source_agent is None
            or not source_agent.is_system
            or source_agent.device_id is not None
    ):
        raise BizError("来源模板不存在", code=400)
    resolve_model(source_agent.model_key)
    resolve_voice_id(source_agent.voice)
    for key in EDITABLE_FIELDS:
        setattr(agent, key, getattr(source_agent, key))
    device = await db.scalar(
        select(Device).where(
            Device.id == device_id,
            Device.deleted == 0,
        )
    )
    if device is None:
        raise BizError("设备不存在", code=404)

    await db.commit()
    await db.refresh(agent)
    if device.active_agent_id == agent.id:
        await notify_agent_config_changed(
            device_id=device_id,
            agent_id=agent.id,
            reason="reset",
        )
    return to_agent_response(agent,
                             active_agent_id=device.active_agent_id,
                             default_template_id=source_agent.id, )


async def save_agent_config(
        db: AsyncSession,
        user_id: int,
        agent_id: int,
        data: AgentUpdate,
) -> AgentResponse:
    """保存 Agent 人设/音色等配置，并通知当前在线语音会话。

    当前版本模型只允许豆包；人设和音色变化仍然需要重建端到端 session，
    因为它们是在 session.create 时加载的。
    """
    agent = await get_agent(db, agent_id)
    if agent is None:
        raise BizError("智能体不存在", code=404)
    if agent.is_system or agent.device_id is None:
        raise BizError(
            "系统模板只读，请先从模板创建副本",
            code=400,
        )
    # 判断是否有修改编辑权限
    device_id = int(agent.device_id)
    await require_device_access(
        db=db,
        user_id=user_id,
        device_id=device_id,
    )
    # 转字典 exclude_unset=True 将没传递的参数去掉 比如  前端只传了
    # {
    #   "model_key": "qwen"
    # }
    # 其他字段 就不转了 就保持 "model_key": "qwen" 不加的话则是
    # {
    #     "name": None,
    #     "system_prompt": None,
    #     "model_key": "qwen",
    #     "voice": None,
    # }
    updates = data.model_dump(exclude_unset=True)
    # 获取要修改的model_key和voice
    next_model_key = updates.get("model_key", agent.model_key)
    next_voice = updates.get("voice", agent.voice)
    # 这里主要校验是否合法参数
    if "model_key" in updates and next_model_key != CURRENT_AGENT_MODEL_KEY:
        raise BizError("当前版本暂只支持豆包模型", code=400)
    resolve_model(next_model_key)
    resolve_voice_id(next_voice)

    for key in EDITABLE_FIELDS:
        if key not in updates:
            continue
        value = updates[key]
        if key in {"name", "system_prompt"} and value:
            value = value.strip() or None
        setattr(agent, key, value)

    device = await db.get(Device, device_id)
    device.active_agent_id = agent.id
    await db.commit()
    await db.refresh(agent)
    await notify_agent_config_changed(
        device_id=device_id,
        agent_id=agent.id,
        reason="updated",
    )
    return to_agent_response(agent, device.active_agent_id)
