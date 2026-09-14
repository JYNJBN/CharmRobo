from fastapi import APIRouter

from app.api.dependencies import CurrentUserId, DbSession
from app.schemas.agent import (
    AgentCreate,
    AgentModelOption,
    AgentOptionsResponse,
    AgentResponse,
    AgentUpdate,
    AgentVoiceOption,
)
from app.schemas.common import ApiResponse
from app.services.agent_service import (
    activate_agent,
    create_agent,
    delete_agent,
    list_agents_for_device,
    reset_agent,
    save_agent_config,
)
from app.services.model_service import MODEL_LABELS, MODEL_REGISTRY
from app.services.voice_service import VOICE_LABELS

router = APIRouter(
    prefix="/agent",
    tags=["智能体"],
)


@router.get(
    "/options",
    response_model=ApiResponse[AgentOptionsResponse],
)
async def get_agent_options() -> ApiResponse[AgentOptionsResponse]:
    """返回当前后端实际可用的模型和音色选项，不暴露敏感配置。"""
    models = [
        AgentModelOption(
            key=key,
            name=MODEL_LABELS.get(key, key),
            provider=str(config.get("provider", "unknown")),
        )
        for key, config in MODEL_REGISTRY.items()
        if key == "doubao"
    ]
    voices = [
        AgentVoiceOption(key=key, name=VOICE_LABELS.get(key, key))
        for key in VOICE_LABELS
    ]
    return ApiResponse(
        data=AgentOptionsResponse(models=models, voices=voices)
    )


@router.get(
    "/",
    response_model=ApiResponse[list[AgentResponse]],
)
async def get_agent(
        device_id: int,
        db: DbSession,
        current_user_id: CurrentUserId

) -> ApiResponse[list[AgentResponse]]:
    """返回该设备的智能体（含系统模板 + 该设备已 fork 的）"""
    data = await list_agents_for_device(
        db=db,
        user_id=current_user_id,
        device_id=device_id,
    )
    return ApiResponse(data=data)


@router.put(
    "/{agent_id}",
    response_model=ApiResponse[AgentResponse],
)
async def update_agent_api(
        agent_id: int,
        data: AgentUpdate,
        db: DbSession,
        current_user_id: CurrentUserId
):
    result = await save_agent_config(
        db=db,
        agent_id=agent_id,
        data=data,
        user_id=current_user_id,
    )
    return ApiResponse(data=result)


@router.post(
    "/device/{device_id}/activate/{agent_id}",
    response_model=ApiResponse[AgentResponse],
)
async def activate_agent_api(
        device_id: int, agent_id: int, db: DbSession,
        current_user_id: CurrentUserId
):
    result = await activate_agent(
        db=db,
        agent_id=agent_id,
        device_id=device_id,
        user_id=current_user_id,
    )
    return ApiResponse(data=result)


@router.delete(
    "/{agent_id}",
    response_model=ApiResponse[bool],
)
async def delete_agent_api(
        agent_id: int,
        db: DbSession,
        current_user_id: CurrentUserId,
):
    result = await delete_agent(
        db=db,
        agent_id=agent_id,
        user_id=current_user_id,
    )
    return ApiResponse(data=result)


@router.post(
    "/create_agent",
    response_model=ApiResponse[AgentResponse],
)
async def create_agent_api(
        db: DbSession,
        current_user_id: CurrentUserId,
        data: AgentCreate
):
    """创建智能体"""
    result = await create_agent(
        db=db,
        data=data,
        user_id=current_user_id,
    )
    return ApiResponse(data=result)


@router.post(
    "/{agent_id}/reset",
    response_model=ApiResponse[AgentResponse],
)
async def reset_agent_api(
        agent_id: int,
        current_user_id: CurrentUserId,
        db: DbSession):
    """恢复默认模板"""
    result = await reset_agent(
        db=db,
        agent_id=agent_id,
        user_id=current_user_id,
    )
    return ApiResponse(data=result)
