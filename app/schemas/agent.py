from pydantic import BaseModel, Field


class AgentCreate(BaseModel):
    device_id: int
    template_id: int
    name: str | None = Field(default=None, min_length=1, max_length=32)
    system_prompt: str | None = Field(default=None, max_length=500)
    model_key: str | None = Field(default=None, max_length=32)
    voice: str | None = Field(default=None, max_length=32)


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=32)
    system_prompt: str | None = Field(default=None, max_length=500)
    model_key: str | None = Field(default=None, max_length=32)
    voice: str | None = Field(default=None, max_length=32)


class AgentModelOption(BaseModel):
    key: str
    name: str
    provider: str


class AgentVoiceOption(BaseModel):
    key: str
    name: str


class AgentOptionsResponse(BaseModel):
    models: list[AgentModelOption]
    voices: list[AgentVoiceOption]


class AgentCreate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=32)
    system_prompt: str | None = Field(default=None, max_length=500)
    model_key: str | None = Field(default=None, max_length=32)
    voice: str | None = Field(default=None, max_length=32)
    device_id: int


class AgentResponse(BaseModel):
    id: int
    device_id: int | None
    name: str
    system_prompt: str | None
    model_key: str
    voice: str
    is_system: bool
    source_agent_id: int | None
    active: bool = False
    # 是否为设备 fork 自默认模板的兜底副本（不可删除，由后端计算）
    is_default: bool = False
