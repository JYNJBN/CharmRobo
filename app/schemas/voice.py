from pydantic import BaseModel, Field


class VoiceChatRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
    )


class VoiceChatResponse(BaseModel):
    reply: str


class TTSRequest(BaseModel):
    """文字转语音请求。"""

    text: str = Field(
        min_length=1,
        max_length=1000,
        description="需要朗读的文字",
    )
    voice: str = Field(
        min_length=1,
        max_length=128,
        description="火山 TTS 音色 ID",
    )
