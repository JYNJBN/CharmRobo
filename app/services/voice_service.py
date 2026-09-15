from app.core.errors import BizError

VOICE_ALIAS = {
    "female_gentle": "zh_female_vv_uranus_bigtts",
    "male_steady": "zh_male_dayi_saturn_bigtts",
    "female_sweet": "zh_female_meilinvyou_saturn_bigtts",
}

VOICE_LABELS = {
    "female_gentle": "度禧禧·阳光女声",
    # 暂时只开放百度 4196 对应的音色；下面两个旧选项先不展示。
    # "male_steady": "度泽言·温暖男声",
    # "female_sweet": "度小柔·温柔女声",
}

# 百度流式 TTS 的发音人编号。数据库仍保存中立的 voice key，避免把某一家
# 厂商的 per/voice_id 直接写进 Agent；不同供应商继续使用各自的转换函数。
BAIDU_TTS_PER = {
    "female_gentle": "4196",
    "male_steady": "4179",
    "female_sweet": "6567",
}


def resolve_voice_id(voice_key: str | None) -> str:
    """把数据库保存的中立音色别名转换成厂商音色 ID。"""
    selected_key = voice_key or "female_gentle"
    voice_id = VOICE_ALIAS.get(selected_key)
    if voice_id is None:
        raise BizError("设备不存在", code=404)
    return voice_id


def describe_voice(voice_key: str | None) -> str:
    """返回给小程序显示的音色名称。"""
    selected_key = voice_key or "female_gentle"
    return VOICE_LABELS.get(selected_key, selected_key)


def resolve_baidu_tts_per(voice_key: str | None) -> str:
    """把 Agent 的中立音色 key 转成百度 TTS 的 per 发音人编号。"""

    selected_key = voice_key or "female_gentle"
    return BAIDU_TTS_PER.get(selected_key, BAIDU_TTS_PER["female_gentle"])
