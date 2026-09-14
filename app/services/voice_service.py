from app.core.errors import BizError

VOICE_ALIAS = {
    "female_gentle": "zh_female_vv_uranus_bigtts",
    "male_steady": "zh_male_dayi_saturn_bigtts",
    "female_sweet": "zh_female_meilinvyou_saturn_bigtts",
}

VOICE_LABELS = {
    "female_gentle": "温柔女声",
    "male_steady": "沉稳男声",
    "female_sweet": "甜美女声",
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
