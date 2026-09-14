from app.core.config import settings
from app.core.errors import BizError
from app.schemas.Model import ModelRegistryObject


def _build_model_registry() -> dict[str, ModelRegistryObject]:
    """构建可供用户选择的模型配置。

    MODEL_REGISTRY 的 key 是数据库和小程序使用的稳定标识；真实模型 ID、
    API Key 和 Base URL 始终只来自后端配置。
    """

    registry: dict[str, ModelRegistryObject] = {
        "doubao": {
            "provider": "ark",
            "api_key": settings.ark_api_key,
            "base_url": settings.ark_base_url,
            "model_id": settings.ark_doubao_model,
            "protocol": "chat_completions",
        },
    }

    if settings.ark_deepseek_model:
        registry["deepseek"] = {
            "provider": "ark",
            "api_key": settings.ark_api_key,
            "base_url": settings.ark_base_url,
            "model_id": settings.ark_deepseek_model,
            "protocol": "chat_completions",
        }

    if settings.qwen_api_key and settings.qwen_model:
        registry["qwen"] = {
            "provider": "qwen",
            "api_key": settings.qwen_api_key,
            "base_url": settings.qwen_base_url,
            "model_id": settings.qwen_model,
            "protocol": "chat_completions",
        }

    return registry


MODEL_REGISTRY = _build_model_registry()

# 给终端用户看的模型名（用于系统提示词和日志），key 与 MODEL_REGISTRY 保持一致。
MODEL_LABELS: dict[str, str] = {
    "doubao": "豆包",
    "deepseek": "DeepSeek",
    "qwen": "千问",
}


def describe_model(model_key: str | None) -> str:
    """返回模型的用户可读名称，未知 key 原样返回，兜底豆包。"""
    key = model_key or "doubao"
    return MODEL_LABELS.get(key, key)


def resolve_model(model_key: str | None) -> ModelRegistryObject:
    selected_key = model_key or "doubao"
    obj = MODEL_REGISTRY.get(selected_key)
    if obj is None:
        raise BizError("设备不存在", code=404)
    return obj
