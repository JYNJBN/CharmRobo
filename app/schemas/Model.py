from typing import TypedDict

from pydantic import SecretStr


class ModelRegistryObject(TypedDict):
    provider: str
    api_key: SecretStr
    base_url: str
    model_id: str
    protocol: str
    # OpenAI 兼容接口之外的厂商请求参数；例如方舟关闭思考时使用 disabled。
    thinking_type: str | None
