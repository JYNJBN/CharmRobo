from typing import TypedDict

from pydantic import SecretStr


class ModelRegistryObject(TypedDict):
    provider: str
    api_key: SecretStr
    base_url: str
    model_id: str
    protocol: str
