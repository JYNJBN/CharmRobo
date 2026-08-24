from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str="Charming Device API"
    debug: bool=True
    # mysql 配置
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str
    mysql_database: str = "charming"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # redis 配置
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: SecretStr | None = None
    redis_db: int = 0
    # 微信小程序的配置
    wechat_app_id: str
    wechat_app_secret: SecretStr
    # jwt配置
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080
    # 语音key 模型配置
    ark_api_key: SecretStr
    ark_model: str
    volc_asr_api_key: SecretStr | None = None
    volc_asr_resource_id: str = "volc.bigasr.auc_turbo"
    volc_stream_asr_resource_id: str = "volc.seedasr.sauc.duration"
    volc_tts_api_key: SecretStr | None = None
    volc_tts_resource_id: str = "seed-tts-2.0"
    # 上传文件保存目录（相对项目根目录），通过 /static 提供访问
    upload_dir: str = "uploads"
    # 火山 STT 读取临时音频文件时使用的公网基础地址，例如 ngrok HTTPS 地址。
    public_base_url: str | None = None
    @property
    def alembic_database_url(self) -> str:
        return (
            f"mysql+pymysql://"
            f"{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}"
            f"/{self.mysql_database}?charset=utf8mb4"
        )
settings = Settings()
