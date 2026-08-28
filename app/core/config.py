from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str="Charming Device API"
    debug: bool=True
    # PostgreSQL 配置
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str
    postgres_database: str = "charming"
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
    # 上下文和摘要配置，单位都是消息条数，不是问答轮数
    short_term_context_messages: int = 20
    summary_batch_messages: int = 6
    @property
    def alembic_database_url(self) -> str:
        return (
            f"postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )
settings = Settings()
