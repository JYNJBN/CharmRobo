from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Charming Device API"
    debug: bool = True
    # 日志级别：DEBUG / INFO / WARNING / ERROR。
    # 容器里用环境变量 LOG_LEVEL 覆盖，例如 LOG_LEVEL=DEBUG。
    # 业务代码用 logging.getLogger(__name__) 打的日志受它控制。
    log_level: str = "INFO"
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
    # llm配置
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_api_key: SecretStr
    ark_deepseek_model: str | None = None
    ark_doubao_model: str
    # 硬件语音链路固定使用的大模型，不随智能体的 model_key 变化，也不影响
    # 小程序端到端模型配置。当前走阿里百炼（千问）的 qwen3.7-flash ——
    # 供应商（api_key / base_url）在 api/v1/hardware_voice.py 里固定为 qwen。
    # ⚠️ 换供应商时下面两个值必须跟着换：model 必须是「目标那一家」托管的
    #    Model ID，否则会 InvalidEndpointOrModel.NotFound（方舟不托管千问，
    #    百炼也不托管 DeepSeek，反之亦然）。
    # 注意：这里刻意**不做** HARDWARE_DOUBAO_MODEL 旧变量名兼容。若服务器 .env
    # 里还留着旧变量，让它被忽略才对——否则旧值会盖掉新模型，表现为"改了没生效"。
    hardware_llm_model: str = "qwen3.7-flash"
    # 写进系统提示词的模型名。用户问"你是什么模型"时模型会照这个回答，
    # 换硬件模型时这里要同步改，否则会答成上一代模型。
    hardware_llm_label: str = "通义千问"
    qwen_api_key: SecretStr | None = None
    qwen_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    qwen_model: str | None = None
    # 阿里 Qwen Audio 实时语音测试链路。实时 WebSocket 地址需要按阿里
    # 控制台/工作空间配置；API Key 可单独配置，也可回退复用 qwen_api_key。
    qwen_audio_realtime_api_key: SecretStr | None = None
    qwen_audio_realtime_ws_url: str | None = None
    qwen_audio_realtime_model: str = "qwen-audio-3.0-realtime-plus"
    qwen_audio_realtime_voice: str = "longanqian"
    qwen_audio_realtime_max_history_turns: int = Field(
        default=20,
        ge=1,
        le=50,
    )
    # 语音key 模型配置
    volc_asr_api_key: SecretStr | None = None
    volc_asr_resource_id: str = "volc.bigasr.auc_turbo"
    volc_stream_asr_resource_id: str = "volc.seedasr.sauc.duration"
    volc_tts_api_key: SecretStr | None = None
    volc_tts_resource_id: str = "seed-tts-2.0"
    # 豆包实时语音模型 3.0（Seeduplex）全双工接口。
    # 使用独立可选 Key；未单独配置时，代码会回退复用 volc_tts_api_key，
    # 方便已有豆包语音应用先做开发测试。
    volc_duplex_api_key: SecretStr | None = None
    volc_duplex_ws_url: str = (
        "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue"
    )
    volc_duplex_model: str = "1.2.6.1"
    volc_duplex_voice: str = "zh_female_vv_jupiter_bigtts"
    # 硬件语音链路使用的百度 ASR/TTS。全部密钥设为可选，避免未部署硬件时
    # 影响原小程序 /api/v1/voice/stream 启动；真正收到硬件语音请求时再校验。
    # OAuth 地址供 ASR/TTS 使用各自的 API Key + Secret Key 换取 Access Token。
    baidu_oauth_url: str = "https://aip.baidubce.com/oauth/2.0/token"
    # 百度短语音识别标准版 RAW 接口：上传完整 16k/16bit/mono 裸 PCM。
    baidu_asr_url: str = "http://vop.baidu.com/server_api"
    baidu_asr_api_key: SecretStr | None = None
    baidu_asr_secret_key: SecretStr | None = None
    # 1537 为普通话近场模型；当前硬件与 ASR 输入均固定 16 kHz。
    baidu_asr_dev_pid: int = 1537
    baidu_asr_sample_rate: int = 16000
    # 百度流式文本在线合成 WebSocket：输入 LLM 短句，返回 MP3 二进制帧。
    baidu_tts_ws_url: str = (
        "wss://aip.baidubce.com/ws/2.0/speech/publiccloudspeech/v1/tts"
    )
    baidu_tts_api_key: SecretStr | None = None
    baidu_tts_secret_key: SecretStr | None = None
    # per=4196 是默认发音人；TTS 输出使用 aue=3 MP3 并降采样到 16 kHz。
    baidu_tts_per: str = "4196"
    baidu_tts_speed: int = 6
    baidu_tts_pitch: int = 5
    baidu_tts_volume: int = 5
    baidu_tts_sample_rate: int = 16000
    # 单轮网络上传总量上限，同时限制 PCM/Speex，防止异常设备无限占用内存。
    hardware_voice_max_audio_bytes: int = 2 * 1024 * 1024
    # 软件动作层：默认关闭，打开后才扫描本地音乐、请求天气或调用 LLM 工具路由。
    #
    # ★ 开关语义：music_enable / weather_enable 是「能力总开关」，不只是关键词开关。
    #   关闭后该能力在【两条路径】上都不可用 —— 关键词路径和 LLM 路由路径都会
    #   回落闲聊。enable=False 时不要指望路由还能用这个能力。
    #   （历史坑：answer_intent() 曾不检查 weather_enable，导致天气开关关掉后
    #    路由路径照样真查真播，同一个开关在两条路径上表现不一致。）
    #   关闭时也不会谎报成资源不存在 —— 不会说「本地曲库里还没有这首歌」，
    #   而是正常走闲聊。
    #
    # ★ 两种玩法（二选一，不要同时开）：
    #   A. 关键词层 + 独立路由：weather_enable=true + tool_router_enable=true。
    #      关键词层负责「XX天气/会不会下雨」这类显式问句（_CITY_IDS 能精确识别
    #      16 个城市、_parse 能提日期），路由层负责「拿伞/下雪/穿外套」这类不在
    #      词表里的口语说法。两层互补、比只开路由准，但每轮固定多一次 LLM 调用。
    #   B. 工具直连（当前采用）：weather_enable=true + hardware_tool_calling_enable=true，
    #     并把 tool_router_enable 关掉。整条 resolve() 短路，不再有独立判断。
    #   两个能力开关如果都关着，工具路由就没有任何可用动作了，resolve() 会直接
    #   短路跳过那次 LLM 调用（省掉每轮约 7 秒）。
    music_enable: bool = False
    music_dir: str = "music"
    music_chunk_bytes: int = 4096
    weather_enable: bool = False
    weather_default_city: str = "深圳"
    qweather_api_host: str = "https://devapi.qweather.com"
    qweather_api_key: SecretStr | None = None
    qweather_timeout_sec: float = 5.0
    tool_router_enable: bool = False
    # 这个超时是真正生效的：resolve() 里用 asyncio.wait_for 包住了整个
    # _tool_decide（建客户端 + 发请求 + 解析 JSON），超时就 return None 回落
    # 闲聊，绝不阻塞对话。12.0 是按实测校准出来的：doubao-seed-2-0-lite
    # 上 14 条样本 min=3.34s / 中位=5.92s / avg=6.96s / p90=10.16s / max=20.28s。
    # 设成 4.0 会让绝大多数请求超时降级、工具能力形同虚设；12.0 只切掉极少数
    # 离群值。注意路由本身的开销是每轮固定多一次 LLM 调用，这个延迟是这条
    # 路径的固有成本，不是超时值能解决的。
    tool_router_timeout_sec: float = 12.0
    tool_router_min_confidence: float = 0.65
    tool_router_music_min_confidence: float = 0.75
    # ★ 工具直连模式：把工具定义挂到**主对话那一次** LLM 请求上，由模型自己
    #   决定「直接回答」还是「发起 get_weather 调用」。开启后 resolve() 整体
    #   短路，关键词词表和 _tool_decide 都不再参与。
    #
    #   和 tool_router_enable 是互斥的两套方案：
    #     - tool_router_enable：关键词层 → 独立的一次 LLM 路由判断 → 执行工具
    #     - hardware_tool_calling_enable：只有一次 LLM 调用，判断和执行合在里面
    #
    #   收益：误报（"我觉得今天天气好差"被截胡去播天气）和漏检（拿伞/下雪/
    #   穿外套不在词表）两类问题一并消失，因为根本不用维护词表了。
    #   代价：每轮约 1 秒（关键词层命中时是 0.2 秒），但远好于独立路由的 3~20 秒。
    #
    #   实测（deepseek-v4-1-flash-260910 + thinking=disabled，2026-09-17）：
    #     · 模型决定调工具时，流里**只有 tool_calls、一个 content 分片都没有**
    #       ⇒ 可以边收边判模式，不存在"先吐半句话再改口"的问题
    #     · 调工具 0.94~1.11 秒；闲聊首字 0.72~0.79 秒
    #     · thinking=disabled 与 tools 可以共存
    #   注意：流式与非流式在"没提城市"时行为不完全一致（流式倾向反问城市），
    #   所以工具描述里写明了"留空即用设备默认城市"。
    hardware_tool_calling_enable: bool = False
    # 上传文件保存目录（相对项目根目录），通过 /static 提供访问
    upload_dir: str = "uploads"
    # 火山 STT 读取临时音频文件时使用的公网基础地址，例如 ngrok HTTPS 地址。
    public_base_url: str | None = None
    # 上下文和摘要配置，单位都是消息条数，不是问答轮数
    short_term_context_messages: int = 6
    summary_batch_messages: int = 6
    # 向量模型配置。变量名刻意做成**供应商无关**的 —— 换供应商只改 .env 三行，
    # 代码一行不用动（embedding.py 用的是标准 OpenAI 兼容接口）。
    # API Key 可单独配置，也可回退复用百炼的 qwen_api_key（与
    # qwen_audio_realtime_api_key 同一个约定，见 resolved_embedding_api_key）。
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "qwen3.7-text-embedding-flash"
    # ★ 换模型时这个数字必须跟着改，因为 Milvus collection 是照它建的：
    #   维度一旦变了，就只能重建 collection 并重新 embedding 全部历史数据 ——
    #   改不动就说明这个模型不能换。当前 1024 与 qwen3.7-text-embedding-flash
    #   的默认维度一致（该模型还支持 768/512/256，靠 dimensions 参数指定）。
    #   ⚠️ 维度相同**不代表能混用**：向量空间不同，换模型后旧向量全部作废，
    #      必须清空 collection 重新灌数据，否则相似度分数无意义。
    embedding_dimension: int = 1024
    embedding_metric_type: str = "COSINE"
    # 长期记忆最低余弦相似度。当前 Milvus 使用 COSINE，分数越大越相关；
    # 低于该值的 Top-K 候选不会回传给模型，避免不相关摘要干扰回答。
    memory_min_similarity: float = Field(default=0.55, ge=-1, le=1)
    # 当前只维护对话摘要这一类向量数据
    milvus_summary_collection: str = "conversation_summary_v1"
    # Milvus 服务配置
    milvus_uri: str = "http://127.0.0.1:19530"

    @property
    def alembic_database_url(self) -> str:
        return (
            f"postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}"
            f"/{self.postgres_database}"
        )

    @property
    def resolved_embedding_api_key(self) -> SecretStr | None:
        """向量模型实际使用的 Key；未单独配置时回退到百炼的 qwen_api_key。

        注意这里**只回退到百炼**，不回退硅基流动。历史坑：一旦代码里留着
        「新变量为空就读旧供应商变量」的逻辑，服务器 .env 里没删干净的老值
        就会悄悄盖掉新配置，表现为「明明改了却还是老模型」。
        """
        return self.embedding_api_key or self.qwen_api_key


settings = Settings()
