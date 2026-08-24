# Charming Device API

Charming 微信小程序和硬件设备的 FastAPI 后端。

当前后端负责：

- 微信小程序用户登录和 JWT 认证；
- 用户资料和头像；
- BLE 配网过程中的一次性设备绑定码；
- 硬件首次初始化和用户设备绑定；
- 查询用户设备、解绑设备；
- 后续设备 WebSocket、STT、AI、TTS 服务的基础框架。

## 技术栈

| 技术 | 作用 |
|---|---|
| FastAPI | HTTP API 和后续 WebSocket 服务 |
| Pydantic | 请求校验、响应结构和配置读取 |
| SQLAlchemy 2.x Async | 异步操作 MySQL |
| Alembic | 数据库表结构迁移 |
| MySQL 8.4 | 永久保存用户、设备和绑定关系 |
| Redis 7 | 保存绑定码、在线状态等临时数据 |
| PyJWT | 用户 Access Token |
| Docker Compose | 本地启动 MySQL 和 Redis |
| uv | Python、虚拟环境和依赖管理 |
| Ruff | Python 静态检查 |

项目要求 Python 3.10 或更高版本。

---

## 项目目录

```text
cml_websit_backend/
├── app/                           # FastAPI 应用代码
│   ├── main.py                    # 应用入口和生命周期
│   ├── api/                       # HTTP 接口层
│   │   ├── dependencies.py        # 数据库、Redis、当前用户依赖
│   │   ├── health.py              # 健康检查接口
│   │   └── v1/                    # 第一版 API
│   │       ├── router.py          # 汇总所有 v1 Router
│   │       ├── auth.py            # 微信登录接口
│   │       ├── users.py           # 用户接口
│   │       ├── device.py          # 小程序设备和硬件设备接口
│   │       └── upload.py          # 文件上传接口
│   ├── core/                      # 全局基础设施和安全工具
│   │   ├── config.py              # 从 .env 读取配置
│   │   ├── database.py            # SQLAlchemy Engine、Session、Base
│   │   ├── redis.py               # Redis 客户端和连接池
│   │   ├── security.py            # 用户 JWT 生成
│   │   ├── device_security.py     # 设备密钥摘要和校验
│   │   └── exception_handlers.py  # 全局异常响应格式
│   ├── models/                    # SQLAlchemy ORM 数据库模型
│   │   ├── user.py                # user 表
│   │   ├── device.py              # device 表
│   │   ├── user_device.py         # 用户设备关系表
│   │   └── __init__.py            # 集中导入 Model，供 Alembic 扫描
│   ├── schemas/                   # Pydantic Request/Response
│   │   ├── common.py              # ApiResponse 通用响应模型
│   │   ├── user.py                # 用户请求和响应结构
│   │   └── device.py              # 设备请求和响应结构
│   ├── services/                  # 业务逻辑和数据库操作
│   │   ├── user_service.py        # 用户业务
│   │   ├── device_service.py      # 设备初始化、查询和解绑业务
│   │   └── device_binding_service.py # Redis 绑定码业务
│   ├── integrations/              # 第三方服务适配层
│   │   ├── ai.py                  # 大模型接口
│   │   ├── stt.py                 # 语音转文字服务
│   │   ├── tts.py                 # 文字转语音服务
│   │   └── device.py              # 外部 IoT 平台接口（如有）
│   └── alembic/                   # Alembic 迁移环境
│       ├── env.py                 # 数据库 URL 和 target_metadata
│       ├── script.py.mako         # 迁移文件模板
│       └── versions/              # 每次表结构修改的迁移文件
├── tests/                         # 自动化测试
├── uploads/                       # 本地上传文件，不提交 Git
├── .env                           # 本地真实配置，不提交 Git
├── .env.example                   # 环境变量示例
├── alembic.ini                    # Alembic 命令配置
├── docker-compose.yml             # MySQL、Redis 容器
├── pyproject.toml                 # 项目信息和依赖
├── uv.lock                        # 锁定后的依赖版本
└── README.md                      # 本文档
```

---

## 各层职责

### `app/main.py`：应用入口

`main.py` 只负责组装应用：

- 创建 `FastAPI` 对象；
- 启动时检查 Redis；
- 关闭时释放 Redis 连接池；
- 注册健康检查和 `/api/v1` 路由；
- 注册全局异常处理；
- 挂载 `/static` 静态文件目录。

不要在 `main.py` 中编写 SQL、用户登录、设备绑定或 AI 调用。

### `app/api/`：接口层

相当于 Spring Boot 的 Controller。

负责：

- 定义 URL 和 HTTP 方法；
- 接收 Pydantic 请求模型；
- 注入数据库、Redis 和当前用户；
- 调用 Service；
- 把结果封装成 `ApiResponse`；
- 把业务异常转换成 HTTP 状态码。

接口层不要堆积复杂 SQL 和业务判断。

### `app/api/dependencies.py`：依赖注入

项目中的常用依赖别名：

```python
DbSession = Annotated[AsyncSession, Depends(get_db)]
RedisClient = Annotated[Redis, Depends(get_redis)]
CurrentUserId = Annotated[int, Depends(get_current_user_id)]
```

路由使用：

```python
async def example_api(
    db: DbSession,
    redis: RedisClient,
    current_user_id: CurrentUserId,
):
    ...
```

FastAPI 会自动创建和注入所需对象。

### `app/core/`：基础设施层

保存全项目公用的底层能力：

- 配置；
- 数据库连接；
- Redis 连接；
- JWT；
- 设备密钥；
- 全局异常处理。

`core` 不应该依赖具体业务 Router。

### `app/models/`：数据库 ORM

Model 有两个用途：

1. SQLAlchemy 使用它查询和更新数据库；
2. Alembic 读取 `Base.metadata` 生成迁移。

Model 不是只给 Alembic 使用的。

当前表：

```text
user
    微信用户和个人资料

device
    设备 SN、型号、版本、密钥摘要和状态

user_device
    用户与设备的绑定关系、别名和角色
```

设备身份和设备归属是两件事：

```text
device_secret → 证明是哪台硬件
user_device   → 说明设备属于哪个用户
```

### `app/schemas/`：请求和响应 DTO

相当于 Java 的 Request、Response、DTO。

Schema 决定：

- 接口接收哪些字段；
- 字段类型和长度；
- 哪些字段必填；
- 接口返回哪些字段。

Schema 不等于数据库表。

例如硬件 Bootstrap 请求需要：

```text
device_sn
device_secret
bind_ticket
product_key
firmware_version（可选）
hardware_version（可选）
```

### `app/services/`：业务逻辑层

相当于 Spring Boot 的 Service。

负责：

- SQLAlchemy 查询和写入；
- 业务规则；
- MySQL 事务；
- Redis 和 MySQL 组合操作；
- 调用第三方 Integration。

固定调用方向：

```text
Router → Service → Model / Redis / Integration
```

不要让 Model 反向导入 Router。

### `app/integrations/`：第三方服务适配层

第三方 SDK 和 HTTP 请求放在这里：

```text
AI 大模型
STT 语音识别
TTS 语音合成
外部 IoT 平台
```

以后更换云服务供应商时，优先修改 Integration，不要让第三方 SDK 散落在 Router 中。

### `app/alembic/`：数据库迁移

Alembic 负责数据库结构版本，不负责日常数据查询。

流程：

```text
修改 SQLAlchemy Model
→ 生成迁移文件
→ 人工检查迁移
→ upgrade head
→ Navicat 检查实际结构
```

---

## MySQL、Redis 和进程内存分工

### MySQL：永久数据

```text
用户
设备
用户设备关系
设备密钥摘要
后续聊天和历史数据
```

### Redis：临时共享状态

```text
一次性设备绑定码
后续设备在线状态和心跳 TTL
认证失败次数
接口限流
短期会话状态
```

### FastAPI 进程内存

```text
当前进程的 WebSocket 对象
连接管理器字典
短生命周期运行对象
```

Redis 不是 MySQL 的替代品。

---

## 当前设备绑定流程

当前使用“硬件生成永久密钥”方案。

```text
1. 硬件生成并保存 device_sn + device_secret
2. 小程序通过 BLE 只读取 device_sn
3. 小程序携带用户 JWT 请求一次性 bind_ticket
4. Redis 保存 bind_ticket → user_id + device_sn，TTL 300 秒
5. 小程序通过 BLE 把 Wi-Fi + bind_ticket 给硬件
6. 硬件连接 Wi-Fi
7. 硬件直接请求 POST /api/v1/device/bootstrap
8. 后端只保存 device_secret_hash
9. 后端同一事务写入 device + user_device
10. MySQL 成功后删除 Redis Ticket
```

小程序不读取、不保存 `device_secret`。

硬件 Bootstrap 请求：

```json
{
  "device_sn": "CHM-A1-20260819-000001",
  "device_secret": "硬件生成并保存在NVS中的永久密钥",
  "bind_ticket": "小程序通过BLE给硬件的一次性绑定码",
  "product_key": "charming-speaker-a1",
  "firmware_version": "1.0.0",
  "hardware_version": "1.0"
}
```

---

## 当前 API

应用启动后访问：

```text
Swagger UI: http://127.0.0.1:8001/docs
OpenAPI:    http://127.0.0.1:8001/openapi.json
```

### 公开接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 健康检查 |
| POST | `/api/v1/auth/login` | 微信登录 |
| POST | `/api/v1/auth/login/phone` | 微信手机号快捷登录/注册 |
| POST | `/api/v1/device/bootstrap` | 硬件首次初始化和绑定 |

### 用户 JWT 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/v1/users/info` | 获取当前用户资料 |
| PATCH | `/api/v1/users/info` | 修改当前用户资料 |
| POST | `/api/v1/upload/avatar` | 上传头像 |
| POST | `/api/v1/device/create/tickets` | 申请设备绑定码 |
| GET | `/api/v1/device` | 查询当前用户绑定的设备 |
| DELETE | `/api/v1/device/{device_id}` | 解绑设备，不删除硬件记录 |

具体以 `/docs` 当前展示为准。

---

## 本地环境配置

复制 `.env.example` 为 `.env`，填写本地配置。

示例字段：

```env
APP_NAME=CharmingDeviceAPI
DEBUG=true

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=change_me
MYSQL_DATABASE=charming

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=change_me
REDIS_DB=0

WECHAT_APP_ID=change_me
WECHAT_APP_SECRET=change_me

JWT_SECRET_KEY=change_me_to_a_long_random_secret
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080

UPLOAD_DIR=uploads
```

注意：

- `.env` 包含真实密钥，不能提交 Git；
- `.env.example` 只能放示例值；
- FastAPI 在电脑本机运行时使用 `MYSQL_HOST=127.0.0.1`、`REDIS_HOST=127.0.0.1`；
- 如果 FastAPI 也放进 Compose，Host 应改成 Compose 服务名 `mysql` 和 `redis`。

---

## 安装和启动

### 1. 安装依赖

```powershell
uv sync
```

### 2. 启动 MySQL 和 Redis

```powershell
docker compose up -d mysql redis
```

查看状态：

```powershell
docker compose ps
```

### 3. 执行数据库迁移

```powershell
uv run alembic upgrade head
```

### 4. 启动 FastAPI

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

如果需要局域网设备访问开发机：

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

生产硬件应通过 HTTPS 公网域名请求后端，不应长期依赖开发机局域网地址。

---

## Alembic 常用命令

修改 Model 后生成迁移：

```powershell
uv run alembic revision --autogenerate -m "describe the change"
```

检查生成的迁移文件后执行：

```powershell
uv run alembic upgrade head
```

查看当前版本：

```powershell
uv run alembic current
```

查看历史：

```powershell
uv run alembic history
```

回退一个版本：

```powershell
uv run alembic downgrade -1
```

不要为了同步 Model 直接删除已有正式表。

---

## SQLAlchemy 查询结果常用方式

只查询一个 ORM Model：

```python
result = await db.execute(select(Device))
devices = result.scalars().all()
```

查询两个 ORM Model：

```python
result = await db.execute(
    select(Device, UserDevice).join(
        UserDevice,
        UserDevice.device_id == Device.id,
    )
)

for device, binding in result.all():
    ...
```

查询指定字段并转换成字典式结果：

```python
result = await db.execute(
    select(
        Device.id.label("device_id"),
        Device.device_sn,
        UserDevice.alias,
    ).join(
        UserDevice,
        UserDevice.device_id == Device.id,
    )
)

rows = result.mappings().all()
```

---

## 统一响应格式

成功响应使用：

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

路由声明：

```python
response_model=ApiResponse[SomeResponse]
```

表示 `data` 必须符合 `SomeResponse`。

业务错误通过 `HTTPException` 交给全局处理器：

```json
{
  "code": 409,
  "message": "设备已经被其他用户绑定",
  "data": null
}
```

不要把数据库原始异常、密钥或第三方云服务敏感响应直接返回客户端。

---

## 新增模块的标准步骤

以新增“设备反馈”为例：

```text
1. models/feedback.py
   定义数据库表

2. schemas/feedback.py
   定义 FeedbackCreate、FeedbackResponse

3. services/feedback_service.py
   编写查询、创建和业务规则

4. api/v1/feedback.py
   定义 HTTP 路由

5. api/v1/router.py
   include_router(feedback_router)

6. Alembic
   生成并执行数据库迁移

7. tests/test_feedback.py
   编写自动化测试
```

---

## 代码检查

运行 Ruff：

```powershell
uv run ruff check app tests
```

自动修复安全的格式问题：

```powershell
uv run ruff check app tests --fix
```

提交代码前至少确认：

```text
Ruff 通过
Alembic 处于 head
MySQL 和 Redis 健康
Swagger 请求响应正确
.env 未被 Git 跟踪
```

---

## 后续目录规划

设备绑定稳定后，建议增加：

```text
app/
└── websocket/
    ├── __init__.py
    ├── device_endpoint.py       # /ws/devices
    ├── device_auth.py           # SN + Secret 认证
    ├── connection_manager.py    # 在线连接管理
    ├── protocol.py              # JSON 控制消息和二进制音频协议
    └── types.py                 # DevicePrincipal 等内部类型
```

后续 WebSocket 第一条认证消息：

```json
{
  "type": "auth",
  "device_sn": "CHM-A1-20260819-000001",
  "device_secret": "硬件NVS中的永久密钥"
}
```

认证成功后再允许心跳、状态上报、音频、STT、AI 和 TTS。

---

## 当前待办

- [ ] JWT 解码后检查用户仍然存在且 `deleted=0`；
- [ ] 增加修改设备别名接口；
- [ ] 完善解绑和换绑测试；
- [ ] 为设备 Bootstrap 编写自动化测试；
- [ ] 实现设备 WebSocket 认证和连接管理；
- [ ] 使用 Redis TTL 保存设备在线状态；
- [ ] 接入流式 STT、AI 和 TTS；
- [ ] 增加对话和设备历史数据表；
- [ ] 正式生产时增加合法 SN 清单或设备初始凭证。

第一版不需要拆成微服务。当前单个 FastAPI 项目足以完成用户、设备、WebSocket 和云端 AI 语音链路。
