# 使用和你本地项目一致的 Python 3.10。
FROM python:3.10-slim-bookworm

# 从 uv 官方镜像中复制 uv 命令。
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

# Python 日志立即输出到 Docker 控制台。
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# uv 安装时复制文件，避免 Docker 文件系统链接警告。
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# 先复制依赖文件，利用 Docker 缓存。
COPY pyproject.toml uv.lock ./

# 安装正式环境依赖，不安装 Ruff 等开发依赖。
RUN uv sync --frozen --no-dev --no-install-project

# 再复制项目代码。
COPY . .

RUN uv sync --frozen --no-dev

# 后面可以直接运行 uvicorn、alembic，不必再写 uv run。
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
