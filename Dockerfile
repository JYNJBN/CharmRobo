# 使用和你本地项目一致的 Python 3.10。
FROM python:3.10-slim-bookworm

# 用腾讯云 PyPI 镜像安装 uv。
# 原来是从 ghcr.io/astral-sh/uv 复制，但 ghcr.io 在国内不稳定（构建中途 EOF），
# 改成 pip 装同一个版本后，Docker 构建不再依赖 github 系域名，也就不再需要代理。
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/ uv==0.12.5

WORKDIR /app

# Python 日志立即输出到 Docker 控制台。
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# uv 安装时复制文件，避免 Docker 文件系统链接警告。
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# 线上服务器访问 PyPI 可能较慢，默认使用腾讯云 PyPI 镜像加快 Docker 构建。
ARG UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/
ENV UV_INDEX_URL=${UV_INDEX_URL}

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
