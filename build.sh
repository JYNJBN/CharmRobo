#!/usr/bin/env bash
# ============================================================
#  构建并重启 backend（生产环境一键脚本）
#
#  - 代理隧道在线  -> 构建时带上代理（HTTP_PROXY=127.0.0.1:17890）
#  - 隧道不可用    -> 自动回退为直连构建，不再阻塞
#
#  ⚠️ 现状说明（2026-09-18 实测）：
#  当前 Dockerfile 已不含 ghcr.io 等境外 registry 依赖，pip / uv 全部走
#  腾讯云镜像源，**构建其实已经不需要代理了**（实测腾讯源下载 16MB 仅 0.1s、
#  pip 装 uv 23.7MB 仅 3.9s）。而 registry 拉取（FROM python:3.10-slim-bookworm）
#  走的是 **dockerd 自身**的代理（docker.service.d/override.conf），
#  **不受本脚本的 BUILD_*_PROXY 影响** —— build args 只作用于容器内的 RUN。
#  这里保留探测逻辑，只是为将来万一又引入境外依赖时能自动兜住。
#
#  用法：  bash build.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="compose.prod.yml"
PROXY="http://127.0.0.1:17890"

# ⚠️ 探针必须用「小目标」。
# 千万别用 https://pypi.org/simple/ —— 那是 46MB 的根索引页，
# 配合 -m 5 必然超时，导致每次都误判「隧道不可用」，自动探测形同虚设。
PROBE="https://github.com"

echo "[build] 检测代理隧道 ${PROXY} ..."
if curl -fsS -m 8 -x "$PROXY" -o /dev/null "$PROBE" 2>/dev/null; then
    echo "[build] 隧道可用 -> 构建时带代理"
    export BUILD_HTTP_PROXY="$PROXY"
    export BUILD_HTTPS_PROXY="$PROXY"
else
    echo "[build] 隧道不可用 -> 直连构建（当前 Dockerfile 直连即可完成）"
    unset BUILD_HTTP_PROXY || true
    unset BUILD_HTTPS_PROXY || true
fi

echo "[build] docker compose build backend ..."
docker compose -f "$COMPOSE" build backend

echo "[build] docker compose up -d backend ..."
docker compose -f "$COMPOSE" up -d backend

echo "[build] 完成，当前状态："
docker compose -f "$COMPOSE" ps
