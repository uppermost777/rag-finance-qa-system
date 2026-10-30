# ============================================================
# 应用镜像（多阶段构建）
# 说明：嵌入与重排模型体积很大（GB 级），不打进镜像，
#       运行时通过 docker-compose 把宿主机 ./models 挂载到 /models。
# ============================================================
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 先装依赖，利用 Docker 层缓存
# 国内构建加速：apt / pip 走镜像源；如需官方源，构建时传
#   --build-arg APT_MIRROR=deb.debian.org --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# torch 源：官方 download.pytorch.org 在国内常只有几 KB/s（200MB 要数小时）。
# 阿里云 ECS 可传内网 wheel 直链，实测 20MB/s+，例如：
#   --build-arg TORCH_WHEEL_URL=http://mirrors.cloud.aliyuncs.com/pytorch-wheels/cpu/torch-2.9.1%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_URL=""
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# 编译工具链：本项目的依赖全部有 manylinux wheel 或为纯 Python 包，无需 gcc。
# 在 4GB 内存机器上安装 gcc-14 会让 apt 峰值内存冲到 3GB+ 并触发 OOM，
# 故默认不装。如后续新增需源码编译的依赖，传 --build-arg INSTALL_BUILD_TOOLS=1。
ARG INSTALL_BUILD_TOOLS=0
COPY requirements.txt .
RUN if [ "${INSTALL_BUILD_TOOLS}" = "1" ]; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources ; \
        else \
            sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list ; \
        fi \
        && apt-get update \
        && apt-get install -y --no-install-recommends gcc build-essential ; \
    fi \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -i ${PIP_INDEX_URL} -r requirements.txt \
    && if [ -n "${TORCH_WHEEL_URL}" ]; then \
           /opt/venv/bin/pip install -i ${PIP_INDEX_URL} "${TORCH_WHEEL_URL}" ; \
       else \
           /opt/venv/bin/pip install --index-url ${TORCH_INDEX_URL} "torch>=2.4,<3" ; \
       fi \
    && /opt/venv/bin/pip install -i ${PIP_INDEX_URL} "sentence-transformers>=3.0,<7" \
    && rm -rf /var/lib/apt/lists/*

# Milvus Lite：pymilvus 3.x 不再默认捆绑，本地嵌入式模式（MILVUS_LITE_PATH）必须显式安装。
# 单独成层，便于上面耗时最长的依赖层保持缓存。
RUN /opt/venv/bin/pip install -i ${PIP_INDEX_URL} "pymilvus[milvus_lite]==3.0.2" aiosqlite

# ------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=Asia/Shanghai

WORKDIR /app

# torch / Milvus Lite 运行期需要 libgomp1（OpenMP）与 libstdc++；
# python:slim 基础镜像不自带 libgomp1，缺失会导致 import torch 直接失败。
ARG APT_MIRROR=mirrors.aliyun.com
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources ; \
    else \
        sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list ; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY migrations ./migrations
COPY alembic.ini ./
COPY .env.example ./.env.example

RUN mkdir -p /app/logs /app/uploads

EXPOSE 8000

# 健康检查：走 /health 探活（依赖不可用时仍返回 200，由 data 字段反映）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4)" || exit 1

# worker 数走环境变量：默认 2（生产）；低配机（4GB）设 UVICORN_WORKERS=1，
# 否则每个 worker 会各自加载一份 bge-m3，多占约 2.9GB 内存。
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
