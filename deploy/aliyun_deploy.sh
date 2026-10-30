#!/usr/bin/env bash
# ============================================================
# 阿里云 ECS 一键部署（Ubuntu 22.04 / 24.04）
#   用法（在项目根目录）：
#     sudo bash deploy/aliyun_deploy.sh --with-models
#     sudo bash deploy/aliyun_deploy.sh --with-models --with-corpus
# 说明：
#   - 自动安装 Docker（若未安装）、配置镜像加速（仅在无 daemon.json 时）
#   - 前端在容器内构建（web/Dockerfile），无需本机 Node
#   - 模型默认放在 ./models，可通过 --with-models 自动下载（约 4.4GB）
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_MODELS=0
WITH_CORPUS=0
SKIP_SEED=0
WITH_SWAP=0
for arg in "$@"; do
  case "$arg" in
    --with-models) WITH_MODELS=1 ;;
    --with-corpus) WITH_CORPUS=1 ;;
    --skip-seed) SKIP_SEED=1 ;;
    --with-swap) WITH_SWAP=1 ;;
    *) echo "未知参数：$arg"; exit 2 ;;
  esac
done

log() { echo -e "\n\033[1;34m[deploy]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warning]\033[0m $*"; }

if [ "$(id -u)" -ne 0 ]; then
  warn "需要 root 权限安装/管理 Docker，正在通过 sudo 重新执行"
  exec sudo bash "$0" "$@"
fi

# ---------------------------------------------------------------- Docker
if ! command -v apt-get >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
  echo "本脚本的自动安装面向 Ubuntu/Debian（apt）。"
  echo "如使用 Alibaba Cloud Linux / CentOS，请先手动安装 docker 与 docker compose 插件，再运行本脚本。"
  exit 4
fi

if ! command -v docker >/dev/null 2>&1; then
  log "安装 Docker Engine 与 Compose 插件"
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  apt-get install -y docker.io docker-compose-v2 || apt-get install -y docker.io docker-compose-plugin
  systemctl enable --now docker
fi
if ! docker compose version >/dev/null 2>&1; then
  warn "未找到 docker compose 插件，尝试安装 compose-v2"
  apt-get install -y docker-compose-v2 || true
fi

if [ ! -f /etc/docker/daemon.json ]; then
  log "配置 Docker 镜像加速（阿里云/国内网络）"
  mkdir -p /etc/docker
  cat > /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": ["https://docker.m.daocloud.io", "https://dockerproxy.com"],
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "3"}
}
EOF
  systemctl restart docker
fi

if [ "$WITH_SWAP" -eq 1 ] && [ ! -f /swapfile ]; then
  log "创建 8GB swap（CPU 推理模型的保底内存）"
  fallocate -l 8G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=8192
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# ---------------------------------------------------------------- .env
if [ ! -f .env ]; then
  log "生成 .env（随机密码）"
  cp .env.example .env
  python3 - <<'PY'
import secrets, string, pathlib
p = pathlib.Path(".env")
s = p.read_text(encoding="utf-8")
alphabet = string.ascii_letters + string.digits
def pw(n): return "".join(secrets.choice(alphabet) for _ in range(n))
mapping = {
    "MYSQL_ROOT_PASSWORD=change-me-root-password": f"MYSQL_ROOT_PASSWORD={pw(24)}",
    "MYSQL_PASSWORD=change-me-db-password": f"MYSQL_PASSWORD={pw(24)}",
    "REDIS_PASSWORD=change-me-redis-password": f"REDIS_PASSWORD={pw(24)}",
    "MINIO_PASSWORD=change-me-minio-password": f"MINIO_PASSWORD={pw(24)}",
    "JWT_SECRET=please-change-this-to-a-random-64-chars-secret": f"JWT_SECRET={pw(64)}",
}
for old, new in mapping.items():
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
p.chmod(0o600)
PY
fi

if grep -q '^LLM_API_KEY=sk-please-replace-me$' .env; then
  warn "检测到 LLM_API_KEY 仍是占位符：问答生成会失败。请先编辑 .env 填入真实的 DeepSeek/通义 Key。"
  warn "可执行：nano .env   然后重新运行本脚本。"
  exit 3
fi

MYSQL_ROOT_PASSWORD="$(grep -E '^MYSQL_ROOT_PASSWORD=' .env | cut -d= -f2-)"
NGINX_PORT="$(grep -E '^NGINX_PORT=' .env | cut -d= -f2- || true)"
NGINX_PORT="${NGINX_PORT:-80}"

# ---------------------------------------------------------------- 模型
if [ "$WITH_MODELS" -eq 1 ]; then
  log "下载嵌入与重排模型（约 4.4GB，可断点续传）"
  python3 scripts/download_models.py --dest models --mirror https://hf-mirror.com
fi
if [ ! -d models/bge-m3 ] || [ ! -d models/bge-reranker-v2-m3 ]; then
  warn "缺少 models/bge-m3 或 models/bge-reranker-v2-m3，可加 --with-models 自动下载"
fi

# ---------------------------------------------------------------- 启动
log "构建并启动全部服务（前端在容器内构建）"
docker compose up -d --build --remove-orphans

log "等待 MySQL 就绪"
for i in $(seq 1 60); do
  if docker compose exec -T mysql mysqladmin ping -h 127.0.0.1 -uroot -p"$MYSQL_ROOT_PASSWORD" --silent >/dev/null 2>&1; then
    echo "  MySQL 已就绪"
    break
  fi
  sleep 5
done

if [ "$SKIP_SEED" -eq 0 ]; then
  log "初始化数据库结构"
  docker compose exec -T app python scripts/init_db.py
  log "写入种子数据"
  docker compose exec -T app python scripts/seed_data.py
fi

if [ "$WITH_CORPUS" -eq 1 ]; then
  if [ -f data/corpus/cards.jsonl ]; then
    log "灌入细粒度语料（360 张知识卡片，CPU 嵌入约 3~6 分钟）"
    docker compose exec -T app python scripts/ingest_corpus.py --cards data/corpus/cards.jsonl
  else
    warn "未找到 data/corpus/cards.jsonl，跳过语料灌入"
  fi
fi

log "健康检查"
sleep 5
curl -fsS "http://127.0.0.1:${NGINX_PORT}/health" && echo

PUBLIC_IP="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || echo '<服务器公网IP>')"
cat <<EOF

============================================================
部署完成
  前端：   http://${PUBLIC_IP}:${NGINX_PORT}/
  接口文档：http://${PUBLIC_IP}:${NGINX_PORT}/docs
  健康检查：http://${PUBLIC_IP}:${NGINX_PORT}/health
  默认账号：admin / Admin@2026     student_lin / Student@2026

常用命令：
  docker compose ps
  docker compose logs -f app worker
  docker compose exec app python scripts/experiment_compare.py --limit 60 --mode service
  docker compose restart app worker
============================================================
EOF
