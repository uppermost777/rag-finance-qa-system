#!/usr/bin/env bash
# ============================================================
# 一键启动脚本（Linux / macOS / WSL）
#   用法：bash scripts/quickstart.sh
# 步骤：准备 .env → 检查模型目录 → 构建镜像 → 启动容器 → 建表 → 灌种子数据
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] 准备环境变量文件"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  已生成 .env，请检查其中的数据库/Redis 密码与大模型 API Key 后重新运行本脚本"
  exit 0
else
  echo "  .env 已存在，跳过"
fi

echo "[2/6] 检查本地模型目录 ./models"
for model in bge-m3 bge-reranker-v2-m3; do
  if [ ! -d "models/$model" ]; then
    echo "  警告：缺少 models/$model，入库与重排功能将不可用"
    echo "  下载方式（任选其一）："
    echo "    huggingface-cli download BAAI/$model --local-dir models/$model"
    echo "    modelscope download --model BAAI/$model --local_dir models/$model"
  fi
done

echo "[3/6] 构建并启动容器（首次约需 5~10 分钟）"
docker compose up -d --build

echo "[4/6] 等待 MySQL 健康"
for _ in $(seq 1 60); do
  if docker compose exec -T mysql mysqladmin ping -h 127.0.0.1 -u"${MYSQL_USER:-finance}" -p"${MYSQL_PASSWORD:-}" --silent >/dev/null 2>&1; then
    echo "  MySQL 就绪"
    break
  fi
  sleep 5
done

echo "[5/6] 初始化数据库表结构"
docker compose exec -T app python scripts/init_db.py

echo "[6/6] 灌入种子数据（1 管理员 / 2 学生 / 6 知识库 / 10 条风控规则）"
docker compose exec -T app python scripts/seed_data.py

echo
echo "启动完成！"
echo "  前端页面：http://localhost:${NGINX_PORT:-80}"
echo "  接口文档：http://localhost:${NGINX_PORT:-80}/docs"
echo "  健康检查：http://localhost:${NGINX_PORT:-80}/health"
echo "  默认账号：admin / Admin@2026（管理员）　student_lin / Student@2026（学生）"
