# 阿里云 ECS 部署指南

> 适用：Ubuntu 22.04 / 24.04（推荐）。Alibaba Cloud Linux / CentOS 需先手动安装 Docker。
> 一键脚本：`deploy/aliyun_deploy.sh`

## 1. 服务器配置建议

| 项目 | 建议 |
|---|---|
| 实例规格 | **4 vCPU / 16 GB 内存**（最低 8 GB，需加 swap）；模型在 CPU 上推理，内存是主要瓶颈 |
| 系统盘 | 100 GB ESSD（镜像约 6 GB + 模型 4.4 GB + 数据） |
| 镜像 | Ubuntu 22.04 LTS 64 位 |
| 安全组 | 放行 **22**（SSH）、**80**（HTTP）；如需直连后端再放行 8000 |
| 公网 IP | 建议分配公网 IPv4 或绑定 EIP |

模型说明：bge-m3（2.17GB）与 bge-reranker-v2-m3（2.17GB）在 CPU 上推理，
16GB 内存可同时容纳 Milvus、MySQL、两个模型与 Python 进程；8GB 建议加 swap（脚本支持 `--with-swap`）。

## 2. 上传项目

### 方式 A：从 Windows 一键上传（推荐）

在**本项目所在 Windows 机器**的项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File deploy\upload_from_windows.ps1 `
  -Server <ECS公网IP> -User root -RemotePath /opt/finance-qa
```

脚本会排除 `.venv`、`node_modules`、`models`、日志，打包上传并解压到 `/opt/finance-qa`。
首次连接会提示确认主机指纹，输入 `yes`。

### 方式 B：git

```bash
sudo mkdir -p /opt && cd /opt
git clone <你的仓库地址> finance-qa
cd finance-qa
```

## 3. 配置环境变量

```bash
cd /opt/finance-qa
cp .env.example .env
nano .env
```

至少修改：

```ini
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的Key
LLM_MODEL=deepseek-chat

MYSQL_ROOT_PASSWORD=改成强密码
MYSQL_PASSWORD=改成强密码
REDIS_PASSWORD=改成强密码
MINIO_PASSWORD=改成强密码
JWT_SECRET=改成64位随机串
```

> 若使用一键脚本且 `.env` 不存在，脚本会用 Python 生成随机密码，但 **LLM_API_KEY 必须自己填**，
> 否则脚本会中止并提示。

## 4. 部署

```bash
sudo bash deploy/aliyun_deploy.sh --with-models --with-corpus --with-swap
```

脚本依次完成：

1. 安装 Docker Engine 与 compose 插件（Ubuntu）；
2. 配置 Docker 镜像加速（仅在 `/etc/docker/daemon.json` 不存在时）；
3. 可选创建 8GB swap；
4. 下载 bge-m3 与 bge-reranker-v2-m3（`--with-models`，断点续传）；
5. `docker compose up -d --build`（**前端在 `web/Dockerfile` 的 Node 阶段构建**，无需本机 Node）；
6. 等待 MySQL 健康 → 建表 → 灌种子数据；
7. 可选灌入 360 张细粒度知识卡片（`--with-corpus`）；
8. 健康检查并打印访问地址。

首次构建需要拉取 MySQL/Redis/Milvus/MinIO/etcd/Node/Nginx 镜像，约 5~10 分钟。

## 5. 验证

```bash
docker compose ps                     # 8 个服务应为 running / healthy
curl http://127.0.0.1/health          # code=0, mysql/redis/milvus 应为 true
docker compose logs -f app worker     # 查看启动日志
```

浏览器访问：

- 前端：`http://<公网IP>/`
- 接口文档：`http://<公网IP>/docs`
- 健康检查：`http://<公网IP>/health`

默认账号：`admin / Admin@2026`（管理员）、`student_lin / Student@2026`（学生）。
**上线前请在数据库中修改默认密码。**

功能验证顺序：登录 → 知识库管理上传 1 篇 PDF/TXT → 等待进度 100% → 回到对话页提问
→ 检查答案下方"参考来源"是否有文件名与页码 → 试问"帮我推荐一只稳赚的基金"应被拦截。

## 6. 跑论文实验

```bash
# 按分类检索（生产配置）
docker compose exec app python scripts/experiment_compare.py --limit 60 --mode service --run-id cloud_fix

# 全库检索（375 切片）
docker compose exec app python scripts/experiment_compare.py --limit 60 --mode service --kb-scope all --run-id cloud_all

# RAGAS 官方指标（10 条跨类）
docker compose exec app python scripts/run_ragas_eval.py --limit 10 --stride 6 --run-id cloud_ragas
```

结果在容器的 `/app/app/evaluation/results/`，映射到宿主机 `app/evaluation/results/`：

```bash
ls -lh app/evaluation/results/
```

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 拉取镜像超时 / TLS 错误 | 确认 `/etc/docker/daemon.json` 镜像加速生效：`sudo systemctl restart docker`；或改用阿里云容器镜像服务 ACR |
| Milvus 反复重启、OOM | 内存不足：升级到 16GB 或加 swap（`--with-swap`）；也可先停 worker 释放内存 |
| `port 80 already in use` | 修改 `.env` 的 `NGINX_PORT=8080` 后 `docker compose up -d` |
| 前端 404 / 空白 | 检查 `docker compose logs nginx`；确认 `web/Dockerfile` 构建成功 |
| 问答报 502 | `docker compose logs app`；多为 LLM Key 无效、Milvus 未就绪或模型路径错误 |
| 上传后一直 pending | worker 未启动：`docker compose logs worker`；确认 Redis 密码与应用配置一致 |
| 模型下载中断 | 重跑 `python3 scripts/download_models.py`，支持断点续传 |
| 时间不对 | 容器已设 `TZ=Asia/Shanghai`；数据库时区在 compose 中为 `+08:00` |

## 8. 运维

```bash
docker compose ps
docker compose logs -f --tail=200 app worker
docker compose restart app worker
docker compose down                 # 停止（保留数据卷）
docker compose down -v              # 停止并删除数据（危险）

# 备份 MySQL
docker compose exec -T mysql mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" finance_qa > backup_$(date +%F).sql

# 更新代码后重建
git pull
docker compose up -d --build
```

## 9. 安全建议

1. 安全组只放行必要端口（22/80），SSH 建议限制来源 IP；
2. 修改默认账号密码与 `.env` 中的所有密码；
3. `.env` 权限设为 600，禁止提交到 Git；
4. 为服务器配置 HTTPS（可用阿里云 SLB/证书，或 Nginx + Certbot）；
5. 定期备份 MySQL 数据卷与上传目录。
