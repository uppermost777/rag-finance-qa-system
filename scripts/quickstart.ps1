# ============================================================
# 一键启动脚本（Windows PowerShell）
#   用法：powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
# ============================================================
$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "[1/6] 准备环境变量文件"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  已生成 .env，请检查数据库/Redis 密码与大模型 API Key 后重新运行本脚本"
    exit 0
} else {
    Write-Host "  .env 已存在，跳过"
}

Write-Host "[2/6] 检查本地模型目录 .\models"
foreach ($model in @("bge-m3", "bge-reranker-v2-m3")) {
    if (-not (Test-Path "models\$model")) {
        Write-Host "  警告：缺少 models\$model，入库与重排功能将不可用"
        Write-Host "  下载：huggingface-cli download BAAI/$model --local-dir models/$model"
    }
}

Write-Host "[3/6] 构建并启动容器（首次约需 5~10 分钟）"
docker compose up -d --build

Write-Host "[4/6] 等待 MySQL 健康"
for ($i = 0; $i -lt 60; $i++) {
    docker compose exec -T mysql mysqladmin ping -h 127.0.0.1 -u"$env:MYSQL_USER" -p"$env:MYSQL_PASSWORD" --silent 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host "  MySQL 就绪"; break }
    Start-Sleep -Seconds 5
}

Write-Host "[5/6] 初始化数据库表结构"
docker compose exec -T app python scripts/init_db.py

Write-Host "[6/6] 灌入种子数据"
docker compose exec -T app python scripts/seed_data.py

Write-Host ""
Write-Host "启动完成！"
Write-Host "  前端页面：http://localhost/"
Write-Host "  接口文档：http://localhost/docs"
Write-Host "  健康检查：http://localhost/health"
Write-Host "  默认账号：admin / Admin@2026（管理员）　student_lin / Student@2026（学生）"
