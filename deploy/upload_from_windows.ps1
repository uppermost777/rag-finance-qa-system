# ============================================================
# 从 Windows 上传项目到阿里云 ECS（不包含 .venv / models / 日志）
# 用法：
#   powershell -ExecutionPolicy Bypass -File deploy\upload_from_windows.ps1 `
#       -Server 47.xx.xx.xx -User root -RemotePath /opt/finance-qa
# 模型建议在服务器上用 scripts/download_models.py 下载，避免上传 4.4GB
# ============================================================
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [string]$User = "root",
    [string]$RemotePath = "/opt/finance-qa",
    [string]$KeyFile = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Parent = Split-Path -Parent $ProjectRoot
$Name = Split-Path -Leaf $ProjectRoot
$Archive = Join-Path $Parent "$Name.tar.gz"

Write-Host "[1/3] 打包项目（排除 .venv / node_modules / models / 日志）"
Push-Location $Parent
try {
    $excludes = @("--exclude=$Name/.venv", "--exclude=$Name/node_modules", "--exclude=$Name/web/node_modules",
                  "--exclude=$Name/models", "--exclude=$Name/logs", "--exclude=$Name/work_logs",
                  "--exclude=$Name/.pytest_cache", "--exclude=$Name/web/dist")
    tar -czf $Archive @excludes $Name
} finally {
    Pop-Location
}

$sshArgs = @("-o", "StrictHostKeyChecking=accept-new")
if ($KeyFile) { $sshArgs += @("-i", $KeyFile) }

Write-Host "[2/3] 创建远程目录并上传"
ssh @sshArgs "$User@$Server" "mkdir -p '$RemotePath'"
scp @sshArgs $Archive "$User@${Server}:/tmp/$Name.tar.gz"

Write-Host "[3/3] 远程解压"
ssh @sshArgs "$User@$Server" "cd '$RemotePath' && tar xzf '/tmp/$Name.tar.gz' --strip-components=1"

Remove-Item $Archive -Force
Write-Host "完成。登录服务器后："
Write-Host "  cd $RemotePath"
Write-Host "  nano .env                     # 填 LLM_API_KEY"
Write-Host "  sudo bash deploy/aliyun_deploy.sh --with-models --with-corpus"
