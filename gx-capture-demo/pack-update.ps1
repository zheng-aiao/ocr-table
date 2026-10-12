param(
  [Parameter(Mandatory = $true)]
  [string]$OutZip
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stage = Join-Path $env:TEMP ("gx-capture-demo-update-" + [guid]::NewGuid().ToString("N"))
$dest = Join-Path $stage "gx-capture-demo"

# 轻量更新：只覆盖业务代码与前端，不含 python / .venv / 本地数据目录
$includeDirs = @(
  "app",
  "static",
  "image-table-extract"
)
$includeFiles = @(
  "config.yaml",
  "run.bat",
  "requirements.txt",
  "README.md"
)

New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($d in $includeDirs) {
  $src = Join-Path $root $d
  if (-not (Test-Path $src)) { throw "缺少目录: $d" }
  robocopy $src (Join-Path $dest $d) /E /NFL /NDL /NJH /NJS /nc /ns /np /XD __pycache__ .pytest_cache | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "复制失败: $d ($LASTEXITCODE)" }
}
foreach ($f in $includeFiles) {
  $src = Join-Path $root $f
  if (-not (Test-Path $src)) { continue }
  Copy-Item $src (Join-Path $dest $f) -Force
}

$note = @"
轻量更新包（覆盖安装）
====================
本包不含：python / .venv / pidll / data（captures、exports、ocr_runs、runs）

现场操作：
1. 停掉正在运行的 uvicorn / run.bat
2. 解压后覆盖 app、static、image-table-extract，以及 config.yaml
3. 若现场 config.yaml 里相机/PI 地址与本包不同，覆盖后改回现场值
   （本次新增：ocr.pi_date_back_days、pi.verify_write）
4. 再双击 run.bat 启动

若缺依赖（极少）：在现场目录执行
  .venv\Scripts\pip install -r requirements.txt
"@
Set-Content -Path (Join-Path $dest "更新说明.txt") -Value $note -Encoding UTF8

if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
Compress-Archive -Path $dest -DestinationPath $OutZip -CompressionLevel Optimal
Remove-Item $stage -Recurse -Force

$info = Get-Item $OutZip
Write-Output ("zip_mb={0:N1}" -f ($info.Length / 1MB))
Write-Output ("zip=" + $info.FullName)
