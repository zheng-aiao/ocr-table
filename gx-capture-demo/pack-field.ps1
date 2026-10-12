param(
  [Parameter(Mandatory = $true)]
  [string]$OutZip
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stage = Join-Path $env:TEMP ("gx-capture-demo-field-" + [guid]::NewGuid().ToString("N"))
$dest = Join-Path $stage "gx-capture-demo"

$includeDirs = @(
  "app",
  "image-table-extract",
  "pidll",
  "python",
  "scripts",
  "static"
)
$includeFiles = @(
  "config.yaml",
  "run.bat",
  "requirements.txt",
  "README.md",
  "运行说明.txt"
)

New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($d in $includeDirs) {
  $src = Join-Path $root $d
  if (-not (Test-Path $src)) { throw "缺少目录: $d" }
  robocopy $src (Join-Path $dest $d) /E /NFL /NDL /NJH /NJS /nc /ns /np /XD __pycache__ | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "复制失败: $d ($LASTEXITCODE)" }
}
foreach ($f in $includeFiles) {
  $src = Join-Path $root $f
  if (-not (Test-Path $src)) { throw "缺少文件: $f" }
  Copy-Item $src (Join-Path $dest $f) -Force
}

# 空目录占位，方便现场直接落盘
foreach ($empty in @("data\captures", "data\exports", "data\ocr_runs", "data\runs")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $dest $empty) | Out-Null
}

if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
Compress-Archive -Path $dest -DestinationPath $OutZip -CompressionLevel Optimal
Remove-Item $stage -Recurse -Force

$info = Get-Item $OutZip
Write-Output ("zip_mb={0:N1}" -f ($info.Length / 1MB))
Write-Output ("zip=" + $info.FullName)