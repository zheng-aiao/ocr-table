param(
  [Parameter(Mandatory = $true)][string]$OutZip,
  [Parameter(Mandatory = $true)][ValidateSet("update", "all")][string]$Mode
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tag = if ($Mode -eq "all") { "field" } else { "update" }
$stage = Join-Path $env:TEMP ("gx-capture-demo-$tag-" + [guid]::NewGuid().ToString("N"))
$dest = Join-Path $stage "gx-capture-demo"

if ($Mode -eq "all") {
  $includeDirs = @(
    "app",
    "python",
    "scripts"
  )
  $includeFiles = @(
    "ocr.bat",
    "config.yaml",
    "requirements.txt",
    "README.md",
    "部署说明.txt"
  )
} else {
  $includeDirs = @(
    "app",
    "scripts"
  )
  $includeFiles = @(
    "ocr.bat",
    "config.yaml",
    "requirements.txt",
    "README.md"
  )
}

New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($d in $includeDirs) {
  $src = Join-Path $root $d
  if (-not (Test-Path $src)) { throw "缺少目录: $d" }
  robocopy $src (Join-Path $dest $d) /E /NFL /NDL /NJH /NJS /nc /ns /np /XD __pycache__ .pytest_cache | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "复制失败: $d ($LASTEXITCODE)" }
}
foreach ($f in $includeFiles) {
  $src = Join-Path $root $f
  if (-not (Test-Path $src)) {
    if ($Mode -eq "all") { throw "缺少文件: $f" }
    continue
  }
  Copy-Item $src (Join-Path $dest $f) -Force
}

if ($Mode -eq "all") {
  foreach ($empty in @("data\captures", "data\exports", "data\ocr_runs", "data\runs")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $dest $empty) | Out-Null
  }
} else {
  $note = @"
轻量更新包（覆盖安装）
====================
本包不含：python / .venv / data

现场操作：
1. ocr.bat stop
2. 解压覆盖 app、scripts、ocr.bat、config.yaml
3. 若 config.yaml 现场地址不同，覆盖后改回
4. ocr.bat start
"@
  Set-Content -Path (Join-Path $dest "更新说明.txt") -Value $note -Encoding UTF8
}

if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
Compress-Archive -Path $dest -DestinationPath $OutZip -CompressionLevel Optimal
Remove-Item $stage -Recurse -Force

$info = Get-Item $OutZip
Write-Output ("zip_mb={0:N1}" -f ($info.Length / 1MB))
Write-Output ("zip=" + $info.FullName)
