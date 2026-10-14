$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$dest = Join-Path $root "python"

if (-not (Test-Path $venvPy)) {
  throw "未找到 .venv\Scripts\python.exe，请先 ocr.bat install"
}

# 解析基座解释器目录（完整安装，非 venv）
$info = & $venvPy -c @"
import sys, os
print(sys.base_prefix)
print('%d.%d' % sys.version_info[:2])
"@
$lines = @($info | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
if ($lines.Count -lt 2) { throw "无法读取 base_prefix" }
$basePrefix = $lines[0]
$ver = $lines[1]
if ($ver -ne "3.12") {
  Write-Warning "当前 .venv 为 Python $ver，推荐 3.12（OCR 库兼容）。继续生成..."
}
if (-not (Test-Path (Join-Path $basePrefix "python.exe"))) {
  throw "基座 Python 无效: $basePrefix"
}

Write-Host "基座 Python: $basePrefix"
Write-Host "目标目录:   $dest"

# 重建 python\
if (Test-Path $dest) {
  Write-Host "删除旧的 python\ ..."
  Remove-Item $dest -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# 复制运行时（排除站点包、文档、无用目录）
$excludeDirs = @(
  "Doc", "docs", "include", "Include", "libs", "Libs",
  "tcl\tcl8*\*\tzdata", "Lib\test", "Lib\idlelib", "Lib\turtledemo",
  "Lib\site-packages"
)
# robocopy 不便嵌套排除 site-packages 后再单独拷；分两步：先拷再换 site-packages
robocopy $basePrefix $dest /E /NFL /NDL /NJH /NJS /nc /ns /np `
  /XD Doc docs include Include libs Libs Scripts Tools site-packages `
  /XF *.pdb *.lib >$null
if ($LASTEXITCODE -ge 8) { throw "复制基座 Python 失败 ($LASTEXITCODE)" }

# 用 .venv 的 site-packages 覆盖
$srcSp = Join-Path $root ".venv\Lib\site-packages"
$dstSp = Join-Path $dest "Lib\site-packages"
if (-not (Test-Path $srcSp)) { throw "未找到 .venv\Lib\site-packages" }
if (Test-Path $dstSp) { Remove-Item $dstSp -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dstSp | Out-Null
# /XD 只接受目录名；通配路径会导致 robocopy 直接退出码 16
robocopy $srcSp $dstSp /E /NFL /NDL /NJH /NJS /nc /ns /np /XD __pycache__ >$null
if ($LASTEXITCODE -ge 8) { throw "复制 site-packages 失败 ($LASTEXITCODE)" }

# 清理便携目录内的 pip（现场不需要联网装包）
foreach ($drop in @(
  (Join-Path $dstSp "pip"),
  (Join-Path $dstSp "pip-*"),
  (Join-Path $dstSp "setuptools"),
  (Join-Path $dstSp "setuptools-*"),
  (Join-Path $dstSp "pkg_resources"),
  (Join-Path $dstSp "_distutils_hack"),
  (Join-Path $dstSp "distutils-precedence.pth")
)) {
  Get-Item $drop -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem $dstSp -Directory -Filter "pip-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem $dstSp -Directory -Filter "setuptools-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem $dstSp -Directory -Filter "wheel-*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# 校验（隔离本机用户 site-packages，避免旧 typing_extensions 等污染）
$probe = Join-Path $dest "python.exe"
$prevNoUserSite = $env:PYTHONNOUSERSITE
$env:PYTHONNOUSERSITE = "1"
try {
  & $probe -c "import uvicorn, fastapi, rapidocr_onnxruntime; print('offline-ok')"
  if ($LASTEXITCODE -ne 0) { throw "便携 python 自检失败" }
} finally {
  if ($null -eq $prevNoUserSite) { Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue }
  else { $env:PYTHONNOUSERSITE = $prevNoUserSite }
}

$size = (Get-ChildItem $dest -Recurse -File -ErrorAction SilentlyContinue |
  Measure-Object -Property Length -Sum).Sum
Write-Host ("python\ 就绪，约 {0:N0} MB" -f ($size / 1MB))
