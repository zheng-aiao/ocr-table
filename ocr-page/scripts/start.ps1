param(
  [Parameter(Mandatory = $true)][string]$PyExe,
  [Parameter(Mandatory = $true)][int]$Port,
  [Parameter(Mandatory = $true)][string]$PidFile,
  [Parameter(Mandatory = $true)][string]$LogFile,
  [Parameter(Mandatory = $true)][string]$WorkDir
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $WorkDir

function Test-PortListening {
  param([int]$ListenPort)
  try {
    $c = Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue
    if ($c) { return $true }
  } catch {}
  $pattern = ":{0}\s+.*LISTENING" -f $ListenPort
  return [bool](netstat -ano | Select-String -Pattern $pattern)
}

if (Test-PortListening -ListenPort $Port) {
  Write-Host ("port {0} already in use" -f $Port)
  exit 1
}

$dataDir = Split-Path -Parent $LogFile
if (-not (Test-Path -LiteralPath $dataDir)) {
  New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
}
# UTF-8（带 BOM）：与 run-server.bat 里 PYTHONUTF8=1 的追加内容一致，编辑器按 UTF-8 打开不乱码
$header = "start {0:o}`r`n" -f (Get-Date)
[System.IO.File]::WriteAllText($LogFile, $header, (New-Object System.Text.UTF8Encoding $true))

# 子进程环境：即使未走 bat 内 set，也能 UTF-8 写日志
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONNOUSERSITE = "1"

$runner = Join-Path $PSScriptRoot "run-server.bat"
$p = Start-Process -FilePath $runner `
  -ArgumentList @($PyExe, "$Port", $LogFile) `
  -WorkingDirectory $WorkDir `
  -WindowStyle Hidden `
  -PassThru

if (-not $p) {
  Write-Host "failed to create process"
  exit 1
}

$ok = $false
for ($i = 0; $i -lt 48; $i++) {
  Start-Sleep -Milliseconds 250
  if ($p.HasExited) { break }
  if (Test-PortListening -ListenPort $Port) {
    $ok = $true
    break
  }
}

Set-Content -LiteralPath $PidFile -Value $p.Id -Encoding ASCII

if (-not $ok) {
  if ($p.HasExited) {
    Write-Host "start failed (process exited)"
    if (Test-Path -LiteralPath $PidFile) {
      Remove-Item -LiteralPath $PidFile -Force
    }
    exit 1
  }
  Write-Host ("process started PID={0}, port not ready yet" -f $p.Id)
}

exit 0
