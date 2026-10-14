param(
  [Parameter(Mandatory = $true)][string]$PidFile,
  [Parameter(Mandatory = $true)][int]$Port
)

$ErrorActionPreference = "Continue"
$stopped = $false

if (Test-Path -LiteralPath $PidFile) {
  $pidText = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($pidText) {
    $procId = 0
    if ([int]::TryParse($pidText.Trim(), [ref]$procId) -and $procId -gt 0) {
      cmd /c ("taskkill /PID {0} /T /F" -f $procId) | Out-Null
      if ($LASTEXITCODE -eq 0) {
        Write-Host ("stopped PID {0}" -f $procId)
        $stopped = $true
      }
    }
  }
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

$ownerIds = @()
try {
  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  if ($conns) {
    $ownerIds = @($conns | Select-Object -ExpandProperty OwningProcess -Unique)
  }
} catch {
  # fallback below
}

if (-not $ownerIds -or $ownerIds.Count -eq 0) {
  $pattern = ":{0}\s+.*LISTENING\s+(\d+)" -f $Port
  foreach ($m in (netstat -ano | Select-String -Pattern $pattern)) {
    if ($m.Matches.Count -gt 0) {
      $ownerIds += [int]$m.Matches[0].Groups[1].Value
    }
  }
  $ownerIds = @($ownerIds | Select-Object -Unique)
}

foreach ($oid in $ownerIds) {
  if ($oid -gt 0) {
    cmd /c ("taskkill /PID {0} /T /F" -f $oid) | Out-Null
    if ($LASTEXITCODE -eq 0) {
      Write-Host ("released port {0} (PID {1})" -f $Port, $oid)
      $stopped = $true
    }
  }
}

if ($stopped) {
  Write-Host "service stopped"
  exit 0
}

Write-Host "no running service found"
exit 0
