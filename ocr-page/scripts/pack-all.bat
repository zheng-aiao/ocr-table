@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
chcp 65001 >nul 2>&1

if exist "python\python.exe" goto :pack

echo 未找到 python\，将自动准备离线环境...
if not exist ".venv\Scripts\python.exe" (
  echo 未找到 .venv，先执行 install...
  call "%~dp0install.bat"
  if errorlevel 1 (
    echo install 失败，无法继续 pack -all。
    exit /b 1
  )
)
echo 执行 offline 生成 python\ ...
call "%~dp0build-offline.bat"
if errorlevel 1 (
  echo offline 失败，无法继续 pack -all。
  exit /b 1
)

:pack
if not exist "python\python.exe" (
  echo 仍未找到 python\python.exe，放弃打包。
  exit /b 1
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set "TS=%%i"
set "OUT=%cd%\..\gx-ocr-all-%TS%.zip"
if exist "%OUT%" del /f /q "%OUT%"

echo 正在打包全量离线包（含 python\）...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -OutZip "%OUT%" -Mode all
if errorlevel 1 (
  echo 打包失败。
  exit /b 1
)
echo.
echo 已生成：%OUT%
echo 现场：解压后改 config.yaml，再 ocr.bat start
exit /b 0
