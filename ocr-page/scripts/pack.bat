@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
chcp 65001 >nul 2>&1

set "OUT=%cd%\..\gx-capture-demo-update.zip"
if exist "%OUT%" del /f /q "%OUT%"

echo 正在打包轻量更新包（不含 python / .venv / data）...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack.ps1" -OutZip "%OUT%" -Mode update
if errorlevel 1 (
  echo 打包失败。
  exit /b 1
)
echo.
echo 已生成：%OUT%
echo 现场：停服务后解压覆盖，再 ocr.bat start
exit /b 0
