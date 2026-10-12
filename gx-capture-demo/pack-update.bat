@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "OUT=%~dp0..\gx-capture-demo-update.zip"
if exist "%OUT%" del /f /q "%OUT%"

echo 正在打包轻量更新包（app / static / image-table-extract / config.yaml）...
echo 不含 python、.venv、pidll、data（captures、exports、ocr_runs、runs）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack-update.ps1" -OutZip "%OUT%"
if errorlevel 1 (
  echo 打包失败。
  exit /b 1
)
echo.
echo 已生成：%OUT%
echo 发给现场：解压后覆盖 app、static、image-table-extract、config.yaml，再 run.bat
exit /b 0
