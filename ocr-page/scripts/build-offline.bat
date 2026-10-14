@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
chcp 65001 >nul 2>&1

if not exist ".venv\Scripts\python.exe" (
  echo 未找到 .venv，请先执行: ocr.bat install
  exit /b 1
)

echo 正在根据 .venv 生成便携 python\ ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-offline.ps1"
if errorlevel 1 (
  echo 生成离线环境失败。
  exit /b 1
)
echo.
echo 完成。现场可用: ocr.bat start
echo 打全量包: ocr.bat pack -all
exit /b 0
