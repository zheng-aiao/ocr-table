@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
chcp 65001 >nul 2>&1

echo 正在重启服务...
call "%~dp0stop.bat"
if errorlevel 1 (
  echo 停止阶段有告警，继续尝试启动...
)
ping -n 2 127.0.0.1 >nul
call "%~dp0start.bat"
exit /b %ERRORLEVEL%
