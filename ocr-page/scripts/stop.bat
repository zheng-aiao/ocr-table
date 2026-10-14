@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
chcp 65001 >nul 2>&1

set "PIDFILE=%cd%\data\ocr.pid"
set "PORT=8088"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" -PidFile "%PIDFILE%" -Port %PORT%
exit /b %ERRORLEVEL%
