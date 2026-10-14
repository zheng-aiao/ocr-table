@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
rem 日志经 >> 重定向；强制 UTF-8，避免中文 Windows 默认 GBK 与 UTF-8 头混写导致乱码
chcp 65001 >nul 2>&1
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYEXE=%~1"
set "PORT=%~2"
set "LOGFILE=%~3"
if "%PYEXE%"=="" exit /b 1
if "%PORT%"=="" set "PORT=8088"
if "%LOGFILE%"=="" set "LOGFILE=%cd%\data\ocr.log"
"%PYEXE%" -m uvicorn app.backend.main:app --host 0.0.0.0 --port %PORT% >> "%LOGFILE%" 2>&1
exit /b %ERRORLEVEL%
