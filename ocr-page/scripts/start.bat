@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."
chcp 65001 >nul 2>&1

set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIDFILE=%cd%\data\ocr.pid"
set "LOGFILE=%cd%\data\ocr.log"
set "PORT=8088"

if not exist "data" mkdir "data"

if exist "%PIDFILE%" (
  set /p OLDPID=<"%PIDFILE%"
  if defined OLDPID (
    tasklist /FI "PID eq !OLDPID!" 2>nul | findstr /I "!OLDPID!" >nul 2>&1
    if not errorlevel 1 (
      echo 服务似已在运行 ^(PID !OLDPID^)。先执行: ocr.bat stop
      exit /b 1
    )
  )
  del /f /q "%PIDFILE%" >nul 2>&1
)

set "PYEXE="
if exist "%cd%\python\python.exe" set "PYEXE=%cd%\python\python.exe"
if not defined PYEXE if exist "%cd%\.venv\Scripts\python.exe" set "PYEXE=%cd%\.venv\Scripts\python.exe"
if not defined PYEXE (
  echo 未找到 python\python.exe 或 .venv。请先: ocr.bat install 或确认离线 python\ 存在。
  exit /b 1
)

echo 使用: %PYEXE%
echo 启动中 ^(端口 %PORT%^) ...

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -PyExe "%PYEXE%" -Port %PORT% -PidFile "%PIDFILE%" -LogFile "%LOGFILE%" -WorkDir "%cd%"
if errorlevel 1 (
  echo 启动失败，详见: %LOGFILE%
  exit /b 1
)

echo 已启动。浏览器: http://127.0.0.1:%PORT%/
echo 日志: %LOGFILE%
echo 停止: ocr.bat stop
exit /b 0
