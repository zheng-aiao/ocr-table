@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul 2>&1

set "ROOT=%cd%"
set "SCRIPTS=%ROOT%\scripts"

if "%~1"=="" goto :menu

set "CMD=%~1"
shift
if /i "%CMD%"=="install" goto :do_install
if /i "%CMD%"=="offline" goto :do_offline
if /i "%CMD%"=="start" goto :do_start
if /i "%CMD%"=="stop" goto :do_stop
if /i "%CMD%"=="restart" goto :do_restart
if /i "%CMD%"=="pack" goto :do_pack
if /i "%CMD%"=="help" goto :usage
if /i "%CMD%"=="-h" goto :usage
if /i "%CMD%"=="--help" goto :usage
echo 未知命令: %CMD%
echo.
goto :usage

:menu
echo.
echo  ==========  珙县 OCR 工具  ==========
echo.
echo   1^) install   安装开发环境 (.venv)
echo   2^) offline   由 .venv 生成离线 python\
echo   3^) start     启动服务
echo   4^) stop      停止服务
echo   5^) restart   重启服务（更新后用）
echo   6^) pack      打轻量更新包
echo   7^) pack -all 打全量离线包(含 python\)
echo   0^) 退出
echo.
set "CHOICE="
set /p "CHOICE=请选择 [0-7]: "
if "%CHOICE%"=="1" goto :do_install
if "%CHOICE%"=="2" goto :do_offline
if "%CHOICE%"=="3" goto :do_start
if "%CHOICE%"=="4" goto :do_stop
if "%CHOICE%"=="5" goto :do_restart
if "%CHOICE%"=="6" goto :run_pack
if "%CHOICE%"=="7" goto :run_pack_all
if "%CHOICE%"=="0" exit /b 0
if "%CHOICE%"=="" exit /b 0
echo 无效选项。
exit /b 1

:usage
echo 用法:
echo   ocr.bat                 交互菜单
echo   ocr.bat install         安装 .venv + 依赖
echo   ocr.bat offline         由 .venv 生成便携 python\
echo   ocr.bat start           后台启动 (http://127.0.0.1:8088/)
echo   ocr.bat stop            停止服务
echo   ocr.bat restart         停止后重新启动
echo   ocr.bat pack            轻量更新包，不含 python
echo   ocr.bat pack -all       全量离线包；缺 python 时自动 install/offline
exit /b 0

:do_install
call "%SCRIPTS%\install.bat"
exit /b %ERRORLEVEL%

:do_offline
call "%SCRIPTS%\build-offline.bat"
exit /b %ERRORLEVEL%

:do_start
call "%SCRIPTS%\start.bat"
exit /b %ERRORLEVEL%

:do_stop
call "%SCRIPTS%\stop.bat"
exit /b %ERRORLEVEL%

:do_restart
call "%SCRIPTS%\restart.bat"
exit /b %ERRORLEVEL%

:do_pack
if /i "%~1"=="-all" goto :run_pack_all
if /i "%~1"=="--all" goto :run_pack_all
if /i "%~1"=="all" goto :run_pack_all
call :run_pack
exit /b %ERRORLEVEL%

:run_pack
call "%SCRIPTS%\pack.bat"
exit /b %ERRORLEVEL%

:run_pack_all
call "%SCRIPTS%\pack-all.bat"
exit /b %ERRORLEVEL%
