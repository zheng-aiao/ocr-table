@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "OUT=%~dp0..\gx-capture-demo-field.zip"
if exist "%OUT%" del /f /q "%OUT%"

echo ���ڴ���ֳ����߰������� .venv / data��...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack-field.ps1" -OutZip "%OUT%"
if errorlevel 1 (
  echo ���ʧ�ܡ�
  exit /b 1
)
echo.
echo �����ɣ�%OUT%
echo �����ֳ�����ѹ��� config.yaml����˫�� run.bat
exit /b 0