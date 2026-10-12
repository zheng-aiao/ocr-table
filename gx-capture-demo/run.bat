@echo off
cd /d "%~dp0"
if exist "%~dp0python\python.exe" (
  "%~dp0python\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8088
  exit /b %ERRORLEVEL%
)
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
  exit /b %ERRORLEVEL%
)
echo δ�ҵ� Python����ȷ�� python\python.exe ���ڱ�Ŀ¼�С�
exit /b 1