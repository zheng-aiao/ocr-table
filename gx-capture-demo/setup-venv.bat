@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PATH=%LocalAppData%\Programs\Python\Launcher;%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%ProgramFiles%\Python312;%PATH%"

set "PYEXE="
call :resolve_python
if defined PYEXE goto :setup

echo δ�ҵ� Python 3.12�������� winget ��װ...
where winget >nul 2>&1
if errorlevel 1 goto :no_winget

winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :refresh_path
set "PATH=%LocalAppData%\Programs\Python\Launcher;%LocalAppData%\Programs\Python\Python312;%LocalAppData%\Programs\Python\Python312\Scripts;%ProgramFiles%\Python312;%PATH%"
set "PYEXE="
call :resolve_python
if defined PYEXE goto :setup

echo.
echo Python 3.12 ��δ��������رձ����ں����´򿪣������� setup-venv.bat��
echo �� winget ��װʧ�ܣ����ֶ���װ Python 3.12����װʱ��ѡ py launcher�������ԡ�
exit /b 1

:no_winget
echo δ�ҵ� winget���޷��Զ���װ Python��
echo ���Ȱ�װ Python 3.12��Ȼ���������� setup-venv.bat��
exit /b 1

:setup
echo ʹ�� Python: %PYEXE%
if exist ".venv\Scripts\python.exe" goto :pip
echo �������⻷�� .venv ...
"%PYEXE%" -m venv .venv
if errorlevel 1 (
  echo �������⻷��ʧ�ܡ�
  exit /b 1
)

:pip
".venv\Scripts\python.exe" -m pip install -U pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
echo.
echo ��װ��ɡ�֮�������� run.bat
exit /b 0

:refresh_path
for /f "skip=2 tokens=1,2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do (
  if /i "%%A"=="Path" set "USER_PATH=%%C"
)
for /f "skip=2 tokens=1,2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do (
  if /i "%%A"=="Path" set "SYS_PATH=%%C"
)
if defined SYS_PATH set "PATH=%SYS_PATH%"
if defined USER_PATH set "PATH=%USER_PATH%;%PATH%"
exit /b 0

:resolve_python
set "PYEXE="
where py >nul 2>&1
if errorlevel 1 goto :try_paths
for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if defined PYEXE exit /b 0

:try_paths
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
  exit /b 0
)
if exist "%ProgramFiles%\Python312\python.exe" (
  set "PYEXE=%ProgramFiles%\Python312\python.exe"
  exit /b 0
)
if exist "%ProgramFiles%\Python\Python312\python.exe" (
  set "PYEXE=%ProgramFiles%\Python\Python312\python.exe"
  exit /b 0
)
where python >nul 2>&1
if errorlevel 1 exit /b 0
for /f "delims=" %%P in ('python -c "import sys; print(sys.executable if sys.version_info[:2]==(3, 12) else '')" 2^>nul') do (
  if not "%%P"=="" set "PYEXE=%%P"
)
exit /b 0
