@echo off
rem 来自 S2R 505b298 的 tools/push_*.bat，2026-11-26 复制并改前缀为 pvb/
chcp 65001 >nul
rem Upload local code  pvb\  ->  Bucket pvb/pvb/   (only new/changed files, never deletes)
rem Requires a WinSCP saved site named below, with the Secret Key saved.
setlocal
set "SITE=C:\Users\Admin（无密码）\Desktop/s2r-bucket"
set "REMOTE=/tenant-vww8n20rmcahx6ur-20260904-1fbu/pvb"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

set "WINSCP=%ProgramFiles(x86)%\WinSCP\WinSCP.com"
if not exist "%WINSCP%" set "WINSCP=%ProgramFiles%\WinSCP\WinSCP.com"
if not exist "%WINSCP%" set "WINSCP=%LOCALAPPDATA%\Programs\WinSCP\WinSCP.com"
if not exist "%WINSCP%" set "WINSCP=D:\Apps\WinSCP\WinSCP\WinSCP.com"
if not exist "%WINSCP%" (
  echo WinSCP.com not found. Edit the WINSCP line in this file.
  pause
  exit /b 1
)
if not exist "%ROOT%\out\logs" mkdir "%ROOT%\out\logs"

rem Record git commit for the runner (.git itself is not uploaded)
set "GITH=nogit"
set "DIRTY=-1"
for /f %%H in ('git -C "%ROOT%\pvb" rev-parse --short^=7 HEAD 2^>nul') do set "GITH=%%H"
if not "%GITH%"=="nogit" for /f %%N in ('git -C "%ROOT%\pvb" status --porcelain --untracked-files^=no ^| find /c /v ""') do set "DIRTY=%%N"
> "%ROOT%\pvb\_git_version.txt" echo git=%GITH% dirty=%DIRTY% pushed=%DATE% %TIME%
echo Git   : %GITH%  (uncommitted files: %DIRTY%)
if not "%DIRTY%"=="0" echo WARNING: uncommitted changes - commit first for a reproducible run
echo Local : %ROOT%\pvb
echo Remote: %REMOTE%/pvb/
"%WINSCP%" /log="%ROOT%\out\logs\winscp_push.log" /command ^
  "option reconnecttime 120" ^
  "open ""%SITE%"" -timeout=60" ^
  "option batch continue" ^
  "mkdir ""%REMOTE%/pvb""" ^
  "option batch abort" ^
  "synchronize remote -filemask=""| .git/; .idea/; .vscode/; __pycache__/; .pytest_cache/; .ipynb_checkpoints/; *.pyc"" ""%ROOT%\pvb"" ""%REMOTE%/pvb""" ^
  "exit"
echo.
if errorlevel 1 (echo PUSH FAILED - see out\logs\winscp_push.log) else (echo PUSH OK)
pause
