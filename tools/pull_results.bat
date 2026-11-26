@echo off
rem 来自 S2R 505b298 的 tools/pull_*.bat，2026-11-26 复制并改前缀为 pvb/
chcp 65001 >nul
rem Download Bucket pvb/data/ and pvb/out/  ->  local data\ and out\   (only new/changed files, never deletes)
rem data/raw/ (42 GB) is excluded on purpose.
rem Big binary intermediates (*.h5 *.npy *.pt) stay in the Bucket by default (external link is ~2-4 MB/s).
rem   pull_results.bat          -> reports, tables, logs, figures only (fast)
rem   pull_results.bat full     -> also the big files (S3 alone is ~9 GB, ~1 hour)
rem   pull_results.bat meta     -> also small sidecars under data/raw/ (*.json *.tsv *.txt *.vhdr *.vmrk CHANGES README*, av_ccep ccepData_*.mat ~22 MB), no signal files
rem   data/raw/_manifest/ and data/raw/_logs/ are always pulled (download lists, logs)  [PVB 2026-11-26]
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
set "MASK=| raw/; s2r_ref/; *.h5; *.npy; *.pt; *.filepart"
if /i "%~1"=="full" set "MASK=| raw/; s2r_ref/; *.filepart"
set RAWMETA=
if /i "%~1"=="meta" set RAWMETA="synchronize local -filemask=""*.json; *.tsv; *.txt; *.vhdr; *.vmrk; CHANGES; README*; ccepData_V1.mat; ccepData_V2.mat; ccepData_init.mat | *.filepart"" ""%ROOT%\data\raw"" ""%REMOTE%/data/raw"""
if not exist "%ROOT%\data\raw\_manifest" mkdir "%ROOT%\data\raw\_manifest"
if not exist "%ROOT%\data\raw\_logs" mkdir "%ROOT%\data\raw\_logs"
if not exist "%ROOT%\out\logs" mkdir "%ROOT%\out\logs"
if not exist "%ROOT%\data" mkdir "%ROOT%\data"

echo Remote: %REMOTE%/data/ and /out/
echo "Exclude: %MASK%"
echo Local : %ROOT%\data and %ROOT%\out
rem Start a fresh log each run, so the skipped-file check below only sees this run
if exist "%ROOT%\out\logs\winscp_pull.log" del "%ROOT%\out\logs\winscp_pull.log"
"%WINSCP%" /log="%ROOT%\out\logs\winscp_pull.log" /command ^
  "option reconnecttime 120" ^
  "open ""%SITE%"" -timeout=60" ^
  "option batch continue" ^
  "mkdir ""%REMOTE%/data""" ^
  "mkdir ""%REMOTE%/out""" ^
  "synchronize local -filemask=""%MASK%"" ""%ROOT%\data"" ""%REMOTE%/data""" ^
  "synchronize local -filemask=""%MASK%"" ""%ROOT%\out"" ""%REMOTE%/out""" ^
  "synchronize local ""%ROOT%\data\raw\_manifest"" ""%REMOTE%/data/raw/_manifest""" ^
  "synchronize local ""%ROOT%\data\raw\_logs"" ""%REMOTE%/data/raw/_logs""" ^
  %RAWMETA% ^
  "exit"
set "RC=%ERRORLEVEL%"
echo.
findstr /C:"(ESkipFile)" "%ROOT%\out\logs\winscp_pull.log" >nul
if not errorlevel 1 goto skipped
if not "%RC%"=="0" (echo PULL FINISHED WITH ERRORS - see out\logs\winscp_pull.log) else (echo PULL OK)
goto done
:skipped
echo WARNING: these files were NOT updated. Close any program that has them open, e.g. an image viewer, then run again:
findstr /C:"(ESkipFile)" "%ROOT%\out\logs\winscp_pull.log"
:done
pause
