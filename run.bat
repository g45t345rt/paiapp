@echo off
rem Launch Pai. On the first run this creates a private .venv next to the app
rem and installs the pinned dependencies; later runs reuse it. No other Python
rem environment (such as a ComfyUI venv) is required or used.
setlocal
set "HERE=%~dp0"
set "BASEPY="

rem Prefer a 3.12 interpreter, then 3.13/3.14 or any Python 3.12+ on PATH.
call :probe py -3.12
if not defined BASEPY call :probe py -3.13
if not defined BASEPY call :probe py -3.14
if not defined BASEPY call :probe py -3
if not defined BASEPY call :probe python

if not defined BASEPY (
    echo Pai needs Python 3.12 or newer on PATH.
    echo Install it from https://www.python.org/downloads/ and run this again.
    exit /b 1
)

%BASEPY% "%HERE%setup_env.py" %*
exit /b %ERRORLEVEL%

:probe
rem Use %~1 %~2 only when it exists and reports Python 3.12+.
where %~1 >nul 2>nul || exit /b 0
%~1 %~2 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul || exit /b 0
set "BASEPY=%~1 %~2"
exit /b 0
