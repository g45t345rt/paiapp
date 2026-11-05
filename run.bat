@echo off
rem Launch Pai. On the first run this creates a private .venv next to the app
rem and installs the pinned dependencies; later runs reuse it. No other Python
rem environment (such as a ComfyUI venv) is required or used.
rem
rem When no supported Python (3.12-3.14) is on PATH the launcher offers to
rem download and install one (winget, then the official python.org installer).
setlocal
set "HERE=%~dp0"
set "BASEPY="

rem Prefer a 3.12 interpreter, then 3.13/3.14 or any supported Python on PATH.
call :probe py -3.12
if not defined BASEPY call :probe py -3.13
if not defined BASEPY call :probe py -3.14
if not defined BASEPY call :probe py -3
if not defined BASEPY call :probe python

if not defined BASEPY (
    echo Pai needs Python 3.12-3.14, which was not found on PATH.
    call :offer_install
)

if not defined BASEPY (
    echo.
    echo Pai needs Python 3.12-3.14 on PATH.
    echo Install one from https://www.python.org/downloads/ and run this again.
    exit /b 1
)

%BASEPY% "%HERE%setup_env.py" %*
exit /b %ERRORLEVEL%

:probe
rem Use %~1 %~2 only when it exists and reports a supported Python (3.12-3.14).
where %~1 >nul 2>nul || exit /b 0
%~1 %~2 -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)" >nul 2>nul || exit /b 0
set "BASEPY=%~1 %~2"
exit /b 0

:offer_install
rem Offer to download and install Python. Only when the helper script is present
rem (a system package may not ship it) and the user opts in.
set "INSTALLER=%HERE%installer\windows\install_python.ps1"
if not exist "%INSTALLER%" exit /b 0
set "ANS="
set /p "ANS=Download and install Python now? [y/N] "
if /i "%ANS%"=="y" goto :do_install
if /i "%ANS%"=="yes" goto :do_install
exit /b 0

:do_install
echo Installing Python (this may take a few minutes)...
set "OUTFILE=%TEMP%\pai-python-path-%RANDOM%.txt"
del "%OUTFILE%" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%" -OutFile "%OUTFILE%"
if exist "%OUTFILE%" for /f "usebackq delims=" %%p in ("%OUTFILE%") do set "BASEPY="%%~p""
del "%OUTFILE%" >nul 2>nul
if defined BASEPY echo Installed Python: %BASEPY%
exit /b 0
