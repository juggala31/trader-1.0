@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
chcp 65001 >NUL

REM Keep window open at the end
set KEEP_OPEN=1

REM Move to this BAT's folder
pushd "%~dp0"

REM If this BAT lives in tools\, go up one level to project root
for %%D in ("%~dp0") do (
  set "_CUR=%%~nxD"
)
if /I "!_CUR!"=="tools\" (
  cd ..
)

echo [Launcher] CWD: %CD%

where python >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  goto END
)

if exist "exosati_trader\app\start_all.py" (
  set "CMD=python -m exosati_trader.app.start_all"
) else if exist "enhanced_trading_dashboard.py" (
  set "CMD=python enhanced_trading_dashboard.py"
) else (
  echo [ERROR] No launcher target found in %CD%
  echo         Expect: exosati_trader\app\start_all.py OR enhanced_trading_dashboard.py
  goto END
)

echo [INFO] Running: !CMD!
!CMD!
set "EXITCODE=!ERRORLEVEL!"
echo [INFO] Python exited with code !EXITCODE!

:END
popd
if defined KEEP_OPEN pause
exit /b %EXITCODE%