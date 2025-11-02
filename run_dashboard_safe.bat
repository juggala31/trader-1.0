@echo off
setlocal
cd /d "%~dp0"
where python >NUL 2>&1 || (echo Python missing & pause & exit /b 1)
if exist "exosati_trader\app\start_all.py" (
  python -m exosati_trader.app.start_all
) else (
  python enhanced_trading_dashboard.py
)