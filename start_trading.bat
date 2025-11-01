@echo off
echo ========================================
echo    FTMO TRADING SYSTEM - STARTUP
echo ========================================
echo.
echo [1] Testing MT5 Integration
echo [2] Start Live Trading
echo [3] Check System Status
echo [4] Stop Trading System
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo Testing MT5 Integration...
    python ftmo_trade.py test
) else if "%choice%"=="2" (
    echo Starting Live Trading...
    python ftmo_trade.py
) else if "%choice%"=="3" (
    echo Checking System Status...
    python -c "from ftmo_mt5_integration import FTMO_MT5_Integration; ftmo = FTMO_MT5_Integration(); print(ftmo.get_system_status())"
) else if "%choice%"=="4" (
    echo Stopping Trading System...
    taskkill /f /im terminal64.exe 2>nul
    echo System stopped.
) else (
    echo Invalid choice.
)

pause
