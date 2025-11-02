@echo off
echo 🚀 FTMO TRADING SYSTEM - ONE CLICK START
echo ======================================
echo.
echo [1] Start GUI Dashboard (Recommended)
echo [2] Start Demo Mode (Terminal)
echo [3] Start Live Challenge
echo.
set /p choice="Select option (1, 2, or 3): "

if "%choice%"=="1" (
    echo Starting Universal GUI Launcher...
    python ftmo_universal_gui.py
) else if "%choice%"=="2" (
    echo Starting Demo Mode...
    python ftmo_minimal.py
) else if "%choice%"=="3" (
    echo Starting Live Challenge...
    python ftmo_launch.py
) else (
    echo Invalid selection
)

pause
