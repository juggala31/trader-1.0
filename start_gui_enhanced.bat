@echo off
chcp 65001 >nul
title FTMO Enhanced Trading System

echo.
echo ========================================
echo   ?? FTMO ENHANCED TRADING SYSTEM
echo ========================================
echo.

echo ?? Initializing Ensemble AI...
python -c "from integrate_ensemble import integrate_ensemble_into_system; ai = integrate_ensemble_into_system(); print('? AI Ready')"

echo.
echo ?? Starting Trading GUI...
python ftmo_tkinter_gui.py

echo.
echo ========================================
echo   ?? Trading Session Ended
echo ========================================
pause
