@echo off
echo ========================================
echo 🎯 FTMO AI SYSTEM - WEEKEND BACKTESTING
echo ========================================
echo.
echo This script will:
echo ✅ Download 2 years of historical data from OANDA
echo ✅ Train XGBoost/LightGBM/Random Forest models
echo ✅ Test model accuracy and performance
echo ✅ Train market regime detection
echo ✅ Save trained models for next week's trading
echo.
echo Estimated time: 10-30 minutes depending on data
echo.
echo Starting backtesting process...
echo.

cd /d "%~dp0"

python weekend_backtester.py

echo.
echo Backtesting process completed!
echo Check the generated report: backtesting_report.txt
echo.
pause
