@echo off
echo ========================================
echo 🎯 FTMO AI - PARAMETER OPTIMIZATION
echo ========================================
echo.
echo This will test thousands of parameter combinations:
echo ✅ Timeframes (D1, H4, H1)
echo ✅ Feature sets (Basic to Comprehensive)  
echo ✅ Technical indicator parameters
echo ✅ Prediction horizons
echo ✅ Train/test splits
echo.
echo Estimated time: 30-60 minutes
echo.
echo Starting optimization process...
echo.

cd /d "%~dp0"

python optimization_backtester.py

echo.
echo Parameter optimization completed!
echo Check the report: parameter_optimization_report.txt
echo Best parameters saved for live trading!
echo.
pause
