@echo off
echo ========================================
echo 🎯 FTMO AI - HIGH-VOLUME OPTIMIZATION
echo ========================================
echo.
echo Testing 3000+ parameter combinations:
echo ✅ 3 Timeframes (D1, H4, H1)
echo ✅ 6 Feature sets (Basic to Comprehensive)
echo ✅ 7 Moving average period combinations
echo ✅ 7 Volatility windows
echo ✅ 5 RSI periods
echo ✅ 6 Prediction horizons
echo ✅ 6 Train/test splits
echo.
echo Expected time: 45-90 minutes
echo.
echo Starting high-volume optimization...
echo.

cd /d "%~dp0"

python high_volume_optimizer.py

echo.
echo High-volume parameter optimization completed!
echo Check the report: high_volume_optimization_report.txt
echo.
pause
