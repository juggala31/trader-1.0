@echo off
chcp 65001 > nul
echo Starting FTMO AI Trading System - OPTIMAL PORTFOLIO EDITION...
echo.
echo 🎯 OPTIMAL SYMBOL PORTFOLIO LOADED:
echo 💰 BTCX25.sim - Bitcoin (1704% historical)
echo 📈 US30Z25.sim - Dow Jones (206% historical) 
echo 🔬 US100Z25.sim - Nasdaq 100 (Tech growth)
echo 🌟 US500Z25.sim - S&P 500 (Broad market)
echo 🏆 XAUZ25.sim - Gold (Safe haven)
echo ⚡ USOILZ25.sim - Crude Oil (Commodity)
echo.
echo 📊 PORTFOLIO FEATURES:
echo ✅ 6 symbols across 4 asset classes
echo ✅ Maximum diversification and risk management
echo ✅ Comprehensive multi-timeframe optimization
echo ✅ Ready for live deployment
echo.
cd /d "%~dp0"
python enhanced_trading_dashboard.py
pause
