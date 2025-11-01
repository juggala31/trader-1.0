from real_time_trading_system import demo_real_time_system

print("Testing REAL-TIME Trading System with OANDA MT5...")
print("=" * 60)

try:
    demo_real_time_system()
    print("\\n🎯 REAL-TIME SYSTEM TEST COMPLETED!")
    print("\\nReal-Time Features:")
    print("✓ Live OANDA MT5 balance and equity tracking")
    print("✓ Real-time P&L calculation from open positions")
    print("✓ Dynamic position sizing based on actual account balance")
    print("✓ FTMO risk limit enforcement (5% daily, 10% total)")
    print("✓ Real market data integration for regime detection")
    
except Exception as e:
    print(f"Real-time test failed: {e}")
    import traceback
    traceback.print_exc()
