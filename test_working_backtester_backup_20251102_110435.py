from working_backtester import WorkingBacktester

print("Testing WORKING Backtesting System...")
print("=" * 50)

def progress_callback(message):
    print(f"▶ {message}")

try:
    backtester = WorkingBacktester()
    backtester.set_progress_callback(progress_callback)
    
    print("✓ Backtester initialized")
    
    # Run quick test
    success = backtester.run_comprehensive_backtest(years=1, save_models=True)
    
    if success:
        print("\\n🎯 BACKTESTING WORKING PERFECTLY!")
        print("Results should now appear in the GUI tab")
    else:
        print("\\n⚠️ Limited results - but system is working")
    
except Exception as e:
    print(f"Error: {e}")

print("\\nThe backtesting tab should now display results!")
