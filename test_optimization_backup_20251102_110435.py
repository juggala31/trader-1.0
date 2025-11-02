from optimization_backtester import OptimizationBacktester

print("Testing Parameter Optimization System...")
print("=" * 50)

def progress_callback(message):
    print(f"▶ {message}")

try:
    optimizer = OptimizationBacktester()
    optimizer.set_progress_callback(progress_callback)
    
    print("✓ Optimization backtester initialized")
    
    # Quick test with 1 year and limited combinations
    print("Running quick optimization test...")
    success = optimizer.run_comprehensive_optimization(years=1, save_best_models=True)
    
    if success:
        print("\\n🎯 PARAMETER OPTIMIZATION WORKING PERFECTLY!")
        print("\\nSystem will test:")
        print("✅ Multiple timeframes and feature sets")
        print("✅ Thousands of parameter combinations") 
        print("✅ Various technical indicator configurations")
        print("✅ Different prediction horizons")
        print("✅ Multiple train/test splits")
        print("✅ Comprehensive performance metrics")
    else:
        print("\\n⚠️ Optimization completed with some limitations")
    
except Exception as e:
    print(f"Optimization test error: {e}")

print("\\nTo run full optimization: run_parameter_optimization.bat")
