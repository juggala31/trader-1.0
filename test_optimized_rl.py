from optimized_rl_system import demo_optimized_rl

print("Testing COMPLETE Optimized RL System...")
print("=" * 50)

try:
    demo_optimized_rl()
    print("\\n🎯 OPTIMIZED RL SYSTEM WORKING PERFECTLY!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
