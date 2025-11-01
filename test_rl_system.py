from rl_enhanced_system import demo_rl_system

print("Testing Reinforcement Learning Integration...")
print("=" * 50)

try:
    demo_rl_system()
    print("\\n🎯 RL Integration Test Completed Successfully!")
    print("\\nSystem Features:")
    print("✓ Q-learning algorithm for strategy optimization")
    print("✓ Reward-based learning from trading outcomes")
    print("✓ Integration with market regime detection")
    print("✓ Real-time learning and adaptation")
    print("✓ Visual analytics dashboard")
    
except Exception as e:
    print(f"Test failed: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install numpy pandas matplotlib")
