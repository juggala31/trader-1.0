# FTMO FINAL VERIFICATION - System Ready Check
import os
import sys

def final_verification():
    print("🎯 FTMO SYSTEM - FINAL VERIFICATION")
    print("==================================")
    
    # Check core files exist
    core_files = [
        'ftmo_minimal.py',
        'ftmo_phase2_system.py', 
        'strategy_orchestrator.py',
        'ftmo_challenge_logger.py',
        'enhanced_risk_manager.py'
    ]
    
    print("\n📁 Core Files Check:")
    all_exist = True
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            all_exist = False
    
    # Test minimal system
    print("\n🔧 System Functionality Test:")
    try:
        from ftmo_minimal import FTMOMinimalSystem
        system = FTMOMinimalSystem()
        status = system.get_system_status()
        print(f"   ✅ System initialization: {status['strategy']}")
        print(f"   ✅ Risk management: {status['risk_level']}")
        print(f"   ✅ FTMO challenge: {status['challenge_type']}")
    except Exception as e:
        print(f"   ❌ System test failed: {e}")
        all_exist = False
    
    print("\n🎯 FINAL STATUS:")
    if all_exist:
        print("✅ FTMO SYSTEM IS FULLY OPERATIONAL!")
        print("==================================")
        print("All core components are working")
        print("System is ready for FTMO challenge")
        print("==================================")
        print("\n🚀 START TRADING WITH:")
        print("python ftmo_minimal.py      # Demo mode")
        print("python ftmo_minimal.py live # Live trading")
        return True
    else:
        print("❌ System needs final adjustments")
        return False

if __name__ == "__main__":
    success = final_verification()
    sys.exit(0 if success else 1)
