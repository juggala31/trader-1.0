# Final FTMO System Check
import os
import sys

def final_system_check():
    print("🔧 FINAL FTMO SYSTEM CHECK")
    print("===========================")
    
    # Check critical files
    critical_files = [
        "strategy_orchestrator.py",
        "ftmo_challenge_logger.py", 
        "ftmo_integrated_system.py",
        "start_trading.py"
    ]
    
    print("\n📁 Critical Files:")
    all_good = True
    for file in critical_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            all_good = False
    
    # Check if system can import
    print("\n🔍 Import Check:")
    try:
        from strategy_orchestrator import StrategyOrchestrator
        print("   ✅ StrategyOrchestrator")
    except Exception as e:
        print(f"   ❌ StrategyOrchestrator: {e}")
        all_good = False
        
    try:
        from ftmo_challenge_logger import FTMOChallengeLogger
        print("   ✅ FTMOChallengeLogger")
    except Exception as e:
        print(f"   ❌ FTMOChallengeLogger: {e}")
        all_good = False
        
    try:
        from ftmo_integrated_system import FTMOIntegratedSystem
        print("   ✅ FTMOIntegratedSystem")
    except Exception as e:
        print(f"   ❌ FTMOIntegratedSystem: {e}")
        all_good = False
    
    print("\n🎯 READY STATUS:")
    if all_good:
        print("✅ SYSTEM READY FOR TRADING!")
        print("\nTo start trading:")
        print("python start_trading.py")
        return True
    else:
        print("❌ SYSTEM NOT READY - Fix issues above")
        return False

if __name__ == "__main__":
    final_system_check()
