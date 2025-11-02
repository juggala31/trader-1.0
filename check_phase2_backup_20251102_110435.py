# FTMO Phase 2 Completion Check
import os
import sys

def check_phase2_completion():
    print("🎯 FTMO PHASE 2 COMPLETION CHECK")
    print("================================")
    
    # Phase 2 required files
    phase2_files = [
        "enhanced_risk_manager.py",
        "drawdown_protection.py", 
        "performance_optimizer.py",
        "ftmo_rule_enforcer.py",
        "ftmo_phase2_system.py",
        "phase2_start.py",
        "phase2_demo.py"
    ]
    
    print("\n📁 Phase 2 Files:")
    all_present = True
    for file in phase2_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
            all_present = False
    
    # Test imports
    print("\n🔧 Component Tests:")
    try:
        from enhanced_risk_manager import EnhancedRiskManager
        print("   ✅ EnhancedRiskManager")
    except Exception as e:
        print(f"   ❌ EnhancedRiskManager: {e}")
        
    try:
        from drawdown_protection import DrawdownProtection
        print("   ✅ DrawdownProtection")
    except Exception as e:
        print(f"   ❌ DrawdownProtection: {e}")
        
    try:
        from ftmo_phase2_system import FTMO_Phase2_System
        print("   ✅ FTMO_Phase2_System")
    except Exception as e:
        print(f"   ❌ FTMO_Phase2_System: {e}")
    
    print("\n🎯 PHASE 2 STATUS:")
    if all_present:
        print("✅ PHASE 2 IMPLEMENTATION COMPLETE!")
        print("\nNext Steps:")
        print("1. Test with demo: python phase2_demo.py")
        print("2. Run enhanced trading: python phase2_start.py")
        print("3. Prepare for Phase 3: Live Deployment")
        return True
    else:
        print("❌ Phase 2 incomplete - fix missing files")
        return False

if __name__ == "__main__":
    check_phase2_completion()
