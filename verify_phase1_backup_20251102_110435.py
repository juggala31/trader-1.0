# Phase 1 Completion Verification
import os
import sys

def check_phase1_completion():
    """Verify all Phase 1 components are present"""
    required_files = [
        "strategy_orchestrator.py",
        "probability_calibrator.py", 
        "ftmo_challenge_logger.py",
        "service_health_monitor.py",
        "ftmo_config.py"
    ]
    
    print("FTMO PHASE 1 COMPLETION CHECK")
    print("=" * 50)
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"? {file} - PRESENT")
        else:
            print(f"? {file} - MISSING")
            all_present = False
            
    print("=" * 50)
    
    if all_present:
        print("?? PHASE 1 COMPLETED SUCCESSFULLY!")
        print("Next: Integrate components and test strategy switching")
        return True
    else:
        print("? PHASE 1 INCOMPLETE - Missing files detected")
        return False

if __name__ == "__main__":
    check_phase1_completion()
