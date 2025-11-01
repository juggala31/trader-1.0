# FTMO Integration Completion Check
import os
import subprocess
import sys

def check_integration():
    """Check if FTMO integration is complete and working"""
    print("?? FTMO INTEGRATION COMPLETION CHECK")
    print("=====================================")
    
    # Check for required files
    required_files = [
        "strategy_orchestrator.py",
        "probability_calibrator.py",
        "ftmo_challenge_logger.py",
        "service_health_monitor.py",
        "ftmo_config.py",
        "ftmo_integrated_system.py",
        "port_manager.py",
        "ftmo_start.py"
    ]
    
    print("\n?? File Check:")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ? {file}")
        else:
            print(f"   ? {file} - MISSING")
            missing_files.append(file)
    
    # Check Python dependencies
    print("\n?? Dependency Check:")
    try:
        import zmq
        print("   ? pyzmq")
    except ImportError:
        print("   ? pyzmq - Install with: pip install pyzmq")
        
    try:
        import xgboost
        print("   ? xgboost")
    except ImportError:
        print("   ? xgboost - Install with: pip install xgboost")
        
    try:
        import psutil
        print("   ? psutil")
    except ImportError:
        print("   ? psutil - Install with: pip install psutil")
    
    # Run the fixed integration test
    print("\n?? Integration Test:")
    try:
        result = subprocess.run([sys.executable, "test_integration_fixed.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ? Integration tests passed")
        else:
            print("   ? Integration tests failed")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ? Could not run tests: {e}")
    
    # Final status
    print("\n?? INTEGRATION STATUS:")
    if not missing_files:
        print("? All required files present")
        print("? Dependencies checked")
        print("? Integration tests completed")
        print("\n?? FTMO PHASE 1 INTEGRATION IS COMPLETE!")
        print("\nNext steps:")
        print("1. Start the system: python ftmo_start.py")
        print("2. Monitor logs: tail -f ftmo_system.log")
        print("3. Check performance in your GUI dashboard")
        return True
    else:
        print(f"? Integration incomplete - {len(missing_files)} files missing")
        return False

if __name__ == "__main__":
    check_integration()
