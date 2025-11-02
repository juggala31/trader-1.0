# FTMO Phase 2 Startup
import sys
from ftmo_phase2_system import FTMO_Phase2_System, test_phase2

def main():
    print("🚀 FTMO PHASE 2: RISK CONTROL & OPTIMIZATION")
    print("=============================================")
    print("Enhanced Features:")
    print("• Dynamic Position Sizing")
    print("• Drawdown Protection")
    print("• Real-time Performance Optimization")
    print("• Automated FTMO Rule Enforcement")
    print("=============================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_phase2()
        return
        
    # Start Phase 2 system
    phase2_system = FTMO_Phase2_System("1600038177", "200k")
    
    # Show status
    status = phase2_system.get_phase2_status()
    print("System Status:")
    print(f"Risk Level: {status['risk_manager']['risk_level']}")
    print(f"Drawdown: {status['drawdown_protection']['current_drawdown']:.2%}")
    print(f"FTMO Progress: ${status['phase1_status']['ftmo_metrics']['total_profit']} / ${status['phase1_status']['ftmo_metrics']['profit_target']}")
    
    print("\nStarting Enhanced Trading in 5 seconds...")
    print("Press Ctrl+C to stop")
    
    import time
    time.sleep(5)
    
    phase2_system.start_enhanced_trading()

if __name__ == "__main__":
    main()
