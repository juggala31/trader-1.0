# Simple FTMO Startup - Minimal version to avoid encoding issues
import sys
import time

def main():
    print("FTMO Trading System - Simple Startup")
    print("=====================================")
    
    try:
        # Import the main system
        from ftmo_integrated_system import FTMOIntegratedSystem
        
        # Initialize system
        print("Initializing FTMO system...")
        system = FTMOIntegratedSystem(account_id="1600038177", challenge_type="200k")
        
        # Show status
        status = system.get_system_status()
        print(f"Current Strategy: {status['current_strategy']}")
        print(f"System Health: {status['health_status']['overall_status']}")
        print(f"FTMO Progress: ${status['ftmo_metrics']['total_profit']} / ${status['ftmo_metrics']['profit_target']}")
        
        print("\nStarting trading in 5 seconds...")
        print("Press Ctrl+C to stop")
        
        time.sleep(5)
        
        # Start trading
        system.start_trading()
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Check if your existing trading system is already running")
        print("Try stopping it first with: python ftmo_start.py stop")

if __name__ == "__main__":
    main()
