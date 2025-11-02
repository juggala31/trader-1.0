# FTMO System Startup - Handles port conflicts and clean startup
# -*- coding: utf-8 -*-
import sys
import time
import logging
from port_manager import setup_ports

def setup_logging():
    """Set up logging for the FTMO system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ftmo_system.log'),
            logging.StreamHandler()
        ]
    )

def start_ftmo_system():
    """Start the FTMO trading system with proper cleanup"""
    print("Starting FTMO Trading System")
    print("==============================")
    
    # Set up logging
    setup_logging()
    
    # Clean up ports
    print("Cleaning up ports...")
    cleaned = setup_ports()
    print(f"Freed {cleaned} ports")
    
    # Wait a moment for cleanup to complete
    time.sleep(2)
    
    # Import and start the system
    try:
        from ftmo_integrated_system import FTMOIntegratedSystem
        
        # Initialize system
        system = FTMOIntegratedSystem(account_id="1600038177", challenge_type="200k")
        
        # Display startup information
        status = system.get_system_status()
        print("\nSystem Startup Information:")
        print(f"Account ID: 1600038177")
        print(f"Challenge Type: 200k")
        print(f"Current Strategy: {status['current_strategy']}")
        print(f"FTMO Profit Target: ${status['ftmo_metrics']['profit_target']}")
        print(f"Daily Loss Limit: ${status['ftmo_metrics']['daily_loss_limit']}")
        print(f"System Health: {status['health_status']['overall_status']}")
        
        print("\nStarting trading system in 3 seconds...")
        print("Press Ctrl+C to stop the system")
        time.sleep(3)
        
        # Start the system
        system.start_trading()
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed to start system: {e}")
        print(f"Error: {e}")
        return False
        
    return True

def stop_ftmo_system():
    """Stop any running FTMO system processes"""
    print("Stopping FTMO system...")
    setup_ports()  # This will clean up any processes using our ports
    print("System stopped")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_ftmo_system()
    else:
        start_ftmo_system()
