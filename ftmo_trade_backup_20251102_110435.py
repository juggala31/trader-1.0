# FTMO Main Trading Script - Complete Integration
import sys
import time
import logging
from ftmo_mt5_integration import FTMO_MT5_Integration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ftmo_trading.log'),
        logging.StreamHandler()
    ]
)

def main():
    print("🎯 FTMO META TRADER 5 TRADING SYSTEM")
    print("====================================")
    print("Account: 1600038177")
    print("Challenge: 200k")
    print("Symbols: US30, US100, UAX")
    print("====================================")
    
    # Initialize the integrated system
    ftmo_system = FTMO_MT5_Integration("1600038177", "200k")
    
    # Display system status
    status = ftmo_system.get_system_status()
    print(f"System Status:")
    print(f"• Strategy: {status['current_strategy']}")
    print(f"• FTMO Target: ${status['ftmo_metrics']['profit_target']}")
    print(f"• Daily Limit: ${status['ftmo_metrics']['daily_loss_limit']}")
    
    if status['account_info']:
        print(f"• Account Balance: ${status['account_info']['balance']}")
    
    print("\nStarting in 5 seconds...")
    print("Press Ctrl+C to stop trading")
    time.sleep(5)
    
    try:
        # Start automated trading
        ftmo_system.start_trading()
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        logging.error(f"Trading error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from ftmo_mt5_integration import test_integration
        test_integration()
    else:
        main()
