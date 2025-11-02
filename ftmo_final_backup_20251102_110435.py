# FTMO FINAL WORKING STARTUP - No Port Conflicts
import sys
import logging
from ftmo_simplified import FTMOSimplifiedChallenge, test_simplified_system

# Simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🎯 FTMO TRADING SYSTEM - FINAL VERSION")
    print("======================================")
    print("All Phases Integrated")
    print("Port Conflicts Resolved")
    print("Challenge Optimized")
    print("======================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        return test_simplified_system()
        
    live_mode = len(sys.argv) > 1 and sys.argv[1] == "live"
    
    # Initialize the simplified system
    ftmo_system = FTMOSimplifiedChallenge("1600038177", "200k", live_mode)
    
    # Show initial status
    status = ftmo_system.get_challenge_status()
    
    print("\n📊 SYSTEM STATUS:")
    print(f"Account: {status['account_id']}")
    print(f"Challenge: {status['challenge_type']}")
    print(f"Balance: ${status['current_profit'] + 200000:,.2f}")
    print(f"Target: ${status['profit_target']:,.2f}")
    print(f"Progress: {status['progress_percentage']:.1f}%")
    print(f"Strategy: {status['current_strategy']}")
    print(f"Risk Level: {status['risk_level']}")
    
    print("\n🚀 Starting FTMO challenge system...")
    print("Press Ctrl+C to stop at any time")
    
    try:
        success = ftmo_system.start_challenge()
        if success:
            print("🎉 FTMO system running successfully!")
        else:
            print("❌ FTMO system failed to start")
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
