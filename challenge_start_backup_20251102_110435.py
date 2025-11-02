# FTMO CHALLENGE FINAL STARTUP
import sys
import logging
from ftmo_challenge_ready import FTMOChallengeReadySystem, test_challenge_system

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🎯 FTMO CHALLENGE - FINAL SYSTEM")
    print("================================")
    print("Optimized for FTMO 200k Challenge")
    print("With Advanced Risk Protection")
    print("And Real-time Analytics")
    print("================================")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_challenge_system()
        return
        
    live_mode = len(sys.argv) > 1 and sys.argv[1] == "live"
    
    # Initialize challenge-ready system
    challenge_system = FTMOChallengeReadySystem("1600038177", "200k", live_mode)
    
    # Show enhanced status
    status = challenge_system.get_challenge_status()
    analytics = challenge_system.get_enhanced_analytics()
    
    print("\n📊 ENHANCED CHALLENGE STATUS:")
    print(f"Account: {status['account_id']}")
    print(f"Challenge: {status['challenge_type']}")
    print(f"Progress: {status['progress_percentage']:.1f}%")
    print(f"Strategy: {status['current_strategy']}")
    print(f"Risk Level: {status['risk_level']}")
    print(f"Challenge Phase: {status['challenge_phase']}")
    print(f"Trading Intensity: {status['trading_intensity']}")
    
    print("\n✅ OPTIMIZATIONS ACTIVE:")
    print("• Challenge-Specific Strategy")
    print("• Advanced Risk Protection")
    print("• Real-time Performance Analytics")
    print("• FTMO Timeline Optimization")
    
    if live_mode:
        print("\n⚠️  LIVE CHALLENGE MODE")
        confirm = input("Type 'CHALLENGE' to start FTMO challenge: ")
        if confirm != 'CHALLENGE':
            print("Challenge start cancelled")
            return
            
    print("\n🚀 Starting in 10 seconds...")
    import time
    time.sleep(10)
    
    # Start challenge trading
    success = challenge_system.start_challenge_trading()
    
    if success:
        print("🎉 FTMO CHALLENGE LAUNCHED!")
    else:
        print("❌ Challenge start failed")

if __name__ == "__main__":
    main()
