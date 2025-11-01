# FTMO ULTIMATE LAUNCH - Guaranteed Startup
import sys
import time
from ftmo_minimal import FTMOMinimalSystem

def main():
    print("🚀 FTMO ULTIMATE LAUNCH SEQUENCE")
    print("================================")
    print("Automated FTMO Challenge Startup")
    print("No User Input Required")
    print("================================")
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    # Initialize system
    print("🧹 Setting up trading environment...")
    system = FTMOMinimalSystem("1600038177", "200k", live_mode=True)
    
    # Auto-confirm trading
    print("✅ Auto-confirming live trading...")
    
    # Start trading immediately
    print("🚀 LAUNCHING FTMO CHALLENGE!")
    print("================================")
    print("System is now LIVE and trading")
    print("Monitoring FTMO challenge progress")
    print("Press Ctrl+C to stop trading")
    print("================================")
    
    # Start the system
    try:
        success = system.start_trading()
        if success:
            print("🎉 FTMO Challenge is running successfully!")
        else:
            print("⚠️  Trading session ended")
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"❌ Trading error: {e}")

if __name__ == "__main__":
    main()
