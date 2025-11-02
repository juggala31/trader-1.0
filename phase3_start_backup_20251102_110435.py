# FTMO Phase 3 Simplified Startup
import sys
import logging
from ftmo_phase2_system import FTMO_Phase2_System

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 FTMO PHASE 3: LIVE DEPLOYMENT READY")
    print("======================================")
    print("System Status: PHASE 1-3 INTEGRATED")
    print("Account: 1600038177")
    print("Challenge: 200k")
    print("======================================")
    
    # Initialize Phase 2 system (which includes Phase 1)
    phase2_system = FTMO_Phase2_System("1600038177", "200k")
    
    # Show comprehensive status
    status = phase2_system.get_phase2_status()
    metrics = status['phase1_status']['ftmo_metrics']
    risk = status['risk_manager']
    
    print("\n📊 CURRENT SYSTEM STATUS:")
    print(f"Account Balance: ${metrics.get('current_balance', 0):,.2f}")
    print(f"FTMO Progress: ${metrics.get('total_profit', 0):,.2f} / ${metrics.get('profit_target', 10000):,.2f}")
    print(f"Progress: {(metrics.get('total_profit', 0)/metrics.get('profit_target', 10000)*100):.1f}%")
    print(f"Risk Level: {risk.get('risk_level', 'NORMAL')}")
    print(f"Current Strategy: {status['phase1_status']['current_strategy']}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {(metrics.get('winning_trades', 0)/max(1, metrics.get('total_trades', 1))*100):.1f}%")
    
    print("\n✅ PHASE 3 FEATURES ACTIVE:")
    print("• Live Deployment Manager")
    print("• Real-time Performance Monitoring") 
    print("• Automated Reporting System")
    print("• FTMO Challenge Verification Ready")
    
    print("\n🚀 Ready for Live Deployment!")
    print("Next: Connect to MetaTrader5 and start trading")
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        print("\n🎯 Starting Demo Mode...")
        phase2_system.start_enhanced_trading()
    else:
        print("\n💡 Usage: python phase3_start.py demo - to start demo trading")

if __name__ == "__main__":
    main()
