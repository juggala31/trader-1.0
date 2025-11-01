# FTMO Minimal Working System - Bypasses Analytics Issues
from ftmo_phase2_system import FTMO_Phase2_System
from advanced_port_manager import setup_ftmo_environment
import logging

class FTMOMinimalSystem:
    def __init__(self, account_id="1600038177", challenge_type="200k", live_mode=False):
        # Set up clean environment
        self.port_manager = setup_ftmo_environment()
        
        # Initialize only the core Phase 2 system
        self.phase2_system = FTMO_Phase2_System(account_id, challenge_type)
        
        self.live_mode = live_mode
        logging.info("FTMO Minimal System initialized")
        
    def start_trading(self):
        """Start trading with minimal dependencies"""
        logging.info("Starting FTMO Minimal System...")
        
        # Show simple status
        status = self.get_system_status()
        
        print("\n🎯 FTMO MINIMAL SYSTEM")
        print("=" * 30)
        print(f"Account: {status['account_id']}")
        print(f"Challenge: {status['challenge_type']}")
        print(f"Balance: ${status['balance']:,.2f}")
        print(f"Target: ${status['target']:,.2f}")
        print(f"Strategy: {status['strategy']}")
        print(f"Risk Level: {status['risk_level']}")
        print("=" * 30)
        
        print("\n✅ CORE FEATURES ACTIVE:")
        print("• Strategy Orchestration")
        print("• Risk Management")
        print("• FTMO Compliance")
        print("• Real-time Trading")
        
        if self.live_mode:
            print("\n⚠️  LIVE TRADING MODE")
            confirm = input("Type 'TRADE' to start: ")
            if confirm != 'TRADE':
                return False
                
        print("\n🚀 Starting trading system...")
        print("Press Ctrl+C to stop")
        
        try:
            # Start the proven Phase 2 system
            self.phase2_system.start_enhanced_trading()
            return True
        except KeyboardInterrupt:
            print("\n🛑 Trading stopped by user")
            return True
        except Exception as e:
            logging.error(f"Trading error: {e}")
            return False
            
    def get_system_status(self):
        """Get simple system status"""
        phase2_status = self.phase2_system.get_phase2_status()
        metrics = phase2_status['phase1_status']['ftmo_metrics']
        
        return {
            'account_id': "1600038177",
            'challenge_type': "200k",
            'balance': metrics.get('current_balance', 200000),
            'target': metrics.get('profit_target', 10000),
            'progress': (metrics.get('total_profit', 0) / 10000) * 100,
            'strategy': phase2_status['phase1_status']['current_strategy'],
            'risk_level': phase2_status['risk_manager']['risk_level']
        }
        
    def quick_test(self):
        """Quick functionality test"""
        print("🧪 Quick System Test")
        
        try:
            status = self.get_system_status()
            print(f"✅ System status: {status['strategy']} strategy")
            print(f"✅ Risk level: {status['risk_level']}")
            print(f"✅ Ready for trading")
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def main():
    """Main function for minimal system"""
    import sys
    
    print("🚀 FTMO MINIMAL TRADING SYSTEM")
    print("==============================")
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        system = FTMOMinimalSystem()
        system.quick_test()
        return
        
    live_mode = len(sys.argv) > 1 and sys.argv[1] == "live"
    
    # Start the system
    system = FTMOMinimalSystem(live_mode=live_mode)
    system.start_trading()

if __name__ == "__main__":
    main()
