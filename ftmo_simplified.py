# FTMO Simplified Challenge System - No Port Conflicts
from ftmo_phase2_system import FTMO_Phase2_System
from ftmo_challenge_optimizer import FTMOChallengeOptimizer
from advanced_risk_protection import AdvancedRiskProtection
from performance_analytics import PerformanceAnalytics
import logging
from advanced_port_manager import setup_ftmo_environment

class FTMOSimplifiedChallenge:
    def __init__(self, account_id="1600038177", challenge_type="200k", live_mode=False):
        # Set up clean environment first
        self.port_manager = setup_ftmo_environment()
        
        # Initialize Phase 2 system directly (avoids Phase 3 complexity)
        self.phase2_system = FTMO_Phase2_System(account_id, challenge_type)
        
        # Add optimizations
        self.challenge_optimizer = FTMOChallengeOptimizer(self.phase2_system)
        self.advanced_protection = AdvancedRiskProtection(self.phase2_system)
        self.performance_analytics = PerformanceAnalytics(self.phase2_system)
        
        self.live_mode = live_mode
        logging.info("Simplified FTMO Challenge System initialized")
        
    def start_challenge(self):
        """Start the FTMO challenge"""
        logging.info("Starting FTMO Challenge...")
        
        # Prepare system
        self.challenge_optimizer.optimize_for_challenge("200k")
        
        # Show status
        status = self.get_challenge_status()
        analytics = self.performance_analytics.generate_analytics_report()
        
        print("\n🎯 FTMO CHALLENGE STATUS")
        print("=" * 40)
        print(f"Account: {status['account_id']}")
        print(f"Challenge: {status['challenge_type']}")
        print(f"Progress: {status['progress_percentage']:.1f}%")
        print(f"Strategy: {status['current_strategy']}")
        print(f"Risk Level: {status['risk_level']}")
        print("=" * 40)
        
        print("\n✅ SYSTEM OPTIMIZATIONS ACTIVE:")
        print("• Challenge-Specific Strategy")
        print("• Advanced Risk Protection")
        print("• Real-time Performance Analytics")
        
        if self.live_mode:
            print("\n⚠️  LIVE MODE - Trading with real money")
            confirm = input("Type 'GO' to continue: ")
            if confirm != 'GO':
                return False
                
        print("\n🚀 Starting challenge trading...")
        
        # Start the Phase 2 system (which includes all functionality)
        try:
            self.phase2_system.start_enhanced_trading()
            return True
        except Exception as e:
            logging.error(f"Challenge start failed: {e}")
            return False
            
    def get_challenge_status(self):
        """Get challenge status"""
        phase2_status = self.phase2_system.get_phase2_status()
        metrics = phase2_status['phase1_status']['ftmo_metrics']
        
        return {
            'account_id': "1600038177",
            'challenge_type': "200k",
            'current_profit': metrics.get('total_profit', 0),
            'profit_target': metrics.get('profit_target', 10000),
            'progress_percentage': (metrics.get('total_profit', 0) / metrics.get('profit_target', 10000)) * 100,
            'current_strategy': phase2_status['phase1_status']['current_strategy'],
            'risk_level': phase2_status['risk_manager']['risk_level'],
            'live_mode': self.live_mode
        }
        
    def quick_test(self):
        """Quick system test"""
        print("🧪 Quick System Test")
        
        try:
            status = self.get_challenge_status()
            print(f"✅ System status: {status['current_strategy']} strategy")
            print(f"✅ Risk level: {status['risk_level']}")
            print(f"✅ Progress: {status['progress_percentage']:.1f}%")
            
            # Test optimizations
            self.challenge_optimizer.optimize_for_challenge("200k")
            print("✅ Challenge optimizations applied")
            
            return True
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

# Simple test function
def test_simplified_system():
    """Test the simplified system"""
    print("Testing Simplified FTMO System...")
    
    system = FTMOSimplifiedChallenge(live_mode=False)
    success = system.quick_test()
    
    if success:
        print("🎉 Simplified system test passed!")
    else:
        print("❌ Simplified system test failed")
        
    return success

if __name__ == "__main__":
    test_simplified_system()
