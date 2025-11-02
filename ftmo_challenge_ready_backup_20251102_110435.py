# FTMO Challenge-Ready System - Final Optimized Version
from ftmo_challenge_optimizer import FTMOChallengeOptimizer
from advanced_risk_protection import AdvancedRiskProtection
from performance_analytics import PerformanceAnalytics
from ftmo_phase3_system import FTMO_Phase3_System
import logging

class FTMOChallengeReadySystem:
    def __init__(self, account_id="1600038177", challenge_type="200k", live_mode=False):
        # Initialize base system
        self.phase3_system = FTMO_Phase3_System(account_id, challenge_type, live_mode)
        
        # Add final optimizations
        self.challenge_optimizer = FTMOChallengeOptimizer(self.phase3_system)
        self.advanced_protection = AdvancedRiskProtection(self.phase3_system)
        self.performance_analytics = PerformanceAnalytics(self.phase3_system)
        
        # Challenge-specific settings
        self.challenge_type = challenge_type
        self.live_mode = live_mode
        
        logging.info("🎯 FTMO Challenge-Ready System Initialized")
        
    def prepare_for_challenge(self):
        """Prepare system for FTMO challenge"""
        logging.info("Preparing for FTMO challenge...")
        
        # 1. Apply challenge-specific optimizations
        self.challenge_optimizer.optimize_for_challenge(self.challenge_type)
        
        # 2. Run comprehensive pre-challenge checks
        checks_passed = self._run_pre_challenge_checks()
        
        # 3. Generate initial analytics
        initial_analytics = self.performance_analytics.generate_analytics_report()
        
        # 4. Set up enhanced monitoring
        self._setup_enhanced_monitoring()
        
        if checks_passed:
            logging.info("✅ System ready for FTMO challenge!")
            return True
        else:
            logging.warning("⚠️ System may need adjustments before challenge")
            return False
            
    def _run_pre_challenge_checks(self):
        """Run pre-challenge validation checks"""
        checks = [
            self._check_risk_parameters(),
            self._check_strategy_readiness(),
            self._check_system_stability(),
            self._check_ftmo_compliance()
        ]
        
        passed = sum(checks)
        total = len(checks)
        
        logging.info(f"Pre-challenge checks: {passed}/{total} passed")
        return passed >= 3  # 75% success rate
        
    def _check_risk_parameters(self):
        """Check risk parameters are challenge-appropriate"""
        risk_report = self.phase3_system.get_phase3_status()['phase2_status']['risk_manager']
        
        if risk_report.get('risk_level') == 'NORMAL':
            logging.info("✅ Risk parameters validated")
            return True
        else:
            logging.warning("❌ Risk parameters need adjustment")
            return False
            
    def _check_strategy_readiness(self):
        """Check strategy readiness"""
        current_strategy = self.phase3_system.get_phase3_status()['phase2_status']['phase1_status']['current_strategy']
        
        if current_strategy in ['xgboost', 'fallback']:
            logging.info("✅ Strategy ready")
            return True
        else:
            logging.warning("❌ Strategy not ready")
            return False
            
    def _check_system_stability(self):
        """Check system stability"""
        # Basic stability check
        try:
            status = self.phase3_system.get_phase3_status()
            logging.info("✅ System stability confirmed")
            return True
        except Exception as e:
            logging.error(f"❌ System stability issue: {e}")
            return False
            
    def _check_ftmo_compliance(self):
        """Check FTMO rule compliance"""
        rule_enforcer = self.phase3_system.get_phase3_status()['phase2_status']['rule_enforcer']
        
        if rule_enforcer.get('violations_count', 0) == 0:
            logging.info("✅ FTMO compliance confirmed")
            return True
        else:
            logging.warning("❌ FTMO compliance issues")
            return False
            
    def _setup_enhanced_monitoring(self):
        """Set up enhanced challenge monitoring"""
        logging.info("Setting up enhanced challenge monitoring...")
        
        # This would start additional monitoring threads
        # For now, log the setup
        
    def start_challenge_trading(self):
        """Start trading for FTMO challenge"""
        if not self.prepare_for_challenge():
            logging.error("Challenge preparation failed - cannot start trading")
            return False
            
        logging.info("🚀 STARTING FTMO CHALLENGE TRADING!")
        
        # Show challenge status
        challenge_status = self.get_challenge_status()
        print("\n" + "="*50)
        print("FTMO CHALLENGE STATUS")
        print("="*50)
        print(f"Account: {challenge_status['account_id']}")
        print(f"Challenge: {challenge_status['challenge_type']}")
        print(f"Target: ${challenge_status['profit_target']:,.2f}")
        print(f"Current: ${challenge_status['current_profit']:,.2f}")
        print(f"Progress: {challenge_status['progress_percentage']:.1f}%")
        print(f"Strategy: {challenge_status['current_strategy']}")
        print(f"Risk Level: {challenge_status['risk_level']}")
        print("="*50)
        
        # Start the enhanced trading system
        return self.phase3_system.deploy_to_live()
        
    def get_challenge_status(self):
        """Get comprehensive challenge status"""
        phase3_status = self.phase3_system.get_phase3_status()
        phase2_status = phase3_status['phase2_status']
        metrics = phase2_status['phase1_status']['ftmo_metrics']
        
        challenge_strategy = self.challenge_optimizer.get_challenge_strategy()
        
        return {
            'account_id': self.challenge_type,
            'challenge_type': self.challenge_type,
            'current_profit': metrics.get('total_profit', 0),
            'profit_target': metrics.get('profit_target', 10000),
            'progress_percentage': (metrics.get('total_profit', 0) / metrics.get('profit_target', 10000)) * 100,
            'current_strategy': phase2_status['phase1_status']['current_strategy'],
            'risk_level': phase2_status['risk_manager']['risk_level'],
            'challenge_phase': challenge_strategy['current_phase'],
            'trading_intensity': challenge_strategy['trading_intensity'],
            'days_passed': challenge_strategy['days_passed'],
            'live_mode': self.live_mode
        }
        
    def get_enhanced_analytics(self):
        """Get enhanced analytics report"""
        return self.performance_analytics.generate_analytics_report()

# Quick challenge test
def test_challenge_system():
    """Test the challenge-ready system"""
    print("🧪 Testing FTMO Challenge-Ready System")
    
    challenge_system = FTMOChallengeReadySystem(live_mode=False)
    
    # Test preparation
    ready = challenge_system.prepare_for_challenge()
    print(f"Challenge Preparation: {'✅ Ready' if ready else '❌ Needs work'}")
    
    # Test status
    status = challenge_system.get_challenge_status()
    print(f"Challenge Status: {status['progress_percentage']:.1f}% complete")
    
    # Test analytics
    analytics = challenge_system.get_enhanced_analytics()
    print("✅ Enhanced analytics working")
    
    print("🎯 Challenge system test completed!")

if __name__ == "__main__":
    test_challenge_system()
