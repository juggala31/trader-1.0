# FTMO Phase 2 Integration - Enhanced Risk & Optimization
from enhanced_risk_manager import EnhancedRiskManager
from drawdown_protection import DrawdownProtection
from performance_optimizer import PerformanceOptimizer
from ftmo_rule_enforcer import FTMO_Rule_Enforcer
from ftmo_mt5_integration import FTMO_MT5_Integration
import logging
import time

class FTMO_Phase2_System:
    def __init__(self, account_id="1600038177", challenge_type="200k"):
        # Initialize Phase 1 system
        self.phase1_system = FTMO_MT5_Integration(account_id, challenge_type)
        
        # Initialize Phase 2 components
        self.risk_manager = EnhancedRiskManager(self.phase1_system.ftmo_logger)
        self.drawdown_protection = DrawdownProtection(self.phase1_system.ftmo_logger)
        self.performance_optimizer = PerformanceOptimizer(self.phase1_system.ftmo_logger)
        self.rule_enforcer = FTMO_Rule_Enforcer(self.phase1_system.ftmo_logger)
        
        # Phase 2 state
        self.phase2_active = True
        self.optimization_enabled = True
        
        logging.info("FTMO Phase 2 System initialized")
        
    def enhanced_trading_cycle(self):
        """Enhanced trading cycle with Phase 2 features"""
        # Step 1: Check rule enforcement
        trading_allowed, reason = self.rule_enforcer.is_trading_allowed()
        if not trading_allowed:
            logging.warning(f"Trading not allowed: {reason}")
            return
            
        # Step 2: Update risk management
        self.risk_manager.update_risk_level()
        
        # Step 3: Monitor drawdown
        current_drawdown = self.drawdown_protection.monitor_drawdown()
        
        # Step 4: Optimize parameters if needed
        if self.optimization_enabled:
            self.performance_optimizer.optimize_parameters()
            
        # Step 5: Execute enhanced trading
        for symbol in self.phase1_system.symbols:
            self._execute_enhanced_trade(symbol)
            
        # Step 6: Enforce rules
        violations = self.rule_enforcer.enforce_rules()
        if violations:
            logging.warning(f"Rule violations detected: {len(violations)}")
            
    def _execute_enhanced_trade(self, symbol):
        """Execute trade with Phase 2 enhancements"""
        # Get trading signal
        signal = self.phase1_system.get_trading_signal(symbol)
        if not signal or signal['action'] == 'HOLD':
            return
            
        # Check risk approval
        approved, reason = self.risk_manager.should_trade_be_executed(symbol, signal['action'])
        if not approved:
            logging.info(f"Trade not approved: {reason}")
            return
            
        # Calculate enhanced position size
        volatility = 1.0  # Would be calculated from market data
        position_size = self.risk_manager.calculate_dynamic_position_size(
            symbol, signal['confidence'], volatility
        )
        
        # Apply drawdown protection
        drawdown_multiplier = self.drawdown_protection.get_position_size_multiplier()
        final_size = position_size * drawdown_multiplier
        
        logging.info(f"Enhanced trade: {signal['action']} {symbol}, Size: {final_size:.4f}")
        
        # Execute trade (would integrate with actual MT5 execution)
        # self.phase1_system.execute_trade(symbol, signal, final_size)
        
    def get_phase2_status(self):
        """Get comprehensive Phase 2 status"""
        return {
            'phase1_status': self.phase1_system.get_system_status(),
            'risk_manager': self.risk_manager.get_risk_report(),
            'drawdown_protection': self.drawdown_protection.get_protection_status(),
            'performance_optimizer': self.performance_optimizer.get_optimization_status(),
            'rule_enforcer': self.rule_enforcer.get_enforcement_status(),
            'phase2_active': self.phase2_active
        }
        
    def start_enhanced_trading(self):
        """Start enhanced trading with Phase 2 features"""
        logging.info("Starting FTMO Phase 2 Enhanced Trading")
        
        if not self.phase1_system.connect_mt5():
            return False
            
        try:
            while True:
                self.enhanced_trading_cycle()
                time.sleep(60)  # 1 minute cycles
                
        except KeyboardInterrupt:
            logging.info("Enhanced trading stopped by user")
        finally:
            self.phase1_system.disconnect_mt5()

# Quick test
def test_phase2():
    """Test Phase 2 components"""
    print("Testing FTMO Phase 2 System...")
    
    phase2_system = FTMO_Phase2_System()
    status = phase2_system.get_phase2_status()
    
    print("✅ Phase 2 System Initialized")
    print(f"Risk Level: {status['risk_manager']['risk_level']}")
    print(f"Drawdown Protection: {status['drawdown_protection']['protection_active']}")
    print(f"Rule Enforcement: {status['rule_enforcer']['trading_halted']}")
    
    print("🎉 Phase 2 Test Completed!")

if __name__ == "__main__":
    test_phase2()

